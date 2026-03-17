import numpy as np
import torch
from utils.utils import *
import os
from threadpoolctl import threadpool_limits
from torch.nn.utils.rnn import pad_sequence
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import recall_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import log_loss
from dataset_modules.dataset_generic import Generic_MIL_Dataset, save_splits, Generic_WSI_Dataset

from sklearn.preprocessing import label_binarize
from sklearn.metrics import auc as calc_auc
from tqdm import tqdm

from collections import OrderedDict

from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score, classification_report,balanced_accuracy_score, f1_score
import time

from tqdm import tqdm
#device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
def _oddize(k: int) -> int:
    k = max(1, int(k))
    return k if k % 2 == 1 else k + 1
class LinearProbe(nn.Module):
    def __init__(self, feature_dim, num_classes):
        super(LinearProbe, self).__init__()
        self.fc = nn.Linear(feature_dim, num_classes)

    def forward(self, x):
        logits = self.fc(x)
        return logits


class Accuracy_Logger(object):
    """Accuracy logger"""
    def __init__(self, n_classes):
        super().__init__()
        self.n_classes = n_classes
        self.initialize()

    def initialize(self):
        self.data = [{"count": 0, "correct": 0} for i in range(self.n_classes)]
    
    def log(self, Y_hat, Y):
        Y_hat = int(Y_hat)
        Y = int(Y)
        self.data[Y]["count"] += 1
        self.data[Y]["correct"] += (Y_hat == Y)

    def log_wsi(self, Y_hat, Y):
        for label_class in torch.unique(Y):
            cls_mask = Y == label_class
            self.data[label_class]["count"] += cls_mask.sum()
            self.data[label_class]["correct"] += (Y_hat[cls_mask] == Y[cls_mask]).sum()

         
    def log_batch(self, Y_hat, Y):
        Y_hat = np.array(Y_hat).astype(int)
        Y = np.array(Y).astype(int)
        for label_class in np.unique(Y):
            cls_mask = Y == label_class
            self.data[label_class]["count"] += cls_mask.sum()
            self.data[label_class]["correct"] += (Y_hat[cls_mask] == Y[cls_mask]).sum()
    
    def get_summary(self, c):
        count = self.data[c]["count"] 
        correct = self.data[c]["correct"]
        
        if count == 0: 
            acc = None
        else:
            acc = float(correct) / count
        
        return acc, correct, count

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=20, stop_epoch=50, verbose=False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 20
            stop_epoch (int): Earliest epoch possible for stopping
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
        """
        self.patience = patience
        self.stop_epoch = stop_epoch
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf

    def __call__(self, epoch, val_loss, model,model_base = None, ckpt_name = 'checkpoint.pt',fm_ckpt_name = ''):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, ckpt_name)
            if model_base is not None:
                self.save_checkpoint(val_loss, model_base, fm_ckpt_name)
        elif score < self.best_score:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience and epoch > self.stop_epoch:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, ckpt_name)
            if model_base is not None:
                self.save_checkpoint(val_loss, model_base, fm_ckpt_name)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, ckpt_name):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), ckpt_name)
        self.val_loss_min = val_loss

def train(datasets, cur, args):
    """   
        train for a single fold
    """
    print('\nTraining Fold {}!'.format(cur))
    writer_dir = os.path.join(args.results_dir, str(cur))
    if not os.path.isdir(writer_dir):
        os.mkdir(writer_dir)

    if args.log_data:
        from tensorboardX import SummaryWriter
        writer = SummaryWriter(writer_dir, flush_secs=15)
    else:
        writer = None
    print('\nInit train/val/test splits...', end=' ')
    train_split, val_split, test_split = datasets
    if val_split is not None:
        save_splits(datasets, ['train', 'val', 'test'], os.path.join(args.results_dir, 'splits_{}.csv'.format(cur)))
    else:
        save_splits(datasets, ['train', 'test'], os.path.join(args.results_dir, 'splits_{}.csv'.format(cur)))
    print('Done!')
    print("Training on {} samples".format(len(train_split)))
    if val_split is not None:
        print("Validating on {} samples".format(len(val_split)))
    print("Testing on {} samples".format(len(test_split)))

    print('\nInit loss function...', end=' ')
    loss_fn = nn.CrossEntropyLoss()
    print('Done!')
    print('\nInit Model...', end=' ')
    model_dict = {"dropout": args.drop_out, 
                  'n_classes': args.n_classes, 
                  "embed_dim": args.embed_dim}
    args.max_window_size=[7]
    if args.model_size is not None and args.model_type != 'mil':
        model_dict.update({"size_arg": args.model_size})

    if args.model_name in ['CARE']:
        from transformers import AutoModel
        model_base = AutoModel.from_pretrained("Zipper-1/CARE",trust_remote_code=True)
        model = None
    else:
        raise NotImplementedError('Model "{}" not supported'.format(args.model_name))


    if model_base is not None:
        if args.model_type not in ['finetuning','random_init']:
            if args.model_name is not None:
                model_base.eval()

                for param in model_base.parameters():
                    param.requires_grad = False


    if args.multi_gpu == 0 or len(args.multi_gpu) == 1:
        torch.cuda.set_device(args.multi_gpu[0])
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if model is not None:
            _ = model.to(device)
        if args.model_name is not None and model_base is not None:
            model_base = model_base.to(device)
    else:
        device = torch.device(f"cuda:{args.multi_gpu[0]}")
        model = nn.DataParallel(model, device_ids=args.multi_gpu)  
        if args.model_name is not None and model_base is not None:
            model_base = model_base.to(device)
            model_base1 = model_base1.to(device)
        model = model.to(device)
    print('Done!')
    print_network(model_base)

    print('\nInit optimizer ...', end=' ')
    if args.model_type in ['finetuning','random_init'] and model_base is not None:
        optimizer = torch.optim.Adam(list(model.parameters()) + list(model_base.parameters()), lr=args.lr, weight_decay=args.reg)
    elif args.model_type in ['finetuning','random_init']:
        optimizer = torch.optim.Adam(list(model.parameters()), lr=args.lr, weight_decay=args.reg)
    else:
        optimizer = None#get_optim(model, args)
    print('Done!')
    print('\nInit Loaders...', end=' ')
    if issubclass(train_split.__class__, Generic_MIL_Dataset):
        train_split.load_from_h5(True)
        train_split.return_slideid()
        test_split.load_from_h5(True)
        test_split.return_slideid()
        if val_split is not None:
            val_split.load_from_h5(True)
            val_split.return_slideid() 
            val_loader = get_coords_id_loader(val_split,training = False, weighted = False, batch_size=args.batch_size,num_workers=1)
        else:
            val_loader = None
        if args.model_type in ['logistic_regression','KNN']:
            train_loader = get_coords_id_loader(train_split,training = True, weighted = False, batch_size=args.batch_size,num_workers=1)
        else:
            train_loader = get_coords_id_loader(train_split,training = True, weighted = True, batch_size=args.batch_size,num_workers=1)
        test_loader = get_coords_id_loader(test_split,training = False, weighted = False, batch_size=args.batch_size,num_workers=1)
    elif issubclass(train_split.__class__, Generic_WSI_Dataset):
        train_split.load_from_h5(True)
        train_split.return_slideid()
        test_split.load_from_h5(True)
        if val_split is not None:
            val_split.load_from_h5(True)
            val_split.return_slideid() 
            val_loader = get_wsi_loader(val_split,training = False, weighted = False, batch_size=args.batch_size,num_workers=1)
        else:
            val_loader = None
        if args.model_type in ['logistic_regression','KNN']:
            train_loader = get_wsi_loader(train_split,training = True, weighted = False, batch_size=args.batch_size,num_workers=1)
        else:
            train_loader = get_wsi_loader(train_split,training = True, weighted = True, batch_size=args.batch_size,num_workers=1)
        test_split.return_slideid() 
        test_loader = get_wsi_loader(test_split,training = False, weighted = False, batch_size=args.batch_size,num_workers=1)
    else:
        raise NotImplementedError('Dataset type "{}" not supported'.format(train_split.__class__))
    print('Done!')

    print('\nSetup EarlyStopping...', end=' ')
    if args.early_stopping:
        early_stopping = EarlyStopping(patience = 15, stop_epoch=20, verbose = True)#(patience = 15, stop_epoch=20, verbose = True)'
    else:
        early_stopping = None
    print('Done!')
    

    
    for epoch in (range(args.max_epochs)):
        t0 = time.time()
        if args.model_type in ['linear','finetuning','random_init']:
            train_linear_loop(epoch, model, train_loader, optimizer, args.n_classes, writer, loss_fn,model_base = model_base,
                       num_region = args.num_region, args=args)
            stop = validate_linear(cur, epoch, model, val_loader, args.n_classes, 
                early_stopping, writer, loss_fn, args.results_dir,model_base = model_base,num_region = args.num_region, args=args)
        else:
            #print(epoch,2)
            stop = True
        if stop: 
            break
        print(epoch,time.time()-t0)
    if args.model_type in ['logistic_regression','KNN','t3_patch']:

        if args.model_name in ['CARE']:
            train_features, train_labels, val_features, val_labels,test_features, test_labels,test_slide_id = get_slide_fea(train_loader,val_loader,test_loader,args,cur,
                                                                                                               model_base,device = device)
        elif args.model_name in ['CHIEF','PRISM','TITAN','TANGLE','FEATHER','GIGAPATH']:
            features=[]
            labels=[]
            for data, label, slide_id in train_loader:
                features.append(data.numpy())
                labels.append(label.numpy())
            train_features=np.concatenate(features)
            train_labels=np.concatenate(labels)
            features=[]
            labels=[]
            if val_loader is not None:
                for data, label, slide_id in val_loader:
                    features.append(data.numpy())
                    labels.append(label.numpy())
                val_features = np.concatenate(features)
                val_labels = np.concatenate(labels)
            else:
                val_features = None
                val_labels = None
            features=[]
            labels=[]
            test_slide_id = []
            for data, label, slide_id in test_loader:
                features.append(data.numpy())
                labels.append(label.numpy())
                test_slide_id.extend(slide_id)
            test_features = np.concatenate(features)
            test_labels = np.concatenate(labels)
        else:
            raise NotImplementedError('Model "{}" not supported'.format(args.model_name))
        if args.model_type in ['KNN']: 
            test_auc, test_acc, val_auc, val_acc = train_knn(train_features, train_labels, val_features, val_labels,test_features, 
                                                                              test_labels,test_slide_id, args=args)
        elif args.model_type in ['logistic_regression']: 
            test_auc, test_acc, val_auc,  val_acc = train_logistic_regression_loop(train_features, train_labels, val_features, val_labels,test_features,
                                                                              test_labels,test_slide_id, args=args)
        return None, test_auc, val_auc, test_acc, val_acc
    else:
        if args.early_stopping:
            model.load_state_dict(torch.load(os.path.join(args.results_dir, "s_{}_checkpoint.pt".format(cur))))
            if model_base is not None:
                model_base.load_state_dict(torch.load(os.path.join(args.results_dir, "model_base_{}_checkpoint.pt".format(cur))))

        else:
            torch.save(model.state_dict(), os.path.join(args.results_dir, "s_{}_checkpoint.pt".format(cur)))
            torch.save(model_base.state_dict(), os.path.join(args.results_dir, "model_base_{}_checkpoint.pt".format(cur)))
        _, val_error, val_auc, _= summary(model, val_loader, args.n_classes,args.model_type, max_window_size=args.max_window_size, 
                                        patch_size =args.step_size,model_base = model_base,num_region = args.num_region, args=args)
        print('Val error: {:.4f}, ROC AUC: {:.4f}'.format(val_error, val_auc))

        results_dict, test_error, test_auc, acc_logger = summary(model, test_loader, args.n_classes,args.model_type, max_window_size=args.max_window_size, 
                                                                patch_size =args.step_size,model_base = model_base,num_region = args.num_region, args=args
                                                                )
        print('Test error: {:.4f}, ROC AUC: {:.4f}'.format(test_error, test_auc))
        for i in range(args.n_classes):
            acc, correct, count = acc_logger.get_summary(i)
            print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))

            if writer:
                writer.add_scalar('final/test_class_{}_acc'.format(i), acc, 0)
        if writer:
            writer.add_scalar('final/val_error', val_error, 0)
            writer.add_scalar('final/val_auc', val_auc, 0)
            writer.add_scalar('final/test_error', test_error, 0)
            writer.add_scalar('final/test_auc', test_auc, 0)
            writer.close()
        return results_dict, test_auc, val_auc, 1-test_error, 1-val_error 


def get_slide_fea(train_loader,val_loader,test_loader,args,cur,model_base = None,device = None):
    if val_loader is not None:
        loaders = [train_loader,val_loader,test_loader]
    else:
        loaders = [train_loader,test_loader]
    datasets = []
    labels = []
    wsi_path = args.cache_path
    if args.subdataset is None:
        save_wsi_dir = os.path.join(wsi_path,args.dataset,'wsi_embedding',args.experiment_target,args.model_name)
    else:
        save_wsi_dir = os.path.join(wsi_path,args.dataset,args.subdataset,'wsi_embedding',args.experiment_target,args.model_name)
    if args.model_name not in ['mean_pooling'] and args.roi_fea == False:
        os.makedirs(save_wsi_dir, exist_ok=True)
    #print(save_wsi_dir)

    test_slide_id = []
    for loader_idx,loader in enumerate(loaders):
        dataset = []
        label_sp = []
        for batch_idx, (data, label,corrds,N_values,slide_id) in enumerate(loader):
            if loader_idx == len(loaders)-1:
                test_slide_id.extend(slide_id)
            #print(slide_id)
            if data.shape[0] != 1:
                raise ValueError(f"Invalid data shape: expected first dimension to be 1, but got {data.shape}")
            if args.task not in ['t3_patch']:
                data = data.to(device)
            



            if model_base is not None:
                slide_path = os.path.join(save_wsi_dir, slide_id[0] + '.pt')
                
                #if (args.auto_skip or cur!=0) and os.path.isfile(slide_path) and args.roi_fea == False:
                if args.auto_skip and os.path.isfile(slide_path) and args.roi_fea == False:
                    #print(args.auto_skip,cur,slide_path)
                    data = torch.load(slide_path,map_location='cpu')
                else:
                    model_base.eval()
                    N_values = N_values.to(device)#data N*1024  
                    for i in range(corrds.shape[0]):
                        diffs = np.linalg.norm(corrds[i][1:] - corrds[i][:-1], axis=1)
                        count_512 = np.sum(diffs == 512)
                        count_1024 = np.sum(diffs == 1024)
                        count_256  = np.sum(diffs == 256)
                        count_128  = np.sum(diffs == 128)
                        counts = {
                            256: count_256,
                            128: count_128,
                            1024: count_1024,
                            512: count_512,
                        }

                        patch_size = max(counts, key=counts.get)
                        if args.model_name in ['CARE']:
                            corrds[i] = corrds[i] // patch_size
                    corrds = corrds.to(device)
                    corrds = corrds#.half()
                    data = data#.half()
                    if args.model_name in ['CARE']:
                        corrds,N_values = corrds.to(device),N_values.to(device)
                        with torch.inference_mode():
                            out = model_base(data,N_values, corrds)     
                            data = out.wsi_embedding
                            aux_loss = out.aux_loss
                    data = data.detach().cpu()
                    if args.roi_fea == False and args.auto_skip:
                        torch.save(data, slide_path)
        label_sp = np.concatenate(label_sp, axis=0)
        dataset = np.concatenate(dataset, axis=0)
        datasets.append(dataset)
        labels.append(label_sp)
    if val_loader is not None:
        return datasets[0], labels[0], datasets[1], labels[1], datasets[2], labels[2],test_slide_id
    else:
        return datasets[0], labels[0], None, None, datasets[1], labels[1],test_slide_id




def _odd(k:int)->int:
    k = max(1, int(k))
    return k if k % 2 == 1 else k + 1
def train_knn(train_features, train_labels, val_features=None, val_labels=None,test_features = None, test_labels = None,test_slide_id = None,args = None):
    
    
    if args.k >10:
        n_train = len(train_labels)
        num_knn = _oddize(np.sqrt(n_train))
        num_knn = int(np.clip(num_knn, 3, 13))
        weights_knn = "distance" 
        knn_clf = make_pipeline(
                StandardScaler(),              
                KNeighborsClassifier(
                            n_neighbors=num_knn,
                            weights=weights_knn,         
                            metric="euclidean"         
                        )
                    )
        knn_clf.fit(train_features, train_labels)

        test_proba = knn_clf.predict_proba(test_features)

        test_pred = knn_clf.predict(test_features)
        if val_labels is not None:
            val_proba  = knn_clf.predict_proba(val_features)  
            val_pred  = knn_clf.predict(val_features)  
            bal_val_acc = balanced_accuracy_score(val_labels, val_pred)

            weighted_val_f1 = f1_score(val_labels, val_pred, average='weighted')
        else:
            bal_val_acc = 0 
            weighted_val_f1 = 0
    else:
        assert (val_features is not None) and (val_labels is not None), \
                "args.k <= 10  val_features / val_labels"
        k_candidates = [3,5,7,9,11,13,15,17,19,23]
        weight_candidates = ["uniform", "distance"]
        metric_candidates = ["euclidean"]#, "manhattan"

        best_score = -np.inf
        best_info = None
        for k in k_candidates:
            for w in weight_candidates:
                for m in metric_candidates:
                    pipe = make_pipeline(StandardScaler(),
                                        KNeighborsClassifier(n_neighbors=k, weights=w, metric=m))
                    pipe.fit(train_features, train_labels)
                    val_pred = pipe.predict(val_features)
                    score = balanced_accuracy_score(val_labels, val_pred)  
                    if score > best_score:
                        best_score = score
                        best_info = {"mode":"with_val_grid",
                                    "n_neighbors":k, "weights":w, "metric":m,
                                    "val_bal_acc": float(score)}
        print('best parameter:',best_info["n_neighbors"],best_info["weights"],best_info["metric"])
        knn_clf = make_pipeline(
            StandardScaler(),
            KNeighborsClassifier(
                n_neighbors=best_info["n_neighbors"],
                weights=best_info["weights"],
                metric=best_info["metric"]
            )
        )

        knn_clf.fit(train_features, train_labels)

        test_proba = knn_clf.predict_proba(test_features)

        test_pred = knn_clf.predict(test_features)
        if val_labels is not None:
            val_proba  = knn_clf.predict_proba(val_features)  
            val_pred  = knn_clf.predict(val_features)   
            bal_val_acc = balanced_accuracy_score(val_labels, val_pred)

            weighted_val_f1 = f1_score(val_labels, val_pred, average='weighted')

    balanced_acc = balanced_accuracy_score(test_labels, test_pred)
    weighted_f1 = f1_score(test_labels, test_pred, average='weighted')

    print('acc:',balanced_acc)
    per_class_acc = recall_score(test_labels, test_pred, average=None)
    for i in range((len(per_class_acc))):
        print('The class %d acc: %s'%(i,per_class_acc[i]))
    #classes, counts = np.unique(test_labels, return_counts=True)


    if test_proba.shape[1] == 2:
        auroc = roc_auc_score(test_labels, test_proba[:, 1])
        if val_labels is not None:
            val_auroc = roc_auc_score(val_labels, val_proba[:, 1])
        else:
            val_auroc = 0
        return auroc, balanced_acc, val_auroc, bal_val_acc

    else:

        if val_labels is not None:
            val_auroc = 0

        else:
            val_auroc = 0
        return weighted_f1, balanced_acc, weighted_val_f1, bal_val_acc


    
def train_logistic_regression_loop(train_features, train_labels, val_features=None, val_labels=None,test_featrures = None, test_labels = None,test_slide_id = None,args = None):

    if val_features is not None and val_labels is not None:

        log_spaced_values = np.logspace(np.log10(10e-6), np.log10(10e5), num=45)
        best_score = -float('inf')
        best_C = None
        logistic_reg_final = None
        for log2_coeff in tqdm(log_spaced_values, desc="Finding best C"):
            # suppress convergence warnings
            import warnings
            warnings.filterwarnings("ignore")
            
            logistic_reg = LogisticRegression(
                C=1/log2_coeff,
                fit_intercept=True,
                max_iter=1000,
                random_state=0,
                solver="lbfgs",
            )

            logistic_reg.fit(train_features, train_labels)
            # predict on val set
            val_loss = log_loss(val_labels, logistic_reg.predict_proba(val_features))
            score = -val_loss
            print(score)
            # score on val set
            if score > best_score:
                best_score = score
                best_C = log2_coeff
                logistic_reg_final = logistic_reg
        print(f"Best C: {best_C}")
    else:
        logistic_reg_final = LogisticRegression(
                C=0.5,
                fit_intercept=True,
                max_iter=10000,
                random_state=0,
                solver="lbfgs",
            )
        logistic_reg_final.fit(train_features, train_labels)


    num_classes = len(np.unique(train_labels))
    if num_classes == 2:
        test_proba = logistic_reg_final.predict_proba(test_featrures)[:, 1]
        roc_kwargs = {}
    else:
        test_proba = logistic_reg_final.predict_proba(test_featrures)
        roc_kwargs = {'multi_class': 'ovo', 'average': 'macro'}
    

    
    if val_labels is not None:
        val_pred = logistic_reg_final.predict(val_features)
        val_proba = logistic_reg_final.predict_proba(val_features)
        bal_val_acc = balanced_accuracy_score(val_labels, val_pred)

        weighted_val_f1 = f1_score(val_labels, val_pred, average='weighted')
    else:
        bal_val_acc = 0
        weighted_val_f1 = 0
    
    test_proba = logistic_reg_final.predict_proba(test_featrures)
    
    test_pred = logistic_reg_final.predict(test_featrures)
    balanced_acc = balanced_accuracy_score(test_labels, test_pred)
    weighted_f1 = f1_score(test_labels, test_pred, average='weighted')
    per_class_acc = recall_score(test_labels, test_pred, average=None)
    for i in range((len(per_class_acc))):
        print('The class %d acc: %s'%(i,per_class_acc[i]))








    if test_proba.shape[1] == 2:
        auroc = roc_auc_score(test_labels, test_proba[:, 1])
        if val_labels is not None:
            val_auroc = roc_auc_score(val_labels, val_proba[:, 1])
        else:
            val_auroc = None
        return auroc, balanced_acc, val_auroc, bal_val_acc

    else:
        if val_labels is not None:
            #val_auroc = roc_auc_score(val_labels, val_proba, multi_class='ovr')
            0
        else:
            val_auroc = None
        return weighted_f1, balanced_acc, weighted_val_f1, bal_val_acc








def train_linear_loop(epoch, model, loader, optimizer, n_classes, writer = None, loss_fn = None,model_base = None,num_region = None,args = None):   
    model.train()
    model.float()
    if args.model_type in ['finetuning','random_init'] or args.model_type== 'linear':
        amp_is = False
    else:
        amp_is = True
    if model_base is not None:
        if args.model_type not in ['finetuning','random_init']:
            model_base.eval()
        else:
            model_base.train()
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    #print(n_classes)
    train_loss = 0.
    train_error = 0.
    from torch.amp import autocast, GradScaler

    scaler = GradScaler('cuda')
    print('\n')
    for batch_idx, datas in enumerate(loader):
        if len(datas) == 3:
            data, label,corrds = datas
        else:
            data, label,corrds,N_values,slide_id = datas
        #print(slide_id)
        data, label = data.to(device), label.to(device)
        #if data.shape[1]>6000:
        #    continue
        if model_base is not None or args.model_name in ['mean_pooling']:
            N_values = N_values.to(device)#data N*1024  

            for i in range(corrds.shape[0]):
                diffs = np.linalg.norm(corrds[i][1:] - corrds[i][:-1], axis=1)
                count_512 = np.sum(diffs == 512)
                count_1024 = np.sum(diffs == 1024)
                patch_size = 512 if count_512 > count_1024 else 1024
                corrds[i] = corrds[i] // patch_size
            corrds = corrds.to(device)        
            corrds = corrds#.half()
            
            data = data#.half()
            #N_values = N_values.half()
            #with torch.amp.autocast('cuda'):
            optimizer.zero_grad()
            from contextlib import nullcontext
            ctx = autocast(device_type='cuda', enabled=amp_is) if amp_is else nullcontext()
            with ctx:
                if args.model_name in ['CARE']:
                        corrds,N_values = corrds.to(device),N_values.to(device)
                        min_vals, _ = torch.min(corrds, dim=1)
                        max_vals, _ = torch.max(corrds, dim=1)
                        max_x, max_y = ((max_vals - min_vals) // args.max_window_size[0] + 1).max(dim=0)[0]
                        max_roi_num =  int(max_x * max_y)
                        data, adapt_region_data,num_adapt_region,results_dict = model_base(data,  N_values, corrds,max_roi_num = max_roi_num, 
                                                                                                return_wsi = True)  
                elif args.model_name in ['CARE']:
                    max_roi_num = int((data.shape[1]**(5/7)/num_region)+1)
                    model_base = model_base#.half()
                    #print(data.shape,end=' ')
                    data,num_adapt_region, _, _, _ = model_base(data,max_roi_num,N_values,corrds,caption_emb = None, rna_emb = None)
                    if data.shape[0] == 1:
                        data = data[:,:num_adapt_region[0],:]
                    else:
                        raise NotImplementedError
                    data = data
                    #print(data.shape,end=' ')

                    #print(data.shape)
                elif args.model_name in ['mean_pooling']:
                    data_perm = data.permute(0, 2, 1)
                    pooled = F.avg_pool1d(data_perm, kernel_size=data_perm.shape[-1])
                    data = pooled.permute(0, 2, 1)
                    data = data.squeeze(0)
                    results_dict = 0
                data = data.squeeze(0)

                logits= model(data)
                Y_hat = torch.argmax(logits, dim = 0)
                #print(label,Y_hat)
                acc_logger.log(Y_hat, label)
                loss = loss_fn(logits.unsqueeze(0), label)+ args.task_loss*results_dict
        elif args.model_type== 'linear':
            optimizer.zero_grad()
            data = data.float()
            target = model(data)
            Y_hat = torch.argmax(target.squeeze(0), dim = 0)
            acc_logger.log(Y_hat, label)
            loss = loss_fn(target, label)
        loss_value = loss.item()
        
        train_loss += loss_value
        error = calculate_error(Y_hat, label)
        train_error += error
        if amp_is:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else: 
            loss.backward()
            #step
            optimizer.step()
        
    # calculate loss and error for epoch
    train_loss /= len(loader)
    train_error /= len(loader)

    print('Epoch: {}, train_loss: {:.4f}, train_error: {:.4f}'.format(epoch, train_loss, train_error))
    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))
        if writer:
            if acc is not None:
                writer.add_scalar('train/class_{}_acc'.format(i), acc, epoch)

    if writer:
        writer.add_scalar('train/loss', train_loss, epoch)
        writer.add_scalar('train/error', train_error, epoch)

def validate_linear(cur, epoch, model, loader, n_classes, early_stopping = None, writer = None, loss_fn = None, results_dir=None,model_base = None,
             num_region =None,model_base1 = None,args = None):
    model.eval()
    if model_base is not None:
        model_base.eval()

    acc_logger = Accuracy_Logger(n_classes=n_classes)
    # loader.dataset.update_mode(True)
    val_loss = 0.
    val_error = 0.
    
    prob = np.zeros((len(loader), n_classes))
    labels = np.zeros(len(loader))

    with torch.no_grad():
        for batch_idx, datas in enumerate(loader):
            if len(datas) == 3:
                data, label,corrds = datas
            else:
                data, label,corrds,N_values,slide_id = datas
            data, label = data.to(device, non_blocking=True), label.to(device, non_blocking=True)
            if model_base is not None or args.model_name in ['mean_pooling']:
                data, label, N_values = data.to(device), label.to(device), N_values.to(device)#data N*1024  

                for i in range(corrds.shape[0]):
                    diffs = np.linalg.norm(corrds[i][1:] - corrds[i][:-1], axis=1)
                    count_512 = np.sum(diffs == 512)
                    count_1024 = np.sum(diffs == 1024)
                    patch_size = 512 if count_512 > count_1024 else 1024
                    corrds[i] = corrds[i] // patch_size
                corrds = corrds.to(device)
                corrds = corrds#.half()
                max_roi_num = int((data.shape[1]**(5/7)/num_region)+1)
                data = data#.half()

                if args.model_name in ['CARE']:
                    corrds,N_values = corrds.to(device),N_values.to(device)
                    min_vals, _ = torch.min(corrds, dim=1)
                    max_vals, _ = torch.max(corrds, dim=1)
                    max_x, max_y = ((max_vals - min_vals) // args.max_window_size[0] + 1).max(dim=0)[0]
                    max_roi_num =  int(max_x * max_y)
                    data, adapt_region_data,num_adapt_region,task_loss = model_base(data,  
                                                                                            N_values, corrds,max_roi_num = max_roi_num, 
                                                                                            return_wsi = True)

                elif args.model_name in ['CARE']:
                    max_roi_num = int((data.shape[1]**(5/7)/num_region)+1)
                    model_base = model_base#.half()
                    #print(data.shape,end=' ')
                    data,num_adapt_region, _, _, _ = model_base(data,max_roi_num,N_values,corrds,caption_emb = None, rna_emb = None)
                    if data.shape[0] == 1:
                        data = data[:,:num_adapt_region[0],:]
                    else:
                        raise NotImplementedError
                    data = data.float()
                elif args.model_name in ['mean_pooling']:
                    data_perm = data.permute(0, 2, 1)
                    pooled = F.avg_pool1d(data_perm, kernel_size=data_perm.shape[-1])
                    data = pooled.permute(0, 2, 1)
                    data = data.squeeze(0)
                    results_dict = 0

            data = data.squeeze(0)
            if args.model_type== 'linear':
                data = data.float()
            logits= model(data)
            Y_hat = torch.argmax(logits, dim = 0)

            #print(label,Y_hat)
            acc_logger.log(Y_hat, label)
            Y_prob = F.softmax(logits.unsqueeze(0), dim = 1)
            loss = loss_fn(logits.unsqueeze(0), label)
            
            prob[batch_idx] = Y_prob.cpu().numpy()
            labels[batch_idx] = label.item()
            
            val_loss += loss.item()
            error = calculate_error(Y_hat, label)
            val_error += error
            

    val_error /= len(loader)
    val_loss /= len(loader)

    if n_classes == 2:
        auc = roc_auc_score(labels, prob[:, 1])
    
    else:
        auc = roc_auc_score(labels, prob, multi_class='ovr')
    
    
    if writer:
        writer.add_scalar('val/loss', val_loss, epoch)
        writer.add_scalar('val/auc', auc, epoch)
        writer.add_scalar('val/error', val_error, epoch)

    print('\nVal Set, val_loss: {:.4f}, val_error: {:.4f}, auc: {:.4f}'.format(val_loss, val_error, auc))
    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))     

    if early_stopping:
        assert results_dir
        early_stopping(epoch, val_loss, model,model_base = model_base, ckpt_name = os.path.join(results_dir, "s_{}_checkpoint.pt".format(cur)),
                       fm_ckpt_name= os.path.join(results_dir, "model_base_{}_checkpoint.pt".format(cur)))
        
        if early_stopping.early_stop:
            print("Early stopping")
            return True

    return False



def summary(model, loader, n_classes,model_type,patch_size = 112,max_window_size = 10,model_base = None,num_region = None,args = None):
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    if model_base is not None:
        model_base.eval()
    model.eval()
    test_loss = 0.
    test_error = 0.

    all_probs = np.zeros((len(loader), n_classes))
    all_labels = np.zeros(len(loader))

    slide_ids = loader.dataset.slide_data['slide_id']
    patient_results = {}
    #for batch_idx, (data, label, corrds, N_values,slide_id) in enumerate(loader):
    for batch_idx, datas in enumerate(loader):
        if len(datas) == 3:
            data, label,corrds = datas
        else:
            data, label,corrds,N_values,slide_id = datas
    #for batch_idx, (data, label, corrds, N_values) in enumerate(loader):
        data, label = data.to(device), label.to(device)
        slide_id = slide_ids.iloc[batch_idx]
        with torch.inference_mode():
            if model_base is not None or args.model_name in ['mean_pooling']:
                data, label, N_values = data.to(device), label.to(device), N_values.to(device)#data N*1024  

                for i in range(corrds.shape[0]):
                    diffs = np.linalg.norm(corrds[i][1:] - corrds[i][:-1], axis=1)
                    count_512 = np.sum(diffs == 512)
                    count_1024 = np.sum(diffs == 1024)
                    patch_size = 512 if count_512 > count_1024 else 1024
                    corrds[i] = corrds[i] // patch_size
                corrds = corrds.to(device)
                corrds = corrds#.half()
                max_roi_num = int((data.shape[1]**(5/7)/num_region)+1)
                data = data#.half()
                
                    
                if args.model_name in ['CARE']:
                    corrds,N_values = corrds.to(device),N_values.to(device)
                    min_vals, _ = torch.min(corrds, dim=1)
                    max_vals, _ = torch.max(corrds, dim=1)
                    max_x, max_y = ((max_vals - min_vals) // args.max_window_size[0] + 1).max(dim=0)[0]
                    max_roi_num =  int(max_x * max_y)
                    wsi_data, adapt_region_data,num_adapt_region,task_loss = model_base(data,  
                                                                                            N_values, corrds,max_roi_num = max_roi_num, 
                                                                                            return_wsi = True)

                    if data.shape[0] == 1:
                        data = data[:,:num_adapt_region[0],:]
                    else:
                        raise NotImplementedError
                elif args.model_name in ['mean_pooling']:
                    data_perm = data.permute(0, 2, 1)
                    pooled = F.avg_pool1d(data_perm, kernel_size=data_perm.shape[-1])
                    data = pooled.permute(0, 2, 1)
                    wsi_data = data.squeeze(0)
                    results_dict = 0
            data = data.squeeze(0)
            
            if args.model_type in ['linear','finetuning','random_init']:
                if args.model_type== 'linear':
                    wsi_data = data.float().unsqueeze(0)
                logits= model(wsi_data.squeeze(0))
                Y_hat = torch.argmax(logits, dim = 0)
                Y_prob = F.softmax(logits.unsqueeze(0), dim = 1)
                acc_logger.log(Y_hat, label)
            else:
                logits, Y_prob, Y_hat, _, _ = model(data)
                acc_logger.log(Y_hat, label)
        probs = Y_prob.cpu().numpy()
        all_probs[batch_idx] = probs
        all_labels[batch_idx] = label.item()
        
        patient_results.update({slide_id: {'slide_id': np.array(slide_id), 'prob': probs, 'label': label.item()}})
        error = calculate_error(Y_hat, label)
        test_error += error

    test_error /= len(loader)
    y_pred = np.argmax(all_probs, axis=1) 
    balanced_acc = balanced_accuracy_score(all_labels, y_pred)
    test_error = 1 - balanced_acc
    if all_probs.shape[1] == 2:
        auc = roc_auc_score(all_labels, all_probs[:, 1])
    else:
        auc = f1_score(all_labels, y_pred, average='weighted')
    if args.model_type in ['linear']:
        acc = accuracy_score(all_labels, y_pred)
        auc = roc_auc_score(all_labels, all_probs,
                              multi_class="ovr",   # one-vs-rest
                              average="macro") 
        test_error = 1 - acc
        return patient_results, test_error, auc, acc_logger
    return patient_results, test_error, auc, acc_logger
