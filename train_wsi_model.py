from __future__ import print_function

import argparse
import pdb
import os
import math
import time
# internal imports
from utils.file_utils import save_pkl, load_pkl
from utils.utils import *
from utils.core_utils import train
from dataset_modules.dataset_generic import  Generic_MIL_Dataset,Generic_WSI_Dataset


# pytorch imports
import torch
from torch.utils.data import DataLoader, sampler
import torch.nn as nn
import torch.nn.functional as F

import pandas as pd
import numpy as np

# nn.MultiheadAttention
def main(args):
    # create results directory if necessary
    if not os.path.isdir(args.results_dir):
        os.mkdir(args.results_dir)

    if args.k_start == -1:
        start = 0
    else:
        start = args.k_start
    if args.k_end == -1:
        end = args.k
    else:
        end = args.k_end

    all_test_auc = []
    all_val_auc = []
    all_test_acc = []
    all_val_acc = []
    folds = np.arange(start, end)
    for i in folds:
        t0=time.time()
        seed_torch(args.seed)
        train_dataset, val_dataset, test_dataset = dataset.return_splits(from_id=False, 
                csv_path='{}/splits_{}.csv'.format(args.split_dir, i))
        
        
        if args.model_type in ['KNN','logistic_regression']:
            train_dataset.return_slideid()
            if val_dataset is not None:
                val_dataset.return_slideid()
            test_dataset.return_slideid()
        datasets = (train_dataset, val_dataset, test_dataset)
        results, test_auc, val_auc, test_acc, val_acc  = train(datasets, i, args)
        all_test_auc.append(test_auc)
        all_val_auc.append(val_auc)
        all_test_acc.append(test_acc)
        all_val_acc.append(val_acc)
        #write results to pkl
        filename = os.path.join(args.results_dir, 'split_{}_results.pkl'.format(i))
        if results is not None:
            save_pkl(filename, results)
        print('%d folds, speed time:%.2f'%(i,time.time()-t0))
    final_df = pd.DataFrame({'folds': folds, 'test_auc': all_test_auc, 
        'val_auc': all_val_auc, 'test_acc': all_test_acc, 'val_acc' : all_val_acc})

    if len(folds) != args.k:
        save_name = 'summary_partial_{}_{}.csv'.format(start, end)
    else:
        save_name = 'summary.csv'
    print("Result:")
    print(final_df)
    print()


    test_acc_mean = final_df['test_acc'].mean()
    test_acc_var  = final_df['test_acc'].var()  
    test_auc_mean = final_df['test_auc'].mean()
    test_auc_var  = final_df['test_auc'].var()
    print(args.exp_code)
    print("'test_acc' 'test_auc'")
    print(f"{test_acc_mean*100:.2f}({test_acc_var*100:.2f}) {test_auc_mean*100:.2f}({test_auc_var*100:.2f})")
    final_df.to_csv(os.path.join(args.results_dir, save_name))

# Generic training settings
parser = argparse.ArgumentParser(description='Configurations for WSI Training')
parser.add_argument('--experiment_target', type = str,default='PTEN',
					choices=['DIED','RAS','BRAF','KRAS','3cls_subtype','7cls_subtype','','KRAS','STK11','KEAP1','PTEN','CTNNB1','SMAD4','CASP8',
              				'VHL','SETD1B','ARID1A','APC','ACVR2A','OT-46-FFPE','OT-46','EGFR','TP53','MSI','BRAF','KRAS_1','ER','PR','HER2',
			  				'PIK3CA','IDH','coarse_subtype','fine_subtype','tumor','subtyping','grading',
							'BAP1','PBRM1','SETD2' ,
                            '1-BRCA','2-BRCA','3-BRCA','4-BRCA','5-BRCA','6-BRCA'],
					help='name about experiment')
parser.add_argument('--task', type = str,default='t1_subtype',choices=['died','t1_subtype','t1_tumor','t1_gene',
                                                                       't2_segmen','t2_combine','t2_cross',
                                                                       't3_zero_shot', 't1_os'],
					help='name about experiment')
parser.add_argument('--dataset', type = str,default='TCGA',#'EBRAIN' 'cptac'
					help='name about dataset')
parser.add_argument('--subdataset', type = str,
					help='name about dataset')
parser.add_argument('--auto_skip', action='store_true',
					help='save wsi feature')
parser.add_argument('--depth', type=int,default=2, choices=[1,2,  4])
parser.add_argument('--wsi_rate', type=float,default=1.)
parser.add_argument('--model_name', type=str, default='CARE', choices=['conch_v1_5','mean_pooling',
                                                                            'TITAN','CARE',
                                                                          'CHIEF','PRISM','TANGLE','FEATHER','GIGAPATH'])
parser.add_argument('--model_type', type=str, choices=['KNN','logistic_regression','finetuning','random_init'], 
                    default='finetuning',#'logistic_regression', 
                    help='type of model (default: clam_sb, clam w/ single attention branch)')#logistic_regression
parser.add_argument('--lr', type=float, default=2e-5,
                    help='learning rate (default: 0.0001)')#lr =2e-4
parser.add_argument('--target_ar', type=float,default=0.5)
parser.add_argument('--batch_size', type=int,default=1, choices=[1,2,  4])
parser.add_argument('--gpu', type=str,default='1')
#default
parser.add_argument('--task_loss', type=float,default=0.1)
parser.add_argument('--num_region', type=float, default=8)
parser.add_argument('--bag_weight', type=float, default=0.5,
                    help='clam: weight coefficient for bag-level loss (default: 0.7)')
parser.add_argument('--roi_fea', action='store_true') #use roi_fea
parser.add_argument('--embed_dim', type=int, default=1024)
parser.add_argument('--step_size', type = int, default=256,
					help='step_size')
parser.add_argument('--csv_path', type = str,default='./dataset_csv',
					help='name about experiment')
parser.add_argument('--data_root_dir', type=str, default='./data', 
                    help='data directory')
parser.add_argument('--results_dir', default='./results/train_wsi_model/', help='results directory (default: ./results)')
parser.add_argument('--exp_code', type=str, help='experiment code for saving results')#default='task_2_RCC_CLAM_50'
parser.add_argument('--model_size', type=str, choices=['small', 'big'], default='small', help='size of model, does not affect mil')
parser.add_argument('--max_epochs', type=int, default=200,
                    help='maximum number of epochs to train (default: 200)')
parser.add_argument('--diff_weight', type=float, default=0.5,
                    help='maximum number of epochs to train (default: 200)')
parser.add_argument('--label_frac', type=float, default=1.0,
                    help='fraction of training labels (default: 1.0)')
parser.add_argument('--reg', type=float, default=1e-5,
                    help='weight decay (default: 1e-5)')
parser.add_argument('--seed', type=int, default=1, 
                    help='random seed for reproducible experiment (default: 1)')
parser.add_argument('--k', type=int, default=5, help='number of folds (default: 10)')
parser.add_argument('--k_start', type=int, default=-1, help='start fold (default: -1, last fold)')
parser.add_argument('--k_end', type=int, default=-1, help='end fold (default: -1, first fold)')
parser.add_argument('--split_dir', type=str, default=None, 
                    help='manually specify the set of splits to use, ' 
                    +'instead of infering from the task and label_frac argument (default: None)')
parser.add_argument('--log_data', action='store_true', default=True, help='log data using tensorboard')
parser.add_argument('--testing', action='store_true', default=False, help='debugging tool')
parser.add_argument('--early_stopping', action='store_true', default=True, help='enable early stopping')
parser.add_argument('--opt', type=str, choices = ['adam', 'sgd'], default='adam')
parser.add_argument('--drop_out', type=float, default=0.25, help='dropout')
parser.add_argument('--bag_loss', type=str, choices=['svm', 'ce'], default='ce',
                     help='slide-level classification loss function (default: ce)')
parser.add_argument('--weighted_sample', action='store_true', default=True, help='enable weighted sampling')


parser.add_argument('--no_inst_cluster', action='store_true', default=False,
                     help='disable instance-level clustering')
parser.add_argument('--inst_loss', type=str, choices=['svm', 'ce', None], default='svm',
                     help='instance-level clustering loss function (default: None)')
parser.add_argument('--subtyping',  default=True, 
                     help='subtyping problem')#action='store_true',

parser.add_argument('--B', type=int, default=8, help='number of positive/negative patches to sample for clam')
parser.add_argument('--cache_path', type=str, default='./data', help='path to cache extracted wsi embeddings')
args = parser.parse_args()
if args.model_name in ['CARE']:
    args.embed_dim =768 #512
    args.step_size=512
    args.output_dim = 512
elif args.model_name in ['TITAN','mean_pooling']:
    args.embed_dim = 768
    args.step_size = 512
    args.output_dim = 768
elif args.model_name in ['CHIEF']:
    args.embed_dim = 768
    args.output_dim = 768
elif args.model_name in ['PRISM']:
    args.embed_dim = 1280
    args.output_dim = 1280
else:
    args.embed_dim = 768
    args.output_dim = 768
print(args)
if args.subdataset is None:
    if args.model_name in ['CARE','mean_pooling']:
        args.data_root_dir = os.path.join(args.data_root_dir,args.dataset,'conch_v1_5')
    elif args.model_name in ['CHIEF','PRISM','TITAN','TANGLE','FEATHER','GIGAPATH']:
        args.data_root_dir = os.path.join(args.data_root_dir,args.dataset,'wsi_embedding',args.model_name.lower())
    else:
        args.data_root_dir = os.path.join(args.data_root_dir,args.dataset,args.model_name)
else:
    if args.model_name in ['CARE','mean_pooling']:
        args.data_root_dir = os.path.join(args.data_root_dir,args.dataset,args.subdataset,'conch_v1_5')
    elif args.model_name in ['CHIEF','PRISM','TITAN','TANGLE','FEATHER','GIGAPATH']:
        args.data_root_dir = os.path.join(args.data_root_dir,args.dataset,args.subdataset,'wsi_embedding',args.model_name.lower())
    else:
        args.data_root_dir = os.path.join(args.data_root_dir,args.dataset,args.subdataset,args.model_name)


if args.subdataset is None:
    args.exp_code=os.path.join(args.task,'%s_%s_%s_%s_%s_%s_%s_%s_%s'%(args.dataset,args.experiment_target, args.model_type,args.model_name, args.num_region, args.wsi_rate, args.task_loss,args.output_dim,args.target_ar           ))
    args.csv_path = os.path.join(args.csv_path,'%s_clean_%s_%s.csv'%(args.task,args.dataset,args.experiment_target))
else:
    args.exp_code=os.path.join(args.task,'%s_%s_%s_%s_%s_%s_%s_%s_%s_%s'%(args.dataset, args.subdataset, args.experiment_target, args.model_type, args.model_name,    args.num_region, args.wsi_rate, args.task_loss ,args.output_dim,args.target_ar         ))
    args.csv_path = os.path.join(args.csv_path,'%s_clean_%s_%s_%s.csv'%(args.task,args.dataset,args.subdataset,args.experiment_target))

    
def seed_torch(seed=7):
    import random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

seed_torch(args.seed)

#encoding_size = 1024
settings = {'num_splits': args.k, 
            'k_start': args.k_start,
            'k_end': args.k_end,
            'task': args.task,
            'max_epochs': args.max_epochs, 
            'results_dir': args.results_dir, 
            'lr': args.lr,
            'experiment': args.exp_code,
            'reg': args.reg,
            'label_frac': args.label_frac,
            'bag_loss': args.bag_loss,
            'seed': args.seed,
            'model_type': args.model_type,
            'model_size': args.model_size,
            "use_drop_out": args.drop_out,
            'weighted_sample': args.weighted_sample,
            'opt': args.opt}

if args.model_type in ['clam_sb', 'clam_mb']:
   settings.update({'bag_weight': args.bag_weight,
                    'inst_loss': args.inst_loss,
                    'B': args.B})
print('\nLoad Dataset')
if args.dataset == 'pandas':
    data_formats = '.tiff'
elif args.dataset in ['DHMC_LUNG']:
    data_formats = '.tif'
elif args.dataset in ['DHMC_RCC']:
    data_formats = '.png'
elif args.dataset in ['BCNB']:
    data_formats = '.jpg'
elif args.dataset in ['EBRAIN']:
    data_formats = '.ndpi'
elif args.dataset in ['SR386']:
    data_formats = '.czi'
else:
    data_formats = '.svs'
if args.dataset in ['pandas']:
    args.suffix = '0_512'
elif args.dataset in ['DHMC-RCC','BCNB']:
    args.suffix = '0_256'
else:
    args.suffix = '0_1024'
if args.model_name in ['PRISM','CHIEF','TITAN','TANGLE','FEATHER','GIGAPATH']:
    dataset_factory = Generic_WSI_Dataset
else:
    dataset_factory = Generic_MIL_Dataset
if args.task in ['t1_gene','died']:
    args.n_classes=2
    dataset = dataset_factory(csv_path = args.csv_path,
                            data_dir= args.data_root_dir,
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = {0:0, 1:1},
                            patient_strat=False,
                            data_format = data_formats,
                            suffix = args.suffix,
                            ignore=[])
elif args.task in ['t2_combine','t2_cross','t1_subtype']:
    if args.dataset == 'EBRAIN' and args.experiment_target == 'fine_subtype':
        args.n_classes=30
    elif args.dataset == 'EBRAIN' and args.experiment_target == 'coarse_subtype':
        args.n_classes=12
    elif args.dataset == 'IMP':
        args.n_classes = 3
    elif args.dataset == 'BRACS' and args.experiment_target in ['coarse_subtype','3cls_subtype']:
        args.n_classes=3
    elif args.dataset == 'BRACS' and args.experiment_target in ['fine_subtype','7cls_subtype']:
        args.n_classes=7
    elif args.dataset == 'TCGA' and args.experiment_target in  ['OT-46-FFPE','OT-46']:
        args.n_classes = 46
    elif args.dataset == 'cptac' and args.experiment_target in ['OT-46']:
        args.n_classes = 10
    elif args.dataset == 'cptac' and args.subdataset == 'CCRCC' and args.task in ['t2_combine','t2_cross']:
        args.n_classes = 3
    elif args.dataset == 'MUT' and args.task in ['t2_combine','t2_cross']:    
        args.n_classes = 2
    label_dict = {i: i for i in range(args.n_classes)}
    print('label_dict',label_dict)
    dataset = dataset_factory(csv_path = args.csv_path,
                            data_dir= args.data_root_dir,
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = label_dict,
                            patient_strat= False,
                            data_format = data_formats,
                            suffix = args.suffix,
                            ignore=[])
elif args.task in ['t1_tumor']:
    args.n_classes=2
    label_dict = {i: i for i in range(args.n_classes)}
    #print(args.csv_path)
    dataset = dataset_factory(csv_path = args.csv_path,
                            data_dir= args.data_root_dir,
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = label_dict,
                            patient_strat= False,
                            data_format = data_formats,
                            suffix = args.suffix,
                            ignore=[])
else:
    raise NotImplementedError
    
if not os.path.isdir(args.results_dir):
    os.mkdir(args.results_dir)

args.results_dir = os.path.join(args.results_dir, str(args.exp_code) + '_s{}'.format(args.seed),args.model_name)
if not os.path.isdir(args.results_dir):
    os.makedirs(args.results_dir, exist_ok=True)
idx = 0

if args.split_dir is None:
    if args.subdataset is None:
        args.split_dir  = os.path.join('./splits', args.task,'%s_%s_%s'%(args.dataset,args.experiment_target,args.label_frac*100))
    else:
        args.split_dir  = os.path.join('./splits', args.task,'%s_%s_%s_%s'%(args.dataset,args.subdataset,args.experiment_target,args.label_frac*100))
else:
    args.split_dir = os.path.join('./splits', args.split_dir)

print('split_dir: ', args.split_dir)
assert os.path.isdir(args.split_dir)

settings.update({'split_dir': args.split_dir})


with open(args.results_dir + '/experiment_parameters_{}.txt'.format('1'), 'w') as f:
    print(settings, file=f)
f.close()

print("################# Settings ###################")
for key, val in settings.items():
    print("{}:  {}".format(key, val))        

if __name__ == "__main__":
    gpu_list = args.gpu.replace(" ", "").split(",")
    try:
        args.multi_gpu = [int(gpu_id) for gpu_id in gpu_list if gpu_id]
        args.batch_size = args.batch_size*len(args.multi_gpu)
    except ValueError:
        raise ValueError(f"Invalid GPU format: {args.gpu}")

    import warnings
    warnings.simplefilter('always')  
    warnings.filterwarnings('error') 
    results = main(args)

    print("finished!")
    print("end script")


