import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
import os
import pandas as pd
from torch.utils.data import DataLoader, Subset,ConcatDataset
from utils.utils import *
from utils.core_utils import Accuracy_Logger
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
from sklearn.metrics import confusion_matrix


def calculate_error_numpy(Y_hat, Y):
	error = 0 if Y_hat == Y else 1
	return error


def confusion_matrix_computing(output_path,loader,model,classes,name):
    labels = []
    pre_labels = []
    t1 = time.time()
    output_path1 = os.path.join(output_path,name+'.csv')
    pred = torch.zeros(len(loader),classes)
    for item_data,batch_data in enumerate(loader):
        #if item_data ==5:
        #    break
        data, target, coords, id = batch_data#[21816, 1024]
        data= data.to(device)
        labels.append(target[0])
        with torch.no_grad():
            logits, Y_prob, Y_hat, _, results_dict = model(data)
            pre_labels.append(Y_hat[0][0].detach().cpu())
            pred[item_data] = Y_prob.detach().cpu()
    print(time.time()-t1)
    labels = np.array(labels)
    pre_labels = np.array(pre_labels)
    cm = confusion_matrix(labels, pre_labels)
    if name == 'id':
        cm_df = pd.DataFrame(cm, index=np.unique(labels), columns=np.unique(pre_labels))
    else:
        cm_df = pd.DataFrame(cm, index=[0,1,-1], columns=[0,1,-1])

    cm_df.to_csv(output_path1, index=True)
    pred_np = pred.numpy()
    pred_df = pd.DataFrame(pred_np)
    output_path2 = os.path.join(output_path,name+'pred.csv')
    pred_df.to_csv(output_path2, index=False)
    


