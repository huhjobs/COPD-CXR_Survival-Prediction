import torchtuples as tt
from pycox.models import LogisticHazard, PMF, MTLR, BCESurv
from pycox.utils import kaplan_meier
from pycox.evaluation import EvalSurv

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader 

import numpy as np
import pandas as pd
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os

import timm
from pprint import pprint

import copy
from tqdm import tqdm
import pickle
from sklearn import metrics
from glob import glob

from load_data import load_data, load_infer_data
from PIL import Image

import sys


def run(opt):
    use_gpu = torch.cuda.is_available()
    gpu_count = torch.cuda.device_count()
    print("Available GPU count:" + str(gpu_count))
    
    NUM_EPOCHS = opt.n_epochs
    BATCH_SIZE = opt.batch_size
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    
    ## Run Path
    if opt.eval_only:
        # test only. it is okay to have duplicate run_path
        os.makedirs(opt.run_path, exist_ok=True)
    else:
        # train from scratch, should not have the same run_path. Otherwise it will overwrite previous runs.
        try:
            os.makedirs(opt.run_path)
        except FileExistsError:
            print("[ERROR] run_path {} exists. try to assign a unique run_path".format(opt.run_path))
            return None, None
        except Exception as e:
            print("exception while creating run_path {}".format(opt.run_path))
            print(str(e))
            return None, None
    
    with open(os.path.join(opt.run_path,'config.yaml'),'w') as fp:
        fp.write('\n'.join(sys.argv[1:]))
    
    ## Load Data
    image_datasets, dataloaders, dataset_sizes, labtrans = load_data(opt.project, opt.augment, BATCH_SIZE, opt.path_to_images, opt.path_to_csv)
    
    
    ## Load Optim
    if opt.optimizer == 'adam':
        optimizer = tt.optim.Adam(opt.lr)
    else:
        f"recheck opt.optimizer = {opt.optimizer}"
    
    
    ## Load Architecture and Weights
    if opt.pretrained_path == 'False':
        model_ft = timm.create_model(opt.arch, pretrained=False, num_classes=labtrans.out_features)
    elif opt.pretrained_path == 'imagenet':
        model_ft = timm.create_model(opt.arch, pretrained=True, num_classes=labtrans.out_features)
    else:
        model_ft = timm.create_model(opt.arch, pretrained=False, num_classes=int(opt.pretrained_path.split('/')[-1].split('_')[0]))
        model_ft.load_state_dict(torch.load(opt.pretrained_path))
        
        if "resnet" in opt.arch:
            num_ftrs = model_ft.fc.in_features
            model_ft.fc = nn.Linear(num_ftrs, labtrans.out_features)
        elif "densenet" in opt.arch:
            num_ftrs = model_ft.classifier.in_features
            model_ft.classifier = nn.Linear(num_ftrs, labtrans.out_features)
        else:
            f"recheck opt.arch = {opt.arch}"

    if opt.surv_module == "LogisticHazard":
        model = LogisticHazard(model_ft, optimizer, duration_index=labtrans.cuts)
    else:
        f"recheck opt.surv_module = {opt.surv_module}"

    callbacks = [tt.cb.EarlyStopping(patience=opt.patience, file_path=f"{opt.run_path}/best_model.pt")] ## save model(sth wrong)
    verbose = True
    log = model.fit_dataloader(dataloaders['train'], NUM_EPOCHS, callbacks, verbose, val_dataloader=dataloaders['val'])
    
    
    ## Evaluation
    dataset_test_x = ImgSimInput(image_datasets['test'])
    dl_test_x = DataLoader(dataset_test_x, BATCH_SIZE, shuffle=False)
    
    df = pd.read_csv(opt.path_to_csv, dtype={opt.filename_column:str})
    sim_test = event_time('test', 
                          image_datasets, 
                          df, 
                          opt.filename_column, 
                          opt.time_column, 
                          opt.event_column
                         )
    
    surv = model.interpolate(10).predict_surv_df(dl_test_x)
    ev = EvalSurv(surv, *sim_test, 'km')
    c_indxe = ev.concordance_td()
    print("="*30)
    print("C-index : ", round(c_indxe, 4))
    
    fpr = dict()
    tpr = dict()
    auc = dict()
    for year in [1, 3, 5]:
        auc[year] = metrics.roc_auc_score(time_label(int(surv.index[20*year]), sim_test), -surv.iloc[20*year].values+1)
        fpr[year], tpr[year], _ = metrics.roc_curve(time_label(int(surv.index[20*year]), sim_test), -surv.iloc[20*year].values+1)
        print(f"{year}year-AUC : ", round(auc[year], 4))
        
        
    ## Plot    
    for year in [1, 3, 5]:
        plt.plot(fpr[year], tpr[year], label=f"{year}-year")
    plt.legend()
    plt.savefig(os.path.join(opt.run_path,'auc.png'))
    
#     for year in [1, 2, 3]:
#         auc[year] = metrics.roc_auc_score(time_label(int(surv.index[year]), sim_test), -surv.iloc[year].values+1)
#         fpr[year], tpr[year], _ = metrics.roc_curve(time_label(int(surv.index[year]), sim_test), -surv.iloc[year].values+1)
#         print(round(auc[year], 4))
        
        
#     ## Plot    
#     for year in [1, 2, 3]:
#         plt.plot(fpr[year], tpr[year], label=f"{year}-year")
#     plt.legend()
    
    for i in range(2):
        idx = sim_test[1] == i
        kaplan_meier(sim_test[0][idx], sim_test[1][idx]).rename(f"{i}_km").plot()
        surv.loc[:, idx].mean(axis=1).rename(f"{i}_nn").plot()
        _ = plt.legend()
    plt.savefig(os.path.join(opt.run_path,'simulate.png'))
    
    
    
    ## Save Model
    torch.save(model.net.state_dict(), f"{opt.run_path}/best_model.pt")
    surv.to_csv(f"{opt.run_path}/surv.csv") #, index=False
    ## TODO : save eval results
    ## TODO : eval_only version
    
    
def infer_run(opt):
    use_gpu = torch.cuda.is_available()
    gpu_count = torch.cuda.device_count()
    print("Available GPU count:" + str(gpu_count))
    
    BATCH_SIZE = opt.batch_size
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    os.makedirs(opt.run_path, exist_ok=True)
    
    ## Load Data
    image_datasets, dataloaders, dataset_sizes, labtrans = load_infer_data(opt.project, opt.augment, BATCH_SIZE, opt.path_to_images, opt.path_to_csv, opt.path_to_train_images, opt.path_to_train_csv)
    
    ## Load Architecture and Weights
    model_ft = timm.create_model(opt.arch, pretrained=False)
        
    if "resnet" in opt.arch:
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, labtrans.out_features)
    elif "densenet" in opt.arch:
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, labtrans.out_features)
    else:
        f"recheck opt.arch = {opt.arch}"
    model_ft.load_state_dict(torch.load(opt.model_path))
    
    if opt.surv_module == "LogisticHazard":
        model = LogisticHazard(model_ft, duration_index=labtrans.cuts)
    else:
        f"recheck opt.surv_module = {opt.surv_module}"

    ## Evaluation
    dataset_test_x = ImgSimInput(image_datasets)
    dl_test_x = DataLoader(dataset_test_x, BATCH_SIZE, shuffle=False)
    
    df = pd.read_csv(opt.path_to_csv, dtype={opt.filename_column:str})
    sim_test = event_time(None, 
                          image_datasets, 
                          df, 
                          opt.filename_column, 
                          opt.time_column, 
                          opt.event_column
                         )
    
    surv = model.interpolate(10).predict_surv_df(dl_test_x)
    ev = EvalSurv(surv, *sim_test, 'km')
    c_indxe = ev.concordance_td()
    print("="*30)
    print("C-index : ", round(c_indxe, 4))
    
    fpr = dict()
    tpr = dict()
    auc = dict()
    for year in [1, 3, 5]:
        auc[year] = metrics.roc_auc_score(time_label(int(surv.index[20*year]), sim_test), -surv.iloc[20*year].values+1)
        fpr[year], tpr[year], _ = metrics.roc_curve(time_label(int(surv.index[20*year]), sim_test), -surv.iloc[20*year].values+1)
        print(f"{year}year-AUC : ", round(auc[year], 4))
        
        
    ## Plot    
    for year in [1, 3, 5]:
        plt.plot(fpr[year], tpr[year], label=f"{year}-year")
    plt.legend()
    
#     for year in [1, 2, 3]:
#         auc[year] = metrics.roc_auc_score(time_label(int(surv.index[year]), sim_test), -surv.iloc[year].values+1)
#         fpr[year], tpr[year], _ = metrics.roc_curve(time_label(int(surv.index[year]), sim_test), -surv.iloc[year].values+1)
#         print(round(auc[year], 4))
        
        
#     ## Plot    
#     for year in [1, 2, 3]:
#         plt.plot(fpr[year], tpr[year], label=f"{year}-year")
#     plt.legend()
    
#     for i in range(2):
#         idx = sim_test[1] == i
#         kaplan_meier(sim_test[0][idx], sim_test[1][idx]).rename(f"{i}_km").plot()
#         surv.loc[:, idx].mean(axis=1).rename(f"{i}_nn").plot()
#         _ = plt.legend()
    
    
    ## Save Model
#     torch.save(model.net.state_dict(), f"{opt.run_path}/best_model.pt")
    surv.to_csv(f"{opt.run_path}/surv.csv") #, index=False
    ## TODO : save eval results
    ## TODO : eval_only version
    
    
def event_time(split, 
               image_datasets, 
               df, 
               filename_column, 
               time_column, 
               event_column):
    time_lst = []
    event_lst = []
    if split == None:
        for i in image_datasets.imgs_lst:
            idx = os.path.splitext(os.path.basename(i))[0]
            time_lst.append(df[df[filename_column] == idx][time_column].values[0])
            event_lst.append(df[df[filename_column] == idx][event_column].values[0])
    else:
        for i in image_datasets[split].imgs_lst:
            idx = os.path.splitext(os.path.basename(i))[0]
            time_lst.append(df[df[filename_column] == idx][time_column].values[0])
            event_lst.append(df[df[filename_column] == idx][event_column].values[0])
    return (np.array(time_lst), np.array(event_lst))


def time_label(index_time, sim_test):
    label_lst = []
    for i in range(len(sim_test[0])):
        dur = sim_test[0][i]
        evt = sim_test[1][i]
        if evt == 0:
            label_lst.append(0)
        elif evt == 1:
            if dur > index_time:
                label_lst.append(0)
            else:
                label_lst.append(1)
        else:
            print("Something wrong!")
    return label_lst


class ImgSimInput(Dataset):
    def __init__(self, img_dataset):
        self.img_dataset = img_dataset

    def __len__(self):
        return len(self.img_dataset)

    def __getitem__(self, index):
        img = self.img_dataset[index][0]
        return img