import torch
import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, Dataset
import os

import pandas as pd
import numpy as np
from PIL import Image
from glob import glob
import cv2

import albumentations as albu
from albumentations.pytorch import ToTensorV2
import random
random.seed(9000)

from pycox.models import LogisticHazard
import torchtuples as tt


class COPDdataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, csv_path, transform=None, mode='train'):
        self.csv_path = csv_path
        self.data_dir = data_dir
        self.transform = transform
        self.mode = mode
        
        df = pd.read_csv(self.csv_path)
        
        if self.mode == 'test':
            idx_lst = list(df[df.test == 1].inclusion)
        elif self.mode == 'val':
            idx_lst = list(df[df.val == 1].inclusion)
        elif self.mode == 'train':
            idx_lst = list(df[df.training == 1].inclusion)
        elif self.mode == 'infer':
            idx_lst = list(df.inclusion)
        else:
            print("Mode error!")
            

        path_lst = [f"{self.data_dir}/{x}.png" for x in idx_lst]        
        time_lst = [df[df.inclusion == x].survival.values[0] for x in idx_lst]
        event_lst = [df[df.inclusion == x].censor.values[0] for x in idx_lst]
        target = (np.array(time_lst), np.array(event_lst))
        
        labtrans = LogisticHazard.label_transform(20) # 최대 기간(3475)을 20등분
        time_event = labtrans.fit_transform(*target)

        self.time, self.event = tt.tuplefy(time_event[0], time_event[1]).to_tensor()

        self.imgs_lst = path_lst
        self.labtrans = labtrans

    def __len__(self):
        return len(self.imgs_lst)

    def __getitem__(self, index):
    
        img = np.array(Image.open(self.imgs_lst[index]).convert("RGB"))
        
        if self.transform:
            img = self.transform(image=img)
        
        data = img['image'], (self.time[index], self.event[index])
                
        return data

class COPDdatasetNP(torch.utils.data.Dataset):
    def __init__(self, data_dir, csv_path, transform=None, mode='train'):
        self.csv_path = csv_path
        self.data_dir = data_dir
        self.transform = transform
        self.mode = mode
        
        csv_data = np.genfromtxt(self.csv_path,
                                 delimiter=',', dtype=str)
        col_names = csv_data[0]
        csv_data = np.genfromtxt(self.csv_path,
                                 delimiter=',', skip_header=1,
                                 dtype=float)
        
        inclusion_idx = list(col_names).index('inclusion')
        training_idx = list(col_names).index('training')
        val_idx = list(col_names).index('val')
        test_idx = list(col_names).index('test')
        
        if self.mode == 'test':
            idx_lst = [csv_data[idx][inclusion_idx] for idx in np.where(csv_data.transpose()[test_idx] == 1)[0]]
        elif self.mode == 'val':
            idx_lst = [csv_data[idx][inclusion_idx] for idx in np.where(csv_data.transpose()[val_idx] == 1)[0]]
        elif self.mode == 'train':
            idx_lst = [csv_data[idx][inclusion_idx] for idx in np.where(csv_data.transpose()[training_idx] == 1)[0]]
        elif self.mode == 'infer':
            idx_lst = [csv_data[:][inclusion_idx]]
        else:
            print("Mode error!")
            
        survival_idx = list(col_names).index('survival')
        censor_idx = list(col_names).index('censor')
        
        path_lst = [f"{self.data_dir}/{int(x)}.png" for x in idx_lst]        
        time_lst = [csv_data[np.where(csv_data.transpose()[inclusion_idx] == x)[0][0]][survival_idx] for x in idx_lst]
        event_lst = [csv_data[np.where(csv_data.transpose()[inclusion_idx] == x)[0][0]][censor_idx] for x in idx_lst]
        target = (np.array(time_lst), np.array(event_lst))
        
        labtrans = LogisticHazard.label_transform(20) # 최대 기간(3475)을 20등분
        time_event = labtrans.fit_transform(*target)

        self.time, self.event = tt.tuplefy(time_event[0], time_event[1]).to_tensor()

        self.imgs_lst = path_lst
        self.labtrans = labtrans

    def __len__(self):
        return len(self.imgs_lst)

    def __getitem__(self, index):
    
        img = np.array(Image.open(self.imgs_lst[index]).convert("RGB"))
        
        if self.transform:
            img = self.transform(image=img)
        
        data = img['image'], (self.time[index], self.event[index])
                
        return data
    
class MTdataset(torch.utils.data.Dataset):
    
    CLASSES = ['heart', 'left lung','right lung']
    
    def __init__(self, data_dir, masks_dir, csv_path, classes=None, transform=None, mode='train'):
        self.csv_path = csv_path
        self.data_dir = data_dir
        self.masks_dir = masks_dir
        self.transform = transform
        self.mode = mode
        
        csv_data = np.genfromtxt(self.csv_path,
                                 delimiter=',', dtype=str)
        col_names = csv_data[0]
        csv_data = np.genfromtxt(self.csv_path,
                                 delimiter=',', skip_header=1,
                                 dtype=float)
        
        inclusion_idx = list(col_names).index('inclusion')
        training_idx = list(col_names).index('training')
        val_idx = list(col_names).index('val')
        test_idx = list(col_names).index('test')
        
        if self.mode == 'test':
            idx_lst = [csv_data[idx][inclusion_idx] for idx in np.where(csv_data.transpose()[test_idx] == 1)[0]]
        elif self.mode == 'val':
            idx_lst = [csv_data[idx][inclusion_idx] for idx in np.where(csv_data.transpose()[val_idx] == 1)[0]]
        elif self.mode == 'train':
            idx_lst = [csv_data[idx][inclusion_idx] for idx in np.where(csv_data.transpose()[training_idx] == 1)[0]]
        elif self.mode == 'infer':
            idx_lst = [csv_data[:][inclusion_idx]]
        else:
            print("Mode error!")

        class_values = list(range(len(classes)))    
        survival_idx = list(col_names).index('survival')
        censor_idx = list(col_names).index('censor')
        
        path_lst = [f"{self.data_dir}/{int(x)}.png" for x in idx_lst]  #images_fps랑 동일 
        masks_lst = [f"{self.masks_dir}/{int(x)}.png" for x in idx_lst]
        time_lst = [csv_data[np.where(csv_data.transpose()[inclusion_idx] == x)[0][0]][survival_idx] for x in idx_lst]
        event_lst = [csv_data[np.where(csv_data.transpose()[inclusion_idx] == x)[0][0]][censor_idx] for x in idx_lst]
        target = (np.array(time_lst), np.array(event_lst))
        
        labtrans = LogisticHazard.label_transform(20) # 최대 기간(3475)을 20등분
        time_event = labtrans.fit_transform(*target)

        self.time, self.event = tt.tuplefy(time_event[0], time_event[1]).to_tensor()

        self.imgs_lst = path_lst
        self.masks_lst = masks_lst
        self.labtrans = labtrans

    def __len__(self):
        return len(self.imgs_lst)

    def __getitem__(self, index):
    
        img = np.array(Image.open(self.imgs_lst[index]).convert("RGB"))
        mask = np.array(Image.open(self.masks_lst[index]).convert("RGB"))
        
        # add background label to the mask
        blank = (np.sum(mask, axis=2) == 0)*255 # 아무것도 해당안되는 픽셀에 값 255 지정한 채널 생성 (shape == (512,512))
        mask = np.stack([mask[:, :, 0], mask[:, :, 1], mask[:, :, 2], blank] , axis=-1).astype('uint8') # shape (512,512,4)
        
        if self.transform:
            sample = self.transform(image=img, mask=mask)

        data = sample['image'], (sample['mask'], (self.time[index], self.event[index]))
                
        return data
    
class PMPdataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, csv_path, transform=None, mode='train'):
        self.csv_path = csv_path
        self.data_dir = data_dir
        self.transform = transform
        self.mode = mode
        
        df = pd.read_csv(self.csv_path)
        
        if self.mode == 'test':
            idx_lst = list(df[df.split == 'test'].filename)
        elif self.mode == 'val':
            idx_lst = list(df[df.split == 'val'].filename)
        elif self.mode == 'train':
            idx_lst = list(df[df.split == 'train'].filename)
        elif self.mode == 'ext':
            idx_lst = list(df[df.split == 'ext'].filename)
        elif self.mode == 'infer':
            idx_lst = list(df.filename)
        else:
            print("Mode error!")
            

        path_lst = [f"{self.data_dir}/{x:04}.png" for x in idx_lst]        
        time_lst = [df[df.filename == x].fu_dur.values[0] for x in idx_lst]
        event_lst = [df[df.filename == x].death.values[0] for x in idx_lst]
        target = (np.array(time_lst), np.array(event_lst))
        
        labtrans = LogisticHazard.label_transform(20)
        time_event = labtrans.fit_transform(*target)

        self.time, self.event = tt.tuplefy(time_event[0], time_event[1]).to_tensor()

        self.imgs_lst = path_lst
        self.labtrans = labtrans

    def __len__(self):
        return len(self.imgs_lst)

    def __getitem__(self, index):
    
        img = np.array(Image.open(self.imgs_lst[index]).convert("RGB"))
        
        if self.transform:
            img = self.transform(image=img)
        
        data = img['image'], (self.time[index], self.event[index])

        return data
    
    
class CSVdataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, df_np, transform=None, mode='train', fusion=False):
        self.data_dir = data_dir
        self.df_np = df_np
        self.transform = transform
        self.mode = mode
        self.fusion = fusion
        
        if self.mode == 'test':
            pos = df_np[:, -1] == 'test'
            imgs_lst = self.data_dir + '/' + df_np[pos][:, 0]
            labels_lst = df_np[pos][:, -2].astype(int)
        elif self.mode == 'val':
            pos = df_np[:, -1] == 'val'
            imgs_lst = self.data_dir + '/' + df_np[pos][:, 0]
            labels_lst = df_np[pos][:, -2].astype(int)
        elif self.mode == 'train':
            pos = df_np[:, -1] == 'train'
            imgs_lst = self.data_dir + '/' + df_np[pos][:, 0]
            labels_lst = df_np[pos][:, -2].astype(int)
        elif self.mode == 'infer':
            imgs_lst = self.data_dir + '/' + df_np[:, 0]
            labels_lst = df_np[pos][:, -2].astype(int)
        else:
            print("Mode error!")            
       
       
        self.classes = np.unique(df_np[:, -3])
        self.imgs_lst = imgs_lst
        self.labels_lst = labels_lst

    def __len__(self):
        return len(self.imgs_lst)

    def __getitem__(self, index):
        
        if self.fusion:
            img_f = Image.open(self.imgs_lst[index]).convert("L")
            img_org = Image.open(f"../JAMA_CXR/JAMA_CH_256/{self.imgs_lst[index].split('/')[-2]}/{self.imgs_lst[index].split('/')[-1]}").convert("L")
            img = np.stack((np.array(img_org), np.array(img_f), np.array(img_f)), axis=-1)
            label = self.labels_lst[index]
        else:
            img = np.array(Image.open(self.imgs_lst[index]).convert("RGB"))
#             img = apply_exposure(img)
            label = self.labels_lst[index]
        
        if self.transform:
            img = self.transform(image=img)
        
        data = img['image'], label

        return data 
    

def load_data(mode, aug, batchsize, data_dir, csv_path):

    print('Project: ', mode)
    print('aug mode:', aug)
    print('batchsize: ', batchsize)
    
    if aug == 'none':
        data_transforms = {
    'train': albu.Compose([
#         albu.Resize(512, 512),
         albu.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],),
        ToTensorV2()
    ]),
    'val': albu.Compose([
#                 albu.Resize(512, 512),
         albu.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],),
        ToTensorV2()
    ]),
    'test': albu.Compose([
#                 albu.Resize(512, 512),
         albu.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],),
        ToTensorV2()
    ]),
    }      
    
    elif aug == 'copd':
        data_transforms = {
    'train': albu.Compose([
        albu.Resize(256, 256),
        albu.OneOf([
            albu.RandomBrightness(limit=.2, p=1), 
            albu.RandomContrast(limit=.2, p=1), 
            albu.RandomGamma(p=1)
        ], p=.3),
        albu.OneOf([
            albu.Blur(blur_limit=3, p=1),
            albu.MedianBlur(blur_limit=3, p=1)
        ], p=.2),
        albu.OneOf([
            albu.GaussNoise(0.002, p=.5),
        ], p=.2),
#         albu.RandomRotate90(p=.5),
        albu.HorizontalFlip(p=.5),
#         albu.VerticalFlip(p=.5),
#         albu.Cutout(num_holes=10, 
#                     max_h_size=int(.1 * size), max_w_size=int(.1 * size), 
#                     p=.25),
        albu.ShiftScaleRotate(shift_limit=0.03, scale_limit=0.01, rotate_limit=10, p=0.4, border_mode = cv2.BORDER_CONSTANT),
        albu.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],),
        ToTensorV2()
    ]),
    'val': albu.Compose([
        albu.Resize(256, 256),
         albu.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],),
        ToTensorV2()
    ]),
    'test': albu.Compose([
        albu.Resize(256, 256),
         albu.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],),
        ToTensorV2()
    ]),
    }
        
 
    elif aug == 'cpmp':
            data_transforms = {
    'train': albu.Compose([
         albu.Resize(256, 256),
        albu.OneOf([
            albu.RandomBrightness(limit=.2, p=1), 
            albu.RandomContrast(limit=.2, p=1), 
            albu.RandomGamma(p=1)
        ], p=.3),
        albu.OneOf([
            albu.Blur(blur_limit=3, p=1),
            albu.MedianBlur(blur_limit=3, p=1)
        ], p=.2),
        albu.OneOf([
            albu.GaussNoise(0.002, p=.5),
        ], p=.2),
         albu.RandomRotate90(p=.5),
         albu.HorizontalFlip(p=.5),
         albu.VerticalFlip(p=.5),
#         albu.Cutout(num_holes=10, 
#                     max_h_size=int(.1 * size), max_w_size=int(.1 * size), 
#                     p=.25),
        albu.ShiftScaleRotate(shift_limit=0.03, scale_limit=0.01, rotate_limit=10, p=0.4, border_mode = cv2.BORDER_CONSTANT),
        albu.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],),
        ToTensorV2()
    ]),
    'val': albu.Compose([
         albu.Resize(256, 256),
         albu.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],),
        ToTensorV2()
    ]),
    'test': albu.Compose([
         albu.Resize(256, 256),
         albu.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],),
        ToTensorV2()
    ]),
    'ext': albu.Compose([
         albu.Resize(256, 256),
         albu.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],),
        ToTensorV2()
    ]),
    }
            
            
    else:
        print(f'recheck opt.augment = {aug}')
    

    if mode == 'copd':
        image_datasets = {x: COPDdataset(data_dir, csv_path, mode=x, transform=data_transforms[x])
                  for x in ['train', 'val', 'test']}
        dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batchsize,
                                             shuffle=False, num_workers=8)
              for x in ['train', 'val', 'test']}
        dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val', 'test']}
        labtrans = COPDdataset(data_dir, csv_path, mode='train', transform=data_transforms['train']).labtrans
    
    elif mode == 'copd_np':
        image_datasets = {x: COPDdatasetNP(data_dir, csv_path, mode=x, transform=data_transforms[x])
                  for x in ['train', 'val', 'test']}
        dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batchsize,
                                             shuffle=False, num_workers=8)
              for x in ['train', 'val', 'test']}
        dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val', 'test']}
        labtrans = COPDdatasetNP(data_dir, csv_path, mode='train', transform=data_transforms['train']).labtrans
    
    elif mode == 'cpmp':
        image_datasets = {x: CPMPdataset(data_dir, csv_path, mode=x, transform=data_transforms[x])
                  for x in ['train', 'val', 'test', 'ext']}
        dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batchsize,
                                             shuffle=True, num_workers=8)
              for x in ['train', 'val', 'test', 'ext']}
        dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val', 'test', 'ext']}
        labtrans = CPMPdataset(data_dir, csv_path, mode='train', transform=data_transforms['train']).labtrans
    
    elif mode == 'csv':
        df = pd.read_csv(df_path)
        df_np = df.values
        image_datasets = {x: CSVdataset(data_dir, df_np, mode=x, transform=data_transforms[x])
                  for x in ['train', 'val', 'test']}
        dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batchsize,
                                             shuffle=True, num_workers=4)
              for x in ['train', 'val', 'test']}
        dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val', 'test']}
        labtrans = None
        
    else:
        print(f'recheck opt.project = {mode}')



    return image_datasets, dataloaders, dataset_sizes, labtrans

def load_infer_data(mode, aug, batchsize, data_dir, csv_path, train_data_dir, train_csv_path):

    print('Project: ', mode)
    print('aug mode:', aug)
    print('batchsize: ', batchsize)
    
    if aug == 'none':
        data_transforms = albu.Compose([
#                 albu.Resize(512, 512),
         albu.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],),
        ToTensorV2()
    ])   
    elif aug == 'copd':
        data_transforms = albu.Compose([
#         albu.Resize(256, 256),
         albu.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],),
        ToTensorV2()
    ])
    elif aug == 'hip':
        data_transforms = albu.Compose([
                albu.augmentations.geometric.resize.LongestMaxSize(max_size=256),
        albu.augmentations.transforms.PadIfNeeded(min_height=256, min_width=256, border_mode=cv2.BORDER_CONSTANT, value=0),
         albu.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],),
        ToTensorV2()
    ])
    elif aug == 'cpmp':
            data_transforms = albu.Compose([
         albu.Resize(256, 256),
         albu.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],),
        ToTensorV2()
    ])    
    else:
        print(f'recheck opt.augment = {aug}')
    
    if mode == 'copd':
        image_datasets = COPDdataset(data_dir, csv_path, mode='infer', transform=data_transforms)
        dataloaders = torch.utils.data.DataLoader(image_datasets, batch_size=batchsize,
                                             shuffle=True, num_workers=8)
        dataset_sizes = len(image_datasets)
        labtrans = COPDdataset(train_data_dir, train_csv_path, mode='train').labtrans
    
    elif mode == 'hip':
        image_datasets = HipCTdataset(data_dir, csv_path, mode='infer', transform=data_transforms)
        dataloaders = torch.utils.data.DataLoader(image_datasets, batch_size=batchsize,
                                             shuffle=True, num_workers=8)
        dataset_sizes = len(image_datasets)
        labtrans = HipCTdataset(train_data_dir, train_csv_path, mode='train').labtrans
    
    elif mode == 'cpmp':
        image_datasets = CPMPdataset(data_dir, csv_path, mode='infer', transform=data_transforms)
        dataloaders = torch.utils.data.DataLoader(image_datasets, batch_size=batchsize,
                                             shuffle=True, num_workers=8)
        dataset_sizes = len(image_datasets)
        labtrans = CPMPdataset(train_data_dir, train_csv_path, mode='train').labtrans
    
    elif mode == 'csv':
        df = pd.read_csv(df_path)
        df_np = df.values
        image_datasets = CSVdataset(data_dir, csv_path, mode='infer', transform=data_transforms)
        dataloaders = torch.utils.data.DataLoader(image_datasets, batch_size=batchsize,
                                             shuffle=True, num_workers=4)
        dataset_sizes = len(image_datasets)
        labtrans = None
        
    else:
        print(f'recheck opt.project = {mode}')


    return image_datasets, dataloaders, dataset_sizes, labtrans

def load_mt_data(mode, aug, batchsize, data_dir, masks_dir, csv_path, classes):

    print('Project: ', mode)
    print('aug mode:', aug)
    print('batchsize: ', batchsize)
    
    if aug == 'none':
        data_transforms = {
    'train': albu.Compose([
#         albu.Resize(512, 512),
#          albu.Normalize(
#         mean=[0.485, 0.456, 0.406],
#         std=[0.229, 0.224, 0.225],),
        ToTensorV2()
    ]),
    'val': albu.Compose([
#                 albu.Resize(512, 512),
#          albu.Normalize(
#         mean=[0.485, 0.456, 0.406],
#         std=[0.229, 0.224, 0.225],),
        ToTensorV2()
    ]),
    'test': albu.Compose([
#                 albu.Resize(512, 512),
#          albu.Normalize(
#         mean=[0.485, 0.456, 0.406],
#         std=[0.229, 0.224, 0.225],),
        ToTensorV2()
    ]),
    }      
    
    elif aug == 'copd':
        data_transforms = {
    'train': albu.Compose([
        albu.Resize(256, 256),
        albu.OneOf([
            albu.RandomBrightness(limit=.2, p=1), 
            albu.RandomContrast(limit=.2, p=1), 
            albu.RandomGamma(p=1)
        ], p=.3),
        albu.OneOf([
            albu.Blur(blur_limit=3, p=1),
            albu.MedianBlur(blur_limit=3, p=1)
        ], p=.2),
        albu.OneOf([
            albu.GaussNoise(0.002, p=.5),
        ], p=.2),
#         albu.RandomRotate90(p=.5),
        albu.HorizontalFlip(p=.5),
#         albu.VerticalFlip(p=.5),
#         albu.Cutout(num_holes=10, 
#                     max_h_size=int(.1 * size), max_w_size=int(.1 * size), 
#                     p=.25),
        albu.ShiftScaleRotate(shift_limit=0.03, scale_limit=0.01, rotate_limit=10, p=0.4, border_mode = cv2.BORDER_CONSTANT),
#         albu.Normalize(
#         mean=[0.485, 0.456, 0.406],
#         std=[0.229, 0.224, 0.225],),
        albu.Lambda(image=to_tensor, mask=to_tensor)
    ]),
    'val': albu.Compose([
        albu.Resize(256, 256),
#          albu.Normalize(
#         mean=[0.485, 0.456, 0.406],
#         std=[0.229, 0.224, 0.225],),
        albu.Lambda(image=to_tensor, mask=to_tensor)
    ]),
    'test': albu.Compose([
        albu.Resize(256, 256),
#          albu.Normalize(
#         mean=[0.485, 0.456, 0.406],
#         std=[0.229, 0.224, 0.225],),
        albu.Lambda(image=to_tensor, mask=to_tensor)
    ]),
    }            
            
    else:
        print(f'recheck opt.augment = {aug}')
        
        
    
    if mode == 'mt_copd':
        image_datasets = {x: MTdataset(data_dir, masks_dir, csv_path, classes, mode=x, transform=data_transforms[x])
                  for x in ['train', 'val', 'test']}
        dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batchsize,
                                             shuffle=False, num_workers=8)
              for x in ['train', 'val', 'test']}
        dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val', 'test']}
        labtrans = MTdataset(data_dir, masks_dir, csv_path, classes, mode='train', transform=data_transforms['train']).labtrans

    else:
        print(f'recheck opt.project = {mode}')



    return image_datasets, dataloaders, dataset_sizes, labtrans


def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')