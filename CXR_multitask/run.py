import sys
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import torch
import pickle
import numpy as np
import cv2
import matplotlib.pyplot as plt
from glob import glob
from PIL import Image
import pandas as pd
from sklearn import metrics as sklearn_metrics
from skimage import io
from skimage import transform as trans
from torch.utils.data import RandomSampler
from torch.utils.data import Dataset, DataLoader
import shutil

import segmentation_models_pytorch as smp

from load_data import load_mt_data
# from load_data import Dataset, get_training_augmentation, get_validation_augmentation, get_preprocessing

import torchtuples as tt
import torch.nn as nn
from pycox.models import loss as coxloss
from mypycox.models.logistic_hazard import LogisticHazard as myLH
from pycox.models import PMF, MTLR, BCESurv
from pycox.utils import kaplan_meier
from pycox.evaluation import EvalSurv

import warnings
warnings.filterwarnings('ignore')

def run(opt):
    # Check PyTorch version and Device
    print('Using PyTorchversion:', torch.__version__)

    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    gpu_count = torch.cuda.device_count()
    print('Device:', DEVICE, 'Available GPU count:' + str(gpu_count))
    print('Mode:', 'Evaluation Only' if opt.eval_only else "Train and Evaluation")


    DATA_DIR = opt.path_to_images
    NUM_EPOCHS = opt.n_epochs
    BATCH_SIZE = opt.batch_size
    CLASSES = opt.classes
    
    # Module Setting
    ENCODER = opt.encoder
    ENCODER_WEIGHTS = opt.encoder_weights
    ACTIVATION = opt.activation # 'sigmoid' # could be None for logits or 'softmax2d' for multiclass segmentation
    preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)
    
    loss = smp.utils.losses.DiceLoss()
    metrics = [
        smp.utils.metrics.IoU(threshold=0.5),
    ]        

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
    image_datasets, dataloaders, dataset_sizes, labtrans = load_mt_data(opt.project, opt.augment, BATCH_SIZE, opt.path_to_images, opt.path_to_masks, opt.path_to_csv, opt.classes)
    
    # Define auxiliary parameters
    aux_params=dict(pooling='avg',             # one of 'avg', 'max'
                    dropout=0.5,               # dropout ratio, default is None
                    activation='softmax2d',      #activation function, default is None
                    classes=labtrans.out_features,                 # define number of output labels
                   )
    
    # Create segmentation model with pretrained encoder
    if opt.arch == 'FPN':
        model_ft = smp.FPN(
            encoder_name=ENCODER, 
            encoder_weights=ENCODER_WEIGHTS, 
            classes=len(CLASSES)+1, 
            activation=ACTIVATION,
            aux_params=aux_params,
        )
    elif opt.arch == 'Unet':
        model_ft = smp.Unet(
            encoder_name=ENCODER, 
            encoder_weights=ENCODER_WEIGHTS, 
            classes=len(CLASSES)+1, 
            activation=ACTIVATION,
            aux_params=aux_params,
        )
    elif opt.arch == 'DeepLabV3Plus':
        model_ft = smp.DeepLabV3Plus(
            encoder_name=ENCODER, 
            encoder_weights=ENCODER_WEIGHTS, 
            classes=len(CLASSES)+1, 
            activation=ACTIVATION,
            aux_params=aux_params,
        )
    elif opt.arch == 'PAN':
        model_ft = smp.PAN(
            encoder_name=ENCODER, 
            encoder_weights=ENCODER_WEIGHTS, 
            classes=len(CLASSES)+1, 
            activation=ACTIVATION,
            aux_params=aux_params,
        )
    elif opt.arch == 'DeepLabV3Plus':
        model_ft = smp.DeepLabV3Plus(
            encoder_name=ENCODER, 
            encoder_weights=ENCODER_WEIGHTS, 
            classes=len(CLASSES)+1, 
            activation=ACTIVATION,
            aux_params=aux_params,
        )
    else:
        f"recheck opt.arch = {opt.optimizer}"
    
    ## TODO: 없는 모델 쓰면 에러 뜨게 만들기
    
    ## Softmax layer 제거
    model_ft.classification_head = nn.Sequential(model_ft.classification_head[0],
                                             model_ft.classification_head[1],
#                                              model_ft.classification_head[2],
                                             model_ft.classification_head[3],)
     
    ## Define loss, metric
    if opt.surv_module == "LogisticHazard":
        loss = [smp.utils.losses.DiceLoss(),
                coxloss.NLLLogistiHazardLoss(),
               ]
    metrics = [smp.utils.metrics.IoU(threshold=0.5),]
    
    ## Load Optim
    if opt.optimizer == 'adam':
        optimizer = tt.optim.Adam(opt.lr)
    else:
        f"recheck opt.optimizer = {opt.optimizer}"
    
    
    
    if opt.surv_module == "LogisticHazard":
        model = myLH(model_ft, optimizer, duration_index=labtrans.cuts, loss = loss)
    else:
        f"recheck opt.surv_module = {opt.surv_module}"

    callbacks = [tt.cb.EarlyStopping(patience=opt.patience, file_path=f"{opt.run_path}/best_model.pt")]
    verbose = True
    log = model.fit_dataloader(dataloaders['train'], NUM_EPOCHS, callbacks, verbose, metrics = metrics, val_dataloader=dataloaders['val'], loss_weight = opt.loss_weight)
    
    
    ## Save log
    fig = log.plot().get_figure()
    fig.savefig(os.path.join(opt.run_path, 'log.png'))
    
    fig = log.to_pandas()[['val_loss', 'train_loss']].plot().get_figure()
    fig.savefig(os.path.join(opt.run_path,'loss.png'))
    
    fig = log.to_pandas()[['val_dice_loss', 'train_dice_loss']].plot().get_figure()
    fig.savefig(os.path.join(opt.run_path,'dice_loss.png'))
    
    fig = log.to_pandas()[['val_iou_score', 'train_iou_score']].plot().get_figure()
    fig.savefig(os.path.join(opt.run_path,'iou.png'))
    
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
        auc[year] = sklearn_metrics.roc_auc_score(time_label(int(surv.index[20*year]), sim_test), -surv.iloc[20*year].values+1)
        fpr[year], tpr[year], _ = sklearn_metrics.roc_curve(time_label(int(surv.index[20*year]), sim_test), -surv.iloc[20*year].values+1)
        print(f"{year}year-AUC : ", round(auc[year], 4))
        
        
    ## Plot    
    for year in [1, 3, 5]:
        plt.plot(fpr[year], tpr[year], label=f"{year}-year")
    plt.legend()
    plt.savefig(os.path.join(opt.run_path,'auc.png'))
    
    for i in range(2):
        idx = sim_test[1] == i
        kaplan_meier(sim_test[0][idx], sim_test[1][idx]).rename(f"{i}_km").plot()
        surv.loc[:, idx].mean(axis=1).rename(f"{i}_nn").plot()
        _ = plt.legend()
    plt.savefig(os.path.join(opt.run_path,'simulate.png'))
    
    ## Save performance
    performance = {"C-index": round(ev.concordance_td(), 4)}
    performance.update({f"{year}y-AUC": round(auc[year], 4) for year in [1,3,5]})
    
    with open(os.path.join(opt.run_path,'performance.pickle'), 'wb') as handle:
        pickle.dump(performance, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
#     with open('performance.pickle', 'rb') as handle:
#         b = pickle.load(handle)
    
    ## Save Model
    torch.save(model.net.state_dict(), f"{opt.run_path}/best_model.pt")
    surv.to_csv(f"{opt.run_path}/surv.csv") #, index=False
    ## TODO : save eval results
    ## TODO : eval_only version
    
def infer_run(opt):
    # Check PyTorch version and Device
    print('Using PyTorchversion:', torch.__version__)
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    gpu_count = torch.cuda.device_count()
    print('Device:', DEVICE, 'Available GPU count:' + str(gpu_count))

    CLASSES = opt.classes

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
    
    if opt.resized:
        INPUT_DIR = opt.path_to_images
    elif not opt.resized:  
        # Resize Images
        os.makedirs(f'{opt.run_path}/inferImage_resize', exist_ok=True)

        if opt.num_samples:
            test_images = glob(opt.path_to_images+'/*.png')[:opt.num_samples]
        else: test_images = glob(opt.path_to_images+'/*.png')

        for path in test_images:
            name = path.split('/')[-1]
            img = io.imread(path)
            img_resize = trans.resize(img,(256,256))
            io.imsave(os.path.join(opt.run_path,'inferImage_resize',name),img_resize)

        INPUT_DIR = os.path.join(opt.run_path,'inferImage_resize')

    # load best saved checkpoint
    best_model = torch.load(opt.model_path)

    # Module Setting
    ENCODER = opt.encoder
    ENCODER_WEIGHTS = opt.encoder_weights
    ACTIVATION = opt.activation # 'sigmoid' # could be None for logits or 'softmax2d' for multiclass segmentation
    preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

    loss = smp.utils.losses.DiceLoss()
    metrics = [
        smp.utils.metrics.IoU(threshold=0.5),
    ]
    
    # create test dataset
    test_dataset = Dataset(
        INPUT_DIR, 
        INPUT_DIR, 
        preprocessing=get_preprocessing(preprocessing_fn),
        classes=CLASSES,
    )

    test_dataloader = DataLoader(test_dataset)

    # evaluate model on test set
    test_epoch = smp.utils.train.ValidEpoch(
        model=best_model,
        loss=loss,
        metrics=metrics,
        device=DEVICE,
    )

    logs = test_epoch.run(test_dataloader)

    # Save figures
    test_dataset_vis = Dataset(
        INPUT_DIR, INPUT_DIR, 
        classes=CLASSES,
    )
    
    os.makedirs(f'{opt.run_path}/predicted_masks', exist_ok=False)
    
    for n in range(len(test_dataset)):
        
        image_vis = test_dataset_vis[n][0].astype('uint8')
        image, gt_mask = test_dataset[n]
        
        gt_mask = gt_mask.squeeze()
        
        x_tensor = torch.from_numpy(image).to(DEVICE).unsqueeze(0)
        pr_mask = best_model.predict(x_tensor)
        pr_mask = (pr_mask.squeeze().cpu().numpy().round())
            
            
        gt_mask = np.transpose(gt_mask, (1,2,0))
        pr_mask = np.transpose(pr_mask, (1,2,0))
        
        save_path = os.path.join(f'{opt.run_path}/predicted_masks',test_dataset.ids[n])
        save_image(save_path, pr_mask[:,:,:-1])
    
    print("")
    print(f"Results saved in: {opt.run_path}")

# # helper function for data visualization
# def visualize(i, run_path, **images):
#     """PLot images in one row."""
#     n = len(images)
#     plt.figure(figsize=(16, 5))
#     for i, (name, image) in enumerate(images.items()):
#         plt.subplot(1, n, i + 1)
#         plt.xticks([])
#         plt.yticks([])
#         plt.title(' '.join(name.split('_')).title())
#         plt.imshow(image)
#     # plt.legend()
#     plt.savefig(os.path.join(run_path,"test_prediction.png"))

# def save_image(save_path, image):
#     image = image * 255
#     image = image.astype(np.uint8)
#     im = Image.fromarray(image)
#     im.save(save_path)

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