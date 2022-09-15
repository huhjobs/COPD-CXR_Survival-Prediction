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
from skimage import io
from skimage import transform as trans
from torch.utils.data import RandomSampler
from torch.utils.data import DataLoader
import shutil

import segmentation_models_pytorch as smp

from load_data import Dataset, get_training_augmentation, get_validation_augmentation, get_preprocessing



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

    # READ ME: Check data directory format (DATA_DIR > train/val/test > images/masks)
    x_train_dir = os.path.join(DATA_DIR, 'train', 'images')
    y_train_dir = os.path.join(DATA_DIR, 'train', 'masks')

    x_valid_dir = os.path.join(DATA_DIR, 'val', 'images')
    y_valid_dir = os.path.join(DATA_DIR, 'val', 'masks')

    x_test_dir = os.path.join(DATA_DIR, 'test', 'images')
    y_test_dir = os.path.join(DATA_DIR, 'test', 'masks')

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

    
    if not opt.eval_only:
        # Load Data
        train_dataset = Dataset(
            x_train_dir, 
            y_train_dir, 
            augmentation=get_training_augmentation(), 
            preprocessing=get_preprocessing(preprocessing_fn),
            classes=CLASSES,
        )

        valid_dataset = Dataset(
            x_valid_dir, 
            y_valid_dir, 
            augmentation=get_validation_augmentation(), 
            preprocessing=get_preprocessing(preprocessing_fn),
            classes=CLASSES,
        )

        train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=12)
        valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=4)


    # Create segmentation model with pretrained encoder
    if opt.arch == 'FPN':
        model = smp.FPN(
            encoder_name=ENCODER, 
            encoder_weights=ENCODER_WEIGHTS, 
            classes=len(CLASSES)+1, 
            activation=ACTIVATION,
        )
    elif opt.arch == 'DeepLabV3':
        model = smp.DeepLabV3(
            encoder_name=ENCODER, 
            encoder_weights=ENCODER_WEIGHTS, 
            classes=len(CLASSES)+1, 
            activation=ACTIVATION,
        )
    elif opt.arch == 'DeepLabV3Plus':
        model = smp.DeepLabV3Plus(
            encoder_name=ENCODER, 
            encoder_weights=ENCODER_WEIGHTS, 
            classes=len(CLASSES)+1, 
            activation=ACTIVATION,
        )
    else:
        f"recheck opt.arch = {opt.optimizer}"

    ## TODO: 없는 모델 쓰면 에러 뜨게 만들기
    
    if opt.optimizer == 'adam':
        optimizer = torch.optim.Adam([dict(params=model.parameters(), lr=0.0001),])
    else:
        f"recheck opt.optimizer = {opt.optimizer}"

    # Create epoch runners
    if not opt.eval_only:
        train_epoch = smp.utils.train.TrainEpoch(
            model, 
            loss=loss, 
            metrics=metrics, 
            optimizer=optimizer,
            device=DEVICE,
            verbose=True,
        )

    valid_epoch = smp.utils.train.ValidEpoch(
        model, 
        loss=loss, 
        metrics=metrics, 
        device=DEVICE,
        verbose=True,
    )

    
    # Train Model
    if not opt.eval_only:
        print("==========Training Start==========")
        max_score = 0

        for i in range(0, opt.n_epochs):

            print('\nEpoch: {}'.format(i))
            train_logs = train_epoch.run(train_loader)
            valid_logs = valid_epoch.run(valid_loader)

            # do something (save model, change lr, etc.)
            if max_score < valid_logs['iou_score']:
                max_score = valid_logs['iou_score']
                torch.save(model, f"{opt.run_path}/best_model.pt")
                print('Model saved!')

            if i == 25:
                optimizer.param_groups[0]['lr'] = 1e-5
                print('Decrease decoder learning rate to 1e-5!')
         
        print(f"==========Done Training {opt.n_epochs} Epochs==========")
              
    
    if opt.eval_only:
        MODEL_PATH = opt.model_path
        shutil.copy(MODEL_PATH, f"{opt.run_path}/best_model.pt")
    elif not opt.eval_only:
        MODEL_PATH = f"{opt.run_path}/best_model.pt"
    
    # Test best saved model
    best_model = torch.load(MODEL_PATH)

    # create test dataset
    test_dataset = Dataset(
        x_test_dir, 
        y_test_dir, 
        augmentation=get_validation_augmentation(), 
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
    
    print(f"==========Validation Start==========")
    logs = test_epoch.run(test_dataloader)
    print(logs)
    
    with open(os.path.join(opt.run_path,'logs.pickle'),'wb') as fp:
        pickle.dump(logs, fp, protocol=pickle.HIGHEST_PROTOCOL)

    # test dataset without transformations for image visualization
    test_dataset_vis = Dataset(
        x_test_dir, y_test_dir, 
        classes=CLASSES,
    )

    # Visualize result and save figure as png
    for i in range(5):
        n = np.random.choice(len(test_dataset))
        
        image_vis = test_dataset_vis[n][0].astype('uint8')
        image, gt_mask = test_dataset[n]
        
        gt_mask = gt_mask.squeeze()
        
        x_tensor = torch.from_numpy(image).to(DEVICE).unsqueeze(0)
        pr_mask = best_model.predict(x_tensor)
        pr_mask = (pr_mask.squeeze().cpu().numpy().round())
        
        gt_mask = np.transpose(gt_mask, (1,2,0))
        pr_mask = np.transpose(pr_mask, (1,2,0))
        
        visualize(i, run_path = opt.run_path,
            image=image_vis, 
            ground_truth_mask=gt_mask[:,:,:-1], 
            predicted_mask=pr_mask[:,:,:-1]
        )
       
    print("")
    print(f"Results saved in: {opt.run_path}")

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


# helper function for data visualization
def visualize(i, run_path, **images):
    """PLot images in one row."""
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image)
    # plt.legend()
    plt.savefig(os.path.join(run_path,"test_prediction.png"))

def save_image(save_path, image):
    image = image * 255
    image = image.astype(np.uint8)
    im = Image.fromarray(image)
    im.save(save_path)
