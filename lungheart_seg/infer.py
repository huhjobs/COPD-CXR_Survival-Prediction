import sys
import argparse
import run
from utils import str2list, str2bool
import os
import datetime
import pytz

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Options to train the model.')

    # parser.add_argument('--path_to_images', type=str, required=True)
    # parser.add_argument('--num_samples', type=int, required=False) # infer directory 에서 이미지 몇 개만 infer해보고 싶을 때.
    # parser.add_argument('--classes', type=str2list, required=True)

    parser.add_argument('--path_to_images', type=str, default = "../DATA/COPD_PNG")
    parser.add_argument('--num_samples', type=int, default = 0) # infer directory 에서 이미지 몇 개만 infer해보고 싶을 때. 0이면 디렉토리 전체 이미지.
    parser.add_argument('--classes', type=str2list, default = "['heart', 'left lung', 'right lung']")

    # parser.add_argument('--path_to_csv', type=str, required=True)
    # parser.add_argument('--path_to_train_images', type=str, required=True)
    # parser.add_argument('--path_to_train_csv', type=str, required=True)
#     parser.add_argument('--run_name', type=str, required=True)
    # parser.add_argument('--filename_column', type=str, required=True, default='inclusion')
    # parser.add_argument('--time_column', type=str, required=True, default='survival')
    # parser.add_argument('--event_column', type=str, required=True, default='censor')
    # parser.add_argument('--project', type=str, required=True, default='copd')
    # parser.add_argument('--augment', type=str, required=True, default='none')
    parser.add_argument('--run_path', type=str, default=None)
    parser.add_argument('--resized', type=str2bool, default='False')

    # parser.add_argument('--input_size', type=int, default=256)
    # parser.add_argument('--random_crop', type=str2bool, default='False')
    parser.add_argument('--gpu_ids', type=str, default="0")
    parser.add_argument('--model_path', type=str,  default='/home/COPD/lungheart_seg/best_model.pth')
    parser.add_argument('--encoder', type=str, default="se_resnext50_32x4d")
    parser.add_argument('--encoder_weights', type=str, default="imagenet")
    parser.add_argument('--activation', type=str, default="softmax2d")

    # module options
    # parser.add_argument('--arch', type=str, default="resnet50")
    # parser.add_argument('--surv_module', type=str, default="LogisticHazard")
    
    # parser.add_argument('--fix_randomness', type=str2bool, default='False')
    # parser.add_argument('--seed', type=int, default=111)
    

    # parser.add_argument('--batch_size', type=int, default=32)
    # parser.add_argument('--num_lr_drops', type=int, default=2)
    

    opt = parser.parse_args(sys.argv[1:])
    
    ## TODO: opt to parameter.json or whatever

    if opt.run_path is None:
        opt.run_path = 'infer_results'
        
    opt.run_path = os.path.join(
        opt.run_path,
     ''.join(datetime.datetime.now(pytz.timezone('Asia/Seoul')).strftime('%Y-%m-%d-%H-%M-%S')))
    
    opt.gpu_ids = [int(gpu_id) for gpu_id in opt.gpu_ids.split(',')]
    
    run.infer_run(opt)