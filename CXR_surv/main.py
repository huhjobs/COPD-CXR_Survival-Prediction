import sys
import argparse
import run
import random
from utils import str2bool
import os
import datetime
import pytz
import yaml

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Options to train the model.')

    parser.add_argument('--path_to_images', type=str, required=True)
    parser.add_argument('--path_to_csv', type=str, required=True)
#     parser.add_argument('--run_name', type=str, required=True)
    parser.add_argument('--filename_column', type=str, required=True, default='inclusion')
    parser.add_argument('--time_column', type=str, required=True, default='survival')
    parser.add_argument('--event_column', type=str, required=True, default='censor')
    parser.add_argument('--project', type=str, required=True, default='copd')
    parser.add_argument('--augment', type=str, required=True, default='none')

    parser.add_argument('--eval_only', type=str2bool, default='False')
    parser.add_argument('--input_size', type=int, default=256)
    parser.add_argument('--random_crop', type=str2bool, default='False')
    parser.add_argument('--gpu_ids', type=str, default="0")
    parser.add_argument('--pretrained_path', type=str, default='False')

    # module options
    parser.add_argument('--arch', type=str, default="resnet50")
    parser.add_argument('--surv_module', type=str, default="LogisticHazard")

    parser.add_argument('--fix_randomness', type=str2bool, default='False')
    parser.add_argument('--seed', type=int, default=111)
    parser.add_argument('--run_path', type=str, default='results')

    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_lr_drops', type=int, default=2)

    # optimization
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--lr_decay_ratio', type=float, default=0.1)
    parser.add_argument('--optimizer', type=str, default='adam')
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--n_epochs', type=int, default=100)
    parser.add_argument('--n_epochs_to_decay', type=str, default='20,10')

    opt = parser.parse_args(sys.argv[1:])
    opt.n_epochs_to_decay = [int(epoch) for epoch in opt.n_epochs_to_decay.split(',')]
    
    ## TODO: opt to parameter.json or whatever

#     if opt.run_path is None:
#         opt.run_path = os.path.join(
#             'results',
#             ''.join(datetime.datetime.now(pytz.timezone('Asia/Seoul')).strftime('%Y-%m-%d-%H-%M-%S'))
#         )
        
    
    if opt.run_path is None:
        opt.run_path = 'results'
        
    opt.run_path = os.path.join(
        opt.run_path,
     ''.join(datetime.datetime.now(pytz.timezone('Asia/Seoul')).strftime('%Y-%m-%d-%H-%M-%S')))
 
            
            
    opt.gpu_ids = [int(gpu_id) for gpu_id in opt.gpu_ids.split(',')]
    

    
    run.run(opt)