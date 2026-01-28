import time
import sys

import torch
import argparse
from torch.backends import cudnn
from data_factory.dataloader import *
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt  # 需导入可视化库
# from solver import Solver
from solver import Solver
import warnings
import json

warnings.filterwarnings("ignore", message="Complex modules are a new feature under active development")

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

class Logger(object):
    def __init__(self, filename='default.log', add_flag=True, stream=sys.stdout):
        self.terminal = stream
        self.filename = filename
        self.add_flag = add_flag

    def write(self, message):
        self.terminal.write(message)
        self.terminal.flush()
        mode = 'a' if self.add_flag else 'w'
        with open(self.filename, mode) as log:
            log.write(message)
            log.flush()

    def flush(self):
        self.terminal.flush()

def main(config):
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    device = torch.device(f"cuda:{config.gpu}" if torch.cuda.is_available() and config.use_gpu else "cpu")
    print(f"device: {device}")
    cudnn.benchmark = True
    solver = Solver(vars(config))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"dataset: {config.dataset}")
    solver.train()
    print(
        f"{config.dataset}\n win_size: {config.win_size},  patch_size: {config.patch_size},batch_size: {config.batch_size}"
        f"\n num_epochs: {config.num_epochs}, anormly_ratio: {config.anormly_ratio}"
        f"\n sw_max_mean: {config.sw_max_mean}, sw_loss： {config.sw_loss}")
    solver.test()
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    dataset_name = ('SMAP')
    parser.add_argument('--dataset', type=str, default=f'{dataset_name}')
    parser.add_argument('--data_path', type=str, default=f'{dataset_name}')
    parser.add_argument('--patch_size',type=int,default=10)
    parser.add_argument('--num_epochs',type=int,default=1)
    parser.add_argument('--batch_size',type=int,default=64)
    parser.add_argument('--input',type=int,default=25)
    parser.add_argument('--anormly_ratio', type=float, default=0.5)
    parser.add_argument('--ratio', type=float, default=0.2, help='重构点和邻居的占比')
    parser.add_argument('--lr',type=float,default=1e-4)
    parser.add_argument('-sw_max_mean', type=int, default=0, help='0:mean , 1:max')
    parser.add_argument('-sw_loss', type=int, default=0, help='0:mse, 1:mae')
    parser.add_argument('--use-gpu',type=bool,default=True, help='use gpu')
    parser.add_argument('--gpu', type=int,default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=True)
    parser.add_argument('--devices',type=str,default='0,1,2', help='device id')
    parser.add_argument('--loss_fuc', type=str, default='MSE', help='MSE, MAE')

    config = parser.parse_args()
    args = vars(config)
    config.win_size = config.patch_size * config.patch_size
    config.min_size = config.batch_size

    config.use_gpu = True if torch.cuda.is_available() and config.use_gpu else False
    if config.use_gpu and config.use_multi_gpu:
        config.devices = config.devices.replace(' ', '')
        device_ids = config.devices.split(',')
        config.device_ids = [int(id_) for id_ in device_ids]
        config.gpu = config.device_ids[0]
    log_file = f'{config.dataset}_log_file.log'
    sys.stdout = Logger(filename="result/" + log_file, add_flag=True, stream=sys.stdout)
    sys.stderr = Logger(filename=log_file, add_flag=True, stream=sys.stderr)
    print("\n")
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    print("\n\n")
    print('================ Hyperparameters ===============')
    for k, v in sorted(args.items()):
        print('%s: %s' % (str(k), str(v)))

    main(config)
