
# coding: utf-8

# In[ ]:

import argparse
import shutil
from os.path import join
import os
import time
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data.dataloader as dataloader
from tensorboardX import SummaryWriter

class Config(object):
    # 数据根目录
    data_root = 'D:\\R\\SIBAE-main\\'
    # model configs
    middle_layer_size = [256, 128, 256]
    # regularized loss, (1-(x1^2+...+xn^2))^p
    p = 2
    # format for saving encoded results
    formt = 'csv'   # 'npy'
from utils import *
from datasets import *
from model import *
from loss import *

config = Config()
device = "cuda" if torch.cuda.is_available() else "cpu"
# middle_layer_size = [256, 128, 256]  # [n_input_features] + middle_layer_size + [n_output_features]
def get_test_data(dataset_name,dataset_number):
    # x = pd.read_csv("D:\\R\\SIBAE-main\\BuSIBAE_github\\results\\dataset_name\\None\\divide_{}.csv".format(dataset_number), sep=',',header=None, index_col=None)
    file_name = 'D:\\R\\SIBAE-main\\scFetalBrain.csv'
    x = pd.read_csv(file_name, sep=',', header=None, index_col=None)

    # return x, y
    print(x.shape)
    return x

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        '--dataset',
        required=True,
        help='dataset_number')
    argparser.add_argument(
        '--exp_id',
        required=True,
        help='experiment id')
    argparser.add_argument(
        '--datasets',
        required=False,
        help='dataset_name')
    argparser.add_argument(
        '--gpu_id',
        default='0',
        help='which gpu to use')
    args = argparser.parse_args()
    # 从命令行参数获取 dataset_number
    dataset_number = args.dataset


 os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

    X = get_test_data(args.datasets,dataset_number)
    X = X.values
    # 然后将 NumPy 数组转换为 PyTorch 张量
    # scdata = torch.from_numpy(scdata)
    X = torch.tensor(X, dtype=torch.float32)  # 改改改改改改改改改改改
    print("6666666666666666")
    print(X.shape)



    # print("7777777777")
    # print(X.shape) X输出(23481, 384)

    in_dim = X.shape[-1]
    # print(in_dim) 384
    model_layer_sizes = [in_dim] + config.middle_layer_size + [in_dim]

    autoencoder = AutoEncoder(model_layer_sizes)
    # ??????????
    ckpts = find_last(log_dir)
    print("7777777777")
    print(ckpts)
    autoencoder.load_state_dict(torch.load(ckpts))
    autoencoder.eval()# 设置模型为评估模式

    autoencoder=autoencoder.to(device)
    y_dec = []
    with torch.no_grad():
        for cell in tqdm(X):

            cell = torch.Tensor(cell).view(1, -1).to(device)
            print(cell.shape)
            ## 输入维度应该是 1x384
            #?????????一个是 1x0 的矩阵，另一个是 384x256 的矩阵
            enc, dec = autoencoder(cell)
            dec = dec.cpu().numpy().squeeze()
            y_dec.append(dec)

    y_dec = np.array(y_dec)
    print("6666666666666666666666666666666666666")
    print(y_dec.shape)

    if config.formt == 'csv':
        #np.savetxt(join(res_dir, 'decoding_{}.csv'.format(dataset_number)), y_dec, delimiter=',', fmt='%g')
        np.savetxt(join(res_dir, 'decoding.csv'.format(dataset_number)), y_dec, delimiter=',', fmt='%g')
        print('decoded results saving to %s' % join(res_dir, 'decoding_{}.csv'.format(dataset_number)))
        directory_path = r"D:\R\SIBAE-main\SIBAE_github\logs\dataset_name"
        try:
            shutil.rmtree(directory_path)
            print(f"目录 {directory_path} 已成功删除")
        except OSError as e:
            print(f"删除目录 {directory_path} 时出错: {e}")
    elif config.formt=='npy':
        np.save(join(res_dir, 'decoding.npy'), y_dec)
        print('decoded results saving to %s' % join(res_dir, 'decoding.npy'))
    else:
        print('Saving error. Wrong saving format')



        ## python infer.py --exp_id experiment id --datasets identify.csv
        # python infer.py --dataset dataset_number --exp_id model_011_0064_loss_0.013.pth


# for dataset_number in 4 5 6 7 8 9 10 11 12 13; do
#
#     python train.py --dataset $dataset_number --exp_id "experiment id" --datasets "dataset_name"
#

#     python infer.py --dataset 1 --exp_id model_011_0064_loss_0.013.pth
# done