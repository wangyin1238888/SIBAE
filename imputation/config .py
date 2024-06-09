import numpy as np

class Config(object):
    # 数据根目录
    data_root = 'D:/R/SIBAE/'

    # model configs
    middle_layer_size = [256, 128, 256]

    # regularized loss, (1-(x1^2+...+xn^2))^p
    p = 2

    # format for saving encoded results
    formt = 'txt'   # 'npy'
