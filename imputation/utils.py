
# coding: utf-8

# In[ ]:

import numpy as np
import os
from os.path import join
from sklearn.preprocessing import MinMaxScaler

def create_dirs(dirs):
    for _dir in dirs:
        os.makedirs(_dir, exist_ok=True)



def preprocess(x):

    column_mins = np.min(x, axis=0)

    column_maxs = np.max(x, axis=0)

    normalized_data = (x - column_mins) / (column_maxs - column_mins)
    return normalized_data

def normalize(x):
    pass

def inverse_preprocess(y,column_sums, scale=10000):#注意column_sums为原数据的列数
    normalized_data = 10 ** y - 1
    x = (normalized_data * column_sums) / scale
    return x

#ckpts = find_last(log_dir)
def find_last(log_dir):
    print(log_dir)
    ckpts = next(os.walk(log_dir))[2]
    ckpts = sorted(filter(lambda x:x.endswith('.pth'), ckpts))

    if not ckpts:
        import errno
        raise FileNotFoundError(
            errno.ENOENT,
            "Could not find model ckpts"
            )

    ckpt = join(log_dir, ckpts[-1])
    return ckpt