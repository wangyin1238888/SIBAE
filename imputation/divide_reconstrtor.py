import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from os.path import join
from functools import partial
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from rawdata_divide import matrices_by_group
from rawdata_divide import columns_to_select_by_group

file_raw = r'D:\R\Bubble-main\splatter\test\test_dropout\many\log_dropout_90.csv'

rawdata = pd.read_csv(file_raw, sep=',',header=0,index_col=0)

## 分块编码，你会得到三个文件，decoding_1.csv,decoding_2.csv,decoding_3.csv,训练完毕后需要返回原来的整个decoding为encoderdata
file_encode_1 = r'D:\R\Bubble-main\splatter\test\test_dropout\many\decoding\decoding_1.csv'
file_encode_2 = r'D:\R\Bubble-main\splatter\test\test_dropout\many\decoding\decoding_2.csv'
file_encode_3 = r'D:\R\Bubble-main\splatter\test\test_dropout\many\decoding\decoding_3.csv'
file_encode_4 = r'D:\R\Bubble-main\splatter\test\test_dropout\many\decoding\decoding_4.csv'
encoderdata_1= pd.read_csv(file_encode_1, sep=',',header=None,index_col=None)
encoderdata_2= pd.read_csv(file_encode_2, sep=',',header=None,index_col=None)
encoderdata_3= pd.read_csv(file_encode_3, sep=',',header=None,index_col=None)
encoderdata_4= pd.read_csv(file_encode_4, sep=',',header=None,index_col=None)
## 整合encoderdata_123为 encoderdata
# 创建一个空的整合后的encoderdata
encoderdata = np.zeros((rawdata.shape[0], rawdata.shape[1]))  # 23481*384列
# 现在 encoderdata 包含了整合后的数据
# 使用字典来存储encoderdata，以标签为键
# 遍历每个标签的列索引列表并整合数据
for X in ['1', '2','3','4']:
    # 获取相应的 encoderdata
    encoderdata_X = eval(f'encoderdata_{X}')
    columns_to_select = columns_to_select_by_group[X]
    # 将 encoderdata_X  encoderdata
    for i, column_index in enumerate(columns_to_select):
        encoderdata[:, column_index] = encoderdata_X.iloc[:, i]

print("------------------------------------------------------------------")

# 现在 encoderdata 包含了整合后的数据
res_dir = 'D:\\R\\SIBAE-main\\splatter\\test\\test_dropout\\many\\decoding\\'
print("原始数据维度",rawdata.shape)
print("encoderdata数据维度",encoderdata.shape)
encoderdata=pd.DataFrame(encoderdata)
np.savetxt(join(res_dir, 'encoderdata_4.csv'), encoderdata, delimiter=',',fmt='%.6f')
