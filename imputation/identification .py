# coding: utf-8
# In[ ]:
import os
import h5py
import sklearn
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from os.path import join
from functools import partial
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
file_path = r'D:\R\Bubble-main\splatter\test\test_dropout\many\log_dropout_90.csv'
# 使用read_csv函数读取CSV文件，指定制表符作为分隔符，无列名，无行索引，只读取第2列及其后面的所有列
rawdata = pd.read_csv(file_path, sep=',',header=0,index_col=0)
# rawdata=preprocess(rawdata)
print("raw的维数")
print(rawdata.shape)
data_norm = rawdata.values
data_norm1 = data_norm.copy()

def find_cluster_cell_idx(l, label):
    '''

    '''
    return label==l
#The identification of dropout events
st = datetime.datetime.now()
def identify_dropout(cluster_cell_idxs, X):#将缺失值暂时填补为-1
    for idx in cluster_cell_idxs:
        # 每一行的dropout就是单元格中列有多少0的比率
        dropout=(X[:,idx]==0).sum(axis=1)/(X[:,idx].shape[1])
        # 根据阈值上下位分数确定最大最小dropout率
        dropout_thr=middle
        dropout_upper_thr,dropout_lower_thr = np.nanquantile(dropout,q=dropout_thr),np.nanquantile(dropout,q=0)
        gene_index1 = (dropout<=dropout_upper_thr)&(dropout>=dropout_lower_thr)
        print(gene_index1)
        cv = X[:,idx].std(axis=1)/X[:, idx].mean(axis=1)
        cv_thr=middle
        cv_upper_thr,cv_lower_thr = np.nanquantile(cv,q=cv_thr),np.nanquantile(cv,q=0)
        # print(cv_upper_thr,cv_lower_thr)
        gene_index2 = (cv<=cv_upper_thr)&(cv>=cv_lower_thr)
        print(gene_index2)
        # include_faslezero_gene= list(np.intersect1d(gene_index1,gene_index2))
        include_faslezero_gene = np.logical_and(gene_index1, gene_index2)
        # print(list(include_faslezero_gene).count(True))
        tmp = X[:, idx]
        tmp[include_faslezero_gene] = tmp[include_faslezero_gene]+(tmp[include_faslezero_gene]==0)*-1
        X[:, idx] = tmp
    return X
# 为每个不同的聚类标签找到对应的细胞索引，比如聚类标签0 对应的细胞索引：[0 3 7]....
label_set = np.unique(label_pr)
cluster_cell_idxs = list(map(partial(find_cluster_cell_idx,label=label_pr), label_set))
data_identi=identify_dropout(cluster_cell_idxs, X=data_norm.T).T
print("身份数据集")
print(data_identi)
num_minus_ones = np.count_nonzero(data_identi == -1)

print("数组中值为-1的个数为:", num_minus_ones)
# 将处理后的数据写入CSV文件.csv
output_file = r'D:\R\Bubble-main\splatter\test\test_dropout\many\identify.csv'  # 指定要保存的文件路径
pd.DataFrame(data_identi).to_csv(output_file, index=False,header=False)
nowdata = pd.read_csv(output_file, sep=',',header=None,index_col=None)
print("now的维数")
print(nowdata .shape)
ed = datetime.datetime.now()
print('identify_dropout ：', (ed - st).total_seconds())

