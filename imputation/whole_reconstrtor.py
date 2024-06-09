import datetime
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from os.path import join
from sklearn.decomposition import PCA
from functools import partial
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
# 注意这个数据的是原始数据没有标准化的数据
file_identify = r'D:\R\SIBAE-main\splatter\test\test_dropout\many\identify.csv'
file_encode = r'D:\R\SIBAE-main\splatter\test\test_dropout\many\decoding\encoderdata_3.csv'
# --------------------------------absolate imputation-------------------------------------
identifydata = pd.read_csv(file_identify, sep=',',header=None,index_col=None)
print(identifydata.isna().any().any())
encoderdata = pd.read_csv(file_encode, sep=',',header=None,index_col=None)
indices = identifydata == -1
print(encoderdata.isna().any().any())

result_data = identifydata.copy()

print("weidu",encoderdata.shape)
print("weidu",result_data.shape)
result_data[indices] = encoderdata[indices]
result_data_nan = result_data.isna().any().any()
print(f"Identifydata 是否有 NaN 值: {result_data_nan}")
# -----------------------------relative imputation-----------------------------------------

n_components = 30  #
pca = PCA(n_components=n_components)
result_data_pca = pca.fit_transform(result_data.T)
result_data_pca=result_data_pca.T
print(result_data_pca.shape)
# 计算result_data_pca相关矩阵
correlation_matrix_pca = pd.DataFrame(result_data_pca).corr()
correlation_matrix_pca=correlation_matrix_pca-np.identity(correlation_matrix_pca.shape[0])
highest_indices = np.argsort(np.abs(correlation_matrix_pca.to_numpy()), axis=1)[:, -4:].astype(int)
non_zero_indices = (result_data == 0) & (encoderdata != 0)
total_false_elements = np.sum(~non_zero_indices.to_numpy())
print(f"原来 元素为 False 的数量: {total_false_elements}")
result_data[non_zero_indices] = -1
##---------------------------------------------------------
result_data_np = result_data.values
count = 0
# 找到所有值为 -1 的元素的行和列索引
rows, cols = np.where(result_data == -1)
# 找到每个元素对应的前十个相关系数列
relevant_columns = highest_indices[cols]
# 检查在 result_data 中对应列的前十个元素是否都为零
values_to_check = result_data_np[rows[:, np.newaxis], relevant_columns]
# 检查每行的所有值是否都为零
mask = np.all(values_to_check == 0, axis=1)
# 将符合条件的元素的值设为0
result_data_np[rows[mask], cols[mask]] = 0
# 计算改变的个数
count = np.sum(mask)
print("改变的个数", count)


result_data_processed = pd.DataFrame(result_data_np, columns=result_data.columns)
indices = result_data_processed == -1
result_data111 = result_data_processed.copy()
# 使用 inputdata 中对应位置的值替换所有值为-1的元素
print("weidu",encoderdata.shape)
print("weidu",result_data.shape)
result_data111[indices] = encoderdata[indices]
res_dir = 'D:\\R\\SIBAE-main\\splatter\\test\\test_dropout\\many\\'
np.savetxt(join(res_dir, 'fx_relative_imputed.csv'), result_data111, delimiter=',',fmt='%.6f')

