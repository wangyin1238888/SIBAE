import pandas as pd
from os.path import join
import numpy as np
# 读取CSV文件
file_path = r'D:\R\Bubble-main\splatter\test\test_dropout\many\log_dropout_0.csv'

data = pd.read_csv(file_path, sep=',',header=0,index_col=0)
# print(data)
# 读取分类标签文件
with open(r'D:\R\Bubble-main\splatter\test\test_dropout\many\group.txt', 'r') as f:
    groups = f.read().splitlines()
data_by_group = {}

for idx, group in enumerate(groups):

    if group not in data_by_group:
        data_by_group[group] = []

    data_by_group[group].append(idx + 1)

columns_to_select_by_group = {}


data_split_by_group = {}
for group, columns in data_by_group.items():

    columns_to_select = [0] + [idx - 1 for idx in columns]
    columns_to_select_by_group[group] =columns_to_select[1:]

    print("Columns to Select:", len(columns_to_select))

    subset_data = data.iloc[:, columns_to_select]
    data_split_by_group[group] = subset_data

matrices_by_group = {}
for group, subset_data in data_split_by_group.items():
    # 去除第一列（Cell_ID列）并将数据转换为矩阵
    matrix = subset_data.iloc[:, 1:].values
    matrices_by_group[group] = matrix
print("111", matrices_by_group['1'].shape)
print("111", matrices_by_group['2'].shape)
print("111", matrices_by_group['3'].shape)
print("111", matrices_by_group['4'].shape)
print("111", matrices_by_group['5'].shape)

res_dir = r'D:\R\Bubble-main\splatter\test\test_dropout\many\divide'
np.savetxt(join(res_dir, 'divide_1.csv'), matrices_by_group['1'], delimiter=',', fmt='%g')
np.savetxt(join(res_dir, 'divide_2.csv'), matrices_by_group['2'], delimiter=',', fmt='%g')
np.savetxt(join(res_dir, 'divide_3.csv'), matrices_by_group['3'], delimiter=',', fmt='%g')
np.savetxt(join(res_dir, 'divide_4.csv'), matrices_by_group['4'], delimiter=',', fmt='%g')
np.savetxt(join(res_dir, 'divide_5.csv'), matrices_by_group['5'], delimiter=',', fmt='%g')


