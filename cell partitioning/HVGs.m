%% 这段代码定义了一个名为 HVGs 的函数，用于基于基因的平均表达和 Fano 因子选择高变异基因（High Variable Genes
function id = HVGs(M,low_mu,high_mu,low_F,iniData)
%选用fano因子low_mu = 0.01;high_mu = 3.5;  low_F = 0.5;
if ~exist('low_mu','var') || isempty(low_mu)
    low_mu = 0; % select gene above this average expression level
end
if ~exist('high_mu','var') || isempty(high_mu)
    high_mu = 10; % select gene below this average expression level
end
if ~exist('low_F','var') || isempty(low_F)
    low_F = 0; % select gene above this Fano factor
end
% M is the raw count data matrix
% id is the set of select gene position


%% select HVGs: calculates the average expression and Fano factor for each gene, places these genes into bins,
% and then calculates a z-score for Fano factor within each bin.
mu = log(1+mean(10.^M-1,2));%进行平均值运算，原来为对数函数转为回来进行平均再转回去对数，参数2表示每一行（每一类基因）

F = log(var(10.^M-1,0,2)./mean(10.^M-1,2));%进行fano因子（一样的先还原再取对数）
mu(isnan(mu)) = 0;F(isnan(F)) = 0;
mu(isinf(mu)) = 0;F(isinf(F)) = 0;
%将平均表达值 mu 分成 20 个离散的区间，返回区间标签 Y 和边界 E。
[Y,E] = discretize(mu,20);%分箱处理,Y是mu数据所属箱子的编号,E有21个范围边界
% E是边界为:最小值--->最大-最小/20 ---> 2*最大-最小/20 ---> 3*最大-最小/20 ..19*最大-最小/20--->最大值
idx = setdiff(1:20,unique(Y));%Y中索引也许没有1--20编号包含完整
if ~isempty(idx)
    %索引没有包含完整1-20，那么则重新分箱，比如没有索引5（防止聚类没有基因）
    E(idx+1) = [];%将本来属于索引5但是没有数据的界限设为空集，
    Y = discretize(mu,E);%没有索引5的话不要索引5，那么可能只有19个区间
end
%计算每个区间内 Fano 因子的平均值和标准差。
mean_y = grpstats(F,Y,"mean");
sd_y = grpstats(F,Y,"std");
F_scaled = (F - mean_y(Y))./sd_y(Y);%每个区间标准化
F_scaled(isnan(F_scaled)) = 0;
%从1000个基因选取标准化fano因子大于0.05并且平均表达在low_mu----high_mu之间的基因
id = find((mu > low_mu) & (mu < high_mu) & F_scaled > low_F);
disp("ggggggggggggggg")
disp(length(id))
%id = sortrows([id, F_scaled(id)], 2, 'descend');  % Sort genes based on Fano factor
%id = id(1:2000, 1);  % Select top 2000 genes
%id = sort(id); % 按照索引值从小到大排序
%disp(id)
%length(id)
figure
scatter(mu,F_scaled,'k.')
xlabel('Average expression');
ylabel('Fano factor')

%bulkdata = readtable('bulkFetalBrain.csv'); % 读取 CSV 文件为表格数据
%selected_bulk = bulkdata(id, :); % 选择特定索引的行

%writetable(selected_bulk, 'bulkFetalBrain_2000.csv'); % 将所选行写入新的 CSV 文件

% 假设 M 是您的原始数据矩阵
selected_M = iniData(id, :); % 选择特定索引的行

writetable(selected_M, 'scFetalBrain_2000.csv'); % 将所选行写入新的 CSV 文件

