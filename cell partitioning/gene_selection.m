function id = gene_selection(M0,iniData,flag,system_used,low_mu,high_mu,low_F)
if ~exist('flag','var') || isempty(flag)
    flag = 0; % select gene based on Fano factor and the mean expression
end
if ~exist('system_used','var') || isempty(system_used)
    system_used = 'Mac'; % select gene based on Fano factor and the mean expression
end
    Rscript = 'D:\R\R-4.3.1\bin\Rscript.exe'; % for 64-bit windows
if ~exist('low_mu','var') || isempty(low_mu)
    %标准的low_mu = 0.01;
    low_mu = 0.01;
end
if ~exist('high_mu','var') || isempty(high_mu)
     %标准的high_mu = 3.5;
     high_mu = 3.5;
end
if ~exist('low_F','var') || isempty(low_F)
     %标准的low_F = 0.01;
     low_F = 0.01;
end

% disp('selecting genes:');
if flag == 0
    disp('选用fano因子')
    id = HVGs(M0,low_mu,high_mu,low_F,iniData);%选用fano因子low_mu = 0.01;high_mu = 3.5;  low_F = 0.5;



   % 假设 selected_genes_matrix 是您的已选择基因的矩阵
    selected_genes_matrix = iniData(id, :);
    disp(size(selected_genes_matrix));
    
    

% % 指定保存的文件路径和文件名
selected_genes_file_path = 'selected_genes_expression_matrix.csv';
% 
% % 使用 writetable 函数将选出的基因表达矩阵保存为 txt 文件
writetable(selected_genes_matrix, selected_genes_file_path);
% 
% 



elseif flag == 1
    %%%%要么进行python进行基尼指数的计算，要么写一个脚本进行R语言的基尼指数
    filefolder = pwd;%获取当前目录
    T = array2table(M0,'RowNames',strcat('gene',cellstr(num2str([1:size(M0,1)]'))));
    writetable(T,'raw_temporal.txt','Delimiter','\t','WriteRowNames',1);
    % Calling R's GiniIndex
    eval([' system([', '''', Rscript, ' GiniIndex.R ', '''', ' filefolder]);']);
    id = importdata('Gini_ID.txt');
    %elseif flag == 1：如果 flag 等于 1，则将数据矩阵 M0 转换为一个表格，并将其保存为文本文件 'raw_temporal.txt'，
    % 然后调用 R 脚本 GiniIndex.R 来计算基因的 Gini 指数，将结果保存在文件 'Gini_ID.txt' 中，
    % 最后将文件中的数据导入到变量 id 中。
else
    id1 = HVGs(M0,low_mu,high_mu,low_F);
    T = array2table(M0,'RowNames',strcat('gene',cellstr(num2str([1:size(M0,1)]'))));
    writetable(T,'raw_temporal.txt','Delimiter','\t','WriteRowNames',1);
    % Calling R's GiniIndex
    filefolder = pwd;%当前文件目录
    eval([' system([', '''', Rscript, ' GiniIndex.R ', '''', ' filefolder]);']);
    id2 = importdata('Gini_ID.txt');
    id = union(id1,id2);%同时选择基因
end


