# -*- coding: utf-8 -*-
"""
Created on Fri Mar  7 08:43:14 2025

@author: 香敏

1，创建分子指纹【仅运行1次】

2，递归特征消除，筛选特征【仅依赖预实验结果，运行一次，重新整理到RFE_FeatureSelection.py】

3，通过多个模型的预测不确定度来 推荐新配方【最终未采用】
4，通过随机森林的预测不确定度来 推荐新配方【采用，但active machine learning过程中有人机交互，
   每个iteration策略不一样，不是纯粹由固定代码产生新配方，未整理成单独可运行代码】
"""

import pandas as pd
import numpy as np
import rdkit
from rdkit import rdBase, Chem
from rdkit.Chem import PandasTools, Descriptors, rdMolDescriptors, MolFromSmiles
from rdkit.Chem import QED,Lipinski
import os
import seaborn as sns
import matplotlib.pyplot as plt

###（一）分子指纹的产生+预处理
# 读取Excel文件
path1 = 'D:/uProjects/AI/PJscripts/DataIn'
path2 = '溶剂性质.xlsx'
fulFile = os.path.join(path1, path2)
excel_file = pd.ExcelFile(fulFile)
# 获取指定工作表中的数据
df = excel_file.parse('Sheet1')
# 查看数据的基本信息
print('数据基本信息：')
df.info()
# 查看数据集行数和列数
rows, columns = df.shape
if rows == 0:
    # 若行数为0，说明数据为空
    print('数据为空')
else:
    # 数据不为空，查看数据前几行信息
    print('数据前几行信息：')
    print(df.head(2).to_csv(sep='\t', na_rep='nan'))
    
# 先将多个分子的片段汇总到一个片段存储器中
ms=list()
for i in np.arange(0,rows):
#     print(sdmsX[i])
    ms.append(Chem.MolFromSmiles(df.smile[i]))

## 直接进行其他方式的分子指纹生成 【最终使用代码段5】
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import MACCSkeys
from rdkit.Chem import DataStructs

# 生成MACCS指纹
fps1 = [MACCSkeys.GenMACCSKeys(x) for x in ms]
print(len(fps1), fps1[0].GetNumBits())
matFP1 = np.array(fps1)

np.savetxt('dataOut/FPMACCS.csv', matFP1 , delimiter = ',')

plt.plot(matFP1.T)
# matFP1里边有很多完全为0的列，需要剔除
indices_to_remove = np.where(matFP1.sum(axis=0) == 0)
matFP1 = np.delete(matFP1, indices_to_remove, axis=1)
# 完全为1的列，剔除
indices_to_remove = np.where(matFP1.sum(axis=0) == matFP1.shape[0])
matFP1 = np.delete(matFP1, indices_to_remove, axis=1)

np.savetxt('dataOut/FPMACCS1.csv', matFP1 , delimiter = ',')

###（二）使用matlab的pca函数进行主成分分析，代码如下

# FP = csvread('C:\Users\10754\.spyder-py3\dataOut\FPMACCS1.csv');
# [coeff,score,latent,tsquared,explained] = pca(FP);
# plot(cumsum(explained));
# disp(cumsum(explained));

# 结果发现前9个主成分，即可得到95.98%的累计方差贡献率
# 于是将score矩阵的前9列复制粘贴到FPMACCS1.csv表格中，作为溶剂的指纹特征矩阵（14*9）


###（二）63组预实验结果整理成X-Y数据集
# 读取Excel配方和结果文件
path3 = '1_结果表格.xlsx'
fulFile2 = os.path.join(path1, path3)
excel_file = pd.ExcelFile(fulFile2)
# 获取指定工作表中的数据
df1 = excel_file.parse('Sheet1')
matFP1=np.array(excel_file.parse('Sheet2'))


#整理成完整的X-Y
Y=np.array(df1.变化百分比)

nSam=df1.shape[0]
nFP=matFP1.shape[1]
nAtrr=df.shape[1]-2

X=np.zeros((nSam,nFP*2))
Xattr=np.zeros((nSam,nAtrr*2))
Xratio=np.zeros((nSam,1))

donePF=np.zeros((nSam,3))
for i in range(nSam):
    if df1.溶剂种类1[i]<df1.溶剂种类2[i]:
        donePF[i,0]=df1.溶剂种类1[i]-1;
        donePF[i,1]=df1.溶剂种类2[i]-1;
        donePF[i,2]=df1.体积比a[i]/(df1.体积比b[i]+df1.体积比a[i]);
    else:
        donePF[i,0]=df1.溶剂种类2[i]-1;
        donePF[i,1]=df1.溶剂种类1[i]-1;
        donePF[i,2]=df1.体积比b[i]/(df1.体积比b[i]+df1.体积比a[i]);
    
    X[i,0:nFP]=matFP1[df1.溶剂种类1[i]-1,:]
    X[i,nFP:] =matFP1[df1.溶剂种类2[i]-1,:]
    Xattr[i,:]=np.concatenate((df[df.columns[1:21]].iloc[df1.溶剂种类1[i]-1],
                               df[df.columns[1:21]].iloc[df1.溶剂种类2[i]-1]),axis=0)

#配方倍增
X1     =     X[:,np.concatenate((range(nFP,nFP*2),range(0,nFP)), axis=0)]
Xattr1 = Xattr[:,np.concatenate((range(nAtrr,nAtrr*2),range(0,nAtrr)), axis=0)]
X      = np.concatenate((X, X1), axis=0)
Xattr  = np.concatenate((Xattr, Xattr1), axis=0)
Xratio=np.concatenate((np.array(df1.体积比a/(df1.体积比b+df1.体积比a)),
                       np.array(df1.体积比b/(df1.体积比b+df1.体积比a))), axis=0)
Xratio=Xratio[:,np.newaxis]

Xall = np.concatenate((X, Xratio, Xattr), axis=1)
Y = np.concatenate((Y, Y), axis=0)
# 完整的X-Y
np.savetxt('DataOut/X.csv', Xall , delimiter = ',')
np.savetxt('DataOut/Y.csv', Y , delimiter = ',')


###（二） 开始特征选择 和 模型优化
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE

# 3. 数据预处理
Xall = np.delete(Xattr,[5,25], axis=1)   #删除重复特征：因为表格中 单一同位素的质量==确切的质量
# 特征标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(Xall)

# 5. 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, Y, test_size=0.1, random_state=42)

# 6. 定义回归模型
model = LinearRegression()
# model.fit(X_train, y_train)

# # 使用中文字体
# plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定字体为黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号'-'显示为方块的问题

# 创建 RFE 对象，指定模型和要选择的特征数量
rfe = RFE(estimator=model, n_features_to_select=7, step=1)
# 拟合 RFE 模型
fit = rfe.fit(X_train, y_train)
# 输出选择的特征数量
print("选择的特征数量: %d" % fit.n_features_)
# 输出特征排名
print("特征排名: %s" % fit.ranking_)
# 输出哪些特征被选中（True 表示选中）
print("特征是否被选中: %s" % fit.support_)
# 将结果以 DataFrame 形式展示，更直观
feature_df = pd.DataFrame({
    'Feature Index': range(X_train.shape[1]),
    'Ranking': fit.ranking_,
    'Selected': fit.support_
})
print(feature_df)

idxGood=np.array([2,3,4,5,9,11,13,17,18,21,22,23,24,28,30,32,36,37]); #matlab分析结果18个=9+9）

#(三) 通过多个模型的预测不确定度来 推荐新配方
# 可以使用选中的特征来训练模型
# X_selected = rfe.transform(Xall)

X_selected = X_scaled[:,idxGood-1]
XMod = np.concatenate((X,Xratio,X_selected), axis=1)

from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer
from sklearn.metrics import r2_score

import warnings
# 忽略所有警告
warnings.filterwarnings('ignore')

def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def print_model_errors(new_model, X, Y):
    # 预测测试集结果
    y_pred = new_model.predict(X)
    
    # 输出一些结果，例如  ++ 均方误差  ++ 
    print("Root Mean squared error: %.2f" % np.sqrt(mean_squared_error(Y, y_pred)))
     
    # 使用交叉验证计算   ++ 调整后的R² ++
    # 注意：scikit-learn本身不直接提供adjusted R²的计算函数，但我们可以自定义一个评分器来使用它
    
    r2 = r2_score(y_pred, Y)
    adjusted_r2 = 1 - (1 - r2) * (len(Y) - 1) / (len(Y) - X.shape[1] - 1)  # 手动计算调整后的R²
    print("Adjusted R^2:", adjusted_r2)
     
    # ++ 调整后的R² ++
    mape = mean_absolute_percentage_error(Y, y_pred)
    print(f'MAPE: {mape:.2f}%')

# 删除离群样本
indices_to_remove = np.array([0, nSam])   #第1个数据变化76%（删除两次是因为配方倍增）
XMod = np.delete(XMod, indices_to_remove, axis=0)
Y = np.delete(Y, indices_to_remove, axis=0)

#1.线性回归
# 创建一个新的线性回归模型并使用选中的特征进行训练
new_model = LinearRegression()
new_model.fit(XMod, Y)

# 预测测试集结果
y_pred = new_model.predict(XMod)

# 输出一些结果，例如  ++ 均方误差  ++ 
print_model_errors(new_model, XMod, Y)

plt.figure(figsize=(10, 6))
plt.scatter(Y ,y_pred, label='线性网络预测')

if 1:
    #2.随机森林回归
    from sklearn.ensemble import RandomForestRegressor
     
    # 创建随机森林回归模型，n_estimators是树的数量参数
    forest = RandomForestRegressor(n_estimators=100)
    forest.fit(XMod, Y)
    
    # 预测测试集结果
    y_pred = forest.predict(XMod)
     
    # 输出一些结果，例如  ++ 均方误差  ++ 
    print_model_errors(forest, XMod, Y)
    plt.scatter(Y ,y_pred, label='随机森林预测')
    
    # 获取每个树对预测的贡献，并计算标准差来估计不确定性
    tree_preds = np.array([tree.predict(XMod) for tree in forest.estimators_]).T
    std_dev = tree_preds.std(axis=1)

#    plt.scatter(Y,std_dev)
    
if 0:
    #4.神经网络回归
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    
    class SimpleFCNN(nn.Module):
        def __init__(self, input_size, hidden_size1,hidden_size2):
            super(SimpleFCNN, self).__init__()
            # 第一个全连接层
            self.fc1 = nn.Sequential(
                nn.Linear(input_size,hidden_size1),
                nn.Tanh())
            
            # 第二个全连接层
            self.fc2 = nn.Sequential(
                nn.Linear(hidden_size1,hidden_size2),
                nn.Tanh()) 
            # 第三个全连接层，通常是输出层
            self.fc3 = nn.Linear(hidden_size2, 1)     
        
        def forward(self, x):
            # 通过第一个全连接层
            x = self.fc1(x)
            # 通过第二个全连接层，并返回结果（通常在输出层后不使用激活函数，除非是多分类问题使用softmax）
            x = self.fc2(x)
            x = self.fc3(x)
            return x
    
    
    import torch.nn.init as init   
    class RegressionModel(nn.Module):
        def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
            super(RegressionModel, self).__init__()
            self.layer1 = nn.Linear(input_size, hidden_size1)  # 第一层全连接层  
            # 随机初始化
            #init.xavier_uniform_(self.layer1.weight)
            init.xavier_normal_(self.layer1.weight)
            init.zeros_(self.layer1.bias)
            
            self.layer2 = nn.Linear(hidden_size1, hidden_size2)  # 第一层全连接层   
            
            
            self.layer3 = nn.Linear(hidden_size2, output_size)  # 输出层
     
        def forward(self, x):
            out = self.layer1(x)
            out = self.layer2(out)
            out = self.layer3(out)
            return out
    
    class RegressionNet(nn.Module):
        def __init__(self, input_size, hidden_size1, hidden_size2):
            super(RegressionNet, self).__init__()
            # 第一个全连接层，输入维度 28，输出维度 98
            self.fc1 = nn.Linear(input_size, hidden_size1)
            # 第一个批量归一化层
            self.bn1 = nn.BatchNorm1d(hidden_size1)
            # ReLU 激活函数
            self.relu = nn.Tanh()
            # 第一个 Dropout 层，丢弃率 0.5
            #self.dropout1 = nn.Dropout(0.5)
            # 第二个全连接层，输入维度 98，输出维度 48
            self.fc2 = nn.Linear(hidden_size1, hidden_size2)
            # 第二个批量归一化层
            self.bn2 = nn.BatchNorm1d(hidden_size2)
            # 第二个 Dropout 层，丢弃率 0.5
            #self.dropout2 = nn.Dropout(0.5)
            # 输出层，输入维度 48，输出维度 1
            self.fc3 = nn.Linear(hidden_size2, 1)
    
        def forward(self, x):
            # 第一个全连接层
            x = self.fc1(x)
            # 第一个批量归一化层
            x = self.bn1(x)
            # ReLU 激活函数
            x = self.relu(x)
            # 第一个 Dropout 层
            #x = self.dropout1(x)
            # 第二个全连接层
            x = self.fc2(x)
            # 第二个批量归一化层
            x = self.bn2(x)
            # ReLU 激活函数
            x = self.relu(x)
            # 第二个 Dropout 层
            #x = self.dropout2(x)
            # 输出层
            x = self.fc3(x)
            return x
    # 使用apply函数的方式进行初始化

    # 在weight_init中通过判断模块的类型来进行不同的参数初始化定义类型
    def weight_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv2d') != -1:
             torch.nn.init.xavier_normal_(m.weight.data)
             torch.nn.init.constant_(m.bias.data, 0.0)
        elif classname.find('fc') != -1:
             torch.nn.init.xavier_normal_(m.weight)
             torch.nn.init.constant_(m.bias, 0.0)
    
    # 创建一个新的偏置并转换为 nn.Parameter
    new_bias = torch.FloatTensor([0.5])  # 1个元素的偏置向量，与输出特征数匹配
    new_bias = nn.Parameter(new_bias)
    
    # 初始化网络
    net = RegressionNet(input_size=37, hidden_size1=60, hidden_size2=32)
    net.apply(weight_init)
    
    #net = SimpleFCNN(input_size=28, hidden_size1=60, hidden_size2=32)
    #net = RegressionModel(input_size=28, hidden_size1=60, hidden_size2=32, output_size=1)
    #print(net)
    
    # 定义损失函数和优化器
    # criterion = nn.MSELoss()
    #optimizer = torch.optim.Adam(net.parameters(), lr=0.0004)
    #optimizer = torch.optim.SGD(net.parameters(), lr=0.1)
    # 定义损失函数和优化器
    criterion = nn.MSELoss()
    #optimizer = torch.optim.ASGD(net.parameters(), lr=0.01, lambd=0.0001, alpha=0.75, t0=1000000.0, weight_decay=0)
    
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9) #
    #optimizer = torch.optim.Adam(net.parameters(), lr=5e-1, betas=(0.9, 0.999))
    # 训练参数
    max_epochs = 50000
    
    # 数据格式转换
    XMod1=torch.from_numpy(XMod).float()
    Y1=torch.tensor(Y).float()
    
    import torch.utils.data as Ddata
    # 创建数据加载器
    batch_size = 30
    dataset = Ddata.TensorDataset(XMod1, Y1)
    train_loader = Ddata.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # 训练网络
    net.train()
    
    cost1tr = []
    cost1ts = []   
    for epoch in range(3000):
        
        for i, (inputs, labels) in enumerate(train_loader):
            outputs = net(inputs)         # 前向传播
            
            loss = criterion(outputs, torch.unsqueeze(labels, dim=1)) # 计算损失
            
            optimizer.zero_grad()   # 清空梯度
            loss.backward()         # 反向传播计算梯度
            optimizer.step()        # 根据梯度更新权重
            
            cost1tr.append(loss.detach().numpy())
            
            
        if (epoch+1) % 100 == 0:  # 每10个epoch打印一次损失值
            print(f'Epoch [{epoch+1}/500], Loss: {loss.item():.4f}')
            print(outputs.detach().numpy().T)
            
            print(net.fc3.bias)
            print(net.fc3.weight)
            
            plt.plot(np.array(cost1tr),'-r')
            plt.ylabel('task 1 cost')
            plt.xlabel('iterations')
            plt.show() 
            
            
            if net.fc3.bias>3:
                net.fc3.bias= new_bias
            
    
    net.eval()
    # 使用训练好的模型进行预测
    y_pred1 = net(XMod1).detach().numpy()  # 对新的输入进行预测
    
    # 输出一些结果，例如  ++ 均方误差  ++ 
    print("nn Root Mean squared error: %.2f" % np.sqrt(mean_squared_error(Y, y_pred1)))
    
    # ++ 调整后的R² ++
    mape = mean_absolute_percentage_error(Y, y_pred1)
    print(f'nn MAPE: {mape:.2f}%')
    
    # # 使用交叉验证计算   ++ 调整后的R² ++
    # # 注意：scikit-learn本身不直接提供adjusted R²的计算函数，但我们可以自定义一个评分器来使用它
    # scorer = make_scorer(r2_score, multioutput='uniform_average')  # 使用uniform_average来处理多输出情况
    # scores = cross_val_score(net, XMod1, Y1, cv=5, scoring=scorer)  # 使用5折交叉验证
    # adjusted_r2 = 1 - (1 - scores.mean()) * (len(Y1) - 1) / (len(Y1) - X.shape[1] - 1)  # 手动计算调整后的R²
    # print("Adjusted R^2:", adjusted_r2)
            
    plt.scatter(Y ,y_pred1, label='全连接网络预测')
    plt.legend()
    plt.show()



#(四) 通过随机森林的预测不确定度来 推荐新配方
#1定义参数空间  
donePF=donePF;  #【已经做过的实验配方】
    
import itertools
import random

elements = range(14)
combinations = list(itertools.combinations(elements, 2))
print(combinations)

ratios=np.array(range(1,10))/10
nRatio=np.shape(ratios)[0]
#2与已有配方的距离进行计算排名（从而剔除一部分配方）

#配方全部距离[done]
nMeshCC=np.shape(combinations)[0]

distFP=np.zeros((nMeshCC,nSam))  #91种配对方式， 与已经做过的nSam次实验都计算 配方距离
for i in range(nMeshCC):
    for j in range(nSam):
        FP1=np.concatenate((matFP1[donePF[j,0].astype(int)],matFP1[donePF[j,1].astype(int)]), axis=0)
        FP2=np.concatenate((matFP1[combinations[i][0]],    matFP1[combinations[i][1]]), axis=0)
        distFP[i,j] = abs(1-np.dot(FP1,FP2)/(np.linalg.norm(FP1)*np.linalg.norm(FP2)))  #归一化的配方距离:=【1-余弦相似度】

distRatio=np.zeros((nRatio,nSam)) #10种配比， 与已经做过的nSam次实验都计算 配比距离
for i in range(nRatio):
    for j in range(nSam): 
        rph1= ratios[i]
        rph2= donePF[j,2]
        distRatio[i,j] = abs(rph1-rph2)
        
pf=np.zeros((nMeshCC*nRatio,4))  #配方表
tem=0
for i in range(nMeshCC):
    for j in range(nRatio):
        pf[tem,0] = combinations[i][0];
        pf[tem,1] = combinations[i][1];
        pf[tem,2] = ratios[j]
        
        DistTemp=np.zeros(nSam)
        for k in range(nSam):  
            DistTemp[k] = distFP[i,k]+distRatio[j,k]
        pf[tem,3]=min(DistTemp)   #新配方[配对号i，比例j]与已经做过的nSam次实验 计算最小距离
        
        tem=tem+1

# pf中已经做过实验的需要剔除
indices_to_remove = np.where(pf[:,3] < 0.07)
pf2Pred = np.delete(pf, indices_to_remove, axis=0)
#pf2Pred=pf
###  剩余配方为候选，  计算预测不确定性排名

#生成输入 特征 矩阵
n2Pred=np.shape(pf2Pred)[0]

X=np.zeros((n2Pred,nFP*2))
Xattr=np.zeros((n2Pred,nAtrr*2))
Xratio=np.zeros((n2Pred,1))

for i in range(n2Pred):
    X[i,0:nFP]=matFP1[pf2Pred[i,0].astype(int),:]
    X[i,nFP:] =matFP1[pf2Pred[i,1].astype(int),:]
    Xattr[i,:]=np.concatenate((df[df.columns[1:21]].iloc[pf2Pred[i,0].astype(int)],
                               df[df.columns[1:21]].iloc[pf2Pred[i,0].astype(int)]),axis=0)
    Xratio[i]=pf2Pred[i,2]+(random.random()-0.5)*0.08 #【配比 增加了随机性】

Xall = np.delete(Xattr,[5,25], axis=1)
# 特征标准化
scaler = StandardScaler()
#X_scaled = scaler.fit_transform(Xall)
X_scaled = scaler.fit_transform(Xall)
X_selected = X_scaled[:,idxGood-1]
XMod2 = np.concatenate((X,Xratio,X_selected), axis=1)

# 获取每个树对预测的贡献，并计算标准差来估计不确定性
tree_preds = np.array([tree.predict(XMod2) for tree in forest.estimators_]).T
std_dev = tree_preds.std(axis=1)

#plt.plot(std_dev)

#计算 稳定性排名
# 预测测试集结果
y2pred = forest.predict(XMod2)

# 相对不确定性
uncertnPer = std_dev/y2pred
# 先取后20%的不确定性样本
threshold = np.percentile(uncertnPer, 80)
print("Threshold:", threshold)

# 完整的相对不确定性
np.savetxt('dataOut/uncertnPer3.csv', uncertnPer , delimiter = ',')

#【早期策略iter1】作图后手工挑选20个极值处对应的配方，无代码
#【早期策略iter2/3】 挑选20个预测不稳定配方，且按预测值进行区间密度划分
indices_to_remove = np.where(uncertnPer < threshold)  #剔除预测稳定的配方
y2pred = np.delete(y2pred, indices_to_remove, axis=0) 
pf2Pred = np.delete(pf2Pred, indices_to_remove, axis=0)
Xratio = np.delete(Xratio, indices_to_remove, axis=0)


thrs=np.logspace(np.log10(min(y2pred)), np.log10(max(y2pred)), num=20, endpoint=True, base=10.0, dtype=None, axis=0)
pf2Do=np.zeros((20,4))  #配方表

tidx = 0 #避免重复
for i in range(20):
    idx = np.argmin(abs(y2pred-thrs[i]))
    if idx == tidx:
        idx = idx+1
    
    pf2Do[i,:]=pf2Pred[idx,:]
    pf2Do[i,2]=Xratio[idx]
    pf2Do[i,3]=y2pred[idx]
    tidx = idx
	
# 【后期策略iter4/5】 直接取预测值最小的约20个配方
# pf2Do= sorted(y2pred)[0:20] #伪代码