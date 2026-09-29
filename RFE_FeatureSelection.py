# -*- coding: utf-8 -*-
"""
Created on Fri Mar  7 08:43:14 2025

@author: 香敏

1，读取预实验结果，获得X-Y数据集
2，递归特征消除，筛选特征
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

###（一）63组预实验结果整理成X-Y数据集
# 读取Excel配方和结果文件
path1 = 'D:/uProjects/AI/PJscripts/DataIn'
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
k=0
datass=np.zeros([38, 4])
for i in range(2):
    for j in range(2):
        rfe = RFE(estimator=model, n_features_to_select=i*2+7, step=j+1)
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
        datass[:,k] = fit.ranking_
        k=k+1
# 考虑配方对称性的情况下求 特征平均排名      
a=np.reshape(sum(datass.T),[2,19])
a=np.mean(a,axis=0)

# 取排序前9个特征，最终的特征选取结果
idxGood=sorted(np.argsort(a)[0:9])

# 在配方X中，有两种溶剂，特征为9*2
idxGood=np.concatenate((np.array(idxGood),np.array(idxGood)+19), axis=0)
# idxGood结果应该是np.array([2,3,4,5,9,11,13,17,18,21,22,23,24,28,30,32,36,37]-1); 
