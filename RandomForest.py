# -*- coding: utf-8 -*-
"""
Created on Tue Apr 29 09:20:23 2025

@author: 香敏
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# 读取Excel配方和结果文件
path1 = 'D:/uProjects/AI/PJHjyy1/1.传感课题执行/PJscripts/DataIn'
path2 = '溶剂性质.xlsx'
path3 = '0123_结果表格.xlsx'

def makeDataMat(path1,path2,path3,dualPF=0):
    fulFile = os.path.join(path1, path2)
    excel_file = pd.ExcelFile(fulFile)
    # 获取指定工作表中的数据
    df = excel_file.parse('Sheet1')
    
    
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
    if dualPF==1:
        #配方倍增
        X1     =     X[:,np.concatenate((range(nFP,nFP*2),range(0,nFP)), axis=0)]
        Xattr1 = Xattr[:,np.concatenate((range(nAtrr,nAtrr*2),range(0,nAtrr)), axis=0)]
        X      = np.concatenate((X, X1), axis=0)
        Xattr  = np.concatenate((Xattr, Xattr1), axis=0)
        Xratio=np.concatenate((np.array(df1.体积比a/(df1.体积比b+df1.体积比a)),
                               np.array(df1.体积比b/(df1.体积比b+df1.体积比a))), axis=0)
        Xratio=Xratio[:,np.newaxis]
        Y = np.concatenate((Y, Y), axis=0)
    else:
        Xratio=np.array(df1.体积比a/(df1.体积比b+df1.体积比a))
        Xratio=Xratio[:,np.newaxis]
        
    # 完整的X-Y
    if 0:
        Xall = np.concatenate((X, Xratio, Xattr), axis=1)
        np.savetxt('dataOut/X.csv', Xall , delimiter = ',')
        np.savetxt('dataOut/Y.csv', Y , delimiter = ',')

    idxGood=np.array([2,3,4,5,9,11,13,17,18,21,22,23,24,28,30,32,36,37]); #matlab分析结果18个=9+9）
    Xatll = np.delete(Xattr,[5,25], axis=1)   #删除重复特征：表格中 单一同位素的质量==确切的质量
    # 特征标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(Xatll)
    X_selected = X_scaled[:,idxGood-1]
    XMod = np.concatenate((X,Xratio,X_selected), axis=1)
    
    return XMod,Y,donePF,X,Xratio, Xattr
    
XMod,Y,donePF, X,Xratio, Xattr   = makeDataMat(path1,path2,path3,1)

# np.savetxt('D:/uProjects/AI/PJHjyy1/1.传感课题执行/csvData/Bth4_dataXY.csv', np.concatenate((XMod,Y[:,np.newaxis]), axis=1), delimiter=',')
# np.savetxt('D:/uProjects/AI/PJHjyy1/1.传感课题执行/csvData/Bth4_donePF.csv', donePF, delimiter=',')


# 删除离群样本
nSam = int(XMod.shape[0]/2)
indices_to_remove = np.array([0, nSam])   #第1个数据变化76%（删除两次是因为配方倍增）
if 1:
    XMod = np.delete(XMod, indices_to_remove, axis=0)
    Y = np.delete(Y, indices_to_remove, axis=0)


from sklearn.metrics import mean_squared_error, r2_score
import warnings
# 忽略所有警告
warnings.filterwarnings('ignore')

def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
def print_model_errors(y_pred, X, Y):
    # 预测测试集结果
    # y_pred = new_model.predict(X)
    
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


if 1:
    #2.随机森林回归
    from sklearn.ensemble import RandomForestRegressor
     
    # 创建随机森林回归模型，n_estimators是树的数量参数
    forest = RandomForestRegressor(n_estimators=100)
    forest.fit(XMod, Y)
    
    # 预测测试集结果
    y_pred = forest.predict(XMod)
     
    # 输出一些结果，例如  ++ 均方误差  ++ 
    print_model_errors(y_pred, XMod, Y)
    plt.scatter(Y ,y_pred, label='随机森林预测')
    
    # 获取每个树对预测的贡献，并计算标准差来估计不确定性
    tree_preds = np.array([tree.predict(XMod) for tree in forest.estimators_]).T
    std_dev = tree_preds.std(axis=1)

# 预测下一批次实验
path4 = 'D:/uProjects/AI/PJHjyy1/1.传感课题执行/PJscripts/DataIn'
path5 = '溶剂性质.xlsx'
path6 = '4_结果表格.xlsx'
XMod2,Y2,donePF2, X2,Xratio2, Xattr2   = makeDataMat(path4,path5,path6,1)
y_pred2 = forest.predict(XMod2)

# 预测结果 平均
nT = int(y_pred2.shape[0]/2)
y_pred2 = (y_pred2[-nT:]+ y_pred2[0:nT])/2

plt.scatter(Y2[-nT:] ,y_pred2, label='随机森林预测')
plt.plot([0,40],[0,40],'-b')
plt.axis('equal')
print_model_errors(y_pred2, XMod2, Y2[-nT:])

# import scipy.io as sio
# boxes = sio.loadmat('D:/uProjects/AI/PJHjyy1/1.传感课题执行/PFy1.mat')['PFy1']
# XMod2[-20:,18] = boxes[:,2]
# 预测mesh

path7 = 'mesh_表格.xlsx'
XMod3,Y3,donePF3, X3,Xratio3, Xattr3   = makeDataMat(path4,path5,path7,1)
y_pred3 = forest.predict(XMod3)
# 预测结果 平均
nT = int(y_pred3.shape[0]/2)
y_pred3 = (y_pred3[-nT:]+ y_pred3[0:nT])/2

if 0:
    import pickle
    # 保存模型
    with open('D:/uProjects/AI/PJHjyy1/1.传感课题执行/PJscripts/DataOut/random_forest_model01234.pkl', 'wb') as f:
        pickle.dump(forest, f)
    # 读取模型
    # forest = pickle.load(f)   #文件打开时需要改成读取模式“rb”