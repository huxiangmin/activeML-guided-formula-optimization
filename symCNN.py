# -*- coding: utf-8 -*-
"""
Created on Sun Jun 15 09:32:58 2025

@author: 香敏
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler

import os
import numpy as np
import pandas as pd
import random
random.seed(7)

# 读取Excel配方和结果文件
path1 = 'D:/uProjects/AI/PJHjyy1/1.传感课题执行/PJscripts/DataIn'
path2 = '溶剂性质.xlsx'
path3 = '全_结果表格.xlsx'

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
XY = np.concatenate((XMod,Y[:,np.newaxis]), axis=1)

from sklearn.cross_decomposition import PLSRegression  
model = PLSRegression(n_components=6)  
model.fit(XMod, Y)  
y_pred = model.predict(XMod)  
Yresiduals = Y- y_pred

idxBad1 = np.where((Yresiduals / Y) < -1.37)[0]
idxBad2 = np.where(abs(Yresiduals) >10)[0]
idx_bad = np.union1d(idxBad1, idxBad2)

if 1:
    XY = np.delete(XY, idx_bad, axis=0)
    
# else:
#     # 分类建模
#     import scipy.io as sio
#     data = sio.loadmat('E:/Matlab_backup/SNN_hsi/rstAll.mat')
#     clas = data['rstAll']
    
#     # 假设XY和rstAll已经定义
#     mask = clas[:, 6] == 3  # 第3类  55个样本
#     mask = clas[:, 6] == 4  # 第4类  74个样本
    
#     idx1 = np.where(mask)[0]
#     idx2 = idx1 + clas.shape[0]
#     combined_idx = np.hstack((idx1, idx2))
#     XY = XY[combined_idx, :]

idx = np.array([i for i in range(0,XY.shape[0])])
random.shuffle(idx)


from sklearn.metrics import mean_squared_error, r2_score
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
    #print("Adjusted R^2:", adjusted_r2)
     
    # ++ 调整后的R² ++
    mape = mean_absolute_percentage_error(Y, y_pred)
    print(f'MAPE: {mape:.2f}%')
    
def getTest(numLast):
    # # 模型的效果验证BBBB
    # XY = np.loadtxt('D:/uProjects/AI/PJHjyy1/1.传感课题执行/csvData/Bth234_dataXY.csv', delimiter=',')
    # PF = np.loadtxt('D:/uProjects/AI/PJHjyy1/1.传感课题执行/csvData/Bth234_donePF.csv', delimiter=',') 
    
    # scaler = StandardScaler()
    # #X_scaled = scaler.fit_transform(Xall)
    # X_scaled = scaler.fit_transform(XY[:,19:-1])
    # XY[:,19:-1]= X_scaled
    
    # num_samples2 = XY.shape[0]
    # # 划分训练集和验证集
    # x_train = XY[:,0:-1]
    # y_train = XY[:,-1]
    
    # numAll = int(num_samples2/2)
    # if numLast==0: #默认取全部样本
    #     x_test = x_train
    #     y_test = y_train
    # else:          #取后numLast个样本
    #     x_test = np.vstack((x_train[(numAll-numLast):numAll,:],x_train[-numLast:num_samples2,:]))
    #     y_test = np.hstack((y_train[(numAll-numLast):numAll],y_train[-numLast:num_samples2]))
    
    path4 = 'D:/uProjects/AI/PJHjyy1/1.传感课题执行/PJscripts/DataIn'
    path5 = '溶剂性质.xlsx'
    path6 = 'mesh_表格.xlsx'
    x_test,y_test,donePF2, X2,Xratio2, Xattr2   = makeDataMat(path4,path5,path6,1)
    x_test1 = x_test[:,0:18].reshape((x_test.shape[0], 9, 2), order='F')
    x_test2 = x_test[:,19:].reshape((x_test.shape[0], 9, 2), order='F')
    x_test3 = x_test[:,18]
    
    # 创建数据集和数据加载器
    x_test1 = torch.from_numpy(x_test1).float()
    x_test2 = torch.from_numpy(x_test2).float()
    x_test3 = torch.from_numpy(x_test3).float()
    y_test = torch.from_numpy(y_test).float()
    
    return x_test1,x_test2,x_test3,y_test
    
# 定义 Transformer 模型
class TransformerRegressor(nn.Module):
    def __init__(self, input_size, hidden_size, num_heads, num_layers, output_size):
        super(TransformerRegressor, self).__init__()
        self.input_embedding = nn.Linear(input_size, hidden_size)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=num_heads)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.input_embedding(x)
        #x = x.permute(1, 0, 2)  # 调整维度以适应 Transformer 输入格式
        x = x.unsqueeze(0)
        output = self.transformer_encoder(x)
        output = output[-1]  # 取最后一个时间步的输出
        output = self.fc(output)
        return output.squeeze(-1)
    
# 定义 CustomNet 模型
import torch
import torch.nn as nn

import torch
import torch.nn as nn

class MultiInputRegressionNet(nn.Module):
    def __init__(self, fea_size, hid_size):
        super(MultiInputRegressionNet, self).__init__()
        self.fea_size = fea_size
        
        # 图像处理分支1
        self.img_conv1 = nn.Conv2d(
            in_channels=1,          # 输入通道数为1（单通道图像）
            out_channels=fea_size,        # 使用32个卷积核
            kernel_size=(9, 1),     # 卷积核尺寸[9,1]
            stride=1,               # 步长
            padding=0               # 不填充
        )
        # 图像处理分支2
        self.img_conv2 = nn.Conv2d(
            in_channels=1,          # 输入通道数为1（单通道图像）
            out_channels=fea_size,        # 使用32个卷积核
            kernel_size=(9, 1),     # 卷积核尺寸[9,1]
            stride=1,               # 步长
            padding=0               # 不填充
        )
        
        # 分通道卷积层
        self.channel_conv = nn.Conv1d(
            in_channels=fea_size,         # 输入通道数为32
            out_channels=fea_size,        # 输出通道数为32
            kernel_size=5,          # 卷积核尺寸[5,1]
            stride=1,               # 步长
            padding=0,              # 不填充
            groups=fea_size               # 分通道卷积
        )
        
        # 特征处理分支 - 使用全连接层扩展特征
        self.feat_fc = nn.Linear(fea_size, hid_size)
        
        # 全连接层将32个特征映射到1个回归值
        self.fc = nn.Linear(hid_size, 1)
        self.dropout = nn.Dropout(p=0.5)
        # 激活函数
        self.relu = nn.ReLU()
        
    def forward(self, img1, img2, feat):
        # 处理图像输入 [batch, 9, 2] -> [batch, 1, 9, 2]
        img1 = img1.unsqueeze(1)  # 添加通道维度
        
        # 图像卷积 [batch, 1, 9, 2] -> [batch, 32, 1, 2]
        img1 = self.img_conv1(img1)
        img1 = self.relu(img1)
        
        # 维度缩减 [batch, 32, 1, 2] -> [batch, 32, 2]
        img1 = img1.squeeze(2)
        
        # 处理图像输入 [batch, 9, 2] -> [batch, 1, 9, 2]
        img2 = img2.unsqueeze(1)  # 添加通道维度
        
        # 图像卷积 [batch, 1, 9, 2] -> [batch, 32, 1, 2]
        img2 = self.img_conv2(img2)
        img2 = self.relu(img2)
        
        # 维度缩减 [batch, 32, 1, 2] -> [batch, 32, 2]
        img2 = img2.squeeze(2)
           
        # 处理特征输入 [batch, 1] -> [batch, 32]
        feat = feat.repeat(self.fea_size,1).T
        
        # 特征扩展 [batch, 32] -> [batch, 32, 1]
        feat = feat.unsqueeze(2)
        
        # 拼接特征 [batch, 32, 2] + [batch, 32, 2] + [batch, 32, 1] -> [batch, 32, 5]
        combined = torch.cat([img1, img2, feat], dim=2)
        
        # 分通道卷积 [batch, 32, 5] -> [batch, 32, 1]
        conv_out = self.channel_conv(combined)
        conv_out = self.relu(conv_out)
        
        # 展平 [batch, 32, 1] -> [batch, 32]
        conv_out = conv_out.squeeze(2)
        
        # 全连接回归 [batch, fea_size] -> [batch, hid_size]
        out = self.feat_fc(conv_out)
        out = self.dropout(out)
        
        # 全连接回归 [batch, hid_size] -> [batch, 1]
        out = self.fc(out)
        
        return out

num_samples = XY.shape[0]
num_features = XY.shape[1]-1

num_val = int(num_samples * 0.2)

testNum = 0   # 20   32  22
x_test1,x_test2,x_test3,y_test = getTest(testNum)

msr5Fold = np.zeros([5,3,2]) #5fold, 3集合, 2指标
yp5Fold = np.zeros([5,910]) #5fold, n预测【修改预测样本量】

for i in range(0,5,1):  #5-fold cross validation
    indVal = num_val*i + np.array([j for j in range(num_val)])
    indTrain = np.setdiff1d(np.array([j for j in range(num_samples)]), indVal)
    
    # 划分训练集和验证集
    x_train = XY[idx[indTrain],0:18]
    x_train1 = x_train.reshape((x_train.shape[0], 9, 2), order='F')
    x_train = XY[idx[indTrain],19:-1]
    x_train2 = x_train.reshape((x_train.shape[0], 9, 2), order='F')
    x_train3 = XY[idx[indTrain],18]
    
    y_train = XY[idx[indTrain],-1]
    
    x_val = XY[idx[indVal],0:18]
    x_val1 = x_val.reshape((x_val.shape[0], 9, 2), order='F')
    x_val = XY[idx[indVal],19:-1]
    x_val2 = x_val.reshape((x_val.shape[0], 9, 2), order='F')
    x_val3 = XY[idx[indVal],18]
    y_val = XY[idx[indVal],-1]
    
    # 创建数据集和数据加载器
    train_dataset = TensorDataset(torch.from_numpy(x_train1).float(),torch.from_numpy(x_train2).float(), torch.from_numpy(x_train3).float(), torch.from_numpy(y_train).float())
    val_dataset = TensorDataset(torch.from_numpy(x_val1).float(),torch.from_numpy(x_val2).float(),torch.from_numpy(x_val3).float(), torch.from_numpy(y_val).float())
    batch_size = 32
    
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last= True)
    val_loader = DataLoader(val_dataset, batch_size=num_val)

    
    # 初始化模型、损失函数和优化器
    input_size = num_features
    hidden_size = 64
    num_heads = 2
    num_layers = 4
    output_size = 1
    model = MultiInputRegressionNet(96,32)
    # model = TransformerRegressor(input_size, hidden_size, num_heads, num_layers, output_size)
    criterion = nn.MSELoss()
    #optimizer = optim.Adam(model.parameters(), lr=0.001)
    optimizer = torch.optim.SGD(model.parameters(),lr = 0.0001)
    # 训练模型
    
    costTr=[]
    costVal=[]
    avgLoss=[]
    
    num_epochs = 20000
    for epoch in range(num_epochs):
        model.train()
        for batch_x1, batch_x2, batch_x3, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_x1, batch_x2, batch_x3)
            loss = criterion(outputs, torch.unsqueeze(batch_y, dim=1))
            loss.backward()
            optimizer.step()
        costTr.append(loss.detach().numpy())
            
        #print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}')
    
        # 在验证集上评估模型
        model.eval()
        with torch.no_grad():
            val_predictions = []
            for batch_x1, batch_x2,  batch_x3, batch_y in val_loader:
                val_predictions = model(batch_x1, batch_x2,  batch_x3)
                val_loss = criterion(val_predictions, torch.unsqueeze(batch_y, dim=1))
                
                costVal.append(val_loss.detach().numpy())
                #print(f'Val Loss: {val_loss.item()}')
        if epoch > 14000:
            if np.mean(costVal[-50:])>np.mean(costVal[-100:]):
                break;

    df = pd.DataFrame({'costTr'+str(i):costTr, 'costVal'+str(i):costVal});
    df.to_excel('loss5fold'+str(i)+'.xlsx', index=False)
     
    plt.plot(costTr,'-*r',costVal,'-*k')
    plt.ylabel('costTr_R + costVal_K')
    plt.xlabel('epoch')
    plt.show()
    
    plt.subplot(1, 3, 1)
    plt.plot(batch_y,val_predictions,'*r',[0,40],[0,40],'-b')
    #计算相对标准差
    xxx=val_predictions.detach().numpy().squeeze()
    residualsPerc = (batch_y.detach().numpy()-val_predictions.detach().numpy().squeeze())/batch_y.detach().numpy()
    residual_stdP = np.std(residualsPerc, ddof=1)
    residual_medP = np.median(residualsPerc)
    print("验证集 相对回归残差的标准差是:", residual_stdP)
    print("      相对回归残差的中值是:", residual_medP)
    msr5Fold[i,1,0]=residual_stdP
    msr5Fold[i,1,1]=residual_medP
    
    plt.title(f'res% med: {residual_medP:.3f}; std: {residual_stdP:.3f}')
    plt.ylabel('predY--val')
    plt.xlabel('realY--val')
    plt.axis('equal')
    
    
    #训练集情况
    train_loader = DataLoader(train_dataset, batch_size=indTrain.shape[0], shuffle=True, drop_last= True)
    for batch_x1,batch_x2, batch_x3,batch_yT in train_loader:
        tr_predictions = model(batch_x1,batch_x2,batch_x3)
    plt.subplot(1, 3, 2)
    plt.plot(batch_yT.detach().numpy(),tr_predictions.detach().numpy(),'*r',[0,40],[0,40],'-b')
    #计算相对标准差
    xx=tr_predictions.detach().numpy().squeeze()
    residualsPerc = (batch_yT.detach().numpy()-tr_predictions.detach().numpy().squeeze())/batch_yT.detach().numpy()
    residual_stdP = np.std(residualsPerc, ddof=1)
    residual_medP = np.median(residualsPerc)
    print("训练集 相对回归残差的标准差是:", residual_stdP)
    print("      相对回归残差的中值是:", residual_medP)
    plt.title(f'res% med: {residual_medP:.3f}; std: {residual_stdP:.3f}')
    plt.ylabel('predY--train')
    plt.xlabel('realY--train')
    plt.axis('equal')

    msr5Fold[i,0,0]=residual_stdP
    msr5Fold[i,0,1]=residual_medP
    
    #测试集情况
    # 预测下一批次实验  
    ts_predictions = model(x_test1,x_test2,x_test3)
    #计算相对标准差
    yR = y_test.detach().numpy()
    yP = ts_predictions.detach().numpy()
    num = int(yP.size/2)
    
    yR = (yR[0:num]+yR[-num:])/2
    yP = (yP[0:num]+yP[-num:])/2
    plt.subplot(1, 3, 3)
    plt.plot(yR,yP,'*y',[0,40],[0,40],'-g')
    
    residualsPerc = (yR- yP.squeeze())/yR
    residual_stdP = np.std(residualsPerc, ddof=1)
    residual_medP = np.median(residualsPerc)
    msr5Fold[i,2,0]=residual_stdP
    msr5Fold[i,2,1]=residual_medP
    print("测试集 相对回归残差的标准差是:", residual_stdP)
    print("      相对回归残差的中值是:", residual_medP)
    plt.title(f'res% med: {residual_medP:.3f}; std: {residual_stdP:.3f}')
    plt.ylabel('predY--test')
    plt.xlabel('realY--test')
    plt.axis('equal')
    plt.show()
    
    yp5Fold[i,:] = yP.squeeze()
    np.savez('matrices'+str(i)+'.npz', matrix1t=batch_yT.detach().numpy(), matrix2t=tr_predictions.detach().numpy().squeeze(),
             matrix3v=batch_y, matrix4v=val_predictions.squeeze(), matrix5m=yR,matrix6m=yP.squeeze())
    
    
    pathOut = 'D:/uProjects/AI/PJHjyy1/1.传感课题执行/PJscripts/DataOut/Model'
    torch.save(model, pathOut+str(i)+'.pth')
#model = torch.load('D:/uProjects/AI/PJHjyy1/1.传感课题执行/csvData/Bth1-4_model1.pth')


plt.plot([1,2,3,4,5],msr5Fold[:,1,0],'-r',
         [1,2,3,4,5],msr5Fold[:,1,1],'--r',
         [1,2,3,4,5],msr5Fold[:,0,0],'-g',
         [1,2,3,4,5],msr5Fold[:,0,1],'--g',
         [1,2,3,4,5],msr5Fold[:,2,0],'-b',
         [1,2,3,4,5],msr5Fold[:,2,1],'--b', 
         )

if 0:
    x_train,y_train = getTest(20)
    predictions = model(x_train)
    
    ina=82; inb=20
    ina=114; inb=32
    ina=136; inb=22
    
    yP=predictions.detach().numpy()
    
    yR=y_train.detach().numpy()
    yP2= (yP[-ina:]+yP[0:ina])/2
    yR2= (yR[0:ina]+yR[-ina:])/2
    plt.plot(yP2[-inb:],yR2[-inb:],'g*',yP2[0:(ina-inb)],yR2[0:(ina-inb)],'r*',[0,40],[0,40],'-b')

    plt.ylabel('predY')
    plt.xlabel('realY')
    plt.axis('equal')
    plt.show()

if 0:
    torch.onnx.export(model, (x_test1,x_test2,x_test3),"modelcnn1.onnx", export_params=True)
    import netron
    # 打开导出的 ONNX 模型文件
    netron.start( "modelcnn1.onnx")  #在网页上显示