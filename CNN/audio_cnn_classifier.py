import os
import torch
import torch.nn as nn
import torchaudio
import numpy as np
import seaborn as sns
from torchaudio.functional import resample
from torch.utils.data import Dataset,DataLoader
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn.metrics import classification_report


"""设置全局参数"""
Sample_Rate = 48000
Data_Path='dataset/train_set'
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
audio_dict={'起飞':0,'降落':1,'前进':2,'后退':3,'升高':4}


"""读取工具函数"""
def audio_processing(waveform,sr):
    # 上采样到48000Hz
    up_waveform=resample(waveform,orig_freq=sr,new_freq=Sample_Rate)
    return up_waveform

def count_parameter(model):
    # 获取模型的总参数量
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def confusion_matrix_plot(y_true,y_pred):
    # 绘制混淆矩阵
    cm=confusion_matrix(y_true,y_pred)
    plt.figure(figsize=(10,8))
    sns.heatmap(cm,annot=True,fmt='d',annot_kws={'size':28})
    plt.xlabel('Predicted',fontsize=20)
    plt.ylabel('Actual Truth',fontsize=20)
    plt.title('CNN Confusion Matrix',fontsize=20)
    plt.show()


"""定义迭代数据加载器"""
class AudioDataset(Dataset):
    def __init__(self,root,transform=None):
        self.audio_paths=[]
        self.labels=[]
        self.transform=transform
        for class_file in os.listdir(root):
            class_path=os.path.join(root,class_file)
            for audio_file in os.listdir(class_path):
                audio_path=os.path.join(class_path,audio_file)

                self.audio_paths.append(audio_path)
                self.labels.append(audio_dict[class_file])

    def __len__(self):
        return len(self.audio_paths)
    
    def __getitem__(self,idx):
        waveform,sr=torchaudio.load(self.audio_paths[idx],channels_first=True)
        label=self.labels[idx]
        if self.transform:
            waveform=self.transform(waveform,sr)
        label=torch.tensor(label)
        return waveform,label
    

"""定义卷积神经网络"""
class MFCC_CNN(nn.Module):
    def __init__(self,in_channels=1,out_channels=5,stride=16,n_channel=32,dropout_rate=0.5):
        super(MFCC_CNN,self).__init__()
        # 卷积层
        self.conv1=nn.Conv1d(in_channels,n_channel,kernel_size=80,stride=stride)
        self.bn1=nn.BatchNorm1d(n_channel)
        self.pool1=nn.MaxPool1d(kernel_size=4)

        self.conv2=nn.Conv1d(n_channel,n_channel,kernel_size=3)
        self.bn2=nn.BatchNorm1d(n_channel)
        self.pool2=nn.MaxPool1d(kernel_size=4)

        self.conv3=nn.Conv1d(n_channel,n_channel*2,kernel_size=3) 
        self.bn3=nn.BatchNorm1d(n_channel*2)
        self.pool3=nn.MaxPool1d(kernel_size=4)

        self.conv4=nn.Conv1d(n_channel*2,n_channel*2,kernel_size=3)
        self.bn4=nn.BatchNorm1d(n_channel*2)
        self.pool4=nn.MaxPool1d(kernel_size=4)

        # 全连接层
        self.dropout=nn.Dropout(dropout_rate)
        self.fc1=nn.Linear(n_channel*2,n_channel*4)
        self.fc2=nn.Linear(n_channel*4,out_channels)
        # self.fc=nn.Linear(n_channel*2,out_channels)

    def forward(self,x):
        x=self.conv1(x)
        x=F.relu(self.bn1(x))
        x=self.pool1(x)
        x=self.dropout(x)

        x=self.conv2(x)
        x=F.relu(self.bn2(x))
        x=self.pool2(x)
        x=self.dropout(x)

        x=self.conv3(x)
        x=F.relu(self.bn3(x))
        x=self.pool3(x)
        x=self.dropout(x)

        x=self.conv4(x)
        x=F.relu(self.bn4(x))
        x=self.pool4(x)
        x=self.dropout(x)
        x=F.avg_pool1d(x,x.shape[-1])
        x=x.permute(0,2,1)
        # logits=self.fc(x).squeeze()
        x=F.relu(self.fc1(x))
        x=self.dropout(x)
        logits=self.fc2(x).squeeze()
        return logits
    

"""train()函数test()函数的处理"""
def train(model,train_loader,optimizer,device,criterion,epochs=50):
    model.train().to(device)
    for epoch in range(epochs):
        running_loss=0.0
        for i,(input,label) in enumerate(train_loader):
            input,label=input.to(device),label.to(device)
            # print(f'----input shape: {input.shape},\tlabel shape: {label.shape}')
            output=model(input)
            # print(f'----output shape: {output.shape},\tlabel shape: {label.shape}')
            loss=criterion(output,label)
            running_loss+=loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if (epoch+1)%10==0 and epoch!=0:
            print(f'----Epoch: {epoch+1}/{epochs}  \tLoss: {running_loss/len(train_loader): .4f}----')
    # 保存模型
    torch.save(model,'CNN/audio_cnn_classifier.pth')
    print('----模型保存成功！')

def test(model,test_loader,device):
    model.eval().to(device)
    correct,total=0,0
    all_labels,all_predictions=[],[]
    with torch.no_grad():
        for i,(input,label) in enumerate(test_loader):
            input,label=input.to(device),label.to(device)
            output=model(input)

            _,predicted=torch.max(output.data,1)
            total+=label.size(0)
            correct+=(predicted==label).sum().item()
            all_labels.extend(label.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
        # accuracy=100*correct/total # 计算准确率
        confusion_matrix_plot(all_labels,all_predictions) # 绘制混淆矩阵
        print(f'----Test Accuracy: {100*correct/total:.4f}%----')
        print(f'----accuracy: {100*accuracy_score(all_labels,all_predictions):.4f}%')
        print(f'----classification report: \n{classification_report(all_labels,all_predictions)}')

            
"""主程序"""
if __name__=='__main__':
    trainset_path='dataset/train_set'
    testset_path='dataset/test_set'
    trainset=AudioDataset(trainset_path,transform=audio_processing)
    testset=AudioDataset(testset_path,transform=audio_processing)
    train_loader=DataLoader(trainset,batch_size=64,shuffle=True,drop_last=True)
    test_loader=DataLoader(testset,batch_size=8,shuffle=True,drop_last=True)
    # 创建model实例
    model=MFCC_CNN()
    optimizer=optim.AdamW(model.parameters(),lr=0.001,weight_decay=0.01)
    criterion=nn.CrossEntropyLoss()
    # 训练和测试模型
    # train(model,train_loader,optimizer,device,criterion,epochs=800)
    my_model=torch.load('CNN/audio_cnn_classifier.pth')
    test(my_model,test_loader,device)