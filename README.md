# :rocket: Guide Introduction :rocket:

This study explores how to effectively use Support Vector Machine (SVM) and Convolutional Neural Network (CNN) for audio recognition under small sample conditions. At the same time, this project uses Mel-frequency cepstral coefficient technology to extract effective features for feature extraction and serves as the input of the support vector machine. At the same time, deep learning technology uses convolutional neural networks (CNN) for audio classification. The waveform of audio is directly input into the convolutional network for processing.

## 1.Runtime Environment :computer:

| Package| Version | Usage |
|---|---|---|
| torch | 2.4.1 | Provide a basic framework|
| librosa | 0.10.2.post1 | Extract audio features |
| seaborn | 0.12.2 | Draw confusion matrix |
| soundfile | 0.12.1 | Parse.wav audio files |
| tqdm | 4.66.4 | Show the time for processing files and training. |
| imbalanced-learn | 0.11.0 | Balanced training set |

## 2.Core Code :key:

___CNN___: The model structure includes four convolutional layers. Each convolutional layer is followed by a batch normalization layer, ReLU activation function, and max pooling layer to extract features from the input data. The convolutional layers gradually increase the number of channels and reduce the spatial dimension of the data to capture more complex patterns. In addition, Dropout is applied after each convolutional block to prevent overfitting. After convolution and pooling operations, the feature map is globally averaged pooled into one dimension and classified through two fully connected layers. Finally, logits matching the number of categories are output to realize the conversion from input to predicted categories. This model design is suitable for audio signal classification tasks.

```python
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
```

___SVM___: The function of mfcc_extraction is used to extract MFCC (Mel-frequency cepstral coefficients) features from audio files in a specified directory, including mean, standard deviation, variance, minimum value, maximum value, and corresponding labels. It traverses all files in the given audio directory audio_dir. For each audio file, after loading the audio data using the librosa.load function, its MFCC features are calculated. For each audio file, the five statistical features (mean, standard deviation, variance, minimum value, and maximum value) obtained by calculation are respectively collected into corresponding lists. At the same time, the file name is added to the label list as a label. Eventually, this function returns a list containing all extracted features, which can be used for subsequent audio classification or analysis tasks. It should be noted that here the label is directly replaced by the file name instead of being obtained through a predefined label dictionary.

```python
# 设置mfcc的参数
num_mfcc_features=25

def mfcc_extraction(audio_dir,labels):
    '''提取mfcc的mean,std,var,min,max,label特征'''
    mean_features=[]
    std_features=[]
    var_features=[]
    min_features=[]
    max_features=[]
    label_features=[]
    for file in os.listdir(audio_dir):
        file_path=os.path.join(audio_dir,file)
        # label= labels_dict[file]
        label=file
        for file_name in tqdm(os.listdir(file_path),desc='----'+file):
            audio_path=os.path.join(file_path,file_name)
            waveform,sr=librosa.load(audio_path)
            mean_features.append(np.mean(librosa.feature.mfcc(y=waveform,sr=sr,n_mfcc=num_mfcc_features).T,axis=0))
            std_features.append(np.std(librosa.feature.mfcc(y=waveform,sr=sr,n_mfcc=num_mfcc_features).T,axis=0))
            var_features.append(np.var(librosa.feature.mfcc(y=waveform,sr=sr,n_mfcc=num_mfcc_features).T,axis=0))
            min_features.append(np.min(librosa.feature.mfcc(y=waveform,sr=sr,n_mfcc=num_mfcc_features).T,axis=0))
            max_features.append(np.max(librosa.feature.mfcc(y=waveform,sr=sr,n_mfcc=num_mfcc_features).T,axis=0))
            label_features.append(label)
    return mean_features,std_features,var_features,min_features,max_features,label_features

mfcc_features_labels=mfcc_extraction(Audio_Data_Path,labels)
print(f'----特征提取完毕')
```

## 3.Project Structure :fire:

```bash
├── CNN
│   ├── audio_cnn_classifier.py
│   ├── audio_cnn_classifier.ipynb
|   └── audio_cnn_model.py
├── SVM
│   ├── audio_svm_classifier.py
│   └── audio_svm_classifier.ipynb
├── README.md
└── dataset
    ├── train_set
    └── test_set
```
