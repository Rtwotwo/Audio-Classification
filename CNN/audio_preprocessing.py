import os
import numpy as np
import torch
import torchaudio
from tqdm import tqdm
import torchaudio.functional as F


"""初始化全局参数"""
trainset_path='./dataset/train_set'
testset_path='./dataset/test_set'
Sample_Rate=48000
Target_Length=2*Sample_Rate


"""定义音频预处理函数工具包"""
def audio_resampling(waveform,sr):
    # 统一音频采样率为48000Hz
    return F.resample(waveform,orig_freq=sr,new_freq=Sample_Rate)

def audio_speed_change(waveform,sr,factor=1.2):
    # 调整音频速率Speed
    transform_speed=torchaudio.transforms.Speed(factor=factor,orig_freq=sr)
    transform_speed_waveform=transform_speed(waveform)
    return transform_speed_waveform[0],sr

def audio_noise_change(waveform,sr,noise_level=0.01):
    # 1.添加随机噪声
    noise=torch.randn_like(waveform)*noise_level
    return waveform+noise,sr

def audio_volume_change(waveform,sr,factor=10.0):
    # 2.调整音频音量
    adjusted_waveform = waveform * factor
    return adjusted_waveform, sr

def audio_time_masking(waveform,sr,time_mask_param=30):
    # 3.时间掩码
    spec=torchaudio.transforms.Spectrogram()(waveform)
    time_mask=torchaudio.transforms.TimeMasking(time_mask_param=time_mask_param)
    masked_spec=time_mask(spec)
    masked_waveform=torchaudio.transforms.GriffinLim()(masked_spec)
    return masked_waveform,sr

def audio_freq_masking(waveform,sr,freq_mask_param=30):
    # 4.频域掩码
    spec=torchaudio.transforms.Spectrogram()(waveform)
    freq_mask=torchaudio.transforms.FrequencyMasking(freq_mask_param=freq_mask_param)
    masked_spec=freq_mask(spec)
    masked_waveform=torchaudio.transforms.GriffinLim()(masked_spec)
    return masked_waveform,sr

def fixed_audio_length(waveform,target_length=Target_Length):
    # 固定音频长度为Target_Length
    if waveform.shape[-1]>target_length:
        waveform=waveform[:,:target_length]
    elif waveform.shape[-1]<target_length:
        padding=torch.zeros((1,target_length-waveform.shape[-1]))
        waveform=torch.cat((waveform,padding),dim=-1)
    return waveform

def save_audio(save_path,waveform, sr):
    # 使用 torchaudio.save 保存音频文件
    torchaudio.save(save_path, waveform, sr)

def audio_preprocessing(path,method):
    # 音频预处理函数
    for cls_file in os.listdir(path):
        cls_path=os.path.join(path,cls_file)
        for audio_file in os.listdir(cls_path):
            audio_path=os.path.join(cls_path,audio_file)
            waveform,sr=torchaudio.load(audio_path)
            # 音频数据增广
            if method=='1' and audio_file[0] not in ['n','v','t','f']:
                noise_save_path=os.path.join(cls_path,'n'+audio_file)
                waveform,sr=audio_noise_change(waveform,sr,noise_level=0.01)
            elif method=='2' and audio_file[0] not in ['n','v','t','f']:
                noise_save_path=os.path.join(cls_path,'v'+audio_file)
                waveform,sr=audio_volume_change(waveform,sr,factor=10.0)
            elif method=='3' and audio_file[0] not in ['n','v','t','f']:
                noise_save_path=os.path.join(cls_path,'t'+audio_file)
                waveform,sr=audio_time_masking(waveform,sr,time_mask_param=30)
            elif method=='4' and audio_file[0] not in ['n','v','t','f']:
                noise_save_path=os.path.join(cls_path,'f'+audio_file)
                waveform,sr=audio_freq_masking(waveform,sr,freq_mask_param=30)
            elif method=='0':
                noise_save_path=os.path.join(cls_path,audio_file)
            fixed_waveform=fixed_audio_length(waveform) # 统一音频长度
            save_audio(noise_save_path,fixed_waveform,sr)


"""主程序"""
if __name__=='__main__':
    for method in tqdm(['0','1','2','3','4'],desc='audio_processing'):
        audio_preprocessing(trainset_path,method)
        audio_preprocessing(testset_path,method)