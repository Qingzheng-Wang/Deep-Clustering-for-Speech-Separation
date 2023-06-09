import json
from pathlib import Path
from collections import OrderedDict
import librosa
import os
import sys
sys.path.append('../')
import torch
import numpy as np
from data_loader import AudioData
from tqdm import tqdm
import pickle
def ensure_dir(dirname):
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)


def read_json(fname):
    fname = Path(fname)
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)


def write_json(content, fname):
    fname = Path(fname)
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)


def read_wav(file_path, sr=8000, is_return_sr=False):
    '''
       file path: wav file path
       is_return_sr: if true, return sr number
    '''
    samp, sr = librosa.load(file_path, sr=sr)
    if is_return_sr:
        return samp, sr
    return samp


def write_wav(file_path, filename, samp, sr=8000):
    '''
       file_path: path of file
       filename: sound of Spectrogram
       sr: sample rate
    '''
    os.makedirs(file_path, exist_ok=True)
    filepath = os.path.join(file_path, filename)
    librosa.output.write_wav(filepath, samp, sr)


def read_scp(scp_file):
    '''
      read the scp file
    '''
    files = open(scp_file, 'r')
    lines = files.readlines()
    wave = {}
    for line in lines:
        line = line.split()
        if line[0] in wave.keys():
            raise ValueError
        wave[line[0]] = line[1]
    return wave


def compute_non_silent(samp, threshold=40, is_linear=True):
    """
       compute the non-silent mask
       samp: Spectrogram
       threshold: threshold(dB)
       is_linear: non-linear -> linear
    """
    # to linear first if needed
    if is_linear:
        samp = np.exp(samp)
    # to dB
    spectra_db = 20 * np.log10(samp)
    max_magnitude_db = np.max(spectra_db)
    threshold = 10**((max_magnitude_db - threshold) / 20)
    non_silent = np.array(samp > threshold, dtype=np.float32)  # [T, F] 非静音的地方为1，静音的地方为0
    return non_silent

def compute_cmvn(scp_file,save_file,**kwargs):
    """
    cvmn是一种特征归一化的方法，对于每个特征维度，减去均值，除以方差
       Feature normalization
       scp_file: the file path of scp
       save_file: the cmvn result file .ark
       **kwargs: the configure setting of file

       return
            mean: [frequency-bins]
            var:  [frequency-bins]
    """
    wave_reader = AudioData(scp_file,**kwargs)
    tf_bin = int(kwargs['nfft']/2+1)
    mean = np.zeros(tf_bin)
    std = np.zeros(tf_bin)
    num_frames = 0
    # 求整个数据集的均值和方差
    for spectrogram in tqdm(wave_reader): # tqdm is a progress bar
        num_frames += spectrogram.shape[0] # the first dimension of spectrogram is the number of frames
        mean += np.sum(spectrogram, 0)
        std += np.sum(spectrogram**2, 0)
    mean = mean / num_frames
    std = np.sqrt(std / num_frames - mean**2) # 方差是平方的期望减去期望的平方
    with open(save_file, "wb") as f:
        cmvn_dict = {"mean": mean, "std": std}
        pickle.dump(cmvn_dict, f) # 将cmvn_dict保存到f中,
        # pickle是python中的序列化模块，将对象转化为字节流，方便存储和传输, dump是将对象序列化后存储到文件中
    print("Totally processed {} frames".format(num_frames))
    print("Global mean: {}".format(mean))
    print("Global std: {}".format(std))
    
def apply_cmvn(samp,cmvn_dict):
    """

      apply cmvn for Spectrogram
      samp: stft Spectrogram
      cmvn: the path of cmvn(python util.py)

      calculate: x = (x-mean)/std
    """
    return (samp-cmvn_dict['mean'])/cmvn_dict['std']

if __name__ == "__main__":
    kwargs = {'window':'hann', 'nfft':256, 'window_length':256, 'hop_length':64, 'center':False, 'is_mag':True, 'is_log':True}
    compute_cmvn("/home/likai/data1/create_scp/tr_mix.scp",'../cmvn.ark',**kwargs)
    #file = pickle.load(open('cmvn.ark','rb'))
    #print(file)
    #samp = read_wav('../1.wav')
    #print(compute_non_silent(samp))
    