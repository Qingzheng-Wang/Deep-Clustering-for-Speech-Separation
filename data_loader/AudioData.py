import sys
sys.path.append('../')
import utils.util as ut
from utils.stft_istft import STFT


class AudioData(object):
    """
        Loading wave file
        scp_file: the scp file path
        other kwargs is stft's kwargs
        is_mag: if True, abs(stft)
    """

    def __init__(self, scp_file, window='hann', nfft=256, window_length=256, hop_length=64, center=False, is_mag=True, is_log=True):
        self.wave = ut.read_scp(scp_file) # scp是一个字典，key是文件名，value是文件路径
        self.wave_keys = [key for key in self.wave.keys()]
        self.STFT = STFT(window=window, nfft=nfft,
                         window_length=window_length, hop_length=hop_length, center=center)
        self.is_mag = is_mag
        self.is_log = is_log

    def __len__(self): # 重载len()函数
        return len(self.wave_keys)

    def stft(self, wave_path):
        samp = ut.read_wav(wave_path)
        return self.STFT.stft(samp, self.is_mag, self.is_log)

    def __iter__(self): # 重载迭代器
        for key in self.wave_keys:
            yield self.stft(self.wave[key]) # yield是一个生成器，每次返回一个stft的结果，这样就可以用for循环来遍历所有的stft结果

    def __getitem__(self, key): # 重载[]操作符
        if key not in self.wave_keys:
            raise ValueError
        return self.stft(self.wave[key])


if __name__ == "__main__":
    ad = AudioData("/home/likai/data1/create_scp/cv_mix.scp", is_mag=True,is_log=True)
    audio = ad['011a010d_0.54422_20do010c_-0.54422.wav']
    print(audio.shape)
    print(ut.compute_non_silent(audio))
