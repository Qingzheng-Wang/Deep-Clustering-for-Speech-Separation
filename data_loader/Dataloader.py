import sys
sys.path.append('../')
from torch.nn.utils.rnn import pack_sequence, pad_sequence
from data_loader import AudioData
import torch
from torch.utils.data import Dataset, DataLoader
from utils import util
import pickle
import numpy as np

class dataset(Dataset):
    def __init__(self, mix_reader, target_readers):
        super(dataset).__init__()
        self.mix_reader = mix_reader
        self.target_readers = target_readers
        self.keys = mix_reader.wave_keys

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        key = self.keys[index]
        if key not in self.keys:
            raise ValueError
        # 返回的是list，list中的每个元素是一个stft结果
        return self.mix_reader[key], [target[key] for target in self.target_readers]

# dataloader的作用是将dataset中的数据按照batch_size进行打包，然后返回
class dataloader(object):
    def __init__(self, dataset, batch_size=40, shuffle=True, num_workers=16, cmvn_file='../cmvn.ark'):
        super(dataloader).__init__()
        self.dataload = DataLoader(
            dataset, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle, collate_fn=self.collate)
        self.cmvn = pickle.load(open(cmvn_file, 'rb'))
    
    def __len__(self):
        return len(self.dataload)

    def transform(self, mix_wave, target_waves):
        frames = mix_wave.shape[0]
        non_slient = util.compute_non_silent(mix_wave)
        mix_wave = util.apply_cmvn(mix_wave, self.cmvn) # apply_cmvn是对stft结果进行归一化
        target_waves = np.argmax(np.array(target_waves), axis=0) # target_waves大小为[frames, spk_num]，
        # argmax是取最大值的索引，因为target_waves是一个one-hot向量，取最大值的索引就是对应的说话人的索引
        return {
            "frames": frames,
            "non_slient": torch.tensor(non_slient, dtype=torch.float32),
            "mix_wave": torch.tensor(mix_wave, dtype=torch.float32),
            "target_waves": torch.tensor(target_waves, dtype=torch.int64)
        }
    
    def collate(self, batchs):
        # batchs是一个list，list中的每个元素是一个tuple，tuple中的第一个元素是mix_wave，第二个元素是target_waves
        trans = sorted([self.transform(mix_wave, target_waves) # 按照frames的大小进行排序，frames越大的越靠前
                        for mix_wave, target_waves in batchs], key=lambda x: x["frames"], reverse=True)
        mix_wave = pack_sequence([t['mix_wave'] for t in trans])
        target_waves = pad_sequence([t['target_waves'] # pad_sequence的作用是将target_waves的长度统一，填充的值为0
                                     for t in trans], batch_first=True)
        non_slient = pad_sequence([t['non_slient']
                                   for t in trans], batch_first=True)
        return mix_wave, target_waves, non_slient

    def __iter__(self):
        for b in self.dataload:
            yield b


if __name__ == "__main__":
    mix_reader = AudioData(
        "/home/likai/Desktop/create_scp/tr_mix.scp", is_mag=True, is_log=True)
    target_readers = [AudioData("/home/likai/Desktop/create_scp/tr_s1.scp", is_mag=True, is_log=True),
                      AudioData("/home/likai/Desktop/create_scp/tr_s2.scp", is_mag=True, is_log=True)]
    dataset = dataset(mix_reader, target_readers)
    dataloader = dataloader(dataset)
    print(len(dataloader))
