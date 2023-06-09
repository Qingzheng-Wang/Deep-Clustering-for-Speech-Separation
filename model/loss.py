import sys
sys.path.append('../') # 把上一级目录加入到环境变量中
import torch


class Loss(object):
    def __init__(self, mix_wave, target_waves, non_slient, num_spks):
        super(Loss).__init__()
        self.mix_wave = mix_wave
        self.target_waves = target_waves
        self.non_slient = non_slient
        self.num_spks = num_spks # number of speakers
        self.device = torch.device('cuda:0')

    def loss(self):
        """
           mix_wave: B x TF x D
           target_waves: B x T x F
           non_slient: B x T x F
        """
        B, T, F = self.non_slient.shape
        
        # B x TF x spks, target_embs is th Y in the paper
        target_embs = torch.zeros([B, T*F, self.num_spks], device=self.device)

        # B x TF x spks 把目标声源的位置置为1, view的作用是reshape
        target_embs.scatter_(2, self.target_waves.view(B, T*F, 1), 1)
        
        # B x TF x 1
        self.non_slient = self.non_slient.view(B, T*F, 1)

        # B x TF x D, 乘法的作用是把非静音的位置的值保留下来，静音的位置的值置为0，D和1维度不同，但是会自动broadcast
        # 自动broadcast是指，如果两个tensor的维度不同，但是维度的值相同或者其中一个tensor的维度为1，那么会自动扩展为相同的维度
        # 也就是说结果mix_wave的维度为B x TF x D，但是只有非静音的位置的值是有效的，静音的位置的值是0
        # 非静音的位置D维每一个值都是1吗？不是，因为mix_wave是经过DPCL网络的输出，所以D维的值是不同的
        # 只是说，乘法的意思是保留非静音位置原来的值，静音位置的值置为0
        self.mix_wave = self.mix_wave * self.non_slient
        self.target_waves = target_embs * self.non_slient

        # B x D x TF bmm的作用是矩阵相乘，transpose的作用是转置，p=2是求2范数,2范数的定义是矩阵的所有元素的平方和的平方根
        vt_v = torch.norm(torch.bmm(torch.transpose(self.mix_wave,1,2), self.mix_wave), p=2)**2
        vt_y = torch.norm(torch.bmm(torch.transpose(self.mix_wave,1,2), self.target_waves), p=2)**2
        yt_y = torch.norm(torch.bmm(torch.transpose(self.target_waves,1,2), self.target_waves), p=2)**2
        
        return (vt_v-2*vt_y+yt_y)/torch.sum(self.non_slient)
