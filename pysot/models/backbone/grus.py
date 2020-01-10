# -*- coding: utf-8 -*-
#! /usr/bin/python3
"""
Created on Thu Sep 21 16:15:53 2017

@author: cx
"""

import torch
import torch.nn as nn


from pysot.models.backbone.convgru import ConvGRU
from tensorboardX import SummaryWriter
import torchvision.utils as vutils

from pysot.core.config import cfg


ALEXNET_GRU_PARAMS = \
    {

        "gru_sets":
            {
                "size": [256, 6, 6],  # 输入gru特征维度
                "hidden_grus": 2,  # 级联的gru个数
                "hidden_dims": [256,256],  # 每个级联的gru隐藏层维度
                "hidden_kernels": [ (3, 3), (3, 3)]  # 每个级联的gru隐藏层卷结合的大小
            }

    }






class GRU_Model(nn.Module):


    def __init__(self,seq_in_len=3,seq_out_len=1,GRU_PARAMS=ALEXNET_GRU_PARAMS,dtype =torch.cuda.FloatTensor):
        super(GRU_Model, self).__init__()

        ###declare some parameters that might be used


        self.seq_in_len = seq_in_len
        self.seq_out_len =seq_out_len
        self.seq_len = seq_in_len+seq_out_len
        gru_set = GRU_PARAMS["gru_sets"]

        self.num_layers = gru_set["hidden_grus"]
        c, h, w = gru_set["size"]
        self.input_channels = c
        self.input_height = h
        self.input_width = w
        self.hidden_dims = gru_set["hidden_dims"]#GRU模块级联个数
        self.gru_in_dim = self.hidden_dims[-1]  # gru输入通道数，和输出通道数保持相等，这样方便将GRU最终的输出结果重新喂给输入，实现多个结果的递推
        self.kernel_size = gru_set["hidden_kernels"]  # gru模块指定不同的卷积核大小
        self.dtype =dtype
        self.return_all_layes= False                    #返回GRU模块组所有gru的隐含状态以及最终输出状态

        # 通道调整#gru的输入通道设置
        self.conv = nn.Conv2d(
            in_channels=self.input_channels,
            out_channels=self.gru_in_dim,
            kernel_size=1,
            stride=1,
            padding=0)

        self.conv_gru = ConvGRU(input_size=( self.input_height, self.input_width),
                        input_dim= self.gru_in_dim,
                        hidden_dim=self.hidden_dims,
                        kernel_size=self.kernel_size,
                        num_layers=self.num_layers,
                        dtype=self.dtype,
                        batch_first=True,
                        bias=True,
                        return_all_layers= self.return_all_layes)


    def forward(self, X):

        # 通道调整
        b, t, c, h, w = X.shape
        gru_in = self.conv(X.reshape(b * t, c, h, w)).reshape(b, t, self.gru_in_dim, h, w)
        hidden_state, last_state = self.conv_gru(gru_in);  # 最后一个gru模块的隐含层，以及它的最后状态，都是list,list中的元素大小分别为【n,t,c,h,w】,[n,c,h,w]




        if self.return_all_layes:
            hidden_state =hidden_state[-1]   #只取最后一个gru模块的输出
            last_state =last_state[-1]
        else:
            hidden_state = hidden_state[0]
            last_state = last_state[0]


        output = [None] *  self.seq_out_len

        _last_state = torch.unsqueeze(last_state,dim=1)  #从n c h w -->n 1 c h w
        output[0]=_last_state;                           #递推的第一个输入状态

        # 承接前面的操作，只不过前面认为序列前5 帧为训练数据，后10帧为输出数据，第6帧要利用第5帧，有一个衔接过程，第7帧要利用第6帧的输入
        for i in range(1, self.seq_out_len):

            _hidden_state, _last_state = self.conv_gru(output[i-1]);  # 最后一个gru模块的隐含层，以及它的最后状态

            if self.return_all_layes:
                _hidden_state = _hidden_state[-1]  # 只取最后一个gru模块的输出
                _last_state = _last_state[-1]
            else:
                _hidden_state = _hidden_state[0]
                _last_state = _last_state[0]

            _last_state = torch.unsqueeze(_last_state, dim=1)  # 从n c h w -->n 1 c h w
            output[i] = _last_state;  # 递推的第一个输入状态

        if(self.seq_out_len<=1):
            return output[0]         #结果要以tensor形式输出为【n,1,c,h,w】
        else:
            tensor = torch.cat(output, dim=1)   ##结果要以tensor形式输出为【n,1,c,h,w】
            return tensor













if __name__=="__main__":

    import os
    # set CUDA device
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # detect if CUDA is available or not
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        dtype = torch.cuda.FloatTensor  # computation in GPU
    else:
        dtype = torch.FloatTensor

    batch_size = 4
    seq_in =5
    seq_out=2
    time_steps = seq_in+seq_out

    model =GRU_Model(seq_in_len=seq_in,seq_out_len=seq_out,GRU_PARAMS=ALEXNET_GRU_PARAMS,dtype=dtype)
    channels =model.input_channels
    height   =model.input_height
    width    =model.input_width


    model = model.to(device)
    print(model)
    for name, param in model.named_parameters():
        print(name)


    input = torch.rand(batch_size, time_steps, channels, height, width)  # (b,t,c,h,w)
    input = input.to(device)
    output = model(input)

    print(input.shape)
    print(output.shape)
