#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time    : 2021/1/22 01:30
# @Author  : glan~
# @FileName: trimodal_attention.py
# @annotation: 模态融合注意力
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from math import sqrt


def bi_modal_attention(x, y):
    '''
        .  代表 dot product
        *  代表 elemwise multiplication
        {} 代表 concatenation

        m1 = x . transpose(y) ||  m2 = y . transpose(x)
        n1 = softmax(m1)      ||  n2 = softmax(m2)
        o1 = n1 . y           ||  o2 = m2 . x
        a1 = o1 * x           ||  a2 = o2 * y

        return {a1, a2}
    '''
    # x 1 512 y 1 512
    # m1 = torch.matmul(x, y.transpose(-1,-2)) # 1 1
    # n1 = F.softmax(m1, dim=1)
    # o1 = torch.matmul(n1, y)
    # a1 = torch.mul(o1, x)  # 相同位置上的元素对应各自相乘
    # m2 = torch.matmul(y, x.transpose(-1,-2))
    # n2 = F.softmax(m2, dim=1)
    # o2 = torch.matmul(n2, x)
    # a2 = torch.mul(o2, y)

    # return torch.cat([a1, a2], dim=1)
    m1 = torch.matmul(x, y.transpose(-1,-2)) # 1 1
    n1 = F.softmax(m1, dim=1)
    o1 = torch.matmul(n1, y)
    a1 = torch.mul(o1, x)  # 相同位置上的元素对应各自相乘

    return a1


# 多模态自注意力融合 #
def multi_sa(video, tri_d, audio):
    vv_att = my_self_attention(video)
    tt_att = my_self_attention(tri_d)
    aa_att = my_self_attention(audio)

    return torch.cat([aa_att, vv_att, tt_att, video, tri_d, audio], dim=1)


# 多模态注意力融合 #
def multi_at(video, tri_d, audio):
    vt_att = bi_modal_attention(video, tri_d)
    av_att = bi_modal_attention(audio, video)
    ta_att = bi_modal_attention(tri_d, audio)

    return torch.cat([vt_att, av_att, ta_att, video, audio, tri_d], dim=1)


def self_attention(query, key, value, mask=None, dropout=None):

    d_k = query.size(-1)
    # (nbatch, h, seq_len, d_k) @ (nbatch, h, d_k, seq_len) => (nbatch, h, seq_len, seq_len)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout:
        p_attn = dropout(p_attn)
    # (nbatch, h, seq_len, seq_len) * (nbatch, h, seq_len, d_k) = > (nbatch, h, seq_len, d_k)
    return torch.matmul(p_attn, value), p_attn


def my_self_attention(x):
    '''
        .  点积 dot
        *  对应元素相乘 elemwise

        m = x . transpose(x)
        n = softmax(m)
        o = n . x
        a = o * x

        return a

    '''

    m = torch.matmul(x, x.transpose(-1,-2))

    n = F.softmax(m, dim=1)

    o = torch.matmul(n, x)

    a = torch.mul(o, x)

    return a


class Self_Attention(nn.Module):
    '''
     举一下：sa = Self_Attention(input.shape[-1], dim_k, dim_v)
     如果想输入输出唯独不变， dim_v 为input.shape[-1]
    '''
    # input : batch_size * seq_len * input_dim
    # q : batch_size * input_dim * dim_k
    # k : batch_size * input_dim * dim_k
    # v : batch_size * input_dim * dim_v
    def __init__(self,input_dim,dim_k,dim_v):
        super(Self_Attention,self).__init__()
        self.q = nn.Linear(input_dim,dim_k)
        self.k = nn.Linear(input_dim,dim_k)
        self.v = nn.Linear(input_dim,dim_v)
        self._norm_fact = 1 / sqrt(dim_k)
        
    
    def forward(self,x):
        Q = self.q(x) 
        print(Q.shape)
        K = self.k(x) 
        print(K.shape)
        V = self.v(x) 
        print(V.shape)
         
        atten = nn.Softmax(dim=-1)(torch.bmm(Q,K.permute(0,2,1))) * self._norm_fact 
        
        output = torch.bmm(atten,V) 
        return output
