import torch
import numpy as np
import torch.nn as nn
import math
class cross_att(nn.Module):
    '''
     举一下：sa = Self_Attention(input.shape[-1], dim_k, dim_v)
     如果想输入输出维度不变， dim_v 为input.shape[-1]
    '''

    def __init__(self, dim_k):
        super(cross_att, self).__init__()

        self.conv1 = nn.Conv2d(512, 512, kernel_size=1, bias=False)
        self.conv2 = nn.Conv2d(512, 512, kernel_size=1, bias=False)
        self.conv3 = nn.Conv2d(512, 512, kernel_size=1, bias=False)
        self._norm_fact = 1 / math.sqrt(dim_k)

    def forward(self, key, query, value):
        res_out = key.squeeze()
        Q = query.squeeze()
        K = res_out
        V = value
        atten = nn.Softmax(dim=-1)(torch.bmm(Q, K.permute(0, 2, 1))) * self._norm_fact
        output = torch.bmm(atten, V) + res_out
        return output
