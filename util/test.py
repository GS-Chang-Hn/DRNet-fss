# -*- coding: utf-8 -*-
# @Time : 2021/10/24 21:06
# @Author : Z.chang
# @FileName: test.py
# @Software: PyCharm
# @Description：
import torch
# a = torch.ones((2,3, 3))
a = torch.tensor([[[2, 1, 3],
         [1, 1, 1],
         [1, 0, 1]],

        [[1, 6, 1],
         [1, 1, 1],
         [1, 1, 1]]])
print(a)
print(a.shape)
#
a1 = torch.sum(a, dim=(1, 2))
print(a1)
print(a1.shape)


