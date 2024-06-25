# DRNet-fss
This is the implementation of our paper DRNet: Disentanglement and Recombination Network for few-shot semantic segmentation that has been submitted to IEEE Transactions on Circuits and Systems for Video Technology (TCSVT).
# Get Started
```
Python 3.7 +
torch 1.7.0
torchvision 0.8.0
scipy 1.7.3
tqdm 4.64.0
```

# Please download the following datasets: 
```
PASCAL-5i is based on the PASCAL VOC 2012 and SBD where the val images should be excluded from the list of training samples.
Images are available at: http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
annotations: https://drive.google.com/file/d/1ikrDlsai5QSf2GiSUR3f8PZUzyTubcuF/view?usp=sharing
This work is built on:
OSLSM: https://github.com/lzzcd001/OSLSM
PANet: https://github.com/kaixin96/PANet
Many thanks to their greak work!
```

# Citation
If you find this project useful, please consider citing:
```
@inproceedings{wang2019panet,

  title={Panet: Few-shot image semantic segmentation with prototype alignment},
  
  author={Wang, Kaixin and Liew, Jun Hao and Zou, Yingtian and Zhou, Daquan and Feng, Jiashi},
  
  booktitle={proceedings of the IEEE/CVF international conference on computer vision},
  
  pages={9197--9206},
  
  year={2019}
}

@article{chang2022mgnet,

  title={MGNet: Mutual-guidance network for few-shot semantic segmentation},
  
  author={Chang, Zhaobin and Lu, Yonggang and Wang, Xiangwen and Ran, Xingcheng},
  
  journal={Engineering Applications of Artificial Intelligence},
  
  volume={116},
  
  pages={105431},
  
  year={2022},
  
  publisher={Elsevier}
}
```
