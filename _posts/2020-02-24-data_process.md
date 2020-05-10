---
title: data_preprocess
date: 2020_03_14
categories: Deep-Learning
tags:
- data process
---

## Data Preprocess
　　模型在进行训练之前，首先需要对输入的数据进行预处理步骤，目的是使输入的数据分布在[-1, 1]之间，帮助模型能够更好地收敛。之前没有对这部分的内容进行过仔细的深入，通过这篇总结对这部分的内容进行一下梳理。<br>
```python
import numpy as np
from PIL import Image		# 好像用PIL库的更多一些，不过相互之间都可以进行转换
import torch
import  torchvision.transform as transform

_imagenet_stats = {
    'mean':[0.485, 0.456, 0.406],
    'std':[0.229, 0.224, 0.225]
}							# 由imagenet抽样得到的处理参数，在三个通道上分别进行标准化的操作，(image[channel] - mean[channel]) / std[channel]
def preprocess():
    transpose_list = [
        transform.ToTensor(),	# 将PIL的数据转换为Tensor，数值分布到0-1，上面的参数也是对于tensor类型的数据来说的
        transform.Normalize(**_imagenet_stats)
    ]
    return transform.Compose(transpose_list)	#按照顺序依次执行图像处理的步骤

# 通常并不会做这样的一个处理，但是在gan等应用中，最终生成的图像是[-1,1]之间的，但是我们需要将生成的图像进行可视化，这就需要将经过归一化的图像再转换回去
def tensor2img(tensor):
    # 图像的三个通道分别进行处理
    for i in range(3):
        tensor[:, i, :, :] = tensor[:, i, :, :] * __imagenet_stats['std'][i] + imagenet_stats['mean'][i]
    return tensor
    
```