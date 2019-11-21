---
title: Useful scripts in torch for training
date: 2019-11-20
categories: Pytorch
tags:
- accumulation of the code
---

　　在自己写网络的过程中经常会需要对网络的每一层输出的尺寸进行分辨率大小的判断，这就需要我们自己写一些简单的程序来进行查看，还有很多类似的相关功能的程序，在本篇博客中进行简单的的整理。<!-- more -->

##  load pretrained model
```python
import torchvision.models as models
import torch
import torch.nn as nn
import numpy as np
from torchsummary import summary			# 可以直接加载出来模型在各个特征层上的特征图上的分辨率

# 从torchvision.models中加载预训练好的模型
resnet18 = models.resnet18(pretrained=True)
# 如果需要对官方给出的模型修改的话，可以直接对网络的层进行修改
resnet18.conv1 = nn.Conv2d(6, 64, kernel_size = 7, stride=2, padding=3, bias=False)

input = torch.random(3, 6, 64, 64)			# 不能直接使用numpy来作为输入数据，可以使用numpy到tensor之间的转换
# 希望能够看到每一层特征图的输出分辨率的大小，如果是使用cuda的话，注意要将模型加载到cuda中
resnet18 = resnet18.to('cuda')
summary(resnet18, (6, 64, 64))				# 输入的分辨率大小为 1*6*64*64

```

```python
# 加载预训练好的部分模型
import torchvision.models as models			# 提供vgg resnet等不同的预训练好的模型
import torch

# 这里涉及两个模型，一个是我们自己的模型，另一个已经训练好的模型，我们要从中加载部分的网络层
if __name__=='__main__':
    #resnet18 = models.resnet18(pretrained=True)		# 加载预训练好的模型，但是如果我们是从一个.pth文件中加载模型的话，就不使用这种方式
    reset18 = models.resnet18()				# 只提供模型的框架，但是并不加载参数
    resnet18.load_state_dict(torch.load('../../..pth'))	# 加载预训练好的模型
    # 上面两行的效果等于reset18 = models.resnet18(pretrained=True)
    
    resnet = ResNet(BasicBlock, [2, 2, 2, 2])			# 加载自己定义的模型
    load_dict = resnet().state_dict()					# 加载模型中所包含的变量，存放在字典中
    pre_dict = resnet18.state_dict()					# 预训练好的模型参数及其权重
    pre_dict = {k: v for k, v in pre_dict if k in load_dict}
    # 更新自己模型中参数对应的权重
    load_dict.update(pre_dict)							# 将预训练好模型中的参数与权重赋值给自己的模型
    resnet.load_state_dict(load_dict)					# 加载模型
    # 已经做过测试，如果不加载预训练好的模型，参数的权重是随机的，上面的方法确实是能够将预训练好的模型中与自己模型对应的参数和权重加载到自己的模型中，对于预训练的模型中没有的参数，将进行随机初始化。
    
    
    
```

