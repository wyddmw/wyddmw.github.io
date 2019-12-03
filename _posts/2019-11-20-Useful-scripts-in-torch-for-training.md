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
    pre_dict = {k: v for k, v in pre_dict.items() if k in load_dict}	# 对预训练好的模型中的参数进行删减，去掉我们自己模型中没有的参数和权重
    # 更新自己模型中参数对应的权重
    load_dict.update(pre_dict)							# 将预训练好模型中的参数与权重赋值给自己的模型
    resnet.load_state_dict(load_dict)					# 加载模型
    # 已经做过测试，如果不加载预训练好的模型，参数的权重是随机的，上面的方法确实是能够将预训练好的模型中与自己模型对应的参数和权重加载到自己的模型中，对于预训练的模型中没有的参数，将进行随机初始化。     
```

## State_dict check

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
summary(resnet18, (6, 64, 64))				# 输入的分辨率大小为 1*6*64*64，查看在各个特征层上的分辨率大小

# 也可以使用state_dict的方式来进行查看
model_dict = resnet18.state_dict()
for i, j in model_dict.items():
    print i									# 各个网络层的名称
```

## Model Visualization

```python
# 模型的可视化，需要用到额外的一些工具
from torchsummary import summary
from torch.autograd import Variable
from tensorBoardX import SummaryWriter
import torchvision.models as models

# 依然是使用resnet来做示例
resnet = models.resnet18()								# 不加载预训练的参数
input = Variable(torch.rand(1, 3, 224, 224))			# 构造假的生成数据
with SummaryWriter(comment='resnet') as w:
    w.add_graph(resnet, input_to_model=input)			# 在当前路径下生成一个runs文件夹，在该文件夹下面运行tensorboard --logdir runs --> localhost:6006

# 在运行的时候报了很多的错误，其中很多的错误是和使用module的版本有关，对一些主要的模块进行降版本之后成功运行，将对应的版本给出
# tensorboard:1.12 和tensorflow的版本如果不对应的话，会出现错误
# tensorboardX: 1.8 
# tensorflow: 1.12
```

## Params update

```python
# 如果希望对指定网络层的参数进行修改，需要用到对应参数的名称
# 对参数的读取，有下面的几种不同的方法
for params in net.parameters():
for name, params in net.named_parameters():

# 在我们自己的程序中，用到了共享的参数层，所以需要对这部分的参数进行平均值处理，具体的操作就是乘上0.5，但是如何对共享部分的参数进行操作，一种方法是在优化器部分，设置需要优化的参数的字典，然后给出对应学习率，这个操作还没有具体看，所以使用的是第二种方法
for name, param in self.netD.named_parameters():
    if name.split('.')[1] == 'discriminator_shared':
        param.grad.data = 0.5 * param.grad.data
        # 这里需要注意的点是，参数的在训练的时候前面会加上一个module的属性，所以在对名称进行分割的时候使用到name.split('.')[1],切出来的名称对应的索引是1
```



## Freeze part params

```python
# 在训练的时候需要对网络部分的参数冻结
'''
补充一个知识点，在获取网络的参数时，上面其实已经写出来了两个不同的方法，一个是net.named_parameters()、net.parameters()，前者可以通过网络层的名称来选取对应的参数，后者则是对整个网络中的参数进行迭代，另一个获取网络参数的方式是state_dict()的方式，二者的区别在于前者能够修改param.requires_grad参数，但是后者对应的param.requires_grad只能是False，所以如果想冻结参数或是修改requires_grad这个属性，需要使用前者。
'''
for name, params in self.netG.named_parameters():
    if name.split('.')[1].split('_')[0] == 'semantic':
        params.requires_grad = False     # 不进行梯度的计算
        self.optimizer_G.step()
        self.cnt += 1
# 对优化器也要进行修改
self.optimizer_G = torch.optim.Adam(filter(lambda p:p.requires_grad,self.netG.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
# 在冻结参数的时候也没有计算对应的损失函数，经过测试，对应的网络层的权重始终保持一个固定的值。

```



