---
title: gan_code_analysis
date: 2019-09-04
categories: Gan-Learning
tags:
- 程序分析
---

　　pytorch实现coupled-gan，结合原理图分析程序
<!-- more -->

## pytorch实现couple gan
```python
import torch
import torch.nn
import torch.nn.functional as nn
import torch.autograd as autograd 
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as pyplot
import matplotlib.gridspec as gridspec
import os
from torch.autograd import Variable
from tensorflow.examples.tutorials.mnist import input_data

#先定义一些参数的设置
mb_size=32
z_dim=100							#噪声的维度
X_dim=mnist.train.images.shape[1]
Y_dim=mnist.train.labels.
h_dim=128
cnt=0
lr=1e-3

#coupled gan的网络结构中，在生成器部分先共享卷积层，然后是两个独立的生成器网络，接着在判别器部分，首先是分别两个独立的网络结构，最后是共享的判别器网络层。

#shared weights 生成器部分的共享网络层
G_shared=torch.nn.Sequential(
	t.nn.Linear(z_dim,h_dim),
	t.nn.Relu(),
)

#共享层之后接的生成器层1
G1_=t.nn.Sequential(
	t.nn.Linear(h_dim,X_dim),
	t.nn.Sigmoid()							#sigmoid函数将输出映射到0~1之间
)

#共享层之后接的生成器2
G2_=t.nn.Sequential(
	t.nn.Linear(h_dim,X_dim),
	t.nn.Sigmoid()
)

#接着是两个独立的判别器网络结构
D1_=t.nn.Sequential(
	t.nn.Linear(X_dim,h_dim),
	t.nn.ReLU()
)

D2_=t.nn.Sequential(
	t.nn.Linear(X_dim,h_dim),
	t.nn.ReLU()
)

#共享的判别器网络
D_shared=torch.nn.Sequetial(
	t.nn.Linear(h_dim,1),			
	t.nn.Sigmoid()				#判别器网络的最后一层输出概率结果，输出的是预测正确图像的概率。
)

def G1(Z):
	h=G_shared(Z)
	x=G1_(h)
	return x
	
def G2(Z):
	h=G_shared(Z)
	x=G1_(h)
	return x

def D1(X):
	h=D1_(X)
	y=D_shared(h)
	return y

def D2(X):
	h=D2_(X)
	y=D_shared(h)
	return y

# 直接补充损失函数部分的细节
# 判别器和生成器要进行交替训练，先训练判别器再训练生成器，交替训练的比例在上一个示例程序中是5:1

for it in range(10000):
	x1=sample_x()						#一个采样函数
	x2=sample_x()						#得到训练用的真实数据
	z=Variable(t.randn(mb_size,z_dim))
	
    #首先是判别器部分
    G1_sample=G1(z)						#首先用生成器去生成假的图片
    D1_real=D1(x1)						#判别器得到真实图片的判断概率
    D1_fake=D1(G1_sample)				#判别器得到假图片的判断概率
    
    G2_sample=G2(z)
    D2_real=D2(x2)
    D2_fake=D2(G2_Sample)					
    
    #计算判别器的损失函数
    D1_loss=t.mean(-t.log(D1_real+1e-8)-t.log(1-D1_fake+1e-8))
    D2_loss=t.mean(-t.log(D2_real+1e-8)-t.log(1-D2_fake+1e-8))
    D_loss=D1_loss+D2_loss
    D_loss.backward()
    
    #generator part
    #生成器的任务是尽可能去欺骗判别器，使判别器无法正确进行区分
    G1_sample=G1(z)
    D1_fake=D1(G1_sample)
    
    G2_sample=G2(z)
    D2_fake=D2(G2_sample)
    
    G1_loss=t.mean(-log(D1_fake+1e-8))
    G2_loss=t.mean(-log(D2_fake+1e-8))
	G_loss=G1_loss+G2_loss
```
## loss function
　　之前在看gan的讲解时，对如何训练gan网络并不理解，讲解的很抽象，一直没有看懂，但是结合着损失函数，其实就很直观了。对于判别而言，其任务就是尽可能识别出来真实图像，区分出来伪造的图像，所以在最小化损失函数的时候，$-log(D_real)$就是使得判别器在真实图片上判断的得分更高，$-log(1-D_fake)$执行的工作就是尽可能使括号中的值更大，尽可能减小判别器在伪造图片上的得分，对判别器的训练通过这样的方式进行来提高判别器的性能。<br>
　　对于生成器的训练来讲，这里是使用生成的假图片来进行训练，然后通过提高判别器将假图片识别为真实图像的概率来提高生成图片的质量。<br>
　　在之前看书的时候，一直觉得，gan网络的训练也是分为两个阶段来进行的，在训练判别器的时候，冻结生成器部分的网络，然后训练生成器的时候冻结判别器的网络，但是目前来看似乎并不是这样的操作。<br>
　　