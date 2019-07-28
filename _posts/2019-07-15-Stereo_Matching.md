---
title: stereo_matching
date: 2019-07-15
categories: Deep-Learning
tags:
- depth estimation
---

　　CVPR2019年的论文，通过立体视觉算法解决深度估计的问题。

<!-- more -->

## Group-wise Correlation Stereo Network

### Four tips
　　解决立体视觉的算法，通常会包含下面四个步骤：
1. 	matching cost computation：匹配代价计算，衡量待匹配像素与候选像素的相关性，尽量减少代价，代价越小，相关性越大，计算左图一个像素和右图一个像素之间的代价
2. 	cost aggregation：代价聚合。基于点之间的匹配很容易受到噪声的影响，往往真实匹配的像素的代价并不是最低。所以有必要在点的周围建立一个window，让像素块和像素块之间进行比较。代价聚合往往是局部算法或是半局部算法才会使用，全局算法抛弃了window，采用基于全图信息的方式建立能量函数。
3. 	disparity optimation：视差优化
4. 	post-processing：后处理

　　匹配代价计算能够提供初始的一个左图和可能与之对应的有图之间的相似度的衡量标准，这对于立体匹配来说是非常重要的一个步骤。
　　
## FlowNet

　　重新又回去看了FlowNet这篇论文，这篇论文是通过两张图像来预测两张图像的光流结果图。FlowNet提出了两个不同的网络模型结构。

![](/pic/flowS.png)

![](/pic/flowC.png)

　　第一个网络结构中，两张输入图像何为一张图像，然后六通道的图像作为输入，最后直接输出光流预测结果。对于第二种网络结构，作者在开始的时候设计了两个相互独立的网络分支，用来分别对输入的图像进行特征的提取，然后在稍后的阶段，将两个输入特征进行融合。通过这样的方式，能够相对独立地计算出有意义的特征，然后在更高层进行特征的合并。作者提出了一个很重要的层——correlation layer，correlation layer的功能是