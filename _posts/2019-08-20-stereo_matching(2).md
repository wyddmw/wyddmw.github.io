---
title: stereo_matching(2)
date: 2019-08-20
categories: Deep-Learning
tags: 
- depth estimation
-  stereo matching
---
　　继续整理论文笔记，这次的论文是对GWCNet论文的补充，GWCNet是建立在这些论文的基础上面的，补充立体匹配中一个重要的概念cost volume.<br>
　　PSMNet.<br>
　　GC-Net.<br>
　　cost volume.<br>
　　3D convolution.

<!-- more -->

## Dense Per Pixel Prediction
　　经常能够看到这个表达，但是没有尝试去理解过这个表达到底想要表达什么。在计算机视觉方面，dense pixel prediction表示的就是从一个图像像素的理解出发，建立对图像的内容的理解。

## PSMNet——Pyramid Stereo Matching Network
　　CVPR2018年的一篇论文，主要讲的是如何添加对上下文信息的利用，提出了一种空间池化层和3D卷积的方法。<br>
　　典型的立体匹配pipeline包含基于匹配代价和后处理找到的一致的点。随着卷积神经网络的应用，CNN已经被用来学习如何找到匹配的像素点。尽管使用卷积神经网络的方法和传统的方法相比确实在精度和速度方面都取得了不错的提升，在一些病态区域（如重叠区，重复出现的表象、低纹理区域，这些需要全局上下文信息进行整合，SPP和膨胀卷积用来扩大感受野，使得PSM-Net可以将像素级的特征扩展至多尺度区域级的特征。）全局信息和局部信息被用来合成cost volume以得到更可靠的视差估计。

### main contributions
1. 提出了一种端到端的立体匹配学习框架，不需要任何额外的后处理。
2. 介绍了空间金字塔池化模块，用来合并局部和全局的上下文信息。
3. 展示了一种堆栈的漏斗3D卷积网络，来延伸cost volume中语义信息的regional support。

### Network Architecture
　　先给出PSMNet的网络结构

![](/pic/PSMNet_Architecture.png)
![](/pic/PSMNet_Architecture2.png)

　　作者在设计网络结构的时候，首先设计了三层卷积核大小为3的网络，用来提取一些基本的特征。作者在这里用到了空洞卷积的方法，对于空洞卷积，后面对这部分内容再进行补充。添加空洞卷积的目的是为了增大感受野，输出的特征图的分辨率是原始分辨率的1/4大小。接下来添加SPP模块，SPP模块的应用是为了收集上下文的语义信息。我们将左图的特征图和右图的特征图串联起来，得到一个cost volume。最终添加回归项，得到最终的输出视差图。

### Spatial Pyramid Pooling Module
　　包含丰富物体语义信息的图像特征图对于一致性估计的问题来说是有益的，尤其是对于病态区域的点来说。在本篇论文中，一个物体和其子区域之间的关系通过SPP模块合并了不同层次的语义信息来学习到。
　　
## GCNet——End-to-End Learning of Geometry and Context for Deep Stereo Regression
　　我们提出了一种创新性的深度学习的网络结构，这个网络通过使用经过标定过之后的图像来回归得到视差结果。使用深度学习得到的特征表达，借助这个问题的几何约束来形成一个cost volume（成本容积）——应该说就是将深度学习得到的特征合在一起，放在所谓的这个容器里面。然后在这个容积的基础上，通过3D卷积的方法来学习合并上下文信息。接着视差图通过一个可微的soft argmin operation来得到最终的视差图。<br>
　　作者在论文中提出了这样的一个问题：我们能否借助我们对于立体几何的理解完全使用深度学习的方法来规划立体匹配的问题。

### contribution
　　The main contribution of this paper is an end-to-end deep learning method to estimate per-pixel disparity from a single rectified image pair.所以说，这篇论文提出了第一个完全使用深度学习的网络结构来解决立体匹配问题吗？<br>
　　It explicitly reasons about geometry by forming a cost volume, while also reasoning about semantics using a deep convolutional network formulation. 不知道应该如何准确地翻译这句，但是其中的大致意思包含了：通过合成这样一个成本容积能够合理化几何关系。所以也就是说，为了更加有效地引入几何约束，这个cost volume是非常重要的一环。<br>
1.  We learn to incorporate context directly from the data, employing 3-D convolutions to learn to filter the cost volume over height、width and disparity dimensions.我们学习去直接从数据中去合并上下文的信息，使用3D卷积去学习在cost volume进行filter处理。但是为什么这个里面会有disparity这个维度？
2.  We use a soft argmin function, which is fully differentiable, and allows us to regress sub-pixel  disparity values from the disparity cost volume. 我们使用了这样的一个可微的函数，允许我们能够从视差代价容积中进一步回归出来子像素（不知道sub-pixel应该如何翻译）的视差。

![](/pic/GCNet_Architecture.png)



　　
　　
　　
　　
　　