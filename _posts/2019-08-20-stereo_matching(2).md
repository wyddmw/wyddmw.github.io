---
title: stereo_matching(2)
date: 2019-08-20
categories: Deep-Learning
tags: 
- depth estimation
-  stereo matching
---
　　继续整理论文笔记，这次的论文是对GWCNet论文的补充，GWCNet是建立在这些论文的基础上面的，补充立体匹配中一个重要的概念cost volume.<br>
　　GC-Net.<br>
　　cost volume.<br>
　　3D convolution.

<!-- more -->

## Dense Per Pixel Prediction
　　经常能够看到这个表达，但是没有尝试去理解过这个表达到底想要表达什么。在计算机视觉方面，dense pixel prediction表示的就是从一个图像像素的理解出发，建立对图像的内容的理解。

## GCNet——End-to-End Learning of Geometry and Context for Deep Stereo Regression
　　我们提出了一种创新性的深度学习的网络结构，这个网络通过使用经过标定过之后的图像来回归得到视差结果。使用深度学习得到的特征表达，借助这个问题的几何约束来形成一个cost volume（成本容积）——应该说就是将深度学习得到的特征合在一起，放在所谓的这个容器里面。然后在这个容积的基础上，通过3D卷积的方法来学习合并上下文信息。接着视差图通过一个可微的soft argmin operation来得到最终的视差图。<br>
　　作者在论文中提出了这样的一个问题：我们能否借助我们对于立体几何的理解完全使用深度学习的方法来规划立体匹配的问题。

### contribution
　　The main contribution of this paper is an end-to-end deep learning method to estimate per-pixel disparity from a single rectified image pair.所以说，这篇论文提出了第一个完全使用深度学习的网络结构来解决立体匹配问题吗？<br>
　　It explicitly reasons about geometry by forming a cost volume, while also reasoning about semantics using a deep convolutional network formulation. 不知道应该如何准确地翻译这句，但是其中的大致意思包含了：通过合成这样一个成本容积能够合理化几何关系。所以也就是说，为了更加有效地引入几何约束，这个cost volume是非常重要的一环。<br>
1.  We learn to incorporate context directly from the data, employing 3-D convolutions to learn to filter the cost volume over height、width and disparity dimensions.我们学习去直接从数据中去合并上下文的信息，使用3D卷积去学习在cost volume进行filter处理。但是为什么这个里面会有disparity这个维度？
2.  We use a soft argmin function, which is fully differentiable, and allows us to regress sub-pixel  disparity values from the disparity cost volume. 我们使用了这样的一个可微的函数，允许我们能够从视差代价容积中进一步回归出来子像素（不知道sub-pixel应该如何翻译）的视差。

![](/pic/GCNet_Architecture.png)

　　作者并没有设计任何需要人为手动设计特征的步骤，希望能够用深度学习的方法直接从一个图像对中学习到端到端的匹配关系。作者在论文中说到，他们的目的并不是设计一个像黑盒子一样的机器学习算法，作者希望从很多多目几何的研究中利用他们的见解来指导设计网络模型结构。

### Unary Features

　　作者提出的方法首先学习了用于立体匹配代价的深层特征表达。选择使用特征表示相较于用生数据的像素点强度来计算立体匹配代价这种方法来说更常用。所谓的unary features是如何形成的呢？在整个网络结构中，首先是简单的2D卷积层，先卷积核大小为5的一层，然后紧跟着残差网络的基本单位。通过将左右立体图像传过这些网络层之后，形成了一元特征（这不就是常规的特征图吗？）<br>
　　首先我们学习到一个深度特征用于计算立体匹配成本，这种特征提取在光照条件 更加复杂的区域更具有鲁棒性，并且能够结合局部上下文的信息。
　　
### Cost Volume
　　这部分的形成方式单纯看论文的话还是没有办法很直接的理解，需要结合着代码才能更好地理解。后面看了代码之后对这部分的内容进一步补充。<br>
　　在得到经过神经网络计算的unary features之后，我们利用这些特征通过形成一个成本容积去计算立体视觉的代价匹配——常规的这些步骤还是不能避免的，但是原始的算法中是直接使用像素点的强度信息来进行匹配代价计算，而这里是使用用作者的话说，更加鲁邦的特征来进行匹配代价计算。构成这样的一个成本容积，允许我们去约束一个模型——能够保留我们对立体视觉中几何的认识。<br>
　　这样的一个成本容积是一个四维的，具体的形成方法是，通过简单地串联左右特征图就可以实现。根据查找到额博客上的介绍，对于这个匹配成本容积，我们将左图的每一个一元特征和右图每一个视差下的特征图串联起来，封装成 一个四维的代价卷。（对于某一个特征，匹配代价卷是三维的，第一层代表视差为0时候的特征图，第二层代表视差为1时的特征图，以此类推，一共有最大的视差+1层特征，长和宽是特征图的长和宽，假设一共有10个特征，那么这样的三维特征快就有10个）<br>
　　按照作者的说法，这步操作保存了特征的维度。这使得我们可以结合上下文的信息并作用于一元特征上。作者发现，通过级联特征图构成匹配代价卷（cost volume）的方式，其效果要优于削减特征特征或是使用距离度量的方法。

### Argmin
　　经典的立体匹配算法中，从每个匹配代价元中获得最终的匹配代价卷。那么原始的argmin是什么呢？google了一下，arg表示的是变元，也就是自变量argument的英文缩写，argmin就是使后面这个式子达到最小值时的变量的取值。所以应用在这里，应该就是当匹配代价取得最小值时对应的视差，目前这样理解。<br>
　　我们在视差维度上采用argmin操作来估算视差值。但是这种操作存在两个问题：
1. 他是离散的并且不能生成亚像素级别的视差估计。
2. 他不能微分因此无法通过反向传播的方式进行训练<br>


　　为了解决这些限制，作者定义了一种柔性Argmin方法，既是完全可微的，又能回归得到一个平滑的视差估计值。首先，通过将匹配代价值c<sub>d</sub>取负数，把匹配代价卷转换为可能性卷（匹配代价越高，可能性越低，从作者在论文中给出的图来看，匹配的代价可能是负的，所以匹配的代价越大，当前对应的视差值越不可能。）我们利用softmax，对可能性卷在视差维度上进行正则化。然后对每个视差值d进行加权求和，权重就是它的可能性。其数学定义如下：


$$
softargmin=\sum_{d=0}^{D_{max}}d\times\sigma(-c_d)
$$

　　但是相比argmin操作，他的输出受到所有值的影响，这使得他对多状态分布很敏感，因为输出没有取得最大可能性。他会估算每一个状态的加权平均值，为了克服这个问题，依赖网络规则化来生成单峰的视差可能性分布图。

### 损失函数

　　作者从随机初始化参数开始端到端得训练整个网络。利用真实的深度数据进行有监督的学习（所以整个网络本质上还是一个有监督的回归问题，虽然添加了几何约束，但是总感觉和我们希望使用的无监督学习去解决深度学习的方法不太一样呢？）对每个标记像素的损失值取平均值，用预测视差值和真实视差值的绝对值差来训练网络（应该也是一个无监督的问题，因为这里用的真实值并不是经过标注的深度值而是视差值？具体的损失函数表达如下：

$$
Loss=\frac{1}{N}\sum_{n=1}^N||d_n-\hat{h}_n||_1
$$
