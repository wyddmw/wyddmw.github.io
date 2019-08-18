---
title: stereo_correspondence	
date: 2019-07-16
categories: Computer-Vision
tags:
- depth esitimation
- stereo correspondence
- stereo matching

---

2001年的计算机顶刊的一篇论文，主要是对当时计算机视觉中的立体匹配算法进行了分类，那个时候还没有用到深度学习的思想，所以从里面可以参考到很多解决立体匹配的传统方法，论文关注的方法是以输出视差图的方法来介绍的。<br>

补充知乎一个很详细的讲解。

<!-- more -->

## A Taxonomy and Evaluation of Dense Two-Frame Stereo Correspondence Algorithms

作者在论文中提到的一点很有意思，写道，任何一个计算机视觉算法的构建，都明确或是不明确的利用了对真实物理世界和图像成像过程的假设。例如，一个算法如何来测量两张图像中的点是匹配的，换句话说，如何证明他们映射到场景中的同一个点。同样重要的是对整个世界或是场景几何以及对物体表面的假设。始于真实物理世界中的包含了分段平滑的表面一样，算法也在构想的过程中添加了平滑的假设。

一个立体匹配算法的目标是去构造一个基于视差空间的函数d(x,y).<br>
The goal of a stereo correspondence algorithm is then to produce a univalued function in disparity space d(x,y) that best describes the shape of the surfaces in the scene. This can be viewed as finding a surface embedded in the disparity space image that has some optimality property

### four steps
Our taxonomy is based on the observation that stereo algorithms generally perform the following four steps:
1. matching cost computation
2. cost aggregation
3. disparity computation/ optimization
4. disparity refinement

#### SSD algorithm——sum of squared differences

---

　　先画一个分割线，补充知乎上的讲解。<br>
　　图像的匹配就是对于两幅图像或是更多的图像来说，找到它们相似的地方。对于立体匹配而言，就是基于同一个场景得到的多张二维图，通过找到相同点进一步还原原始场景的三维信息，通常是使用双目图像。立体匹配的基本操作就是上面说的：
1. 匹配代价计算（Matching Cost Computation)
2. 代价聚合（Cost Aggregation）
3. 视差计算（Disparity Computation）
4. 视差精化（Disparity Refinement）

### 全局匹配
　　
　　全局立体匹配算法通常会省略第二步，主要是采用了全局的优化理论方法估计视差，建立全局能量函数，通过最小化全局能量函数得到最优视差值，能量函数由数据项和平滑项构成。

$$
E(d)=E_{data}(d)+E_{smooth}(d)
$$

　　虽然写了上面的这些内容，但是还是不知道这个算法是什么意思，日后有机会继续补充。

### 局部匹配

　　基本原理是给定在一幅图像上的某一点，选取该像素点领域内的一个子窗口，在另一幅图像中的一个区域中，根据某种相似性判断依据，寻找与子窗口最为相似的子图，而其匹配的子图中对应的像素点就是该像素的匹配点。通常方法有SAD、SSD等（上面写道的SSD等，但是好像没有写完整）

### 代价匹配计算
　　通常来说，匹配代价的计算是对左右两幅图像的每一个像素点而言的，可以认为定义了一个处理左右两幅图像中匹配像素点的函数：

$$
f(I_L(x,y),I_R(x+d,y))
$$

　　这里的d是我们定义的视差范围，也就是对于左图中的一个像素点，我们给它在右图中定义一个寻找视差的范围。<br>
　　看到作者给出的一个有趣的思路，在给定的视差范围内，如果找到最小的代价匹配值，是不是就可以认为这个最小值就代表是正确的匹配点，而这个最小值对应的d就是我们要找的视差。如果对每一个像素进行这样的操作，就可以得到最后的视差图。
　　
### 代价聚合
　　
　　上面给出的通过单一像素点去计算代价匹配值之后得到视差的方式，其结果并不理想，很容易受到噪声的影响。所以我们需要在点的周围建立一个窗口，让像素块和像素块之间进行比较。代价聚合往往是局部算法或是半全部局部算法才会使用的，全局算法抛弃了window，采用基于全图信息的方式建立能量函数。这样来看，代价聚合可以看做是对匹配代价计算的结果进行一个滤波的过程。
