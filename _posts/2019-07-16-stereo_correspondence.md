---
title: stereo_correspondence	
date: 2019-07-16
categories: Computer-Vision
tags:
- depth esitimation
- stereo correspondence

---

2001年的计算机顶刊的一篇论文，主要是对当时计算机视觉中的立体匹配算法进行了分类，那个时候还没有用到深度学习的思想，所以从里面可以参考到很多解决立体匹配的传统方法，论文关注的方法是以输出视差图的方法来介绍的。

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