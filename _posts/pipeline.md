---
title: Pipeline
date: 2019-10-14
categories: Experiment
tags: 
- description of our experiment
---

　　这篇文档主要用来记录在组织我们自己的网络模型的过程中，遇到的问题以及想到的解决问题的方法，当我们的这篇文档已经不需要再进行内容添加的时候，我们自己的网络模型就完成了。

<!-- more -->

目前能够想起来的关于实验结果对标的问题

1. 在图像合成方面：gan生成的视图要和deep3D得到的结果进行对标<br>
2. 在深度估计方面的结果 一方面要和使用单目视图进行深度估计的方法比较最终的结果，另一方面也要和使用双目视图进行深度估计的方法做结果的比对。<br>
3. 整理一下gan的程序，在之前的博客上面添加对代码的注释。<br>
4. 条件生成模型中，除了pix2pix中提出的损失函数，还需要额外添加L1范式的约束条件，这样的目的是为了使生成的图像和真实的图像更加接近。除了上面的这个损失函数之外，还可以再额外添加一个用来衡量图像之间相似程度的约束条件。可以尝试借鉴stylegan中的方法，将输入的latent code先转换到另一个空间intermediate latent code，用mlp做一个映射，然后再配合着条件一起输入到判别器中的。噪声的输入可能是导致合成视图中物体外观以及颜色等存在差异的重要原因，这部分可以通过对比实验来进一步说明，目前的思路是不采用噪声作为输入的一环。<br>
5. 一方面需要继续看gan的论文，另一方面需要看一下其他优秀的视图合成的工作。目前找到的两篇相关的论文，一个是LLFF，在image_synthesis的文件夹中<br>

输入图像经过mlp之后得到的特征向量<br>
使用DCGAN可以先来试一试 不添加使用输入的噪声 后面可以考虑添加使用circleGAN<br>
看pix2pix的程序 看stylegan的网络结构 手写<br>

6. 看完了DCGAN，在博客上做一下整理，国庆的时候整理一下建模时用到的相关知识，整理整理博客<br>

7. 定向图像的生成，同样也取决于判别器，然后我们就可以综合来利用circleGAN+pix2pixHD的综合来得到高分辨率的合成图。<br>

～～在写程序的时候遇到的问题是如何处理输入图像分辨率 长宽相差较大的问题 需要参看其他的程序来调整分辨率～～

8. pix2pix_simple 的输入图像的分辨率的大小调整为了286*286 还是需要看原始的程序来看
需要引入参差网络的网络结构。需要看一下这些初始化方式之间的差别<br>

9. 从目前的实验来看，在不添加噪声的前提下，是可以完成定向的图像生成的，接下来的任务就是如何提高生成图像的质量。两个方向，一个是改善网络的结构，另一个方向是添加一个后处理的步骤，因为将生成一致性的约束添加到了网络结构部分，所以这里的后处理主要强调的内容是图像的去模糊。<br>
10. 关于上面分辨率的问题，发现是自己在想问题的时候想复杂了，因为这部分并不需要降到很低的分辨率之后再进行上采样，是可以将分辨率降低2倍或是降低3倍之后就进行上采样的，这部分是对生成器的网络结构来说的，而对于判别器，也并不需要降采样到1之后计算最后的概率值，算损失值，这对于网络结构的设计可以说是非常大的一个思路解放。

下载了一些用于去模糊的论文，其中也有使用对抗生成网络进行的工作，放在paper的去模糊的文件夹里。

现在遇到的主要问题一个方面是如何赶快把程序编写出来，需要实现的功能是将语义分割和原始图像进行融合之后输入网络中然后尝试去合成新的图像，其次就是如何解决数据集数据量可能不充足的问题，如何在数据增强方面做一些工作出来，具体程序的编程方面，还是要参照Deblur的程序，里面有很多可以用的小的点。
