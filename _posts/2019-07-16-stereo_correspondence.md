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

　　基本原理是给定在一幅图像上的某一点，选取该像素点邻域内的一个子窗口，在另一幅图像中的一个区域中，根据某种相似性判断依据，寻找与子窗口最为相似的子图，而其匹配的子图中对应的像素点就是该像素的匹配点。通常方法有SAD、SSD等（上面写道的SSD等，但是好像没有写完整）

### 匹配代价计算
　　通常来说，匹配代价的计算是对左右两幅图像的每一个像素点而言的，可以认为定义了一个处理左右两幅图像中匹配像素点的函数：

$$
f(I_L(x,y),I_R(x+d,y))
$$

　　这里的d是我们定义的视差范围，也就是对于左图中的一个像素点，我们给它在右图中定义一个寻找视差的范围。<br>
　　看到作者给出的一个有趣的思路，在给定的视差范围内，如果找到最小的代价匹配值，是不是就可以认为这个最小值就代表是正确的匹配点，而这个最小值对应的d就是我们要找的视差。如果对每一个像素进行这样的操作，就可以得到最后的视差图。
　　
### 代价聚合
　　
　　上面给出的通过单一像素点去计算代价匹配值之后得到视差的方式，其结果并不理想，很容易受到噪声的影响。所以我们需要在点的周围建立一个窗口，让像素块和像素块之间进行比较。代价聚合往往是局部算法或是半全部局部算法才会使用的，全局算法抛弃了window，采用基于全图信息的方式建立能量函数。这样来看，代价聚合可以看做是对匹配代价计算的结果进行一个滤波的过程。<br>
　　目前来看，代价聚合是在代价匹配计算之前需要完成的一个预处理的步骤。看一下作者列举出来的例子，我们假设有一个3^2的window，在这个window上进行代价聚合的函数是g(x)，首先对原始的左图上一个**像素点**进行聚合：

$$
\sum_{(x_L,y_L)\varepsilon(W_{xL},W_yL)}g(x_L,y_L)
$$

　　经过上面计算之后得到的是窗口的中心点的值，同理对于右边的图像，需要在视差的范围内进行聚合操作：

$$
\sum_{(x_R+d,y_R)\varepsilon(W_xR,W_yR)}g(x_R+d,y_R)
$$

　　上面的两步处理完成之后，下一步就需要对处理之后的像素计算匹配代价，并且找到代价的最小值：

$$
min(f(I_L(x\prime,y\prime),I_R((x+d)\prime,y\prime))),d\varepsilon(d_{mii},d_{max})
$$

### 视差计算

　　在上面匹配代价计算和代价聚合的步骤里，都提及了视差计算。视差计算最常用的策略就是WTA策略，就是上面说的，在一定视差的范围内，如果找到了最小的代价匹配值，就可以认为这个最小值就代表是正确的匹配点，这个最小值对应的d就是我们需要找的视差。这一步可以分为局部算法和全局算法，局部算法直接优化代价聚合模型，对于全局算法，需要建立一个能量函数，输出的是一个粗略的视差图。
　　
### SAD源码简读

```c++
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace std;
using namespace cv;

int main()
{
    int hWin=3;	//window的大小是3*3
    cout<<"hWin: "<<hWin<<endl;
    cout<<"SAD test"<<endl;
    
    Mat left_image=imread("disp2.png");
    Mat right_image=imread("disp4.png");
    
    int imagWidth=left_image.cols;
    int imageHeight=left_image.rows;
    
    Mat SAD_image=Mat(left_image.size(),CV_8UC1,1);
    Mat MatchLevel_image = Mat(left_image.size(),CV_8UC1,1);  
    
    //设定视差的范围d是0~7
    int minDBounds=0;
    int maxDBounds=7;
    
    namedWindow("Left");
    namedWindow("Right");
    namedWindow("SAD");
    namedWindow("MatchLevel1");
    
    imshow("left",left_image);
    imshow("right",right_image);
    
    /*SAD Transform*/
    int i,j,m,n,k;
    unsigned char centerPixel=0;
    unsigned char neighborPixel=0;
    int bitCount=0;
    unsigned int bigger=0;
    
    int sum=0;
    unsigned int *match_level=new unsigned intp[maxDBounds-minDBounds+1];
    int temp_min=0;
    int temp_index=0;
    
    unsigned char* dst;
    unsigned char* left_src=NULL;
    unsigned char* right_src=NULL;
    
    unsigned char left_pixel=0;
    unsigned char right_pixel=0;
    unsigned char sub_pixel=0;
    
    //应该是先进行代价聚合的计算
    for (i=0;i<imageHeight;i++)
    {
        for (j=0;j<imageWidth;j++)
        {       
            for (k=minDBounds;k<=maxDBounds;k++)
            //只在x方向上进行搜索
            {
                sum=0;
                //在整个窗口中进行计算，在x方向和y方向上进行循环，比较左右两个图像上的差值然后求和，这一步相当于是起到了滤波的效果
                //m表示的是y轴
                for (m=i-hWin;i<=i+hWin;i++)
                {                
                    for (n=j-hWin;n<=i+hWin;i++)
                    //n表示的是y轴
                    {
                        if (m<0||m>=imageHeight||n<0||n>=imageWidth)
                        {
                            sub_pixel=0;
                        }
                        else if(n+k>=imageWidth)
                        {
                            sub_pixel=0;
                        }
                        else
                        {
                            left_src=(unsigned char*)left_image.data+m*left_image.step+n+k;
                            right_src=(unsigned char*)right_image.data+m*right_iamge.step+n;
                            
                            left_pixel=*left_src;
                            right_pixel=*right_src;
                            if (left_pixel > right_pixel)
                            {
                                sub_pixel=left_pixel-right_pixel;
                            }
                            else
                            {
                                sub_pixel=right_pixel-left_pixel;
                            }
                        }
                     sum+=sub_pixel;   
                    }
                }
                match_level[k]=sum;		//保存在当前d下计算得到的差值
            }
            //d范围内的差值全部计算完成，接下来进行最佳匹配点的寻找
            
            temp_min=0;
            temp_index=0;
            //通过循环比较计算出来的最小值，差值最小的认为是最佳匹配点
            for(m=1;m<maxDBounds-minDBounds+1;m++)
            {
                if (match_level[m]<match_level[temp_index])
                {
                    temp_min=match_level[m];
                    temp_index=m;			//这个m表示的就是视差
                }
            }
            dst=(unsigned char *)SAD_image.data+i*SAD_image.step+j;
            *dst=temp_index*8;
            
            dst=(unsigned char*)MatchLevel_image.data+i*MatchLevel_image.step+j;
            *dst=temp_min;
        }
    }
    
    //剩下的就是输出显示的图像了
}
```