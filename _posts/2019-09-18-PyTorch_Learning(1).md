---
title: PyTorch_Learning(1)
date: 2019-09-18
categories: PyTorch-Learning
tags:
- auto_grad
---

 　　PyTorch逐渐成为更加主流的深度学习框架，从tf转战到torch，需要重新开始做学习笔记，记录一些torch在使用过程中的主要内容，学习的资料来自pytorch的官方说明文档。

<!-- more -->

　　在pytorch中，所有神经网络的中心内容是autograd包，autograd包给所有tensors上的运算提供了自动计算微分的功能。<br>
　　torch.Tensor是这个包的核心类。如果我们设置它的属性.requires_grad是True的话，它就会追踪它上面所有的运算。当我们完成我们的运算之后，我们需要调用.backward()函数，然后所有的微分运算都会被自动执行，这个tensor的微分就会被存储在.grad属性中。如果想要阻止跟踪运算，我们可以使用with torch.no_grad()函数，在这个函数的作用域内，即使参数的requires_grad属性被设置为True，也不会进行运算。<br>
　　之所以能够实现autograd机制，和variable以及function是分不开的，上面简单写了sensor，那么sensor和variable是什么关系呢？要进行autograd，必须先将tensor包成variable。variable和tensor基本一致，区别在于额外多了一些属性，autograd.Variable的属性有data、grad、creator。<br>
　　对于autograd来说，还有一个非常重要的类——function。function和tensor是紧密相连的，二者构建了一个无环图，对运算的历史进行了编码。每一个tensor都有一个参考"Function"构建tensor的.grad_fn属性，除了那些用户自己创建的tensor，它们的grad_fn是None。<br>
　　如果我们想要计算偏导数，我们可以调用一个tensor上的.backward()函数。
　　
```python
import torch 
#创建一个tensor，并设置requires_grad属性为True
x=torch.ones(2,2,requires_grad=True)
y=x+2		#由于y是作为一个运算的结果，所以它也是有.grad_fn属性的。
z=Y^2*3
out=z.mean()

#来看一下用户自己的数据
a=torch.randn(2,2)		#如果是用户自己定义的数据，
a=((a*3)/(a-1))
print(a.requires_grad)	#这个时候输出的结果是false
a.requires_grad_(True)
print(a.requires_grad)	#经过修改之后现在的结果是True
b=(a*a).sum()
print(b.grad_fn)		#grad_fn表示的是这个tensor对应的grad的类型是什么
#如果a没有被设置为requires_grad为True的话，那么后续和它相关的tensor的requires_grad就都是False
out.backward()			#out是一个标量，所以out.backward()是等同于out.backward(torch.tensor(1.))
x=torch.randn(3,requires_grad=True)
print(x.requires_grad)
print((x^2).requires_grad)
with torch.no_grad():
	print((x^2).requires_grad)		#在这个作用域下，输出的结果是false
print((x^2).requires_grad)			#出了这个作用域之后输出的结果恢复为True

```