---
title: Customized Conv
date: 2020-12-21
categories: PyTorch
tags:
- PyTorch
- DeepLearning
---
　　实现使用PyTorch自定义卷积，包括使用分组卷积等操作，相当于使用PyTorch实现自定义算子，将传统的方法嵌入到深度学习中
<!-- more -->

## Customized Conv Kernel
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

# 实现高斯kernel
class Gaussian(nn.Module):
	def __init__(self, input_tensor, out_planes):
		super(Gaussian, self).__init__()
		kernel = [[0.03797616, 0.044863533, 0.03797616],
                   [0.044863533, 0.053, 0.044863533],
                   [0.03797616, 0.044863533, 0.03797616]]
		B, C, H, W = input_tensor.size()
		kernel = torch.FloatTensor(kernel).expand(out_planes, C, 3, 3).cuda()
		# 设置为参数固定的权重，不进行梯度更新
		self.weight = torch.nn.Parameter(data=kernel, requires_grad=False)
	
	def forward(self, x):
		# 使用F.conv2d()函数可以自己设置卷积核
		return F.conv2d(x, self.weight, stride=1, padding=1)
```

```python
# 分组卷积操作
def conv_in_groups():
	# 实现一个简单的矩形框，kernel的权重大小是固定的
	input_tensor = torch.arange(1, 21).reshape(1, 2, 2, 5)
	N, C, H, W = input_tensor.size()
	kernel = [[1, 1, 1],
              [1, 1, 1],
              [1, 1, 1]]
    kernel = torch.FloatTensor(kernel).expand(C, 1, 3, 3)
	kernel = torch.nn.Parameter(data=kernel, requires_grad=False)
	out_tensor = F.conv2d(input_tensor, kernel, stride=1, padding=1, groups=C)
	# 分组卷积的分组数应该和kernel的个数是相同的
	

```