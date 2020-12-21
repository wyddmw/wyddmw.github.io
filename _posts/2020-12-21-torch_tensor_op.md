---
title: Torch Tensor Operation
date: 2020-12-21
categories: PyTorch
tags: 
- DeepLearning
- PyTorch
---
　　整理pytorch中对输入张量的一些操作，主要是一些尺寸方面的修改
<!-- more -->

## torch.chunk()
```python
import torch

input_tensor = torch.randn(1, 3, 4, 4)
out_tensor = torch.chunk(input_tensor, chunks=3, dim=1)
# chunk函数的作用是将输入的tensor张量按照指定的维度进行分块处理，返回的结果是tuple类型
for tensor in out_tensor:
	print(tensor.shape) 

# 输出的结果是(1, 1, 4, 4)
```

## torch.view()与torch.transpose()
```python
# 使用view函数和reshape的效果其实是相同的，只是处理的对象有差别
import torch
input_tensor = torch.randn(1, 3, 3, 4)
out_tensor = input_tensor.reshape(1, 3, 4, 3)	# 注意和transpose之间的区别，reshape是不改变tensor的顺序的，但是transpose是直接将两个维度上的数据进行交换，数据的顺序是发生了改变的
a = torch.arange(1, 21).reshape(1, 2, 2, 5)

#view()函数只能用torch.Tensor().view()函数来进行调用，与transpose函数的区别在于，前者只能操作contituous的tensor，比如cost volume-> cost.contiguous()，view处理过后的tensor与原tensor是共享存储的，reshape函数对于是否contiguous的tensor都可以进行操作
# 但是经过测试，review函数之后的tensor与原始的tensor内存地址并不是相同的，应该并不会共享存储吧？
```

## torch.flatten()
```python
import torch
#torch.flatten()的输入是tensor
torch.flatten(input, start_dim=0, end_dim=-1)
# 作用是将输入的tensor的第start_dim到第end_dim维之间的数据拉平到一维tensor
# 输入tensor的顺序并不会发生改变
```

## torch.expand()与torch.repeat()
```python
import torch
a = torch.randn(1, 1, 3, 768)
# torch.expand函数只能对维度为1的数据进行处理
output_tensor = a.expand_tensor(2, -1, -1, -1)

# torch.repeat()函数不只是可以对维度等于1的数据进行处理
```
## torch.cat()实现的就是concat的功能
```python
import torch

a = torch.randn(3, 4)
b = torch.cat(a, dim=1)	# 
```

