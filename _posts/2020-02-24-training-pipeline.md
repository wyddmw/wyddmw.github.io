---
title: pytorch_training
date: 2020_02_24
categories: Deep-Learning
tags:
- training process
- training pipeline
---

<!-- more -->
## 关于CUDA

　　之前一直没有注意过关于模型部署到GPU上的问题，在这次自己写程序进行inference的时候出现了模型加载过程中的问题。遇到的第一个问题是，如果设置了多GPU的训练，网络的变量前会有module.的关键字，所以模型在训练的过程中如果没有使用多GPU，在测试的时候，如果将模型加载到多GPU上，就会出现模型加载失败的错误，解决的方法就是去掉多GPU:
```python
import torch
# 在多GPU训练的时候，需要的参数是GPU的IDs
gpu_ids =  []
model = MSMNet()
model.to(gpu_ids[0])
model = nn.DataParallel(model, gpu_list)	# 后面的参数是多GPU的ID
# 通过这样的写法，模型的变量前面会有module.的关键字，如果不想使用多GPU训练的话，就将dataparall()去掉，并指定GPU
```
```python
# 完整的训练流程
# 先将模型加载到第一个GPU上，然后使用并行
if len(gpu_ids)>0:
	assert(torch.cuda.is_available())
	torch.cuda.set_device(gpu_ids[0])
	net.to(gpu_ids[0])
	net = torch.nn.DataParallel(net, gpu_ids)
	
# 上面是将网络模型加载到GPU上，接下来还需要将Tensor加载到GPU上
device = torch.device('cuda{}'.format(gpu_ids[0]) if gpu_ids else torch.device('cpu'))
# 损失函数tensor添加到GPU上
critirien_loss = torch.nn.L1Loss().to(device)
input_left = input_left.to(device)
# 基本的多GPU训练的流程便是上面的内容，但是在存储模型和加载模型的过程中，也需要针对多GPU进行补充

def savemodel(save_path):
	if len(gpu_ids) > 0 and torch.cuda.is_available():
		torch.save(net.module.cpu().state_dict(), save_path)
		# 因为是多GPU训练，所以要添加module的关键字
		net.cuda(device)
	else:
		torch.save(net.cpu().state_dict(), save_path)

def loadmodel(model_path):
	# 在训练的过程中遇到的一个问题就是在gpu0上训练的模型，如果不进行修改就无法使用gpu1进行加载，所以我们需要指定加载的gpu id
	if isinstance(model, nn.DataParallel):
		net = net.module
		# 针对多GPU的模型，添加module关键字
	state_dict = torch.load(model_path, map_location=str(device))	#加载到指定的模型上
	if hasattr(state_dict, '_metadata'):
		del state_dict._metadata
	net.load_state_dict(state_dict)
	
```
　　关于state_dict的一些补充，内容来自网上的一篇博客，model.state_dict()返回的是一个OrderDict，存储了网络结构的名字和参数

```python
import torch
# import ...
# 这个脚本程序主要是记录一下最新训练过程中用到的一些修改，现在的问题是不知道是不是在模型加载的过程中出现了问题，所以导致模型一直没有办法按照预期的方向收敛。
if isinstance(model, nn.DataParallel()):
	model = model.module		# 这个是如果模型是多GPU训练，加载的是单GPU模型时可能会用到的程序
	model.load_state_dict()
# 但是另一个常见的问题是在多GPU上训练之后在单GPU上做预测，这个时候就不能使用datapallel了，对应就需要将多GPU训练的模型中关键字去掉，自己写的一个不算正规的处理程序：
for v in state_dict[]:
	if v.split('.')[0] == 'module':
		model.load_state_dict({k.replace('module.', ''):v for k,v in state_dict[].items()})		#就是将模型中的module关键字去掉

# 查看变量在什么设备上的时候，使用的语句：
a = torch.FloatTensor()
a = a.cuda()
a.get_device()		# 默认是部署到了cuda:0上，所以当我们指定了设备，应该也还是部署到了cuda0上
# 个人感觉还是指定device的方式更加直观有效一些
```
