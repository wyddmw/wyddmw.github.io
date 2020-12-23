---
title: Dockerfile
date: 2020-11-26
categories: Docker
tags:
- Linux
---
　　学习使用dockerfile对镜像进行构建。
<!-- more -->

　　首先是dockerfile的写法，可以直接在dockerfile中定义需要使用的软件等，可以从宿主机上现有的docker镜像直接进行修改。<br>

```dockerfile
ARG ubuntu=20.04
# 从现有的基础镜像上进行修改
FROM nvidia/cuda:11.0-devel-ubuntu20.04
LABEL maintainer "spyder"

# 更换软件源 但是不知道为什么不好用啊
RUN rm /etc/apt/sources.list && \
		touch /etc/apt/sources.list && \
		echo "deb http://mirrors.aliyun.com/ubuntu/ focal main restricted universe multiverse" > /etc/apt/sources.list && \
		echo "deb http://mirrors.aliyun.com/ubuntu/ focal-security main restricted universe multiverse" >> /etc/apt/sources.list && \
		echo "deb http://mirrors.aliyun.com/ubuntu/ focal-updates main restricted universe multiverse" >> /etc/apt/sources.list && \
		echo "deb http://mirrors.aliyun.com/ubuntu/ focal-proposed main restricted universe multiverse" >> /etc/apt/sources.list && \
		echo "deb http://mirrors.aliyun.com/ubuntu/ focal-backports main restricted universe multiverse" ?> /etc/apt/sources.list && \
		echo "deb-src http://mirrors.aliyun.com/ubuntu/ focal main restricted universe multiverse" >> /etc/apt/sources.list && \
		echo "deb-src http://mirrors.aliyun.com/ubuntu/ focal-security main restricted universe multiverse" >> /etc/apt/sources.list && \
		echo "deb-src http://mirrors.aliyun.com/ubuntu/ focal-updates main restricted universe multiverse" >> /etc/apt/sources.list && \
		echo "deb-src http://mirrors.aliyun.com/ubuntu/ focal-proposed main restricted universe multiverse" >> /etc/apt/sources.list && \
		echo "deb-src http://mirrors.aliyun.com/ubuntu/ focal-backports main restricted universe multiver" >> /etc/apt/sources.list

# 并不是一定需要的一些流程
# ENV CUDA_VERSION 11.0		# 设置全局的环境变量

#ENV PATH /usr/local/bin:${PATH}		# 将cuda添加到全局环境变量中

RUN apt-get update && apt-get upgrade && apt-get install -y python3 && \ 																			python3-pip \
															vim \
															
RUN pip3 install -no-cache-dir numpy \
								torch==1.7 \
								torchvision \
								scipy \
								cython \
								pandas
															
```

---

2020.12.22 分割线一下

　　使用上述的更换软件源的方式并不好用，在创建一个torch版本的镜像时，目前采用的方式是直接从dockerhub上复制一个镜像，然后在镜像中将软件更换一下，目前是最直接的方式了，不是从cuda的官方docker库中进行选择，不知道为什么这里的apt-get总是会出错，找对应版本的torch镜像库然后下载。<br>

　　从pytorch下载了docker镜像之后才发现，原来下载的镜像使用的conda的环境，这里发现了之前一直没有发现的一个问题，其实可能下载了一个镜像之后并不需要去更新update，因为并不需要额外安装软件，之前在安装的过程中一直是使用pip3 install的方式，但是后来发现如果电脑上只安装了一个版本的python的话，就可以直接使用pip进行安装，这个时候就不存在pip3命令找不到的问题了。<br>

　　所以现在的环境部分其实就可以打通了，还是使用pytorch官方提供的镜像，然后直接使用pip进行软件包的安装。如果存在cuda环境的问题，部分也可以直接通过conda的命令直接进行安装：

```python
conda search cudatoolkit
conda search cudnn
conda install cuda==version cudnn==version
```

　　使用vscode远程连接ssh，这样就能够直接修改远程服务器上的代码，而不需要每次修改完之后再传输。安装插件remote ssh，然后remote explorer添加远程的ssh，打开需要使用的文件夹然后直接在vscode上修改代码，打开一个终端在，这个时候其实已经是远程的终端了，进入docker之后，在vscode上面修改代码，终端里直接运行。

