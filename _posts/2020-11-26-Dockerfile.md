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

# 更换软件源
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

