---
title: docker visualize
date: 2020-08-05
categories: Docker
tags:
- Linux
---
　　实现docker下的可视化操作
<!-- more -->

　　为了在docker中使用gaas以及ros等相关仿真的工具，需要实现docker的可视化。尝试了很多的方法，终于调通了，现在记录一下方法：

```shell
$ sudo apt-get install -y vnc4server	# 　　首先要在ubuntu上安装vncserver：
$ vnc4server
$ # 查看可以用的端口 以localhost:8001为例
$ export DISPLAY=localhost:8001
$ xhost +
$ # 这个时候需要看到输出：access control disabled, clients can connect from any host 说明可以通过clients来进行访问
```

　　接下来是启动docker时需要使用的参数命令：

```bash
$ sudo docker run -itd --name container_id \ 
 -v /home/:/home \
 -v /tmp/.X11-unix:/tmp/.X11-unix \   #共享本地unix端口
 -e DISPLAY=unix$DISPLAY \             #修改环境变量DISPLAY
 -e GDK_SCALE \     
 -e GDK_DPI_SCALE \
 image_id
```

　　在docker中运行ros的时候，需要额外添加相应网络的配置，在/etc/hosts中添加下面的内容：

```shell
151.101.84.133  raw.githubusercontent.com
```

