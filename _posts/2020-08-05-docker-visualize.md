---
title: docker visualize
date: 2020-08-05
categories: Docker
tags:
- Linux
---
　　实现docker下的可视化操作
<!-- more -->

　　为了在docker中使用gaas以及ros等相关仿真的工具，需要实现docker的可视化。方法如下：

```shell
$ sudo apt-get install apt-get update
$ sudo apt-get install apt-get upgrade
$ sudo apt-get install apt-get install lightdm	# 安装lightdm成功解决了向日葵远程连接的时候中断的问题，也成功实现了可视化，所以如果是重新开始安装的话，建议还是安装lightdm
$ sudo apt-get install -y vnc4server	# 　　首先要在ubuntu上安装vncserver：
$ vnc4server
$ # 查看可以用的端口 以localhost:8001为例
$ export DISPLAY=localhost:8001	# 以终端中具体显示的为主
$ xhost +			# 每次重启以后 需要重新执行这个命令来开启远程的访问
$ # 这个时候需要看到输出：access control disabled, clients can connect from any host 说明可以通过clients来进行访问
```

　　接下来是启动docker时需要使用的参数命令：

```bash
$ sudo docker run -itd --name container_id \ 
 -v /home/:/home \
 -v /tmp/.X11-unix:/tmp/.X11-unix \   	#共享本地unix端口
 -e DISPLAY=:0.0 \              #修改环境变量DISPLAY
 -e GDK_SCALE \     
 -e GDK_DPI_SCALE \
 image_id
```

　　为了测试在docker中可视化是否成功，可以在docker中运行下面的指令：

```bash
apt-get install xarclcok
xarclock 		# 如果可视化安装正常，这个时候可以看到出现一个时钟
```

　　docker在下拉镜像的过程中会可能会出现过慢的情况，可以通过阿里云来实现镜像加速，具体的操作方法为：登录<https://cn.aliyun.com/>，然后右上角登录，可以直接支付宝扫码登录，登录之后，在控制台搜索：容器镜像服务，然后选中镜像加速器，可以看到下面的说明：

```bash
sudo mkdir -p /etc/docker 		# 这一步可以省略，在安装docker2的时候已经生成了这个文件夹
sudo tee /etc/docker/daemon.json <<-'EOF'
{
  "registry-mirrors": ["https://02dn6dgi.mirror.aliyuncs.com"]
}
EOF
sudo systemctl daemon-reload
sudo systemctl restart docker
```

　　运行完上面的命令行之后会覆盖原始docker对应的json文件，需要注意的是上述命令行生成的json对拉取镜像是能够起到加速作用的，但是在后续运行镜像时会报错找不到docker对应runtime文件，所以原始的json文件不能直接被进行覆盖，一个解决的方案就是保存两个json文件，一个是拉取镜像的时候使用，另一个是运行镜像的时候使用，比如在拉取镜像时，原始的镜像可以命名为daemon_runtime.json，然后daemon.json文件存放加速的配置。每次修改了json文件之后需要运行最下面的两条systemctl命令来使修改生效。

　　目前制作的镜像可以在dockerhub中通过搜索spyderzsy来进行查看。

　　在docker中运行ros的时候，需要额外添加相应网络的配置，在/etc/hosts中添加下面的内容：

```shell
151.101.84.133  raw.githubusercontent.com
```

