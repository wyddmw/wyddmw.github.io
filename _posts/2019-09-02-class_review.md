---
title: class_review
date: 2019-09-02
categories: Python
tags:
- class programming
---

看论文的工作暂时先告一段落，准备实验的代码部分，从基本的pytorch学习开始，先复习一些关于类的内容。
<!-- more -->

### object的问题
　　在写程序的时候，发现有的程序在定义类的时候会继承父类object，但是有的程序却没有继承这个类，发现了这个区别之后，通过编写程序来进行查看二者的区别。<br>
　　是否需要继承object对于python3来说明显，在python2中，如果没有继承object，会有很多的特性无法利用，该类下可以操作的对象只有三个，如果继承了object，这个类的命名空间下会有更多的成员变量可以操作，这些可操作的对象都是类中的高级特性。这些高级的特性主要是用于手写框架或是写大项目的人来用的。但是在python3中，在创建类的时候默认就会继承object，所以即使不写也没有关系。



