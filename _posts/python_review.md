# Pytorch Learning
## 定义网络
　　定义网络的时候，需要继承nn.Module，病实现它的forward方法，把网络中具有可学习参数的层放在构造函数__init__中。如果某一层不具有可学习的参数，则既可以放在构造函数中也可以不放在构造函数，不建议放在其中，在forward中使用nn.functional进行代替。<br>

```python
import torch.nn as nn       # this module is designed for neural networks
import torch.nn.functional as F

class Net(nn.Module):        #
      def __init__(self):
        #nn.Module子类的函数必须在函数必须在构造函数中执行父类的构造函数
        super(Net, self).__init__( )

        #conv_1
        self.conv1=nn.Conv2d(1,6,5) #kernel size=5*5 6 indicates the number of output channels
        self.conv2=nn.Conv2d(6,16,5)
        self.fc1=nn.Linear(16*5*5,120)
        self.fc2=nn.Linear(120,84)
        self.fc3=nn.Linear(84,10)

      def forward(self, x):         #定义forward方法
        # 卷积->激活->池化
        x=F.max_pool2d(F.relu(self.conv1(x)),(2,2))
        x=F.max_pool2d(F.relu(self.conv2(x)),2)
        x=x.view(x.size()[0],-1)
        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        x=self.fc3(x)
        return x
```
　　只要在nn.Module的子类中定义了forward函数，backward函数就会自动被实现（利用autograd），网络可学习参数通过net.parameters()返回，net.named_parameters可以同时返回可学习的参数及名称。<br>
　　 
```python
#pytorch中可以直接查看参数的名称以及输出的spatial_size
for name,params in net.parameters():
    print(name,":",params.shape)
```

```python
# 补充完整的训练的程序
from __future__ import print_function
import torch as t
import torchvision as tv
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.transforms import ToPILImage
from torch import optim


class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.conv1=nn.Conv2d(3,6,5)
        self.conv2=nn.Conv2d(6,16,5)
        self.fc1=nn.Linear(16*5*5,120)
        self.fc2=nn.Linear(120,84)
        self.fc3=nn.Linear(84,10)

    def forward(self,x):
        x=F.max_pool2d(F.relu(self.conv1(x)),(2,2))
        x=F.max_pool2d(F.relu(self.conv2(x)),2)
        #将特征图展成二维的
        x=x.view(x.size()[0],-1)
        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        x=self.fc3(x)
        return x


net=Net()
print(net)

criterion=nn.CrossEntropyLoss()             #创建一个对象
optimizer=optim.SGD(net.parameters(),lr=0.001,momentum=0.9)

t.set_num_threads(8)
for epoch in range(2):
    running_loss=0.0
    for i,data in enumerate(trainloader,0):
        inputs,labels=data
        optimizer.zero_grad()

        #forward plus backward
        outputs=net(inputs)
        loss=criterion(outputs,labels)
        loss.backward()

        #update parameters
        optimizer.step()

        #print log information
        #loss是一个scalar，需要使用loss.item()来获取数值，不能使用loss[0]
        running_loss+=loss.item()
        if i % 2000 == 1999:    # print training status every 2000 batches
            print('[%d,%5d] loss : %.3f'\
                  %(epoch+1,i+1,running_loss/2000))
            running_loss=0.0
print('training finished')

```

```python
#关于super的使用，附上一个测试代码
class A():
  def __init__(self):
    print('call class A')
    print('leave class A')

class B(A):     #继承基类A
  def __init__(self):
    print('call class B')
    print('leave class B')

class C(A):     #继承基类A
  def ___init__(self):
    print('call class C')
    super(C,self).__init__()    #自动调用A的__init__函数，不需要进行显式的声明
    print('leave class C')

class E(B,C):   #继承基类B C
  def __init__(self):
    print('call class E')
    #显式调用基类的构造函数
    B.__init__()
    C.__init__()              #虽然B和C都继承了基类A，但是因为B没有使用super，所以不会自动调用A中的__init__函数
    print('leave class E')
```