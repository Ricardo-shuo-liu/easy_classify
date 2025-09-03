"""
神经网络搭建模块(neural_network_NLP.py)

本模块提供了对语音设别问题的简易的神经网络架构,用于之后对于训练或者预测调用

核心内容:
    - neural_network_11: 神经网络模型对象

依赖:
    - python 3.13.5
    - 第三方库: torch.nn

"""

from torch.nn  import (
    Linear,
    Sigmoid,
    CrossEntropyLoss,
    Module,
    Sequential
)


class neural_network_11(Module):
    """
    搭建神经网络架构简易的语音识别神经网络
    通过多层Linear和activation function结合实现通过分类来对语音进行识别
    
    Args:
        None

    Attributes:
        module_framework:Sequential -  神经网络架构主体
    
    Methods:
        forward(inputs) - 神经网络前向传播函数,获取outputs
        cross_entropy(pre,labesl) - 交叉熵损失函数用于计算损失值
    """
    def __init__(self, *args, **kwargs):
        """
        初始化函数，实现模型框架的搭建和损失函数的调用

        Args:
            None
        Returns:
            None
        
        """
        super().__init__(*args, **kwargs)
        self.module_framework = Sequential(
            Linear(429, 1024),
            Sigmoid(),
            Linear(1024, 512),
            Sigmoid(),
            Linear(512, 128),
            Sigmoid(),
            Linear(128, 39) 
        )
        self.lossfunction = CrossEntropyLoss()
    def forward(self,inputs):
        """
        前向传播函数，用于将输入数据转换成目标数据
        
        Args:
            inputs:tensor - 输入的样本特指集
        
        Returns:
            outputs:tensor - 转变后的输出集,对应39个数据的分数
        """
        outputs = self.module_framework(inputs)
        return outputs
    def cross_entropy(self,pre,labels):
        """
        损失函数调用方法,为反向传播提供依据
        
        Args:
            pre:tensor - 预测值
            labels:tensor - 真实值
        
        Returns:
            loss:tensor - 交叉熵损失值
        
        Notes:
            在对CrossEntropyloss进行调用的时候会在之前对pre进行一个softmax
            所以在架构的最后一个不需要经过激活函数
        """
        return self.lossfunction(pre,labels)