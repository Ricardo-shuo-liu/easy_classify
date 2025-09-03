"""
本模块实现对数据的处理,依赖神经网络进行训练,验证和测试

核心内容:
    - mydataset类: 用于数据集的处理
    - dataloader函数: 实现对处理后的数据进行打包批处理
    - dev函数: 验证函数
    - train函数: 训练神经网络
    - test函数: 测试最终模型
    - get_device: 获得运行主机是不是可以使用GPU训练网络
    - plt_acc_curve: 绘制图观察随着训练次数预测准确率变化情况
    - save_pre: 保存预测结果

依赖:
    - python 3.13.5
    - 第三方库 : torch
                os
                numpy
                gc
                matplotlib
                csv 
"""

from neural_network_NLP import neural_network_11
from torch.utils.data import (
    Dataset,
    DataLoader
)
import os
import numpy as np
import torch
import gc
import matplotlib.pyplot as plt
import csv


class mydataset(Dataset):    
    """
    数据集处理类
    通过对文件地址的访问将其转化为tensor类型为之后打包提供原始数据

    Args:
        path_doc : str - 存储数据资料的文件夹目录路径
        path_file_featurs : str - 存储特征资料文件路径
        mode : str - 模式
        path_file_labels : str - 存储标签资料的文件路径
    Attributes:
        mode : str - 模式用于实现全局处理
        featurs : tensor - 特征数据集合
        labels : tensor - 标签数据集合,对于test不存在
    Methods:
        process_dataset(path_doc,
                        path_file_featurs,
                        path_file_labels,mode) - 读取数据以及数据转化函数
        
    """
    def __init__(self,
                 path_doc,
                 path_file_featurs,
                 mode,
                 pathe_file_labels = None
                 ):
        super().__init__()
        self.mode = mode
        self.process_dataset(path_doc=path_doc,
                             path_file_featurs=path_file_featurs,
                             path_file_labels=pathe_file_labels,
                             mode=mode)
    def process_dataset(self,path_doc,
                        path_file_featurs,
                        path_file_labels,mode):
        """
        数据处理方法,将数据从文件中读取并且转化成tensor类型

        Args:
            path_doc : str - 文件夹路径
            pathe_file_featurs : str - 特征文件存储路径
            pathe_file_labels : str - 标签文件存储路径
            mode : str - 模式选择
        Returns:
            None
        """
        if mode == 'test':
            path_fearure = os.path.join(path_doc,path_file_featurs)
            #路径合并获取文件的主路径地址用于读取数据
            self.features = torch.from_numpy(np.load(path_fearure)).float()

        else:
            path_fearure = os.path.join(path_doc,path_file_featurs)
            
            featurs = np.load(path_fearure)
            
            path_labels = os.path.join(path_doc,path_file_labels)

            labels = np.load(path_labels)

            if mode == 'train':
                indices = [i for i in range(len(featurs)) if i % 10 != 0]
            elif mode == 'dev':
                indices = [i for i in range(len(featurs)) if i % 10 == 0]
            self.features = torch.from_numpy(featurs[indices]).float()
            #from_numpy 将numpy类型转化为tensor类型,共用底层资源,防止资源浪费
            self.labels = torch.LongTensor(labels[indices].astype(np.int64))
            #将labels转为长整数形,用于交叉熵计算
            #拆分数据集用于实现早停机制
            del featurs,labels
            gc.collect()
            #手动调用一次垃圾回收机制,防止大数据占用过多资源
    def __getitem__(self, index):
        if self.mode == 'test':
            return self.features[index]
        else:
            return self.features[index],self.labels[index]
    def __len__(self):
        return self.features.shape[0]
    
def dataloader(path_doc,
               path_file_featurs,
               mode,
               batch_size,
               path_file_labels=None):
    """
    实现数据的处理并且进行打包

    Args:
        path_doc : str - 文件夹路径
        pathe_file_featurs : str - 特征文件存储路径
        pathe_file_labels : str - 标签文件存储路径
        mode : str - 模式选择
        batch_size: int - 每批次样本数目
    Returns:
        dataloader : Dataloader - 可迭代的对象用于神经网络使用

    """
    dataset = mydataset(path_doc=path_doc,
                        path_file_featurs=path_file_featurs,
                        mode=mode,
                        pathe_file_labels=path_file_labels)
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        drop_last=(mode=='train'),
        shuffle=(mode=='train'),
        num_workers=0,
        pin_memory=False
    )
    return dataloader
def dev(dataloader_dev,model,config):
    """
    验证函数用于实现早停止机制并且可以作为一个准确率的检测

    Args:
        dataloader_dev : Dataloader - 验证数据集
        model : Model - 使用的模型对象
        config : dict - 配置文件
    Returns:
        acc : float - 准确率 
    """
    device = config['device']
    model.eval()
    with torch.no_grad():
        acc = 0
        for data_featurs,data_labels in dataloader_dev:
            data_featurs = data_featurs.to(device)
            data_labels = data_labels.to(device)
            pre = model(data_featurs)
            list_index = torch.argmax(pre,dim=1)
            acc += (list_index.cpu() == data_labels.cpu()).sum().item()
        acc /= len(dataloader_dev.dataset)
        return acc*100
def train(dataloader_train,datloader_dev,model,config):
    """
    训练函数,实现对目标的完成

    Args:
        dataloder_train : Dataloader - 训练数据集
        dataloader_dev : Dataloader - 验证数据集
        model : Model - 使用的模型对象
        config : dict - 配置文件
    Returns:
        acc_list : list - 每一轮训练结束后的平均准确率
    """
    early_epoch = 0
    device = config['device']
    acc_list = []
    optimizer = getattr(torch.optim, config['optimizer'])(
        model.parameters(), **config['optim_hparas'])
    n_epochs = config['n_epoch']
    model.train()
    for i in range(n_epochs):
        acc = 0
        for data_featurs,data_labels in dataloader_train:
            data_featurs = data_featurs.to(device)
            data_labels = data_labels.to(device)
            pre = model(data_featurs)
            loss = model.cross_entropy(pre,data_labels).to(device)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            list_index_max = torch.argmax(pre,dim=1)
            acc += (list_index_max.cpu() == data_labels.cpu()).sum().item()
        acc /=len(dataloader_train.dataset)
        acc_list.append(acc*100)
        #早停机制--->如果持续一个limit的准确率下降将停止训练
        dev_acc = dev(dataloader_dev=datloader_dev,
                      model=model,
                      config=config)
        max_acc = 0
        limit_early_epoch = config['limit_early_epoch']
        if dev_acc > max_acc:
            max_acc = dev_acc
            torch.save(model.state_dict(),config['save_path'])
            early_epoch = 0
        else:
            early_epoch +=1
        if early_epoch >= limit_early_epoch:
            break
    print('最终训练次数为{},模型验证正确率为{}'.format(i+1,max_acc))
    return acc_list
def test(dataloader_test,model,config):
    """
    测试函数,基于未知数据

    Args:
        dataloader_test : Dataloader - 测试数据集
        model : Model - 使用的模型对象
        config : dict - 配置文件
    Returns:
        list_index : list - 预测结果列表
    """
    device = config['device']
    model.eval()
    list_index = []
    with torch.no_grad():
        for data_featurs in dataloader_test:
            data_featurs = data_featurs.to(device)
            pre = model(data_featurs)
            list_index += torch.argmax(pre,dim=1)
    return list_index
def get_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'
def plt_acc_curve(acc_list, config):
    plt.scatter(acc_list,range(config['n_epoch']),label='accury')
    plt.show()
def save_pred(preds, save_file_path):
    print('Saving results to {}'.format(save_file_path))
    with open(save_file_path, 'w') as fp:
        writer = csv.writer(fp)
        writer.writerow(['id', 'tested_results'])
        for i, p in enumerate(preds):
            writer.writerow([i, p])
if __name__ == '__main__':
    #配置基本参数文件
    device = get_device()
    config = {
        'device':device,
        'limit_early_epoch':10,
        'save_path':'./model/NLP_easy_test.pth',
        'n_epoch':20,
        'optim_hparas':{
            'lr' : 0.001
        },
        'optimizer' : 'Adam',
        'save_file_path' : 'pred.csv',
        'batch_size' : 64
    }
    path_doc='./timit_11/'
    path_file_train = 'train_11.npy'
    path_file_train_labels = 'train_label_11.npy'
    path_file_test = 'test_11.npy'    
    dataloader_train = dataloader(path_doc=path_doc,
                                  path_file_featurs=path_file_train,
                                  path_file_labels=path_file_train_labels,
                                  mode='train',
                                  batch_size=config['batch_size'])
    dataloader_test = dataloader(path_doc=path_doc,
                                 path_file_featurs=path_file_test,
                                 mode='test',
                                 batch_size=config['batch_size'])
    dataloader_dev = dataloader(path_doc=path_doc,
                                  path_file_featurs=path_file_train,
                                  path_file_labels=path_file_train_labels,
                                  mode='dev',
                                  batch_size=config['batch_size'])
    
    neural_network = neural_network_11().to(config['device'])
    acc_list = train(dataloader_train=dataloader_train,
               datloader_dev=dataloader_dev,
               model=neural_network,
               config=config)
    plt_acc_curve(acc_list=acc_list,config=config)
    del neural_network
    model = neural_network_11().to(config['device'])
    best_model = torch.load(config['save_path'], map_location='cpu')
    model.load_state_dict(best_model)
    list_index = test(dataloader_test=dataloader_test,model=model,config=config)
    save_pred(preds=list_index,save_file_path=config['save_file_path'])