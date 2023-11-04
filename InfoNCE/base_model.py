from abc import *
import torch.nn as nn
import torch
import torch.nn.functional as F


class BaseModel(nn.Module, metaclass=ABCMeta):
    def __init__(self, last_dim=300, num_classes=10, simclr_dim=400):
        super(BaseModel, self).__init__()
        self.linear = nn.Linear(last_dim, num_classes)
        self.out_num=1
        self.weight3 = nn.Parameter(torch.Tensor(3 + self.out_num, 300))
        self.simclr_layer = nn.Sequential(
            nn.Linear(last_dim, last_dim),
            nn.ReLU(),
            nn.Linear(last_dim, simclr_dim),
        )
        self.shift_cls_layer = nn.Linear(last_dim, 4)
        self.joint_distribution_layer = nn.Linear(last_dim, 4 * num_classes)

    @abstractmethod
    def penultimate(self, inputs, all_features=False):
        pass

    def forward(self, inputs, penultimate=False, simclr=False, shift=False):
        _aux = {}
        _return_aux = False

        features = self.penultimate(inputs)#这里是MLP最后一层
        #print("feature",features.shape)

        output = F.linear(features,self.weight3)#再跑一个线性层，变成分类任务

        if penultimate:
            _return_aux = True
            _aux['penultimate'] = features#这里跑的是没head的输出

        if simclr:
            _return_aux = True
            _aux['simclr'] = self.simclr_layer(features)#这里跑的是simclr，128

        if shift:
            _return_aux = True
            _aux['shift'] = self.shift_cls_layer(features)#这里是预测shift，4

        if _return_aux:
            return output[:,:self.out_num], _aux

        return output[:,:self.out_num]
