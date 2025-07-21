#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import print_function

import torch.nn as nn
import torch
from torch.nn.parameter import Parameter
import math
import numpy as np
import matplotlib.pyplot as plt

class GraphConvolution(nn.Module):
    """
    adapted from : https://github.com/tkipf/gcn/blob/92600c39797c2bfb61a508e52b88fb554df30177/gcn/layers.py#L132
    """

    def __init__(self, in_features, out_features, bias=True, node_n=48, times = 1):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        # self.att = Parameter(torch.FloatTensor(node_n, node_n))        
        self.att = nn.Parameter(torch.FloatTensor(0.01 + 0.99 * np.eye(node_n)[np.newaxis, ...]))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.times = times
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        # self.weight.data.uniform_(-stdv, stdv)
        # self.att.data.uniform_(-stdv, stdv)
        torch.nn.init.xavier_normal_(self.weight)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
    def show_A(self):
        att_numpy = self.att.cpu().detach().numpy()
        
        D = np.diag(np.sum(att_numpy[0], axis=1))

        # 计算D的-1/2次方
        D_half_inv = np.power(D, -0.5)
        D_half_inv[np.isinf(D_half_inv)] = 0.


        # 计算对称归一化的邻接矩阵
        A_hat = D_half_inv @ att_numpy[0] @ D_half_inv
        # print(att_numpy[0])
        # 使用 matplotlib 的 imshow 函数创建热图
        plt.imshow(A_hat, cmap='hot', interpolation='nearest')

        # 显示热图
        # plt.show()

        # 如果你想保存热图，可以使用 plt.savefig 函数
        plt.savefig('heatmap'+str(self.times)+'.png')
    def forward(self, input):
        support = torch.matmul(input, self.weight)
        output = torch.matmul(self.att, support)
        # self.show_A()
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GC_Block(nn.Module):
    def __init__(self, in_features, p_dropout, bias=True, node_n=48):
        """
        Define a residual block of GCN
        """
        super(GC_Block, self).__init__()
        self.in_features = in_features
        self.out_features = in_features

        self.gc1 = GraphConvolution(in_features, in_features, node_n=node_n, bias=bias, times = 1)
        self.bn1 = nn.BatchNorm1d(node_n * in_features)

        self.gc2 = GraphConvolution(in_features, in_features, node_n=node_n, bias=bias, times = 2)
        self.bn2 = nn.BatchNorm1d(node_n * in_features)

        self.do = nn.Dropout(p_dropout)
        self.act_f = nn.Tanh()

    def forward(self, x):
        y = self.gc1(x)
        b, n, f = y.shape
        y = self.bn1(y.view(b, -1)).view(b, n, f)
        y = self.act_f(y)
        y = self.do(y)

        y = self.gc2(y)
        b, n, f = y.shape
        y = self.bn2(y.view(b, -1)).view(b, n, f)
        y = self.act_f(y)
        y = self.do(y)

        return y + x

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GCN(nn.Module):
    def __init__(self, input_feature, hidden_feature, p_dropout, num_stage=1, node_n=48):
        """

        :param input_feature: num of input feature
        :param hidden_feature: num of hidden feature
        :param p_dropout: drop out prob.
        :param num_stage: number of residual blocks
        :param node_n: number of nodes in graph
        """
        super(GCN, self).__init__()
        self.num_stage = num_stage

        self.gc1 = GraphConvolution(input_feature, hidden_feature, node_n=node_n)
        self.bn1 = nn.BatchNorm1d(node_n * hidden_feature)

        self.gcbs = []
        for i in range(num_stage):
            self.gcbs.append(GC_Block(hidden_feature, p_dropout=p_dropout, node_n=node_n))

        self.gcbs = nn.ModuleList(self.gcbs)

        self.gc7 = GraphConvolution(hidden_feature, input_feature, node_n=node_n)

        self.do = nn.Dropout(p_dropout)
        self.act_f = nn.Tanh()

    def forward(self, x):
        y = self.gc1(x)
        b, n, f = y.shape
        y = self.bn1(y.view(b, -1)).view(b, n, f)
        y = self.act_f(y)
        y = self.do(y)

        for i in range(self.num_stage):
            y = self.gcbs[i](y)

        y = self.gc7(y)
        y = y + x

        return y
if __name__ == '__main__':
    
    GCN = GCN(1024,512, p_dropout = 0.2, num_stage=2, node_n=8)
    x = torch.randn(12,8,1024)
    output = GCN(x)
    print(x.shape)
