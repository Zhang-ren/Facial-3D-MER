import torch.nn as nn
import torch.nn.functional as F
from pointnet2_utils import PointNetSetAbstraction, Trans2feat
import numpy as np
import torch
import matplotlib.pyplot as plt
from GCN import GCN
class TransformerModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, num_heads, dropout):
        super(TransformerModel, self).__init__()
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(input_dim, num_heads, hidden_dim, dropout),
            num_layers)
        # b, c, n -> b, 1, n
        self.avg_pool = nn.AdaptiveAvgPool1d(1) #
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x = self.transformer_encoder(x)
        # x = self.avg_pool(x.permute(0, 2, 1))
        # x = x.permute(0, 2, 1)
        # x = x.squeeze()
        x = x.mean(dim=1)
        x = self.fc(x)
        return x
from point_transformer_pytorch import PointTransformerLayer
class get_model(nn.Module):
    def __init__(self,num_class,normal_channel=True):
        super(get_model, self).__init__()
        in_channel = 6 if normal_channel else 3
        self.normal_channel = normal_channel
        self.sa1 = PointNetSetAbstraction(npoint=512, radius=1, nsample=32, in_channel=in_channel, mlp=[64, 64, 128], group_all=False)
        self.sa2 = PointNetSetAbstraction(npoint=128, radius=0.4, nsample=64, in_channel=128 + 3, mlp=[128, 128, 256], group_all=False)
        self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=256 + 3, mlp=[256, 512, 1024], group_all=True)
        self.t2f1 = Trans2feat(in_channel=in_channel, mlp=[64, 64, 128], npoint=512, group_all=False,first_layer=True)
        self.t2f2 = Trans2feat(in_channel=128+3, mlp=[128, 128, 256], npoint=128, group_all=False)
        self.t2f3 = Trans2feat(in_channel=256+3, mlp=[256, 512, 1024], npoint=None, group_all=True)
        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.2)
        self.fc3 = nn.Linear(512, num_class)
        self.transformer = TransformerModel(input_dim=1024, hidden_dim=256, output_dim=7, num_layers=2, num_heads=4, dropout=0.3)
        self.GCN1 = GraphConvolution(1024,1024, node_n=8,times = 2)
        self.GCN2 = GraphConvolution(1024,1024, node_n=8,times = 3)
        self.GCN3 = GraphConvolution(1024,1024, node_n=8,times = 4)
        self.GCN4 = GraphConvolution(1024,1024, node_n=8,times = 5)
        self.GCN = GCN(1024,512, p_dropout = 0.2, num_stage=2, node_n=8)
        self.attn = PointTransformerLayer(
                dim = 256,
                pos_mlp_hidden_dim = 256,
                attn_mlp_hidden_mult = 4,
                num_neighbors = 16          # only the 16 nearest neighbors would be attended to for each point
            )

        

    def forward(self, xyz):
        B, _, _ = xyz.shape
        if self.normal_channel:
            norm = xyz[:, 3:, :]
            xyz = xyz[:, :3, :]
        else:
            norm = None
        l1_xyz, l1_points, l1_color = self.t2f1(xyz,norm,norm) #([B, D, L, N])
        l2_xyz, l2_points, l2_color = self.t2f2(l1_xyz, l1_points, l1_color)
        
        # l2_x = l2_xyz.permute(0, 2, 3, 1)
        # l2_x = l2_x.reshape(l2_x.shape[0], -1 , l2_x.shape[3])
        # l2_p = l2_points.permute(0, 2, 3, 1)
        # l2_p1 = l2_p.reshape(l2_p.shape[0], -1 , l2_p.shape[3])
        # l2_points = self.attn(l2_p1, l2_x, None)
        # l2_points = l2_points.reshape(l2_p.shape[0], l2_p.shape[1], l2_p.shape[2], l2_p.shape[3])
        # l2_points = l2_points.permute(0, 3, 1, 2)
        
        l3_xyz, l3_points, l3_color = self.t2f3(l2_xyz, l2_points, l2_color)
        x0 = l3_points.permute(0, 2, 1)
        # x = self.GCN(x0)
        # x = self.GCN1(x)
        # x1 = self.GCN2(x)+x0
        # x = self.GCN3(x1)
        # x = self.GCN4(x)+x1
        x = self.GCN(x0)
        x = x.mean(dim=1)
        # x = self.transformer(x)


        # import pdb; pdb.set_trace()
        # l1_xyz, l1_points = self.sa1(xyz, norm)
        # l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        # l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        # x = l3_points.view(B, 1024)
        x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        # x = self.drop2(F.relu(self.bn2(self.fc2(x))))
        x = self.fc3(x)
        #(B,1024)
        x = F.log_softmax(x, -1)
        # print(x.shape)


        return x, l3_points
class GraphConvolution(nn.Module):
    """
    adapted from : https://github.com/tkipf/gcn/blob/92600c39797c2bfb61a508e52b88fb554df30177/gcn/layers.py#L132
    """

    def __init__(self, in_features, out_features, bias=True, node_n=48,times = 1):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        # self.att = Parameter(torch.FloatTensor(node_n, node_n))
        self.att = nn.Parameter(torch.FloatTensor(0.01 + 0.99 * np.eye(node_n)[np.newaxis, ...]))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.times = times
        self.reset_parameters()
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
    def reset_parameters(self):
        # stdv = 1. / math.sqrt(self.weight.size(1))
        # self.weight.data.uniform_(-stdv, stdv)
        # self.att.data.uniform_(-stdv, stdv)
        torch.nn.init.xavier_normal_(self.weight)
        if self.bias is not None:
            # self.bias.data.uniform_(-stdv, stdv)
            self.bias.data.zero_()

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


class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()

    def forward(self, pred, target, trans_feat):
        total_loss = F.nll_loss(pred, target)

        return total_loss
if __name__ == '__main__':
    input1 = torch.rand(32, 8, 1024)
    model = GraphConvolution(1024,1024, node_n=8)
    output = model(input1)
    print(output.shape)
