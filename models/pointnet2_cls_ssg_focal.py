import torch.nn as nn
import torch.nn.functional as F
from pointnet2_utils import PointNetSetAbstraction
from FocalLoss import FocalLoss
import torch
import torch.nn as nn
class get_model(nn.Module):
    def __init__(self,num_class,normal_channel=True):
        super(get_model, self).__init__()
        in_channel = 6 if normal_channel else 3
        self.normal_channel = normal_channel
        self.sa1 = PointNetSetAbstraction(npoint=512, radius=0.2, nsample=32, in_channel=in_channel, mlp=[64, 64, 128], group_all=False)
        self.sa2 = PointNetSetAbstraction(npoint=128, radius=0.4, nsample=64, in_channel=128 + 3, mlp=[128, 128, 256], group_all=False)
        self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=256 + 3, mlp=[256, 512, 1024], group_all=True)
        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.4)
        self.fc3 = nn.Linear(256, num_class)

    def forward(self, xyz):
        B, _, _ = xyz.shape
        if self.normal_channel:
            norm = xyz[:, 3:, :]
            xyz = xyz[:, :3, :]
        else:
            norm = None
        l1_xyz, l1_points = self.sa1(xyz, norm)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        x = l3_points.view(B, 1024)
        x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        x = self.drop2(F.relu(self.bn2(self.fc2(x))))
        x = self.fc3(x)
        x = F.log_softmax(x, -1)


        return x, l3_points
# class FocalLoss(nn.Module):
#     def __init__(self, gamma=2, weight=None, reduction='mean'):
#         super(FocalLoss, self).__init__()
#         self.gamma = gamma
#         self.weight = weight
#         self.reduction = reduction

#     def forward(self, output, target):
#         # convert output to presudo probability
#         out_target = torch.stack([output[i, t] for i, t in enumerate(target)])
#         probs = torch.sigmoid(out_target)
#         focal_weight = torch.pow(1-probs, self.gamma)

#         # add focal weight to cross entropy
#         ce_loss = F.cross_entropy(output, target, weight=self.weight, reduction='none')
#         focal_loss = focal_weight * ce_loss

#         if self.reduction == 'mean':
#             focal_loss = (focal_loss/focal_weight.sum()).sum()
#         elif self.reduction == 'sum':
#             focal_loss = focal_loss.sum()

#         return focal_loss


class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()
        # self.weights = [0.05, 0.08, 0.15 ,0.07, 0.2, 0.15, 0.3]
        self.weights = [0.25, 0.3, 0.1 ,0.15, 0.05, 0.1, 0.05]
        self.class_weights = torch.FloatTensor(self.weights)
        self.class_weights = self.class_weights.cuda()


    def forward(self, pred, target, trans_feat):
        # total_loss = F.nll_loss(pred, target)
        # preds = pred.data.max(1)[1]
        # preds = np.zeros((len(pred[:,0]),1))
        # for i in range(len(pred[:,0])):
        #     preds[i,0] = np.argmax(pred[i,:])
        # print(preds.shape,preds)
        loss = FocalLoss()
        # loss = nn.CrossEntropyLoss(weight = self.class_weights)
        # loss = FocalLoss()
        total_loss = loss(pred, target)
        return total_loss