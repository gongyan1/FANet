### For latest triplet_attention module code please refer to the corresponding file in root. 

import torch
import math
import torch.nn as nn
import torch.nn.functional as F



# conv + bn + relu
class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

        self.conv = self.conv.half()
        self.bn = self.bn.half()

    def forward(self, x):


        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


# Maxpooling 和 average pooling
class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )



#  channelpool + conv + sigmoid
class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        # 自适应padding
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)
    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = torch.sigmoid_(x_out) 
        return x * scale,scale






class TripletAttention(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False):
        super(TripletAttention, self).__init__()
        self.ChannelGateH = SpatialGate()
        self.ChannelGateW = SpatialGate()

        # self.a = nn.Parameter(torch.ones(4),requires_grad=True)



        self.no_spatial=no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate()

    def eca_layer(self, x, gamma=2, b=1):
        # eca-net
        # 原理：通过GPA（全局平均池化）转为1*1*C的向量，再通过1维conv进行权重更新
        N, C, H, W = x.size()
        t = int(abs((math.log(C, 2) + b) / gamma))
        k_size = t if t % 2 else t + 1
        # k_size = 3
        avg_pool_eca = nn.AdaptiveAvgPool2d(1).cuda().half()
        # conv1d_eca = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        conv1d_eca = nn.Conv1d(1, 1, kernel_size=k_size, padding=int(k_size // 2), bias=False).cuda().half()
        sigmoid_eca = nn.Sigmoid().cuda().half()
        y = avg_pool_eca(x)
        # print(x)
        # print(y)
        # y = y.cpu()
        # print(type(y))
        y = conv1d_eca(y.squeeze(-1).transpose(-1, -2))
        # print("dasdasada")
        y = y.transpose(-1, -2).unsqueeze(-1)
        y = sigmoid_eca(y)
        # y = y.cuda()
        # 将 y 变成和 x 一样的 shape
        return x * y.expand_as(x)



    def eca_layer_sort(self, x, gamma=2, b=1):
        # eca-net
        # 原理：通过GPA（全局平均池化）转为1*1*C的向量，再通过1维conv进行权重更新


        N, C, H, W = x.size()
        t = int(abs((math.log(C, 2) + b) / gamma))
        k_size = t if t % 2 else t + 1
        # k_size = 3
        avg_pool_eca = nn.AdaptiveAvgPool2d(1).cuda().half()
        # conv1d_eca = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        conv1d_eca = nn.Conv1d(1, 1, kernel_size=k_size, padding=int(k_size // 2), bias=False).cuda().half()
        sigmoid_eca = nn.Sigmoid().cuda().half()

        # 通过全局平均池化
        y = avg_pool_eca(x)     # torch.Size([16, 128, 1, 1])

        # 按绝对值进行排序
        # y_abs = y.abs()
        # sorted, indices = torch.sort(y_abs, dim=1, descending=True)
        # new_y = y.gather(dim=1, index=indices)
        # y = new_y


        # 随即排序channel
        # perm = torch.randperm(C)
        # y = y[:, perm]

        # 按正、负排序
        sorted, indices = torch.sort(y, dim=1, descending=True)
        b, rank = torch.sort(indices, dim=1)


        y = sorted


        # print(x)
        # print(y)
        # y = y.cpu()
        # print(type(y))
        y = conv1d_eca(y.squeeze(-1).transpose(-1, -2))
        # print("dasdasada")
        y = y.transpose(-1, -2).unsqueeze(-1)
        y = sigmoid_eca(y)
        # y = y.cuda()
        # 将 y 变成和 x 一样的 shape


        # 排序回去
        y_c = torch.clone(y)
        len = x.size()[0]
        # 按 batch循环
        for i in range(len):
            ranki = rank[i, :, :, :].squeeze()
            # print(ranki)
            y_c[i, :, :, :] = y[i, :, :, :].index_select(0, ranki)
        y = y_c

        return x * y.expand_as(x)




    def forward(self, x ,weight):
        # gamma 和 b 仍是超参
        #  0 1 2 3    b c w h

        # 模仿ECA,确定卷积核的大小
        # B,C,W,H = x.size()
        # t = int(abs((math.log(C, 2) + b) / gamma))
        # k_size = t if t % 2 else t + 1
        # t_w = int(abs((math.log(W,2))+b) / gamma)
        # k_w_size = t_w if t_w% 2 else t_w+1
        # print(k_size,k_w_size)   # 5,3

        # x = self.eca_layer(x)


        #  0 2 1 3    b w c h  （建立c与h之间）
        x_perm1 = x.permute(0,2,1,3).contiguous()
        x_out1,scale_hc = self.ChannelGateH(x_perm1)
        # 0 2 1 3   b c w h
        x_out11 = x_out1.permute(0,2,1,3).contiguous()

        # 0 3 2 1     b h w c  （建立w与c之间的联系）
        x_perm2 = x.permute(0,3,2,1).contiguous()
        x_out2,scale_wc = self.ChannelGateW(x_perm2)
        # 0 3 2 1     b c w h
        x_out21 = x_out2.permute(0,3,2,1).contiguous()


        # 尝试对传入的x进行重要性排序

        x_out31 = self.eca_layer_sort(x)
        # x_out31 = self.eca_layer(x)



        if not self.no_spatial:
            x_out,scale_wh = self.SpatialGate(x)


            # b = F.softmax(weight,0)
            # print(b)
            # x_out = (b[0]*x_out + b[1]*x_out11 + b[2]*x_out21 +b[3]*x_out31)



            # fixed_weight
            # x_out = (0.22119 * x_out + 0.24731 * x_out11 + 0.23474 * x_out21 + 0.29688 * x_out31)

            # fixed_weight    better
            x_out = (0.23779 * x_out + 0.24695 * x_out11 + 0.24365 * x_out21 +  0.27173* x_out31)




            # big-data MBM
            # x_out = (0.07251 * x_out + 0.07874 * x_out11 + 0.05896 * x_out21 + 0.78955 * x_out31)
            # x_out = (1/4 * x_out + 1/4 * x_out11 + 1/4 * x_out21 + 1/4 * x_out31)


            # [0.07251, 0.07874, 0.05896, 0.78955]


            # x_out = (1/3)*(x_out + x_out11 + x_out21)
            # print(self.weight)


            # x_out =  (self.weight[0]*x_out + self.weight[1]*x_out11 + self.weight[2]*x_out21 + self.weight[3]*x_out31)



        else:
            x_out = (1/2)*(x_out11 + x_out21)
            # exit()
        return x_out


if __name__ == '__main__':
    Triplet = TripletAttention(128)
    for i in Triplet.named_parameters():
        print(i)