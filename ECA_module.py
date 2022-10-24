import torch.nn as nn
import math
import torch

class Model(nn.Module):
    def __init__(self):
        super(Model,self).__init__()
        self.gy1 = nn.Conv2d(1,1,1)
        self.weight = nn.Parameter(torch.Tensor([1.35]),requires_grad=True)
        self.gy2 = nn.Conv2d(1, 1, 1)



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
    def forward(self,x):
        # print(x.shape)
        weight = self.weight
        return x*weight


if __name__ == '__main__':
    gy = Model()
    # print(gy)
    for k,v in gy.named_modules():
        print(v)

    for i in gy.named_parameters():
        print(i)



