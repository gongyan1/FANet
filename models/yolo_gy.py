# YOLOv5 YOLO-specific modules

import torch.nn as nn
import argparse
import logging
import sys
from copy import deepcopy






sys.path.append('../')
sys.path.append('./')  # to run '$ python *.py' files in subdirectories
logger = logging.getLogger(__name__)
from models.common import *
from models.experimental import *
from utils.autoanchor import check_anchor_order
from utils.general import make_divisible, check_file, set_logging
from utils.torch_utils import time_synchronized, fuse_conv_and_bn, model_info, scale_img, initialize_weights, \
    select_device, copy_attr

try:
    import thop  # for FLOPS computation
except ImportError:
    thop = None
from models.triplet_attention_gy import *
from models.fca_layer import FcaLayer as FCA

class Detect(nn.Module):
    stride = None  # strides computed during build
    export = False  # onnx export

    def __init__(self, nc=80, anchors=(), ch=()):  # detection layer
        super(Detect, self).__init__()
        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.grid = [torch.zeros(1)] * self.nl  # init grid
        a = torch.tensor(anchors).float().view(self.nl, -1, 2)
        self.register_buffer('anchors', a)  # shape(nl,na,2)
        self.register_buffer('anchor_grid', a.clone().view(self.nl, 1, -1, 1, 1, 2))  # shape(nl,1,na,1,1,2)
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv

    def forward(self, x):
        # x = x.copy()  # for profiling
        z = []  # inference output
        self.training |= self.export
        for i in range(self.nl):
            x[i] = self.m[i](x[i])  # conv
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if not self.training:  # inference
                if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i] = self._make_grid(nx, ny).to(x[i].device)

                y = x[i].sigmoid()
                y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[i]  # xy
                y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                z.append(y.view(bs, -1, self.no))

        return x if self.training else (torch.cat(z, 1), x)

    @staticmethod
    def _make_grid(nx=20, ny=20):
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()



# class eca_layer(nn.Module):
#     """Constructs a ECA module.
#     Args:
#         channel: Number of channels of the input feature map
#         k_size: Adaptive selection of kernel size
#     """
#     def __init__(self, k_size=3):
#         super(eca_layer, self).__init__()
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, x ,gamma=2, b=1):
#         # x: input features with shape [b, c, h, w]
#         # b, c, h, w = x.size()
#         # t = int(abs((math.log(c, 2) + b) / gamma))
#         # k_size = t if t % 2 else t + 1
#
#         # feature descriptor on the global spatial information
#         y = self.avg_pool(x)
#
#         # Two different branches of ECA module
#         y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
#
#         # Multi-scale information fusion
#         y = self.sigmoid(y)
#
#         return x * y.expand_as(x)




## CBAM
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1).half()
        self.max_pool = nn.AdaptiveMaxPool2d(1).half()

        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // 16, 1, bias=False).half(),
                                nn.ReLU().half(),
                                nn.Conv2d(in_planes // 16, in_planes, 1, bias=False).half())
        self.sigmoid = nn.Sigmoid().half()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False).half()
        self.sigmoid = nn.Sigmoid().half()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)




class Model(nn.Module):
    def __init__(self, cfg='yolov5s.yaml', ch=3, nc=None, anchors=None):  # model, input channels, number of classes
        super(Model, self).__init__()

        if isinstance(cfg, dict):
            self.yaml = cfg  # model dict
        else:  # is *.yaml
            import yaml  # for torch hub
            self.yaml_file = Path(cfg).name
            # print(cfg)   # models/yolov5s_gy.yaml
            with open(cfg) as f:
                self.yaml = yaml.load(f, Loader=yaml.SafeLoader)  # model dict

        # Define model
        ch = self.yaml['ch'] = self.yaml.get('ch', ch)  # input channels
        if nc and nc != self.yaml['nc']:
            logger.info(f"Overriding model.yaml nc={self.yaml['nc']} with nc={nc}")
            self.yaml['nc'] = nc  # override yaml value
        if anchors:
            logger.info(f'Overriding model.yaml anchors with anchors={anchors}')
            self.yaml['anchors'] = round(anchors)  # override yaml value
        self.model, self.save = parse_model(deepcopy(self.yaml), ch=[ch])  # model, savelist
        self.save.append(3)

        self.names = [str(i) for i in range(self.yaml['nc'])]  # default names

        self.weight = nn.Parameter(torch.ones(4), requires_grad=True)
        # print(self.w1.requires_grad)
        # exit()


        # print([x.shape for x in self.forward(torch.zeros(1, ch, 64, 64))])


        # self.conv1x1 = nn.Conv2d(128,64,1)
        # self.BN = nn.BatchNorm2d(64)
        # self.SLU = nn.SiLU()


        # Build strides, anchors
        m = self.model[-1]  # Detect()

        # print(m)
        # Detect(
        #     (m): ModuleList(
        #       (0): Conv2d(128, 18, kernel_size=(1, 1), stride=(1, 1))
        #       (1): Conv2d(256, 18, kernel_size=(1, 1), stride=(1, 1))
        #       (2): Conv2d(512, 18, kernel_size=(1, 1), stride=(1, 1))
        # )
        # )

        if isinstance(m, Detect):
            s = 256  # 2x min stride
            # m.stride = torch.tensor([s / x.shape[-2] for x in self.forward(torch.zeros(1, ch, s, s),Tag=True)])  # forward
            # tensor([8., 16., 32.])
            m.stride = torch.tensor([8., 16., 32.])   # yolov5s



            m.anchors /= m.stride.view(-1, 1, 1)
            check_anchor_order(m)
            self.stride = m.stride
            self._initialize_biases()  # only run once
            # print('Strides: %s' % m.stride.tolist())

        # Init weights, biases
        initialize_weights(self)
        self.info()
        logger.info('')

    def SENet(self,x):
        N, C, H, W = x.size()
        filter3 = C
        se = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)).half(),
            nn.Conv2d(filter3,filter3//16,kernel_size=1).half(),
            nn.ReLU().half(),
            nn.Conv2d(filter3//16,filter3,kernel_size=1).half(),
            nn.Sigmoid().half()
        )
        se = se.cuda()
        y = se(x)

        return x * y.expand_as(x)

    def FCA(self,x):
        N, C, H, W = x.size()
        planes = C

        reduction = 16
        resolution = 32
        self.se = FCA(planes, reduction, resolution, resolution).half().cuda()

        x = self.se(x)

        return x






    def CBAM(self,x):
        N, C, H, W = x.size()
        planes = C

        ca = ChannelAttention(planes).cuda()
        sa = SpatialAttention().cuda()



        x = ca(x) * x
        x = sa(x) * x


        return x




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
        y = avg_pool_eca(x)  # torch.Size([16, 128, 1, 1])

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

    def forward(self, x, augment=False, profile=False,Tag=False):
        if not Tag:
            imgs = x['imgs']
            thermal_img = x['thermal_img']
            # print(imgs.shape,thermal_img.shape)   # torch.Size([16, 3, 640, 640]) torch.Size([16, 3, 640, 640])
            # exit()
        if augment:
            if not Tag:
                x = imgs
                # x_t = thermal_img
            img_size = x.shape[-2:]  # height, width
            s = [1, 0.83, 0.67]  # scales
            f = [None, 3, None]  # flips (2-ud, 3-lr)
            y = []  # outputs
            for si, fi in zip(s, f):
                xi = scale_img(x.flip(fi) if fi else x, si, gs=int(self.stride.max()))
                print(xi.shape)
                yi = self.forward_once(xi)[0]  # forward
                print(yi.shape)
                exit()
                yi[..., :4] /= si  # de-scale
                if fi == 2:
                    yi[..., 1] = img_size[0] - yi[..., 1]  # de-flip ud
                elif fi == 3:
                    yi[..., 0] = img_size[1] - yi[..., 0]  # de-flip lr
                y.append(yi)

                # if not Tag:
                #     xi_t = scale_img(x_t.flip(fi) if fi else x_t, si, gs=int(self.stride.max()))
                #     yi_t = self.forward_once(xi_t)[0]
                #     yi_t[..., :4] /= si  # de-scale
                #     # cv2.imwrite(f'img_{si}.jpg', 255 * xi[0].cpu().numpy().transpose((1, 2, 0))[:, :, ::-1])  # save
                #     if fi == 2:
                #         yi_t[..., 1] = img_size[0] - yi_t[..., 1]  # de-flip ud
                #     elif fi == 3:
                #         yi_t[..., 0] = img_size[1] - yi_t[..., 0]  # de-flip lr
                #     y.append(yi_t)
                #     print("augment,may be have errors!")
            return torch.cat(y, 1), None  # augmented inference, train
        else:
            return self.forward_once(x, profile,Tag)  # single-scale inference, train

    def forward_once(self, x, profile=False,Tag=False):


        # 特征图可视化
        show = False

        if show:
            gy_thermal_dict = {}
            gy_rgb_dict ={}
            gy_fused_dict = {}

        if not Tag:
            imgs = x['imgs']
            thermal_img = x['thermal_img']
            x = imgs
            x_t = thermal_img
        y, dt = [], []  # outputs
        # y_t = []

        for m in self.model:
            # print(type(m))  # f, n, m, args


            # x中放多个值
            # [-1, 6],2     [-1, 4],2     [-1, 14],2      [-1, 10],2      [17, 20, 23],3
            # cat 操作(当前特征图和按序排好的第x个特征图)
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers

            if profile:
                o = thop.profile(m, inputs=(x,), verbose=False)[0] / 1E9 * 2 if thop else 0  # FLOPS
                t = time_synchronized()
                for _ in range(10):
                    _ = m(x)
                dt.append((time_synchronized() - t) * 100)
                print('%10.1f%10.0f%10.1fms %-40s' % (o, m.np, dt[-1], m.type))

            # print(m.i,type(m))    #  detect:Tag = True    一般：Tag = Fasle
            # if not Tag:
            #     if m.i <= 3:
            #         x_t = m(x_t)
            #         y_t.append(x_t if m.i in self.save else None)
            #     if m.i == 4:
            #         # print(type(m))
            #         x = m(x,x_t)
            #         y.append(x if m.i in self.save else None)
            #         continue


            if m.i <= 3:
                # print(m.i, type(m))
                x, x_t = m(x, x_t)  # run

                if show:
                    gy_rgb_dict[m.i] = x
                    gy_thermal_dict[m.i] = x_t


            elif m.i == 4:
                x = m(x, x_t)
                # x = torch.tensor(x,dtype=torch.float)
                # print(x.dtype)
                if show:
                    gy_fused_dict['4-1'] = x

                # MBM
                N, C, H, W = x.size()

                x = x.half()
                weight = torch.softmax(self.weight,0)
                triplet_attention = TripletAttention(C, 16).cuda()
                x = triplet_attention(x,weight)
                # x = x.float()

                if show:
                    gy_fused_dict['4-2'] = x

                # ECA_layer
                # x = self.eca_layer(x)

                # ECA_layer_sort
                # x = self.eca_layer_sort(x)

                # SENet
                # x = self.SENet(x)

                # CBAM
                # x = self.CBAM(x)

                # FCA
                # x = self.FCA(x)

            else:
                if show:
                    gy_fused_dict[m.i] = x
                x = m(x)


            # x = m(x)
            y.append(x if m.i in self.save else None)  # save output


        # exit()

            # if not Tag:
            #     if m.i <=3:
            #         x_t = m(x_t)
            #     y_t.append(x_t if m.i in self.save else None)
                # if m.i == 4:
                #     if y[-1] is not None and y_t[-1] is not None:
                #         # y_out = torch.cat((y[-1],y_t[-1]),1)
                #         # y_out = self.conv1x1(y_out)
                #         # y_out = self.BN(y_out)
                #         # y_out = self.SLU(y_out)
                #         # y[-1] = y_out
                #         print(m.i , type(m))
                #         # print(y[-1].shape, y_t[-1].shape, y_out.shape)
                #         exit()
                # else:
                #     print("y or y_t is none!")
                #     exit()


        if profile:
            print('%.1fms total' % sum(dt))

        if show:
            return x, gy_thermal_dict,gy_rgb_dict,gy_fused_dict

        return x

    def _initialize_biases(self, cf=None):  # initialize biases into Detect(), cf is class frequency
        # https://arxiv.org/abs/1708.02002 section 3.3
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.
        m = self.model[-1]  # Detect() module
        for mi, s in zip(m.m, m.stride):  # from
            b = mi.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
            b.data[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
            b.data[:, 5:] += math.log(0.6 / (m.nc - 0.99)) if cf is None else torch.log(cf / cf.sum())  # cls
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def _print_biases(self):
        m = self.model[-1]  # Detect() module
        for mi in m.m:  # from
            b = mi.bias.detach().view(m.na, -1).T  # conv.bias(255) to (3,85)
            print(('%6g Conv2d.bias:' + '%10.3g' * 6) % (mi.weight.shape[1], *b[:5].mean(1).tolist(), b[5:].mean()))

    # def _print_weights(self):
    #     for m in self.model.modules():
    #         if type(m) is Bottleneck:
    #             print('%10.3g' % (m.w.detach().sigmoid() * 2))  # shortcut weights

    def fuse(self):  # fuse model Conv2d() + BatchNorm2d() layers
        print('Fusing layers... ')
        for m in self.model.modules():
            if type(m) is Conv and hasattr(m, 'bn'):
                m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
                delattr(m, 'bn')  # remove batchnorm
                m.forward = m.fuseforward  # update forward
        self.info()
        return self

    def nms(self, mode=True):  # add or remove NMS module
        present = type(self.model[-1]) is NMS  # last layer is NMS
        if mode and not present:
            print('Adding NMS... ')
            m = NMS()  # module
            m.f = -1  # from
            m.i = self.model[-1].i + 1  # index
            self.model.add_module(name='%s' % m.i, module=m)  # add
            self.eval()
        elif not mode and present:
            print('Removing NMS... ')
            self.model = self.model[:-1]  # remove
        return self

    def autoshape(self):  # add autoShape module
        print('Adding autoShape... ')
        m = autoShape(self)  # wrap model
        copy_attr(m, self, include=('yaml', 'nc', 'hyp', 'names', 'stride'), exclude=())  # copy attributes
        return m

    def info(self, verbose=False, img_size=640):  # print model information
        model_info(self, verbose, img_size)


def parse_model(d, ch):  # model_dict, input_channels(3)
    logger.info('\n%3s%18s%3s%10s  %-40s%-30s' % ('', 'from', 'n', 'params', 'module', 'arguments'))
    anchors, nc, gd, gw = d['anchors'], d['nc'], d['depth_multiple'], d['width_multiple']
    na = (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors  # number of anchors
    no = na * (nc + 5)  # number of outputs = anchors * (classes + 5)


    layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out
    for i, (f, n, m, args) in enumerate(d['backbone'] + d['head']):  # from, number, module, args
        m = eval(m) if isinstance(m, str) else m  # eval strings
        for j, a in enumerate(args):
            try:
                args[j] = eval(a) if isinstance(a, str) else a  # eval strings
            except:
                pass

        n = max(round(n * gd), 1) if n > 1 else n  # depth gain
        if m in [Conv,Conv_t, GhostConv, Bottleneck, GhostBottleneck, SPP, DWConv, MixConv2d, Focus,Focus_t, CrossConv, BottleneckCSP,
                 C3,C3_t,C4]:
            c1, c2 = ch[f], args[0]
            if c2 != no:  # if not output
                c2 = make_divisible(c2 * gw, 8)

            args = [c1, c2, *args[1:]]
            if m in [BottleneckCSP, C3,C3_t,C4]:
                args.insert(2, n)  # number of repeats
                n = 1
        elif m is nn.BatchNorm2d:
            args = [ch[f]]
        elif m is Concat:
            c2 = sum([ch[x] for x in f])
        elif m is Detect:
            args.append([ch[x] for x in f])
            if isinstance(args[1], int):  # number of anchors
                args[1] = [list(range(args[1] * 2))] * len(f)
        elif m is Contract:
            c2 = ch[f] * args[0] ** 2
        elif m is Expand:
            c2 = ch[f] // args[0] ** 2
        else:
            c2 = ch[f]

        m_ = nn.Sequential(*[m(*args) for _ in range(n)]) if n > 1 else m(*args)  # module
        t = str(m)[8:-2].replace('__main__.', '')  # module type
        np = sum([x.numel() for x in m_.parameters()])  # number params
        m_.i, m_.f, m_.type, m_.np = i, f, t, np  # attach index, 'from' index, type, number params
        logger.info('%3s%18s%3s%10.0f  %-40s%-30s' % (i, f, n, np, t, args))  # print
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
        layers.append(m_)
        if i == 0:
            ch = []
        ch.append(c2)
    # print(ch)   # [32, 64, 64, 128, 128, 256, 256, 512, 512, 512, 256, 256, 512, 256, 128, 128, 256, 128, 128, 256, 256, 256, 512, 512, 512]
    return nn.Sequential(*layers), sorted(save)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='yolov5s.yaml', help='model.yaml')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    opt = parser.parse_args()
    opt.cfg = check_file(opt.cfg)  # check file
    set_logging()
    device = select_device(opt.device)

    # Create model
    model = Model(opt.cfg).to(device)

    # model.train()

    print(model)














    # Profile
    # img = torch.rand(8 if torch.cuda.is_available() else 1, 3, 640, 640).to(device)
    # y = model(img, profile=True)

    # Tensorboard
    # from torch.utils.tensorboard import SummaryWriter
    # tb_writer = SummaryWriter()
    # print("Run 'tensorboard --logdir=models/runs' to view tensorboard at http://localhost:6006/")
    # tb_writer.add_graph(model.model, img)  # add model to tensorboard
    # tb_writer.add_image('test', img[0], dataformats='CWH')  # add model to tensorboard
