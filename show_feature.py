import torch
import torchvision.transforms as transforms
import skimage.data
import skimage.io
import skimage.transform
import numpy as np
import matplotlib
import pylab
import matplotlib.pyplot as plt
import os

import cv2
# from completion_segmentation_model import DepthCompletionFrontNet
# from completion_segmentation_model_v3_eca_attention import DepthCompletionFrontNet
import math
from models.yolo_gy import Model
import yaml
from utils.torch_utils import intersect_dicts




# https://blog.csdn.net/missyougoon/article/details/85645195
# https://blog.csdn.net/grayondream/article/details/99090247


# 定义是否使用GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 定义数据预处理方式(将输入的类似numpy中arrary形式的数据转化为pytorch中的张量（tensor）)
transform = transforms.ToTensor()


def get_picture(picture_dir, transform):
    '''
    该算法实现了读取图片，并将其类型转化为Tensor
    '''
    img = skimage.io.imread(picture_dir)
    img256 = skimage.transform.resize(img, (128, 256))
    img256 = np.asarray(img256)
    img256 = img256.astype(np.float32)

    return transform(img256)


def get_picture_rgb(picture_dir):
    '''
    该函数实现了显示图片的RGB三通道颜色
    '''
    img = skimage.io.imread(picture_dir)
    img256 = skimage.transform.resize(img, (256, 256))
    skimage.io.imsave('4.jpg', img256)

    # 取单一通道值显示
    # for i in range(3):
    #     img = img256[:,:,i]
    #     ax = plt.subplot(1, 3, i + 1)
    #     ax.set_title('Feature {}'.format(i))
    #     ax.axis('off')
    #     plt.imshow(img)

    # r = img256.copy()
    # r[:,:,0:2]=0
    # ax = plt.subplot(1, 4, 1)
    # ax.set_title('B Channel')
    # # ax.axis('off')
    # plt.imshow(r)

    # g = img256.copy()
    # g[:,:,0]=0
    # g[:,:,2]=0
    # ax = plt.subplot(1, 4, 2)
    # ax.set_title('G Channel')
    # # ax.axis('off')
    # plt.imshow(g)

    # b = img256.copy()
    # b[:,:,1:3]=0
    # ax = plt.subplot(1, 4, 3)
    # ax.set_title('R Channel')
    # # ax.axis('off')
    # plt.imshow(b)

    # img = img256.copy()
    # ax = plt.subplot(1, 4, 4)
    # ax.set_title('image')
    # # ax.axis('off')
    # plt.imshow(img)

    img = img256.copy()
    ax = plt.subplot()
    ax.set_title('image')
    # ax.axis('off')
    plt.imshow(img)

    plt.show()


def visualize_feature_map_sum(item, name, save_path):
    '''
    将每张子图进行相加
    :param feature_batch:
    :return:
    '''
    feature_map = item.squeeze(0)
    c = item.shape[1]
    print(feature_map.shape)
    feature_map_combination = []
    for i in range(0, c):
        feature_map_split = feature_map.data.cpu().numpy()[i, :, :]

        feature_map_combination.append(feature_map_split)

    feature_map_sum = sum(one for one in feature_map_combination)


    # feature_map = np.squeeze(feature_batch,axis=0)
    plt.figure()
    plt.title("combine figure")
    plt.imshow(feature_map_sum)
    # print(feature_map_sum.dtype)   # float32
    # exit()
    pylab.show()
    # exit()
    # plt.show()
     # <class 'module'>
    # exit()
    plt.savefig(save_path+'/feature_map_' + name + '.png')  # 保存图像到本地


def visualize_feature_map_sum_1(item, name, save_path, h_img):
    '''
    将每张子图进行相加
    :param feature_batch:
    :return:
    '''
    feature_map = item.squeeze(0)
    c = item.shape[1]
    # print(type(feature_map))
    print(feature_map.shape)


    feature_map_sum = 0
    for i in range(0, c):
        feature_map_sum = feature_map.data.cpu().numpy()[i, :, :] + feature_map_sum


    # heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # cv2.imwrite(save_path + '/cam.png', superimposed_img)

    plt.figure()
    plt.title("combine figure")

    heatmap = cv2.resize(feature_map_sum, (h_img.shape[1], h_img.shape[0]))
    Min = np.min(heatmap)
    Max = np.max(heatmap)
    heatmap = (heatmap - Min) / (Max - Min)

    heatmap = heatmap * 150

    # heatmap = np.uint8(255 * heatmap)
    # superimposed_img = np.sum(heatmap * 0.4,h_img)

    # print(h_img.shape[2])
    # exit()

    h_img[:,:,0] = h_img[:,:,0] + heatmap

    plt.imshow(h_img)
    plt.axis('off')
    plt.savefig(save_path + '/thermal_night_fam.png')  # 保存图像到本地





def get_feature():
    # 输入数据
    # root_path = '/home/ubuntu/dataset/kaist dataset/gy_copy/test/'
    # pic_dir = 'set06_V003_lwir_I02799.png'
    # pc_path = root_path + 'thermal/' + pic_dir
    # rgb_path = root_path + 'visible/' + pic_dir


    root_path = '/home/ubuntu/dataset/kaist_pixel_level/preparing_data/train/'
    pic_dir = 'set00_V004_I01225.png'
    pic_dir_1 = 'set00_V004_I01225.png'
    pic_dir_2 = 'set00_V004_I01225.png'

    pc_path = root_path + 'images/' + pic_dir_1
    rgb_path = root_path + 'RGB/' + pic_dir
    th_path = root_path + 'images/' + pic_dir_2
    save_path = '/home/ubuntu/code/yolov5-master-copy-copy/feature_map/nosaliency/'
    dir_name = pic_dir.split('.')[0]
    save_path = save_path+dir_name
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    img_rgb = get_picture(rgb_path, transform)
    # 插入维度
    img_rgb = img_rgb.unsqueeze(0)
    img_rgb = img_rgb.to(device)

    img_pc = get_picture(pc_path, transform)
    # 插入维度
    img_pc = img_pc.unsqueeze(0)
    img_pc = img_pc.to(device)

    # 加载模型
    ckpt = torch.load('/home/ubuntu/code/yolov5-master-copy-copy/runs/train/early_train/RGB+thermal+all-train+0.23779-0.24695-0.24365-0.27173*MBM/weights/best.pt')
    # ckpt = torch.load('/home/ubuntu/code/yolov5-master-copy-copy/runs/train/early_train/small_data/RGB+thermal+MBM-0.237-0.246-0.243-0.271-+saliency/weights/best.pt')
    cfg = ckpt['model'].yaml
    hyp = 'data/hyp.scratch.yaml'

    with open(hyp) as f:
        hyp = yaml.load(f, Loader=yaml.SafeLoader)


    resume = False
    model = Model(cfg, ch=3, nc=1, anchors=hyp.get('anchors')).to(device)  # create
    exclude = ['anchor'] if (cfg or hyp.get('anchors')) and not resume else []  # exclude keys
    state_dict = ckpt['model'].float().state_dict()  # to FP32
    state_dict = intersect_dicts(state_dict, model.state_dict(), exclude=exclude)  # intersect
    model.load_state_dict(state_dict, strict=False)  # load
    # print(model)


    # myexactor = FeatureExtractor(model, exact_list)
    img1 = {
        'thermal_img': img_pc, 'imgs': img_rgb
    }
    # print(img1['pc'])
    # x = myexactor(img1)
    result,gy_thermal_dict,gy_rgb_dict,gy_fused_dict = model(img1)
    outputs = []



    x_rgb = []
    x_thermal = []
    x_fused = []
    for item in gy_rgb_dict:
        # print(item)
        # continue
        x_rgb.append(gy_rgb_dict[item])



    for item in gy_thermal_dict:
        # print(item)
        # continue
        x_thermal.append(gy_thermal_dict[item])



    for item in gy_fused_dict:
        # print(item)
        # continue
        x_fused.append(gy_fused_dict[item])


    # 特征输出可视化

    outputs = [x_rgb,x_thermal,x_fused]
    namex = ['rgb','thermal','fused']


    # h_img = cv2.imread(rgb_path)
    # h_img = cv2.imread(pc_path)
    h_img = cv2.imread(th_path)

    for i in range(len(namex)):
        k = 0
        xname = namex[i]


        for item in outputs[i]:
            # c = item.shape[1]

            plt.figure()
            name = "{}{}".format(xname,k)

            name = str(name)
            plt.suptitle(name)


            item = x_fused[0]

            # for i in range(c):
            #     wid = math.ceil(math.sqrt(c))
            #     ax = plt.subplot(wid, wid, i + 1)
            #     ax.set_title('Feature {}'.format(i))
            #     ax.axis('off')
            #     figure_map = item.data.cpu().numpy()[0, i, :, :]
            #     plt.imshow(figure_map, cmap='jet')
            #     plt.savefig('/home/ubuntu/code/yolov5-master-copy-copy/feature_map/feature_map_' + name + '.png')  # 保存图像到本地
            visualize_feature_map_sum_1(item, name,save_path,h_img)
            k = k + 1
            exit()


    # plt.show()


# 训练
if __name__ == "__main__":
    # get_picture_rgb(pic_dir)
    get_feature()

