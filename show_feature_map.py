import torch
import torchvision.transforms as transforms
import skimage.data
import skimage.io
import skimage.transform
import numpy as np
import matplotlib.pyplot as plt
from completion_segmentation_model import DepthCompletionFrontNet
# from completion_segmentation_model_v3_eca_attention import DepthCompletionFrontNet
import math

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


def visualize_feature_map_sum(item, name):
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
    plt.savefig('E:/Dataset/qhms/feature_map/feature_map_sum_' + name + '.png')  # 保存图像到本地
    plt.show()


def get_feature():
    # 输入数据
    root_path = 'E:/Dataset/qhms/data/small_data/'
    pic_dir = 'test_umm_000067.png'
    pc_path = root_path + 'knn_pc_crop_0.6/' + pic_dir
    rgb_path = root_path + 'train_image_2_lane_crop_0.6/' + pic_dir

    img_rgb = get_picture(rgb_path, transform)
    # 插入维度
    img_rgb = img_rgb.unsqueeze(0)
    img_rgb = img_rgb.to(device)

    img_pc = get_picture(pc_path, transform)
    # 插入维度
    img_pc = img_pc.unsqueeze(0)
    img_pc = img_pc.to(device)

    # 加载模型
    checkpoint = torch.load('E:/Dataset/qhms/all_result/v3/crop_0.6_old/hah/checkpoint-195.pth.tar')
    args = checkpoint['args']
    print(args)
    model = DepthCompletionFrontNet(args)
    print(model.keys())
    model.load_state_dict(checkpoint['model'])
    model.to(device)

    exact_list = ["conv1", "conv2", "conv3", "conv4", "convt4", "convt3", "convt2_", "convt1_", "lane"]
    # myexactor = FeatureExtractor(model, exact_list)
    img1 = {
        'pc': img_pc, 'rgb': img_rgb
    }
    # print(img1['pc'])
    # x = myexactor(img1)
    result, all_dict = model(img1)
    outputs = []

    # 挑选exact_list的层
    for item in exact_list:
        x = all_dict[item]
        outputs.append(x)

    # 特征输出可视化
    x = outputs
    k = 0
    print(x[0].shape[1])
    for item in x:
        c = item.shape[1]

        plt.figure()
        name = exact_list[k]
        plt.suptitle(name)

        for i in range(c):
            wid = math.ceil(math.sqrt(c))
            ax = plt.subplot(wid, wid, i + 1)
            ax.set_title('Feature {}'.format(i))
            ax.axis('off')
            figure_map = item.data.cpu().numpy()[0, i, :, :]
            plt.imshow(figure_map, cmap='jet')
            plt.savefig('E:/Dataset/qhms/feature_map/feature_map_' + name + '.png')  # 保存图像到本地
        visualize_feature_map_sum(item, name)
        k = k + 1
    plt.show()


# 训练
if __name__ == "__main__":
    # get_picture_rgb(pic_dir)
    get_feature()
