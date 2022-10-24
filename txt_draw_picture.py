import os
import os.path
import xml.etree.cElementTree as ET
import cv2
import numpy as np


image_path = '/home/ubuntu/dataset/CVC-14-processed/Train/RGB/2014_05_04_15_42_53_069000_.tif'
# root_saved_path = '/home/ubuntu/dataset/CVC-14-processed/Test/Thermal_label/2014_05_01_22_22_42_020000_.txt/'
img = cv2.imread(image_path, 1)
# img = img.transpose(1,0,2)


# print(img.shape)  #  (512, 640, 3)
# exit()
# print(img)
# exit()   387 211 34 59
name = 'person'
file_name = 'xxx1'
f = open('/home/ubuntu/dataset/CVC-14-processed/Train/RGB_label_normalization/2014_05_04_15_42_53_069000_.txt', 'r')


# 505 212 20 50 0 0 0 0 0 0 0
txt = True
x_y_w_h = True
xmin_ymin_w_h = False
guiyihua = True

width_o = 670
height_o = 471
#
# width_o = 1
# height_o = 1


x_center = []
y_center = []
width = []
height = []


if txt:
    # 256 211 42 110    48 206 38 88       535 230 46 110         403 223 29 59
    # 286 211 45 110    106 212 33 77      74 207 35 86            140 222 31 62


    f_lines = f.readlines()
    # f_lines = f_lines[1:]
    print(f_lines.__len__())
    for item in f_lines:

        item_split = item.split(' ')
        # item_split = item_split[1:]
        # print(item_split)
        x_center.append(float(item_split[1]))
        y_center.append(float(item_split[2]))
        width.append(float(item_split[3]))
        height.append(float(item_split[4]))
        print(x_center,y_center,width,height)
        # exit()
    # exit()
    if guiyihua:
        x_center = np.array(x_center)*width_o
        y_center = np.array(y_center)*height_o
        width = np.array(width)*width_o
        height = np.array(height)*height_o
    else:
        x_center = np.array(x_center)
        y_center = np.array(y_center)
        width = np.array(width)
        height = np.array(height)

    if x_y_w_h:
        x_min = x_center - width/2
        x_max = x_center + width/2
        y_min = y_center - height / 2
        y_max = y_center + height / 2

    elif xmin_ymin_w_h:
        x_min = x_center
        x_max = x_center + width
        y_min = y_center
        y_max = y_center + height

    print(x_min,x_max,y_min,y_max)
    # exit()

    for x1, x2, y1, y2 in zip(x_min, x_max, y_min, y_max):
        x1 = int(x1)
        x2 = int(x2)
        y1 = int(y1)
        y2 = int(y2)
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), thickness=2)
        # 字为绿色
        cv2.putText(img, name, (x1, y1), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 0), thickness=2)


else:

    # 0.2921875 0.41796875 0.059375 0.14453125

    x1 = 0.292187
    y1 =  0.4179687
    x2 =  0.059375
    y2 =  0.1445312
    # 352.0, 241.25, 40.5, 97.5

    #  0.801451


    x1_c = int(x1*640-x2*640)
    x2_c = int(x1*640+x2*640)
    y1_c = int(y1*512-y2*512)
    y2_c = int(y1*512+y2*512)
    cv2.rectangle(img, (x1_c, y1_c), (x2_c, y2_c), (255, 0, 0), thickness=2)
    # 字为绿色
    cv2.putText(img, name, (x1_c, y1_c), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 0), thickness=2)


cv2.imshow('aaa',img)
cv2.waitKey(0)
# cv2.imwrite(os.path.join(root_saved_path, file_name + '.jpg'), img)

#
#
# def draw(image_path, root_saved_path):
#     """
#     图片根据标注画框
#     """
#     src_img_path = image_path
#     # 获取 路径下的文件
#     for file in os.listdir(src_img_path):
#         # print(file)
#         # 分离前缀 和 后缀
#         file_name, suffix = os.path.splitext(file)
#         if suffix == '.xml':
#             # print(file)
#             xml_path = os.path.join(src_img_path, file)
#             image_path = os.path.join(src_img_path, file_name+'.jpeg')
#             # image_path = os.path.join(src_img_path, file_name + '.bmp')
#             img = cv2.imread(image_path)
#             tree = ET.parse(xml_path)
#             root = tree.getroot()
#
#             # 读取 object
#             i = 1;
#             for obj in root.iter('object'):
#                 name = obj.find('name').text
#
#                 xml_box = obj.find('bndbox')
#                 x1 = int(xml_box.find('xmin').text)
#                 x2 = int(xml_box.find('xmax').text)
#                 y1 = int(xml_box.find('ymin').text)
#                 y2 = int(xml_box.find('ymax').text)
#
#
#                 # 小 ----->  大
#                 # x1 = int(x1 * 1800./640)
#                 # x2 = int(x2 * 1800./640)
#                 # y1 = int(y1 * 1600./512)
#                 # y2 = int(y2 * 1600./512)
#                 # x1 = x1+50
#                 # x2 = x2+50
#
#                 cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), thickness=2)
#                 # 字为绿色
#                 cv2.putText(img, name+str(i), (x1, y1), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 0), thickness=2)
#                 i=i+1;
#             # cv2.imwrite(os.path.join(root_saved_path, file_name+'.bmp'), img)
#             cv2.imwrite(os.path.join(root_saved_path, file_name + '.jpg'), img)
#
#
# if __name__ == '__main__':
#     image_path = r"E:\Dataset\jsonToXml\test1_thermal"
#     root_saved_path = r"E:\Dataset\jsonToXml\test1_result_thermal"
#     draw(image_path, root_saved_path)
