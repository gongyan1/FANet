import cv2
import os

root_path = "/home/ubuntu/save/small/saliency/time-mp4/"
to_path = "/home/ubuntu/save/small/saliency/"
# 读取时序图中的第一张图片
img = cv2.imread(root_path+'set06_V000_I00459.jpg_pred.jpg')
# 设置每秒读取多少张图片
fps = 3
imgInfo = img.shape

# 获取图片宽高度信息
size = (imgInfo[1], imgInfo[0])
# print(size)
# exit()
fourcc = cv2.VideoWriter_fourcc(*"mp4v")

# 定义写入图片的策略
videoWrite = cv2.VideoWriter(to_path+'output.mp4', fourcc, fps, size)
img_list = os.listdir(root_path)

img_list.sort()

# path = '/home/ubuntu/save/small/saliency/'
out_num = len(img_list)

for i in range(0, out_num):
    # 读取所有的图片
    # fileName = path + 'in' + str(i).zfill(6)+'.jpg'
    fileName = img_list[i]
    img = cv2.imread(root_path+fileName)

    # 将图片写入所创建的视频对象
    videoWrite.write(img)

# videoWrite.release()
print('finish')