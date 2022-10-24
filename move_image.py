import shutil
import os
# txt = open('/home/ubuntu/dataset/kaist_pixel_level/preparing_data/test/val.txt','r')
img_path = '/home/ubuntu/dataset/kaist_pixel_level/preparing_data/val/images/'
root_path ='/home/ubuntu/dataset/kaist_pixel_level/preparing_data/test/RGB/'
moved_path ='/home/ubuntu/dataset/kaist_pixel_level/preparing_data/val/RGB/'


txt_list = os.listdir(img_path)
# txt_list = txt.readlines()
for item in txt_list:
    # print(item)
    # exit()
    item = item[:-4]+'png'
    print(item)
    # print(root_path+item,moved_path+item)
    # exit()
    shutil.move(root_path+item,moved_path+item)
