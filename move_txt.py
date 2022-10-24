import shutil
import os
# txt = open('/home/ubuntu/dataset/kaist_pixel_level/preparing_data/test/val.txt','r')
img_path = '/home/ubuntu/dataset/kaist dataset/test_thermal/'
root_path ='/home/ubuntu/dataset/kaist dataset/test_thermal/'
moved_path ='/home/ubuntu/dataset/KAIST_ALL/kaist_test/annotaation/'


txt_list = os.listdir(img_path)
# txt_list = txt.readlines()
for item in txt_list:
    if item[-3:] == 'txt':
        print(item)
        shutil.copy(root_path + item, moved_path + item)
    # print(item[-3:])
    # exit()
    # item = item[:-4]+'png'
    # print(item)
    # print(root_path+item,moved_path+item)
    # exit()

