import random
import os

img_list = os.listdir('/home/ubuntu/dataset/kaist_pixel_level/preparing_data/test/image/')
f = open('/home/ubuntu/dataset/kaist_pixel_level/preparing_data/test/val.txt','w')
random.shuffle(img_list)
x = int(img_list.__len__()/2)


for i in img_list[:x]:
    # print(i)
    f.write(i+'\n')
