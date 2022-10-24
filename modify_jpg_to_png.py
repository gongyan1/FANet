import os

root_path = '/home/ubuntu/dataset/kaist_pixel_level/preparing_data/train/image/'
img_list = os.listdir(root_path)
for item in img_list:
    # print(item)
    old_name = item
    old_name = os.path.splitext(old_name)[0]
    new_name = old_name+'.png'
    os.rename(root_path+item,root_path+new_name)

