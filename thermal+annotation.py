from PIL import Image
import numpy as np
import os

annotation_path = '/home/ubuntu/dataset/kaist_pixel_level/preparing_data/train/annotations/'
image_path = '/home/ubuntu/dataset/kaist_pixel_level/preparing_data/train/images/'
save_path = '/home/ubuntu/dataset/kaist_pixel_level/preparing_data/train/images+anno/'

annotation_list = os.listdir(annotation_path)   # .jpg
image_list = os.listdir(image_path)             # .png

for item in annotation_list:
    anno_name = item
    img_name = item[:-3]+'png'
    # print(anno_name,img_name)
    annotation = Image.open(annotation_path+anno_name)    #  L
    img = Image.open(image_path+img_name)           #  RGB
    annotation_np = np.array(annotation)
    img_np = np.array(img)
    img_np[:, :, 2] = annotation_np
    im = Image.fromarray(img_np)
    # im.show()
    # exit()
    im.save(save_path + anno_name)



# bool_equal = img_np[:,:,0]==img_np[:,:,2]     #  第一维和第三维度相同






