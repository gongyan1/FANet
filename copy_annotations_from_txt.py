# coding=utf-8
import os
import os.path
import shutil  # Python文件复制相应模块
import glob
# 主要用到的命令是 shutil 命令
# 如：shutil.copy('A/ReadMe.txt', 'B')
# 就是把 目录 A 下面的 readme 复制到 目录B 下面
# 目录 A 和 B 都是绝对路径

# 读取 txt 文件
# txt_dir = 'E:/Dataset/kaist_pixel_level/saliency_imagesetfile_train.txt'

def file_filter(f):
    if f[-4:] in ['.png']:
        return True
    else:
        return False

# 选择文件列表
train_dir = '/home/ubuntu/dataset/kaist_pixel_level/preparing_data/test/images/'

# 准备移动图片的目录
# xml 的所在目录（注意最后加 / ，因为一会要去拼接 txt读取的文件名  的路径）
anno_dir = '/home/ubuntu/dataset/kaist dataset/test_visible/'

# 要复制到的 xml 文件夹
copy_anno_dir = '/home/ubuntu/dataset/kaist_pixel_level/preparing_data/test/a/'

train_list = os.listdir(train_dir)

txt_list = os.listdir(anno_dir)

# txt_list = list(filter(file_filter,txt_list))

print(train_list.__len__(),txt_list.__len__())
# exit()
# set00_V000_visible_I01225.txt
# set00_V000_I00980.jpg

train_list_new = []
for i in train_list:
    anno_name = i.split('.')[0]
    anno_name_split = anno_name.split('_')
    anno_name = anno_name_split[0]+'_'+anno_name_split[1]+'_'+'visible'+'_'+anno_name_split[2]
    # print(anno_name_split)
    print(anno_name)
    # exit()
    train_list_new.append(anno_name)

# exit()
for i in train_list_new:
    anno_name = i + '.png'
    anno_name_absolute_path = os.path.join(anno_dir, anno_name)
    if anno_name in txt_list:
        # print(anno_name)
        shutil.copy(anno_name_absolute_path, copy_anno_dir)



# for i in str:
    # anno_name = i.split('/')[7].split('.')[0]+'.xml'   #  anno_name 是这样的形式  ：  FLIR_10109.xml
    # anno_name_absolute_path = os.path.join(anno_dir,anno_name)
    # shutil.copy( anno_name_absolute_path,copy_anno_dir)

