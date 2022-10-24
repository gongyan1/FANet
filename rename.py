import os

root_path = '/home/ubuntu/dataset/KAIST_ALL/kaist/test/annotation/'
txt_list = os.listdir(root_path)
# set06_V000_visible_I00899.txt
# set06_V000_I00899.txt
# set06_V000_I00459.png


# set06_V002_lwir_I01459.png
# set09_V000_lwir_I01679.png
# set06_V000_visible_I00019.png


# set05_V000_I02919.png
for item in txt_list:
    item_new = item[:11] + item[16:22]+item[-4:]
    # print(item)
    print(item, item_new)
    os.rename(root_path+item,root_path+item_new)
    # exit()


