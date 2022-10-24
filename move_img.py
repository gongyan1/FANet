import os
import shutil
import os
# dir = "FCA/"
root_path = "/home/ubuntu/dataset/KAIST_ALL/annotations/"
to_path = "/home/ubuntu/dataset/KAIST_ALL/kaist/train/annotation/"

# save_list = ['set06_V000_I01099.jpg_pred.jpg','set06_V002_I01219.jpg_pred.jpg','set06_V003_I00659.jpg_pred.jpg','set06_V003_I02959.jpg_pred.jpg','set08_V000_I00379.jpg_pred.jpg','set08_V000_I02159.jpg_pred.jpg','set08_V002_I00859.jpg_pred.jpg','set10_V000_I03599.jpg_pred.jpg']
# save_list = ['set06_V000_I01099.jpg_labels.jpg','set06_V002_I01219.jpg_labels.jpg','set06_V003_I00659.jpg_labels.jpg','set06_V003_I02959.jpg_labels.jpg','set08_V000_I00379.jpg_labels.jpg','set08_V000_I02159.jpg_labels.jpg','set08_V002_I00859.jpg_labels.jpg','set10_V000_I03599.jpg_labels.jpg']
# save_list = ['set06_V003_I01439.png','set06_V003_I02799.png','set10_V000_I00139.png','set11_V000_I01399.png']

img_list = os.listdir(root_path)

set_name = ['set00', 'set01', 'set02', 'set03', 'set04', 'set05']
for set_n in img_list:
    print(set_n)
    if set_n not in set_name:
        continue
    png_path_s = os.path.join(root_path, set_n)
    v_list = os.listdir(png_path_s)
    for v_n in v_list:
        png_path_v = os.path.join(png_path_s, v_n)
        type_list = os.listdir(png_path_v)
        for type_n in type_list:
            png_path_t = os.path.join(png_path_v, type_n)
            new_png_name = set_n + '_' + v_n + '_' + type_n.split('.')[0] +  '.' + type_n.split('.')[1]
            print(png_path_t)
            print(new_png_name)
            shutil.copy(png_path_t, to_path + new_png_name)
            # exit()
            # png_list = os.listdir(png_path_t)
            # for img_png in png_list:
            #     # set06_V000_I00059_visible
            #     new_png_name = set_n + '_' + v_n +'_'+ img_png.split('.')[0] +'_' + type_n + '.'+img_png.split('.')[1]
            #     png_path_p = os.path.join(png_path_t, img_png)
            #     print(png_path_p)
            #     print(new_png_name)
            #     exit()
            #

                # if 'visible' in png_path_p:
                #     shutil.copy(png_path_p, to_path+'visible/' + new_png_name)
                # else:
                #     shutil.copy(png_path_p, to_path + 'lwir/' + new_png_name)





        #












