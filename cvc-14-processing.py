import shutil
import os

def file_copy(src_path, dst_path, file_tag = ".tif"):

    file_list = os.listdir(src_path)
    for item in file_list:
        if file_tag not in item:
            continue

        shutil.copy(src_path+item, dst_path+item)

def count_file(root_path):
    RGB_path = root_path + 'RGB/'
    Thermal_path = root_path + 'Thermal/'
    RGB_label_path = root_path + 'RGB_label/'
    Thermal_label_path = root_path + 'Thermal_label/'

    RGB_list = os.listdir(RGB_path)
    Thermal_list = os.listdir(Thermal_path)
    RGB_label_list = os.listdir(RGB_label_path)
    Thermal_label_list = os.listdir(Thermal_label_path)

    print(len(RGB_list), len(Thermal_list), len(RGB_label_list), len(Thermal_label_list))

    RGB_list.sort()
    Thermal_list.sort()
    RGB_label_list.sort()
    Thermal_label_list.sort()

    min_len = min(len(RGB_list), len(Thermal_list), len(RGB_label_list), len(Thermal_label_list))

    for i in range(min_len):
        print(i, RGB_list[i], Thermal_list[i], RGB_label_list[i], Thermal_label_list[i])

    # k = 0
    #
    # save_list = []
    #
    # for item in RGB_list:
    #     name = item.split('.')[0]
    #
    #     label_name = name + '.txt'
    #
    #     # print(item, label_name)
    #
    #     if item in Thermal_list and label_name in RGB_label_list and label_name in Thermal_label_list:
    #         print(k, item)
    #         save_list.append(item)
    #     else:
    #         print(k, "xxxx")
    #     k = k + 1
    #
    # print(len(save_list))   # 1417
    #
    # for item in Thermal_label_list:
    #     name = item.split('.')[0]
    #     label_name = name + '.tif'
    #     if label_name not in save_list:
    #         os.remove(Thermal_label_path+item)

if __name__ == '__main__':
    # src_path = "/home/ubuntu/dataset/CVC-14/Day/FIR/NewTest/FramesPos/"
    # dst_path = "/home/ubuntu/dataset/CVC-14-processed/Test/Thermal/"

    # file_copy(src_path, dst_path)

    root_path = "/home/ubuntu/dataset/CVC-14-processed/Train/"
    count_file(root_path)

