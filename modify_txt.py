import os

root_path = '/home/ubuntu/dataset/CVC-14-processed/Train/Thermal_label/'
to_path = '/home/ubuntu/dataset/CVC-14-processed/Train/Thermal_label_normalization/'


fig_w = 670
fig_h = 471

txt_list = os.listdir(root_path)

# sum_no_person = 0
for txt_name in txt_list:
    txt_path = os.path.join(root_path, txt_name)
    print(txt_path)
    f = open(txt_path, 'r')
    f_lines = f.read().splitlines()
    # f_lines = f_lines[1:]
    # print(f_lines)
    f_w = open(os.path.join(to_path, txt_name), 'w')
    for f_line in f_lines:
        # print(f_lines.__len__())
        # if 'person' not in f_line:
        #     sum_no_person = sum_no_person + 1
        #     continue
        f_line_split = f_line.split(' ')
        f_line_split = f_line_split[:4]
        print(f_line_split)
        item = f_line_split
        x_c = float(item[0])
        y_c = float(item[1])
        w = float(item[2])
        h = float(item[3])

        save_list = []
        save_list.append(0)
        save_list.append(x_c/fig_w)
        save_list.append(y_c/fig_h)
        save_list.append(w / fig_w)
        save_list.append(h / fig_h)
        print(save_list)

        for k in save_list:
            f_w.write(str(k) + " ")
        f_w.write('\n')
    f_w.close()
    f.close()


# print(sum_no_person)  # 10902
