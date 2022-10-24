import glob
img_list = glob.glob('/home/ubuntu/dataset/kaist dataset/train_thermal/*.png')
f1 = open('/home/ubuntu/dataset/kaist dataset/task-kaist/train_thermal.txt','w')
for i in img_list:
    print(i)
    f1.write(i)
    f1.write('\n')




