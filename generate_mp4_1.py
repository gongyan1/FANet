import cv2
root_path = "/home/ubuntu/save/small/saliency/time-mp4/"
to_path = "/home/ubuntu/save/small/saliency/"
# 读取时序图中的第一张图片
img = cv2.imread(root_path+'set06_V000_I00459.jpg_pred.jpg')
# 设置每秒读取多少张图片
fps = 25
imgInfo = img.shape

size = (imgInfo[1], imgInfo[0])


videowriter = cv2.VideoWriter("a.avi", cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps, size)

for i in range(1, 200):
    img = cv2.imread('%d'.jpg % i)
    videowriter.write(img)







