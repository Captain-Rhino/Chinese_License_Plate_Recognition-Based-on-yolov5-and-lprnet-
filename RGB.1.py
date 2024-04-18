'''
qhy
2018.12.3
'''
import os
import numpy as np
import cv2

ims_path = 'C:/Users/王王王/Desktop/啊啊啊啊啊啊啊啊啊啊啊啊~/综合课程设计/中期/'  # 图像数据集的路径
ims_list = os.listdir(ims_path)
R_means = []
G_means = []
B_means = []
for im_list in ims_list:
    im = cv2.imdecode(np.fromfile(ims_path + im_list,dtype=np.uint8),flags=cv2.IMREAD_COLOR)
    # extrect value of diffient channel
    im_B = im[:, :, 0]
    im_G = im[:, :, 1]
    im_R = im[:, :, 2]
    # count mean for every channel
    im_R_mean = np.mean(im_R)
    im_G_mean = np.mean(im_G)
    im_B_mean = np.mean(im_B)
    # save single mean value to a set of means
    R_means.append(im_R_mean)
    G_means.append(im_G_mean)
    B_means.append(im_B_mean)
    print('图片：{} 的 RGB平均值为 \n[{}，{}，{}]'.format(im_list, im_R_mean, im_G_mean, im_B_mean))

    if (im_B_mean > im_G_mean and im_B_mean > im_R_mean):
        print('蓝')
    elif (im_G_mean > im_B_mean and im_G_mean > im_R_mean):
        print('绿')
    else:
        print('黄')
