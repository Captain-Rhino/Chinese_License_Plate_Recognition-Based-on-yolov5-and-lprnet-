import numpy as np
import cv2
import torch
from LPRNet_test_5_15 import build_lprnet
import time
import os
def greedy_decode(preds, pred_char=False):
    last_chars_idx = len(CHARS) - 1

    # 贪婪解码 (greedy decode)
    pred_labels = []
    for i in range(preds.shape[0]):
        pred = preds[i, :, :]  # 第i张图片对应的结果，即维度为(66, 18)的二维数组
        pred_label = []
        for j in range(pred.shape[1]):  # 遍历每一列，找到每一列最大值的索引（index）
            pred_label.append(np.argmax(pred[:, j], axis=0))
        no_repeat_blank_label = []
        pre_c = -1
        for c in pred_label:  # 合并重复的索引值部分，删除空白标签，即为-1的值(dropout repeate label and blank label)
            if (pre_c == c) or (c == last_chars_idx):
                if c == last_chars_idx:
                    pre_c = c
                continue
            no_repeat_blank_label.append(c)
            pre_c = c
        pred_labels.append(no_repeat_blank_label)

    # 解码成字符串
    if pred_char:
        labels = []
        for label in pred_labels:
            lb = ""
            for i in label:
                lb += CHARS[i]
            labels.append(lb)
        return pred_labels, labels
    else:
        return pred_labels

CHARS = ['京', '沪', '津', '渝', '冀', '晋', '蒙', '辽', '吉', '黑',
         '苏', '浙', '皖', '闽', '赣', '鲁', '豫', '鄂', '湘', '粤',
         '桂', '琼', '川', '贵', '云', '藏', '陕', '甘', '青', '宁',
         '新', '港', '澳', '警', '使', '领', '学',
         '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
         'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K',
         'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
         'W', 'X', 'Y', 'Z', '-'
         ]

model = build_lprnet(lpr_max_len=8, class_num=len(CHARS), dropout_rate=0.5)  # 建立模型

path = 'demo_test_92.57.pth'  # ./weights/LPR_mix.pth ./weights/LPR_demo.pth
# 使用map_location，防止GPU训练的模型导入到CPU上测试会出错
dic = torch.load(path, map_location=torch.device("cpu"))

model.load_state_dict(dic, strict=False)

# file_path = 'E:/BaiduNetdiskDownload/CBLPRD-330k_v1/train/贵PRGS48_普通蓝牌.jpg'
def load_singlepic(file_path):
    image = cv2.imdecode(np.fromfile(file_path,dtype=np.uint8),flags=cv2.IMREAD_COLOR)
    height, width, _ = image.shape
    if width != 94 or height != 24:
        image = cv2.resize(image, (94, 24))  # 缩放

    image = image.astype('float32')
    image -= 127.5
    image *= 0.0078125
    # (height, width, channel) -> (channel, height, width) eg:(24,94,3)->(3,24,94)
    image = np.transpose(image, (2, 0, 1))
    # 改变shape
    image = image.reshape(1, 3, 24, 94)
    image = torch.from_numpy(image)
    return image
# image = load_singlepic(file_path)

img_dir = r"G:\yolo-train\yolov5-master\data\tem_5_15"
false = 0
for img_name in os.listdir(img_dir):
    file_path = os.path.join(img_dir, img_name)
    image = load_singlepic(file_path)
    imgname = img_name.split("_")[0]
    imgname = imgname.split('.')[0]

    model.eval()
    with torch.no_grad():
        output = model(image)
        output = output.cpu().detach().numpy()
        labels_idx, labels = greedy_decode(output, True)
    if labels[0] != imgname:
        print('true:'+imgname+'\n')
        print('false:' + labels[0] + '\n\n')
        false += 1
print(false)



