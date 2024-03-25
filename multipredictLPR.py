import numpy as np
from torch.utils.data import Dataset
import random
import cv2
import os
import torch
from torch.utils.data import DataLoader
from LPRNet_test import build_lprnet
CHARS = ['京', '沪', '津', '渝', '冀', '晋', '蒙', '辽', '吉', '黑',
         '苏', '浙', '皖', '闽', '赣', '鲁', '豫', '鄂', '湘', '粤',
         '桂', '琼', '川', '贵', '云', '藏', '陕', '甘', '青', '宁',
         '新', '港', '澳', '警', '挂', '使', '领', '学',
         '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
         'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K',
         'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
         'W', 'X', 'Y', 'Z', '-'
         ]
CHARS_DICT = {char:i for i, char in enumerate(CHARS)}
class LPRDataSet(Dataset):
    def __init__(self, img_dir, imgSize, lpr_max_len, PreprocFun=None, mode='train'):
        self.img_dir = img_dir
        self.img_paths = []
        if type(img_dir) == str:  # 一个路径
            for img_name in os.listdir(img_dir):
                if 'True' not in img_name:  # 只添加单层车牌，带有True的文件名是双层车牌
                    img_name = os.path.join(img_dir, img_name)
                    self.img_paths.append(img_name)
        elif type(img_dir) == list:  # 多个路径
            for dir_path in img_dir:
                for img_name in os.listdir(dir_path):
                    if 'True' not in img_name:  # 只添加单层车牌，带有True的文件名是双层车牌
                        img_name = os.path.join(dir_path, img_name)
                        self.img_paths.append(img_name)
        else:
            print('error in img_dir, img_dir must be str or list')
        random.shuffle(self.img_paths)        # 打乱顺序
        self.img_size = imgSize
        self.lpr_max_len = lpr_max_len
        if PreprocFun is not None:
            self.PreprocFun = PreprocFun
        elif PreprocFun is None and mode == 'test':
            self.PreprocFun = self.test_transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        filename = self.img_paths[index]
        image = cv2.imdecode(np.fromfile(filename,dtype=np.uint8),flags=cv2.IMREAD_COLOR)
        height, width, _ = image.shape
        if width != self.img_size[0] or height != self.img_size[1]:
            image = cv2.resize(image, self.img_size)  # 缩放
        image = self.PreprocFun(image)

        basename = os.path.basename(filename)
        imgname, suffix = os.path.splitext(basename)
        imgname = imgname.split("_")[0]
        label = list()
        for c in imgname:
            label.append(CHARS_DICT[c])


        if len(label) > self.lpr_max_len:
            print(imgname)
            assert 0, "Error label ^~^!!!"

        return image, label, len(label)

    def test_transform(self, img):
        img = img.astype('float32')
        img -= 127.5
        img *= 0.0078125
        # (height, width, channel) -> (channel, height, width) eg:(24,94,3)->(3,24,94)
        img = np.transpose(img, (2, 0, 1))
        return img


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

def collate_fn(batch):
    imgs = []
    labels = []
    lengths = []
    for sample in batch:
        img, label, length = sample
        imgs.append(torch.from_numpy(img))
        # imgs.append(img)
        labels.extend(label)
        lengths.append(length)
    labels = np.asarray(labels).flatten().astype(np.int32)
    return (torch.stack(imgs, 0), torch.from_numpy(labels), lengths)
#yolo切割车牌，放到test_data
test_data = LPRDataSet('G:/licenseplate/LPRNet/data/challage', (94, 24), 8, mode='test')
batch_size = 128

test_loader = DataLoader(dataset=test_data, batch_size=batch_size,
                         shuffle=True, drop_last=True, collate_fn=collate_fn)


model = build_lprnet(lpr_max_len=8, class_num=len(CHARS), dropout_rate=0.5)  # 建立模型

path = 'demo_test.pth'  # ./weights/LPR_mix.pth ./weights/LPR_demo.pth
# 使用map_location，防止GPU训练的模型导入到CPU上测试会出错
dic = torch.load(path, map_location=torch.device('cpu'))
model.load_state_dict(dic)

model.eval()  # 关闭dropout
total_test_loss = 0
total_acc_num = 0
Tp, Tn = 0.0, 0.0
with torch.no_grad():  # 关闭梯度计算
    for images, labels, lengths in test_loader:
        # labels: 1D -> 2D
        targets = []
        start = 0
        for length in lengths:
            label = labels[start:start + length]
            targets.append(label.tolist())
            start += length
        # forward
        prebs = model(images)
        prebs = prebs.cpu().detach().numpy()
        # greedy decode
        preb_labels = greedy_decode(prebs)
        # calculate
        for i, label in enumerate(preb_labels):
            if len(label) != len(targets[i]):  # 长度不一致
                Tn += 1
                continue
            if targets[i] == label:
                Tp += 1
                print('true')
            else:
                Tn += 1
                print('false')
    total_test_acc = float(Tp) / float(Tp + Tn)
    print("整体测试集上的acc: {}".format(total_test_acc))
