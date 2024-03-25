import numpy as np
import cv2
import torch
from LPRNet_test import build_lprnet

class LicensePlateRecognizer:
    def __init__(self, model_path='demo_test.pth'):
        self.CHARS = ['京', '沪', '津', '渝', '冀', '晋', '蒙', '辽', '吉', '黑',
                      '苏', '浙', '皖', '闽', '赣', '鲁', '豫', '鄂', '湘', '粤',
                      '桂', '琼', '川', '贵', '云', '藏', '陕', '甘', '青', '宁',
                      '新', '港', '澳', '警', '挂', '使', '领', '学',
                      '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
                      'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K',
                      'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
                      'W', 'X', 'Y', 'Z', '-'
                     ]
        self.model = build_lprnet(lpr_max_len=8, class_num=len(self.CHARS), dropout_rate=0.5)  # Build model
        self.load_model(model_path)

    def load_model(self, model_path):
        dic = torch.load(model_path, map_location=torch.device("cpu"))
        self.model.load_state_dict(dic)
        self.model.eval()

    def preprocess_image(self, image):
        height, width, _ = image.shape
        if width != 94 or height != 24:
            image = cv2.resize(image, (94, 24))  # Resize if necessary
        image = image.astype('float32')
        image -= 127.5
        image *= 0.0078125
        image = np.transpose(image, (2, 0, 1))  # (height, width, channel) -> (channel, height, width)
        image = image.reshape(1, 3, 24, 94)  # Reshape
        return torch.from_numpy(image)

    def predict(self, image_path):
        # image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), flags=cv2.IMREAD_COLOR)
        image = self.preprocess_image(image_path)
        with torch.no_grad():
            output = self.model(image)
            output = output.cpu().detach().numpy()
            _, labels = self.greedy_decode(output, True)
            return labels

    def greedy_decode(self, preds, pred_char=False):
        last_chars_idx = len(self.CHARS) - 1
        pred_labels = []
        for i in range(preds.shape[0]):
            pred = preds[i, :, :]
            pred_label = []
            for j in range(pred.shape[1]):
                pred_label.append(np.argmax(pred[:, j], axis=0))
            no_repeat_blank_label = []
            pre_c = -1
            for c in pred_label:
                if (pre_c == c) or (c == last_chars_idx):
                    if c == last_chars_idx:
                        pre_c = c
                    continue
                no_repeat_blank_label.append(c)
                pre_c = c
            pred_labels.append(no_repeat_blank_label)
        if pred_char:
            labels = []
            for label in pred_labels:
                lb = ""
                for i in label:
                    lb += self.CHARS[i]
                labels.append(lb)
            return pred_labels, labels
        else:
            return pred_labels

# 示例用法
if __name__ == "__main__":
    lpr = LicensePlateRecognizer()
    file_path = 'E:/BaiduNetdiskDownload/CBLPRD-330k_v1/train/贵PRGS48_普通蓝牌.jpg'
    predicted_labels = lpr.predict(file_path)
    print(predicted_labels)
