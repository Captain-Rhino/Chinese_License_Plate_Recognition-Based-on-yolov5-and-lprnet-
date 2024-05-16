import numpy as np
import cv2
import torch
from A_LPRNet_test_5_15 import build_lprnet

class LicensePlateRecognizer:
    CHARS = ['京', '沪', '津', '渝', '冀', '晋', '蒙', '辽', '吉', '黑',
             '苏', '浙', '皖', '闽', '赣', '鲁', '豫', '鄂', '湘', '粤',
             '桂', '琼', '川', '贵', '云', '藏', '陕', '甘', '青', '宁',
             '新', '港', '澳', '警', '使', '领', '学',
             '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
             'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K',
             'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
             'W', 'X', 'Y', 'Z', '-']

    def __init__(self, model_path='demo_test_final.pth'):
        self.model = build_lprnet(lpr_max_len=8, class_num=len(self.CHARS), dropout_rate=0.5)  # 建立模型
        dic = torch.load(model_path, map_location=torch.device("cpu"))
        self.model.load_state_dict(dic, strict=False)
        self.model.eval()

    @staticmethod
    def preprocess_image(image):
        height, width, _ = image.shape
        if width != 94 or height != 24:
            image = cv2.resize(image, (94, 24))  # Resize if necessary
        image = image.astype('float32')
        image -= 127.5
        image *= 0.0078125
        image = np.transpose(image, (2, 0, 1))  # (height, width, channel) -> (channel, height, width)
        image = image.reshape(1, 3, 24, 94)  # Reshape
        return torch.from_numpy(image)

    @staticmethod
    def greedy_decode(preds, pred_char=False):
        last_chars_idx = len(LicensePlateRecognizer.CHARS) - 1

        pred_labels = []
        for i in range(preds.shape[0]):
            pred = preds[i, :, :]
            pred_label = [np.argmax(pred[:, j], axis=0) for j in range(pred.shape[1])]
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
                lb = "".join([LicensePlateRecognizer.CHARS[i] for i in label])
                labels.append(lb)
            return pred_labels, labels
        else:
            return pred_labels

    @classmethod
    def recognize_plate(cls, image, model_path='demo_test_final.pth'):
        instance = cls(model_path)
        preprocessed_image = instance.preprocess_image(image)
        with torch.no_grad():
            output = instance.model(preprocessed_image)
            output = output.cpu().detach().numpy()
            labels_idx, labels = instance.greedy_decode(output, True)
        #print(labels[0])
        return labels[0]

# Example usage:
# plate = LicensePlateRecognizer.recognize_plate(r"G:\yolo-train\yolov5-master\data\tem_5_15\川A8K10Y_3181.jpg")
# print("Recognized Plate:", plate)
