import cv2
import os
import numpy as np
import csv

from PIL import Image

# CCPD车牌有重复，应该是不同角度或者模糊程度
path = r'G:\yolo-train\yolov5-master\data\ccpd_weather'  # 改成自己的车牌路径

provinces = ["皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "京", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤", "桂", "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新", "警", "学", "O"]
alphabets = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'O']
ads = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'O']

# CSV 文件路径
csv_file_path = r'G:\yolo-train\lpr_dataset\save_test\plate_results.csv'

# 打开 CSV 文件，并创建 CSV 写入器
with open(csv_file_path, mode='w', newline='', encoding='utf-8-sig') as csv_file:
    csv_writer = csv.writer(csv_file)
    # 写入表头
    #csv_writer.writerow(['Filename', 'Plate'])

    num = 0
    for filename in os.listdir(path):
        num += 1
        result = ""
        _, _, box, points, plate, brightness, blurriness = filename.split('-')
        list_plate = plate.split('_')  # 读取车牌
        result += provinces[int(list_plate[0])]
        result += alphabets[int(list_plate[1])]
        result += ads[int(list_plate[2])] + ads[int(list_plate[3])] + ads[int(list_plate[4])] + ads[int(list_plate[5])] + ads[int(list_plate[6])]
        # 新能源车牌的要求，如果不是新能源车牌可以删掉这个if
        # if result[2] != 'D' and result[2] != 'F' \
        #         and result[-1] != 'D' and result[-1] != 'F':
        #     print(filename)
        #     print("Error label, Please check!")
        #     assert 0, "Error label ^~^!!!"
        
        img_path = os.path.join(path, filename)
        img = cv2.imread(img_path)
        assert os.path.exists(img_path), "image file {} dose not exist.".format(img_path)

        # 将文件名和车牌结果写入 CSV 文件
        csv_writer.writerow([result])

        print(result)
