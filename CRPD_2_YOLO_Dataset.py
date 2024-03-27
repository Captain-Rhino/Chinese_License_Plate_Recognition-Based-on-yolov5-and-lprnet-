import os
import cv2

# 读取图像
img = cv2.imread('images/1 (1).jpg')
h, w, _ = img.shape

# 获取 labels 文件夹中的所有文件
label_files = os.listdir('labels')

# 对每个文件执行操作
for file_name in label_files:
    file_path = os.path.join('labels', file_name)
    
    # 读取标注数据文件
    with open(file_path, 'r') as f:
        lines = f.readlines()

    # 解析标注数据并计算新的数据
    for i, line in enumerate(lines):
        temp = line.split()
        x1, x2, x3, x4 = eval(temp[0]), eval(temp[2]), eval(temp[4]), eval(temp[6])
        y1, y2, y3, y4 = eval(temp[1]), eval(temp[3]), eval(temp[5]), eval(temp[7])
        label = 0
        x_ = (x1 + x2) / (2 * w)
        y_ = (y1 + y3) / (2 * h)
        w_ = (x2 - x1) / w
        h_ = (y3 - y1) / h

        # 将新的数据替换到原始数据中
        lines[i] = f"{label} {x_} {y_} {w_} {h_} \n"

    # 将修改后的数据写回原始文件
    with open(file_path, 'w') as f:
        f.writelines(lines)
