import os
from PIL import Image

# 设置文件夹路径
folder_path = r"G:\yolo-train\yolov5-master\data\green_seg"
save_path = r"G:\yolo-train\yolov5-master\data\green_seg2"
num = 0

# 遍历文件夹中的所有文件
for filename in os.listdir(folder_path):
    # 检查文件是否是图像文件
    if (filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".png")) and num < 20:
        # 打开图像文件
        image_a = Image.open(os.path.join(folder_path, filename))
        
        # 打开背景图像文件
        image_b = Image.open(r"G:\yolo-train\yolov5-master\data\background.jpg")
        
        # 将图像B调整为320*320的尺寸
        image_b = image_b.resize((320, 320))
        
        # 将图像A叠加在图像B上
        image_b.paste(image_a, (96, 136))
        
        # 保存叠加后的图像为图像A的文件名
        image_b.save(os.path.join(save_path, filename))
        
        # 关闭图像文件
        image_a.close()
        image_b.close()
        
        num += 1
