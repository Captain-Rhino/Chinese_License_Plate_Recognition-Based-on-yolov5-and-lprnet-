#将原车牌信息记录在一个csv表格中
import os
import csv
Plate = []

# 获取 labels 文件夹中的所有文件
label_files = os.listdir('data/CRPD_1000/labels')

# 对每个文件执行操作
for file_name in label_files:
    file_path = os.path.join('data/CRPD_1000/labels', file_name)
    
    # 读取标注数据文件
    with open(file_path, 'r') as f:
        lines = f.readlines()

    # 解析标注数据并计算新的数据
    for i, line in enumerate(lines):
        temp = line.split()
        Plate.append(temp[-1])#提取车牌信息
print(Plate)#打印车牌



# 指定要保存的CSV文件路径
csv_file = 'data/CRPD_1000/plates.csv'

# 将Plate列表数据写入CSV文件
with open(csv_file, 'w', newline='', encoding='utf-8-sig') as f:
    writer = csv.writer(f)
    # 写入标题行（如果有需要）
    # writer.writerow(['Plate'])
    # 写入数据行
    for plate in Plate:
        writer.writerow([plate])

print("CSV文件已成功生成:", csv_file)

