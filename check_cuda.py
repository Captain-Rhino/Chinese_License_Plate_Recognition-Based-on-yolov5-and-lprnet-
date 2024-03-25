import torch
print(torch.__version__)
print(torch.version.cuda)
print(torch.backends.cudnn.version())
# 检查CUDA是否可用
if torch.cuda.is_available():
    # 获取GPU数量
    num_devices = torch.cuda.device_count()
    print("Number of available GPUs:", num_devices)

    # 遍历每个GPU并输出设备编号和名称
    for i in range(num_devices):
        print("CUDA Device", i, ":", torch.cuda.get_device_name(i))
else:
    print("CUDA is not available. Please make sure you have CUDA enabled GPU.")

