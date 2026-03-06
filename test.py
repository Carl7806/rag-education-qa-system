import torch
# 查看 PyTorch 关联的 CUDA 版本
print("CUDA 版本:", torch.version.cuda)
# 检查 CUDA 是否可用
print("CUDA 是否可用:", torch.cuda.is_available())
# 如果有 GPU，查看 GPU 名称
if torch.cuda.is_available():
    print("GPU 型号:", torch.cuda.get_device_name(0))
