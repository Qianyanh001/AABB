# ABCD Project
## 1. 环境配置 (Environment Setup)
本项目基于 Python 3.10+ 和 PyTorch 2.4 (CUDA 12.1)。
### 1.1 创建虚拟环境
请确保项目根目录下已存在 `environment.yml` 和 `requirements.txt` 文件。
```bash
# 1. 根据 environment.yml 创建基础环境 (仅包含 User 手动安装的 Conda 包)
conda env create -f environment.yml
# 2. 激活环境 (假设 environment.yml 中定义的名称为 face_distill，如果不同请自行调整)
conda activate face_distill
```
### 1.2 安装 PyTorch (CUDA 12.1)
推荐优先安装 PyTorch，以确保 CUDA 依赖正确：
```bash
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```
### 1.3 安装其他依赖
安装剩余的 Python 依赖包 (包括 `transformers`, `tensorboard` 等)：
```bash
pip install -r requirements.txt
```
---
## 2. 数据准备 (Data Preparation)
请严格按照以下目录结构存放数据，以确保 `train_distill.py` 能正确读取。
### 2.1 数据集 (Face Emore)
- **下载链接**: [点击这里下载 (GDrive)](https://drive.google.com/file/d/1SXS4-Am3bsKSK615qbYdbA_FMVh3sAvR/view)
- **存放路径**: 建议存放在 `datasets/faces_emore`。
- **⚠️ 重要配置**: 代码中的 `data_root` 默认为绝对路径 (`/media/Storage1/qyh/datasets/faces_emore/`)。**请务必修改 `train_distill.py` 中的 `data_root` 变量，使其指向您实际存放数据集的路径。**
### 2.2 教师模型特征 (Teacher Features)
- **文件夹名称**: `teacher_features_12k`
- **存放路径**: 项目根目录 (与 `train_distill.py` 同级)
- **内容**: 需包含 `teacher_embedding.npy` 和 `teacher_feat_map.npy`。
### 2.3 预训练权重 (BLIP-2 Weights)
- **下载脚本**: 运行 `download_weight.py`。
- **存放路径**: `blip2_weights` (脚本会自动下载到此目录)。
- **操作**:
  ```bash
  python download_weight.py
  ```
### **推荐的文件结构**
```text
[Project Root]/
├── train_distill.py       # 主训练脚本
├── download_weight.py     # 权重下载脚本
├── FaceDataset.py         # 数据集加载类
├── environment.yml        # Conda 环境配置
├── requirements.txt       # Pip 依赖列表
├── teacher_features_12k/  # 教师特征
│   ├── teacher_embedding.npy
│   └── teacher_feat_map.npy
├── blip2_weights/         # BLIP-2 权重 (自动下载)
├── datasets/              # 数据集根目录
│   └── faces_emore/       # 人脸数据集
├── logs/                  # 训练日志
└── checkpoints/           # 模型保存路径
```
---
## 3. 训练 (Training)
### 3.1 启动训练
脚本支持自动检测多卡，默认使用所有可见 GPU。您可以通过 `GPU_LIST` 环境变量控制使用的显卡。
**基本运行命令**:
```bash
python train_distill.py
```
**指定显卡运行** (例如使用 GPU 0 和 1):
Windows PowerShell:
```powershell
$env:GPU_LIST="0,1"; python train_distill.py
```
或者直接修改 `train_distill.py` 中的 `GPU_LIST` 默认值。
### 3.2 训练参数配置
主要参数在 `train_distill.py` 中定义。请根据硬件显存情况调整：
| 参数 | 默认值 | 说明 |
| :--- | :--- | :--- |
| `per_gpu_batch` | 64 | 单卡 Batch Size (总 Batch Size = 64 * GPU数) |
| `EPOCHS` | 100 | 总训练轮数 |
| `LR` | 1e-4 | 初始学习率 (CosineAnnealingLR) |
| `num_classes` | 600 | 分类类别数 |
| `LAMBDA_AT` | Dynamic | 0 -> 1.0 (随 Epoch 线性增加) |
| `LAMBDA_RKD` | Dynamic | 0 -> 1000.0 (随 Epoch 线性增加) |
### 3.3 显卡与内存优化
- **显存优化**: 代码已启用 `PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"` 以减少碎片化。
- **混合精度**: 已启用 `fp16` (AMP) 训练。
---
## 4. 结果保存与上传 (Result Saving & Uploading)
训练完成后，主要关注以下输出文件：
1. **模型权重 (`checkpoints/`)**:
    - `best_student_adapter.pth`: 验证 Loss 最低的最优模型权重 (包含 `state_dict` 和 loss 信息)。
2. **训练日志 (`logs/adapter_distill/`)**:
    - 包含 TensorBoard 事件文件，可使用 `tensorboard --logdir=logs/adapter_distill` 查看 Loss 曲线。
**打包命令示例**:
```bash
tar -czvf training_results.tar.gz checkpoints/ logs/
```
---
## 5. 常见问题 (FAQ)
- **ImportError: FaceDataset.py not found**
  - 确保 `FaceDataset.py` 位于项目根目录。
- **FileNotFoundError: .../faces_emore/...**
  - 请检查 `train_distill.py` 中的 `data_root` 变量是否已修改为您本地的正确路径。
- **RuntimeError: CUDA out of memory**
  - 尝试减小 `per_gpu_batch` (例如从 64 减至 32 或 16)。
- **Environment Issues**
  - 如果遇到 `transformers` 相关错误，请确保均已安装 `requirements.txt` 中的依赖。
