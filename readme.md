# ABCD Project
## 1. 环境配置 (Environment Setup)
本项目基于 Python 3.10+ 和 PyTorch 2.1+ (CUDA 12.1)。
### 1.1 创建虚拟环境并安装 PyTorch
建议使用 Conda 创建环境：
```bash
conda create -n face_distill python=3.10 -y
conda activate face_distill
# 安装 PyTorch (CUDA 12.1)
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```
### 1.2 安装其他依赖
请确保项目目录下有 `requirements.txt` 文件，然后运行：
```bash
pip install -r requirements.txt
```
---
## 2. 数据准备 (Data Preparation)
请严格按照以下目录结构存放数据，以确保 `train_distill.py` 能正确读取。建议在项目根目录下创建 `datasets` 和 `blip2_weights` 文件夹。
### 2.1 数据集 (Face Emore)
- **下载链接**: [点击这里下载 (GDrive)](https://drive.google.com/file/d/1SXS4-Am3bsKSK615qbYdbA_FMVh3sAvR/view)
- **存放路径**: 请将下载的数据解压到项目根目录下的 `datasets/faces_emore`。
- **说明**: 该目录应包含人脸图像数据，供 `FaceDataset.py` 读取。
### 2.2 教师模型特征 (Teacher Features)
- **文件夹名称**: `teacher_features_12k`
- **存放路径**: 请将该文件夹直接放在项目根目录下 (即与 `train_distill.py` 同级)。
- **内容**: 包含 `teacher_embedding.npy` 和 `teacher_feat_map.npy`，用于计算蒸馏损失。
### 2.3 预训练权重 (BLIP-2 Weights)
- **下载脚本**: 运行项目中的 `download_weight.py` 自动下载权重。
- **存放路径**: 脚本会自动将权重保存到项目根目录下的 `blip2_weights` (确保 `download_weight.py` 中的保存路径与此一致)。
- **代码对应**: `train_distill.py` 中的 `BLIP_PATH` 默认为 `./blip2_weights`。
**推荐的文件结构**:
```text
[Project Root]/
├── train_distill.py       # 主训练脚本
├── download_weight.py     # 权重下载脚本
├── FaceDataset.py         # 数据集加载类
├── requirements.txt       # 依赖列表
├── teacher_features_12k/  # 教师特征 (文件夹)
│   ├── teacher_embedding.npy
│   └── teacher_feat_map.npy
├── blip2_weights/         # BLIP-2 预训练权重 (由 download_weight.py 下载)
├── datasets/              # 数据集目录
│   └── faces_emore/       # 脸部数据集 (解压至此)
├── logs/                  # 训练日志 (自动生成)
└── checkpoints/           # 模型保存路径 (自动生成)
```
---
## 3. 训练 (Training)
### 3.1 启动训练
使用多卡 (GPU 0, 1, 2, 3) 进行分布式训练。
**运行命令**:
```bash
python train_distill.py
```
### 3.2 训练参数配置
主要参数在 `train_distill.py` 的 configuration 区域中定义，默认路径已修改为相对路径：
```python
    PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
    
    # 默认路径 (相对于项目根目录)
    BLIP_PATH = os.path.join(PROJECT_ROOT, "blip2_weights")
    DATA_ROOT = os.path.join(PROJECT_ROOT, "datasets", "faces_emore")
    TEACHER_FEAT_DIR = os.path.join(PROJECT_ROOT, "teacher_features_12k")
```
其他超参数：
| 参数 | 默认值 | 说明 |
| :--- | :--- | :--- |
| `BATCH_SIZE` | 128 | 全局批量大小 (4张卡时每张卡约32) |
| `EPOCHS` | 100 | 总训练轮数 |
| `LR` | 5e-4 | 初始学习率 (配合 CosineAnnealingLR) |
| `LAMBDA_AT` | 1000.0 | Attention Transfer 损失权重 |
| `LAMBDA_RKD` | 1.0 | Relational Knowledge Distillation 损失权重 |
### 3.3 显卡设置
代码默认使用 GPU 0, 1, 2, 3。
- `os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"`
### 3.4 训练特性
- **混合精度**: 暂未使用 (代码中未显式开启 amp)。
- **早停 (Early Stopping)**: 当 Loss 连续 5 个 Epoch 不再下降时，自动停止训练。
- **权重保存**:
  - 最优模型: `checkpoints/best_student_adapter_face.pth` (Loss 最低时保存)
  - 定期保存: 每 5 个 Epoch 保存一次，格式 `checkpoints/student_adapter_face_epoch_*.pth`
- **日志监控**: TensorBoard 日志保存在 `logs/adapter_face_distill`。
### 3.5 查看日志
```bash
tensorboard --logdir=./logs/adapter_face_distill
```
---
## 4. 结果保存与上传 (Result Saving & Uploading)
训练完成后，请务必保存并上传以下两类文件，以便后续分析和部署：
1. **模型权重 (Checkpoints)**
    - 路径: `checkpoints/` 文件夹
    - 关键文件:
        - `best_student_adapter_face.pth`: 训练过程中验证损失最低的最优模型权重 (务必上传)。
        - `student_adapter_face_epoch_*.pth`: 定期保存的中间权重 (可视情况上传)。
2. **训练日志 (Logs)**
    - 路径: `logs/adapter_face_distill/` 文件夹
    - 内容: 包含 TensorBoard 事件文件 (如 `events.out.tfevents...`)，记录了 Loss 曲线、学习率变化等重要信息。
**操作建议**:
可以使用 `tar` 或 `zip` 命令打包后再上传/下载：
```bash
# 打包 checkpoints 和 logs
tar -czvf training_results.tar.gz checkpoints/ logs/
```
---
## 5. 常见问题
- **ImportError: FaceDataset.py not found**: 请确保 `FaceDataset.py` 在项目根目录下。
- **RuntimeError: CUDA out of memory**: 请尝试减小 `BATCH_SIZE`。
- **Invalid device id**: 代码已自动适配 `CUDA_VISIBLE_DEVICES`，请勿手动修改 `device_ids` 列表，除非您了解 PyTorch 的设备映射机制。
