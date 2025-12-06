# 水下多模态时序预测项目 - 实施计划

## 项目信息
- **项目名称**: underwater_multimodal_prediction
- **项目路径**: `D:\Vscodeprogram\underwater_multimodal_prediction`
- **数据源**: `D:\水下小目标项目\数据集-收集中\新数据\Sunboat_03-09-2023\2023-09-03-07-58-37`
- **GPU**: NVIDIA GeForce RTX 3060 Laptop (6GB VRAM)
- **Python版本**: 3.13

## 项目目标
基于Camera图像、Sonar声纳图像和Navigation数据，构建多模态LSTM时序预测模型，预测未来的航行状态（经纬度、深度、姿态角、速度等）。

---

## 数据对齐机制（已验证 ✓）

### 数据结构
```
数据集目录/
├── camera/
│   ├── camera.csv          # 3829行（1行header + 3828行数据）
│   └── 00001.png - 03828.png  # 3828张图像 (1692x1355)
├── sonar/
│   ├── sonar.csv           # 3829行（1行header + 3828行数据）
│   └── 00001.png - 03828.png  # 3828张图像 (1692x1355)
├── navigation/
│   └── navigation.csv      # 3829行（1行header + 3828行数据）
└── samples.json            # 3828个样本（索引0-3827）
```

### 关键对齐规则
1. **CSV文件**: 第1行是header，数据从第2行开始，共3828行数据
2. **samples.json**: 索引0-3827，共3828个样本
3. **图像文件**: 00001.png - 03828.png，共3828张
4. **索引对应**: samples[i] → CSV数据iloc[i] → 图像文件 {i+1:05d}.png

### 时间戳对齐验证结果
三个模态的timestamp存在微小差异（0.01-0.05秒），但samples.json已完成对齐：

示例（样本0）：
- Camera:     1693728637.417
- Sonar:      1693728637.458
- Navigation: 1693728637.408
- 时间差: < 0.05秒

**数据加载策略**:
```python
# 根据samples.json索引加载对齐的数据
sample = samples_json["samples"][i]
cam_idx = sample["camera"][0]      # 索引i
son_idx = sample["sonar"][0]        # 索引i
nav_idx = sample["navigation"][0]  # 索引i

# 从CSV加载
cam_filename = camera_csv.iloc[cam_idx]['filename']  # 如 "00001.png"
son_filename = sonar_csv.iloc[son_idx]['filename']
nav_data = nav_csv.iloc[nav_idx]  # 包含10个预测目标

# 加载图像
camera_img = load_image(f"camera/{cam_filename}")
sonar_img = load_image(f"sonar/{son_filename}")
```

---

## 图像预处理（已测试 ✓）

### Camera图像
- 原始尺寸: 1692×1355
- 目标尺寸: 224×224 (RGB)
- 处理: 直接resize + ImageNet归一化

### Sonar图像（关键）
**测试结果**（已验证100张图像）:
- 原始尺寸: 1692×1355
- **ROI裁剪范围**: [293:497, 296:637] (扩大版本，保证完整)
- ROI尺寸: 204×341像素
- 目标尺寸: 320×320 (保留声纳细节)

**裁剪效果**: ✓ 成功保留紫色扇形声纳谱，无信息丢失

**处理流程**:
```python
# 使用PIL读取（支持中文路径）
pil_img = Image.open(img_path)
img = np.array(pil_img)

# 使用配置文件中的ROI坐标裁剪
y1, y2, x1, x2 = 293, 497, 296, 637

# 裁剪并resize
cropped = pil_img.crop((x1, y1, x2, y2))
resized = cropped.resize((320, 320), Image.BILINEAR)

# 归一化
normalized = (np.array(resized) / 255.0 - 0.5) / 0.5
```

---

## 时序参数设计

### 采样间隔
- 平均间隔: **0.5-0.7秒/帧**
- 数据特性: 水下航行，惯性大，变化平稳

### 序列参数（可配置）
```yaml
seq_length: 20        # 输入历史长度（约10-14秒）
pred_horizon: 5       # 预测未来步长（约2.5-3.5秒）
stride: 1             # 滑动窗口步长
```

### 有效样本数计算
- 总样本: 3828
- 训练集: 2680样本 → 实际序列: 2655
- 验证集: 574样本 → 实际序列: 550
- 测试集: 574样本 → 实际序列: 551

---

## 预测目标（10个参数）

根据navigation.csv的列：

**位置信息 (3个)**:
- latitude (纬度)
- longitude (经度)
- depth (深度)

**姿态信息 (3个)**:
- yaw (航向角)
- pitch (俯仰角)
- roll (翻滚角)

**速度信息 (3个)**:
- velocity_x
- velocity_y
- velocity_z

**高度信息 (1个)**:
- altitude

---

## 模型架构设计

### 整体流程
```
输入: [B, T=20, ...]
    ↓
Camera[B,T,3,224,224] → CNN → [B,T,512]
Sonar[B,T,3,320,320]  → CNN → [B,T,512]
Navigation[B,T,10]    → FC  → [B,T,128]
    ↓
Concat → [B,T,1152]
    ↓
LSTM(hidden=256, layers=2) → [B, 256]
    ↓
FC → [B, 10, 5]  # 10个参数 × 5个未来时间步
```

### 详细组件

#### 1. CNN特征提取器
```python
# Camera和Sonar共享相似架构
Camera Encoder:
  ResNet18 (pretrained on ImageNet)
  Input: [3, 224, 224]
  Output: [512]

Sonar Encoder:
  ResNet18 (pretrained, 允许fine-tune)
  Input: [3, 320, 320]
  Output: [512]
```

#### 2. Navigation编码器
```python
FC Encoder:
  Input: [10]
  Linear(10 → 64) → ReLU → Dropout(0.2)
  Linear(64 → 128) → ReLU
  Output: [128]
```

#### 3. 多模态融合
```python
Fusion:
  Concat([cam_feat, sonar_feat, nav_feat])
  → [512+512+128 = 1152]
  Linear(1152 → 512) → ReLU → Dropout(0.3)
  Output: [512]
```

#### 4. LSTM时序建模
```python
LSTM:
  input_size: 512
  hidden_size: 256
  num_layers: 2
  dropout: 0.3
  Output: [256]
```

#### 5. 预测头
```python
Predictor:
  Linear(256 → 128) → ReLU → Dropout(0.2)
  Linear(128 → 50)  # 10参数 × 5步长
  Reshape: [10, 5]
```

### 损失函数
```python
# 加权MSE Loss
loss = 2.0 * mse(位置) + 1.0 * mse(姿态) + 1.0 * mse(速度) + 1.0 * mse(高度)
```

---

## 训练策略

### 优化器配置
```yaml
optimizer: Adam
lr: 0.001
weight_decay: 0.0001
batch_size: 8  # RTX 3060 6GB显存，较小batch避免OOM

lr_scheduler: CosineAnnealingLR
T_max: 50
min_lr: 1e-6
```

### 训练技巧
- **梯度裁剪**: max_norm=1.0（防止LSTM梯度爆炸）
- **Early Stopping**: patience=10
- **模型保存**: 保存验证集最优模型
- **Mixed Precision**: 使用torch.cuda.amp（节省显存）

### 数据增强（可选）
- Navigation数据添加轻微高斯噪声
- 时间步随机dropout

---

## 评估指标

### 回归指标（每个参数）
- **MAE** (Mean Absolute Error)
- **RMSE** (Root Mean Square Error)
- **MAPE** (Mean Absolute Percentage Error)

### 时序指标
- **Horizon-wise MAE**: 每个预测步长的误差
- **早期 vs 后期预测**: 第1步 vs 第5步误差对比

### 关键指标
重点关注位置预测准确性：
- Latitude RMSE < 0.00001° (约1米)
- Longitude RMSE < 0.00001° (约1米)
- Depth RMSE < 0.5m

---

## 可视化方案

### 1. 训练过程（TensorBoard）
- Loss曲线（train vs val）
- 学习率变化
- 各参数MAE曲线

### 2. 预测结果对比图（5组）

**图1: 位置预测** (Latitude, Longitude, Depth)
```python
3个子图，每个显示真实值 vs 预测值的时间序列
```

**图2: 姿态预测** (Yaw, Pitch, Roll)
```python
3个子图，时间序列折线图
```

**图3: 速度预测** (Vx, Vy, Vz)
```python
3个子图，时间序列折线图
```

**图4: 高度预测** (Altitude)
```python
单图，时间序列折线图
```

**图5: 2D轨迹地图**
```python
在经纬度平面绘制:
- 真实路径（蓝色实线）
- 预测路径（红色虚线）
- 起点（绿色圆点）
- 终点（红色×）
```

### 3. 误差分析
- 10×5误差热力图（参数×时间步）
- 误差分布直方图
- Horizon误差曲线

---

## 项目目录结构

```
D:\Vscodeprogram\underwater_multimodal_prediction\
├── README.md                          # 本文档
├── requirements.txt                   # Python依赖
├── test_sonar_crop.py                 # Sonar裁剪测试脚本 ✓
│
├── configs/
│   └── config.yaml                    # 配置文件
│
├── data/
│   ├── raw/                           # 原始数据（软链接，不复制）
│   ├── processed/                     # 预处理后数据
│   │   ├── train/                    # .npz序列文件
│   │   ├── val/
│   │   └── test/
│   ├── statistics.json               # 归一化统计信息
│   └── split_indices.json            # 数据集划分索引
│
├── src/
│   ├── preprocessing/
│   │   ├── data_loader.py            # 加载CSV和图像
│   │   ├── image_processor.py        # 图像预处理（ROI、resize）
│   │   ├── sequence_builder.py       # 构建时序序列
│   │   └── normalization.py          # 数据归一化
│   │
│   ├── models/
│   │   ├── cnn_encoder.py            # CNN特征提取器
│   │   ├── nav_encoder.py            # Navigation编码器
│   │   ├── fusion_module.py          # 多模态融合
│   │   ├── lstm_predictor.py         # LSTM时序预测器
│   │   └── multimodal_model.py       # 完整模型
│   │
│   ├── dataset.py                     # PyTorch Dataset
│   ├── train.py                       # 训练脚本
│   ├── evaluate.py                    # 评估脚本
│   ├── visualize.py                   # 可视化脚本
│   └── utils.py                       # 工具函数
│
├── checkpoints/                       # 模型权重
│   └── best_model.pth
│
├── results/                           # 结果输出
│   ├── predictions/                  # 预测结果.npz
│   ├── figures/                      # 可视化图像
│   │   └── sonar_crop_test.png      # ✓ Sonar裁剪测试结果
│   └── metrics.json                  # 评估指标
│
└── logs/                              # 训练日志
    └── tensorboard/
```

---

## 实施步骤

### 第1步: 环境搭建 ✓
- [x] 创建项目目录
- [x] 验证GPU可用性（RTX 3060）
- [x] 安装基础依赖（pandas, numpy, opencv, PIL, matplotlib）

### 第2步: 数据验证与探索 ✓
- [x] 验证数据对齐机制（3828组样本）
- [x] 测试Sonar图像ROI裁剪效果

### 第3步: 配置文件 ✓
- [x] 编写config.yaml
- [x] 定义所有超参数

### 第4步: 数据预处理管道 ✓
- [x] 实现图像加载器（支持中文路径）
- [x] 实现Sonar ROI裁剪
- [x] 实现Camera resize
- [x] 实现序列构建器（滑动窗口）
- [x] 实现归一化模块
- [x] 生成预处理数据（.npz格式）

### 第5步: 模型实现 ✓
- [x] CNN编码器（ResNet18）
- [x] Navigation编码器
- [x] 融合模块
- [x] LSTM预测器
- [x] 完整模型组装
- [x] 模型forward测试

### 第6步: 数据集与加载器 ✓
- [x] PyTorch Dataset类
- [x] DataLoader配置
- [x] 数据加载测试

### 第7步: 训练脚本 ✓
- [x] 训练循环
- [x] 验证循环
- [x] TensorBoard日志
- [x] 模型保存与恢复
- [x] Early Stopping

### 第8步: 评估与可视化 ✓
- [x] 评估指标计算
- [x] 生成5类预测对比图
- [x] 误差分析可视化
- [x] 保存结果

---

## 配置文件示例 (config.yaml)

```yaml
# ============ 项目路径 ============
project:
  root: "D:/Vscodeprogram/underwater_multimodal_prediction"
  data_root: "D:/水下小目标项目/数据集-收集中/新数据/Sunboat_03-09-2023/2023-09-03-07-58-37"

# ============ 数据配置 ============
data:
  total_samples: 3828
  seq_length: 20
  pred_horizon: 5
  stride: 1

  split:
    train_ratio: 0.70  # 2680样本
    val_ratio: 0.15    # 574样本
    test_ratio: 0.15   # 574样本

# ============ 图像处理 ============
image:
  camera:
    target_size: [224, 224]
    normalize_mean: [0.485, 0.456, 0.406]  # ImageNet
    normalize_std: [0.229, 0.224, 0.225]

  sonar:
    roi: [358, 482, 371, 549]  # [y1, y2, x1, x2]
    target_size: [320, 320]
    normalize_mean: [0.5, 0.5, 0.5]
    normalize_std: [0.5, 0.5, 0.5]

# ============ 模型配置 ============
model:
  cnn:
    backbone: "resnet18"
    pretrained: true
    feature_dim: 512

  nav_encoder:
    input_dim: 10
    hidden_dim: 64
    output_dim: 128
    dropout: 0.2

  fusion:
    input_dim: 1152
    output_dim: 512
    dropout: 0.3

  lstm:
    input_size: 512
    hidden_size: 256
    num_layers: 2
    dropout: 0.3

  predictor:
    hidden_dim: 128
    dropout: 0.2

# ============ 训练配置 ============
training:
  batch_size: 8
  epochs: 50
  num_workers: 2

  optimizer:
    type: "adam"
    lr: 0.001
    weight_decay: 0.0001

  scheduler:
    type: "cosine"
    T_max: 50
    min_lr: 1.0e-6

  loss_weights:
    position: 2.0
    orientation: 1.0
    velocity: 1.0
    altitude: 1.0

  early_stopping:
    patience: 10

  gradient_clip: 1.0
  mixed_precision: true

# ============ 硬件配置 ============
device: "cuda"
seed: 42
```

---

## 关键技术点

### 1. 中文路径处理
- 使用PIL Image.open()而非cv2.imread()
- 所有路径使用forward slash或pathlib.Path

### 2. 显存优化（6GB VRAM）
- batch_size=8（较小）
- Mixed precision训练
- 梯度累积（可选）
- 及时清理中间变量

### 3. 时序数据特殊性
- 按时间顺序划分数据集
- 不跨边界构建序列
- 梯度裁剪防止LSTM爆炸

### 4. 数据归一化
- 位置：Min-Max归一化
- 角度：转换为弧度后归一化
- 速度：Z-score标准化
- 保存统计信息用于反归一化

---

## 预期结果

### 性能目标
- Latitude/Longitude RMSE < 0.00001° (约1米)
- Depth RMSE < 0.5m
- Yaw RMSE < 5°
- 训练时间: 约4-6小时（50 epochs）

### 交付物
1. 训练好的模型权重 (.pth)
2. 完整源代码
3. 5类可视化结果图
4. 评估报告 (metrics.json)
5. README文档

---

## 后续优化方向

1. **数据增加**: 使用11-37数据集（5011组）
2. **模型改进**: 尝试Transformer、Attention机制
3. **多任务学习**: 同时预测障碍物检测
4. **在线推理**: 部署实时预测系统

---

## 快速开始 🚀

### 1. 数据预处理

**运行预处理脚本**（在VSCode终端中）：
```bash
cd D:\Vscodeprogram\underwater_multimodal_prediction
python preprocess.py
```

**预处理过程**：
- 加载3828组样本
- 按70/15/15划分训练/验证/测试集
- 处理图像（Camera 224×224, Sonar ROI裁剪+320×320）
- 构建时序序列（seq_length=20, pred_horizon=5）
- 保存为.npz格式

**预计时间**：2-3小时（解除CPU功率限制后）

**输出**：
```
data/processed/
├── train/     # 2655个序列
├── val/       # 550个序列
└── test/      # 550个序列
```

### 2. 模型训练

**运行训练脚本**：
```bash
python train.py
```

**训练特性**：
- 加权MSE损失（位置权重2.0）
- Adam优化器 + Cosine学习率衰减
- 梯度裁剪 + Mixed Precision
- TensorBoard可视化
- Early Stopping（patience=10）
- 自动保存最优模型

**预计时间**：4-6小时（50 epochs，RTX 3060）

**查看训练进度**：
```bash
tensorboard --logdir logs/tensorboard
```

### 3. 模型评估

**运行评估脚本**：
```bash
python evaluate.py
```

**输出**：
- 10个参数的MAE、RMSE、MAPE
- 每个预测步长的误差
- 关键指标达标检查
- 保存预测结果到 `results/predictions/test_predictions.npz`

### 4. 结果可视化

**运行可视化脚本**：
```bash
python visualize.py
```

**生成7类图表**：
1. `position_prediction.png` - 位置预测对比
2. `orientation_prediction.png` - 姿态预测对比
3. `velocity_prediction.png` - 速度预测对比
4. `altitude_prediction.png` - 高度预测对比
5. `trajectory_map.png` - 2D轨迹地图
6. `error_heatmap.png` - 误差热力图
7. `horizon_error_curve.png` - 预测步长误差曲线

### 5. 推理使用

**加载模型进行预测**：
```python
from predict import Predictor
from pathlib import Path
import numpy as np

# 加载预测器
predictor = Predictor(config, checkpoint_path)

# 准备输入（20帧历史数据）
camera_paths = [Path('camera/00001.png'), ..., Path('camera/00020.png')]
sonar_paths = [Path('sonar/00001.png'), ..., Path('sonar/00020.png')]
navigation_data = np.array([...])  # shape: [20, 10]

# 预测未来5个时间步
predictions = predictor.predict(camera_paths, sonar_paths, navigation_data)
# predictions shape: [5, 10]
```

---

## 项目文件说明 📁

### 主要脚本
- `preprocess.py` - 数据预处理主脚本
- `train.py` - 模型训练脚本
- `evaluate.py` - 模型评估脚本
- `visualize.py` - 结果可视化脚本
- `predict.py` - 推理脚本

### 核心模块
- `src/utils.py` - 工具函数
- `src/dataset.py` - PyTorch Dataset
- `src/preprocessing/` - 预处理模块（5个文件）
- `src/models/` - 模型模块（5个文件）

### 配置文件
- `configs/config.yaml` - 完整配置
- `requirements.txt` - Python依赖

---

## 性能优化建议 ⚡

### 预处理加速
- **解除CPU功率限制** → 速度提升50-100%
- 使用SSD存储数据
- 减少不必要的图像操作

### 训练加速
- 使用Mixed Precision（已实现）
- 适当调大batch_size（如果显存足够）
- 冻结CNN前几层（减少计算）

### 显存优化
- 当前batch_size=8（6GB VRAM安全）
- 如果OOM，降低到batch_size=4
- 使用梯度累积模拟大batch

---

## 常见问题 ❓

### Q1: 预处理很慢怎么办？
**A**: 解除CPU功率限制，或者使用多进程加速（需修改代码）

### Q2: 训练时GPU利用率低？
**A**: 增大batch_size或num_workers，检查数据加载是否成为瓶颈

### Q3: 验证集loss不下降？
**A**: 检查学习率是否过大，尝试调整loss权重，增加数据增强

### Q4: 如何使用其他数据集？
**A**: 修改 `config.yaml` 中的 `data_root` 路径，重新运行预处理

### Q5: 如何调整预测步长？
**A**: 修改 `config.yaml` 中的 `pred_horizon`，重新运行预处理和训练

---

## 已知问题与修复记录 🔧

### 问题1：混合精度训练 CUDA 错误 ✅ 已修复

**现象**：
- 训练第1个epoch后，验证阶段报错：
  ```
  RuntimeError: GET was unable to find an engine to execute this computation
  ```

**根本原因**：
- 训练时启用了 `torch.cuda.amp.autocast()`（FP16混合精度）
- 验证时直接使用 FP32 推理
- FP16/FP32 类型不匹配导致 CUDA 无法找到合适的计算引擎

**修复方案**：
- 在 `train.py` 的 `validate()` 方法中也使用 `autocast()`
- 在验证前清理 CUDA 缓存（`torch.cuda.empty_cache()`）

**详细记录**：见 `DEBUG_REPORT.md`

**状态**：✅ 已完全修复（2025-12-05）

### 问题2：断点恢复不完整 ⚠️ 部分实现

**现状**：
- 当前只恢复模型权重（`model_state_dict`）
- 未恢复优化器状态（`optimizer_state_dict`）和学习率调度器状态（`scheduler_state_dict`）

**影响**：
- 恢复训练时学习率调度器重置
- 可能轻微影响收敛速度

**计划修复**：优先级 P1

---

## 更新日志 📝

- **2025-12-06**: 模型训练进行中（第四轮）
- **2025-12-06**: 添加断点恢复和 Ctrl+C 安全保存功能
- **2025-12-05**: 修复混合精度训练 CUDA 错误
- **2025-12-05**: 完成所有核心模块实现（13个文件）
  - ✅ 数据预处理模块（5个文件）
  - ✅ 多模态LSTM模型（5个文件）
  - ✅ 训练脚本（TensorBoard + Early Stopping）
  - ✅ 评估脚本（完整指标计算）
  - ✅ 可视化脚本（7类图表）
  - ✅ 推理脚本
- **2025-12-05**: 完成数据预处理（3756个序列）
- **2025-12-04**: 初始计划创建
- **2025-12-04**: 完成数据对齐验证（3828组样本）
- **2025-12-04**: 完成Sonar ROI裁剪测试（ROI: [293:497, 296:637]）
- **2025-12-04**: 确认GPU环境（RTX 3060 Laptop 6GB）

---

## 项目状态 ✅

**所有核心功能已完整实现并运行！**

**已完成的工作**：
1. ✅ 数据预处理（已生成 2655+550+551 个序列文件）
2. ✅ 模型实现（13个Python模块，总计约2000行代码）
3. ✅ 训练脚本（已开始训练，存在模型权重）
4. ✅ 评估与可视化脚本（完整实现）

**当前状态**：
- 📊 模型训练进行中/已完成（检查 `checkpoints/` 目录）
- 💾 已保存模型：`best_model.pth` 和 `latest_model.pth`
- 🎯 可运行完整的评估和可视化流程

**快速开始**：
- 如需继续训练：`python train.py`（自动从断点恢复）
- 评估已训练模型：`python evaluate.py`
- 生成可视化结果：`python visualize.py`
- 使用模型推理：见 `predict.py` 示例代码
