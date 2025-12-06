# 训练错误诊断报告

## 🔴 错误信息

```
RuntimeError: GET was unable to find an engine to execute this computation
```

**出现位置**：Epoch 1 验证步骤（Val）第一个 batch
**触发模块**：ResNet18 的 conv2 层

---

## 🔍 根本原因分析

### 问题 1：混合精度（Mixed Precision）不一致

**配置**：`configs/config.yaml` 第 143 行
```yaml
mixed_precision: true           # 启用混合精度（FP16）
```

**代码问题**：
- **训练时** ✅（train_epoch 第 145 行）：使用了 `torch.cuda.amp.autocast()`
  ```python
  if self.use_amp:
      with torch.cuda.amp.autocast():
          pred = self.model(camera, sonar, navigation)  # FP16 模式
          loss = self.criterion(pred, target)
  ```

- **验证时** ❌（validate 第 202 行）：直接调用，没有 autocast
  ```python
  pred = self.model(camera, sonar, navigation)  # FP32 模式
  loss = self.criterion(pred, target)
  ```

**结果**：
- 训练时：CNN 权重用 FP16 优化
- 验证时：试图用 FP32 推理 FP16 权重 → 数据类型不匹配 → CUDA 无法找到合适计算引擎

### 问题 2：显存压力

训练完成后显存接近满（RTX 3060 仅 6GB），验证时如果没有缓存清理可能导致显存溢出。

---

## ✅ 已实施的修复

### 修复 1：验证函数中使用一致的 autocast

**文件**：`train.py` 第 187-216 行

```python
def validate(self, epoch):
    """验证"""
    self.model.eval()
    val_loss = 0.0

    with torch.no_grad():
        pbar = tqdm(self.val_loader, desc=f'Epoch {epoch+1}/{self.epochs} [Val]')

        for batch in pbar:
            camera = batch['camera'].to(self.device)
            sonar = batch['sonar'].to(self.device)
            navigation = batch['navigation'].to(self.device)
            target = batch['target'].to(self.device)

            # 与训练保持一致的混合精度设置 ✅
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    pred = self.model(camera, sonar, navigation)
                    loss = self.criterion(pred, target)
            else:
                pred = self.model(camera, sonar, navigation)
                loss = self.criterion(pred, target)

            val_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    avg_val_loss = val_loss / len(self.val_loader)
    self.writer.add_scalar('Val/Loss', avg_val_loss, epoch)
    return avg_val_loss
```

### 修复 2：在验证前清理 CUDA 缓存

**文件**：`train.py` 第 239-257 行

```python
def train(self):
    """完整训练流程"""
    # ... 初始化代码 ...

    for epoch in range(self.epochs):
        # 训练
        train_loss = self.train_epoch(epoch)

        # 清理CUDA缓存（防止验证时显存不足）✅
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # 验证
        val_loss = self.validate(epoch)

        # 学习率调度
        self.scheduler.step()
        # ... 后续代码 ...
```

---

## 🚀 重新开始训练

### 步骤 1：停止当前运行（如果还在运行）

在 VSCode 终端按：`Ctrl + C`

### 步骤 2：清理旧的日志和检查点（可选）

```bash
# 删除旧的训练日志
rmdir /s /q logs\tensorboard

# 删除旧的模型权重
del /q checkpoints\*.pth
```

### 步骤 3：重新开始训练

```bash
python train.py
```

**预期输出**：
```
Loading config...
Setting device to cuda...
Building model...
Creating dataloaders...
模型参数量: 2,234,560

开始训练，共50个epoch

Epoch 1/50 [Train]: 100%|████████| 331/331 [45:19<00:00, 8.22s/it, loss=0.2995]
Epoch 1/50 [Val]:  100%|████████|  69/69  [12:30<00:00, 10.87s/it, loss=0.0825]

Epoch 1/50:
  Train Loss: 0.299532
  Val Loss: 0.082512
  Learning Rate: 0.000999
```

---

## 📊 预期训练进度

| Epoch | 预期时间 | 说明 |
|-------|---------|------|
| 1 | ~60分钟 | 包括训练(45min)和验证(15min) |
| 2-50 | 各~60分钟 | 总计约50小时 |
| **总计** | **~50小时** | 在 RTX 3060 上 |

---

## ⚠️ 如果还是出现错误

### 备选方案 A：关闭混合精度

编辑 `configs/config.yaml` 第 143 行：

```yaml
mixed_precision: false          # 禁用混合精度（更稳定但更慢）
```

**优点**：更稳定，避免数据类型问题
**缺点**：显存占用增加 20%，速度慢 10-15%

### 备选方案 B：降低 batch_size

编辑 `configs/config.yaml` 第 106 行：

```yaml
batch_size: 4                   # 从8降低到4（更安全）
```

**优点**：显存占用减半
**缺点**：需要重新预处理数据，训练变慢

### 备选方案 C：同时应用 A 和 B

```yaml
mixed_precision: false
batch_size: 4
```

此时需要重新预处理：
```bash
python preprocess.py
python train.py
```

---

## 📝 检查清单

运行前请确认：

- [ ] 已修改 `train.py`（混合精度一致性）
- [ ] CUDA 缓存清理已添加
- [ ] `configs/config.yaml` 配置正确
- [ ] `data/processed/` 文件夹存在
- [ ] Python 环境正常（`python --version` = 3.13.1）
- [ ] PyTorch 可用（`python -c "import torch; print(torch.cuda.is_available())"` = True）

---

## 🎯 验证修复成功

训练能否顺利完成第 1 个 epoch 的验证步骤，说明问题已解决。

如果顺利通过，将继续训练 50 个 epoch，大约 2-3 天内完成。

---

**生成时间**：2025-12-05
**Python 版本**：3.13.1
**PyTorch 版本**：2.7.1+cu118
**GPU**：NVIDIA GeForce RTX 3060 Laptop (6GB)
