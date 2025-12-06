"""
训练脚本
多模态LSTM时序预测模型训练
"""
import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np

# 添加src到路径
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from utils import load_config, set_seed, get_device, count_parameters
from dataset import create_dataloaders
from models.multimodal_model import build_model


class WeightedMSELoss(nn.Module):
    """加权MSE损失"""

    def __init__(self, config):
        super().__init__()

        loss_config = config['training']['loss']
        param_groups = loss_config['param_groups']
        weights = loss_config['weights']

        # 构建权重向量 [10]
        weight_vector = torch.ones(10)

        for group_name, indices in param_groups.items():
            weight = weights[group_name]
            for idx in indices:
                weight_vector[idx] = weight

        self.register_buffer('weight_vector', weight_vector)

    def forward(self, pred, target):
        """
        Args:
            pred: [B, 10, pred_horizon]
            target: [B, pred_horizon, 10]

        Returns:
            loss: scalar
        """
        # 转换target维度 [B, pred_horizon, 10] -> [B, 10, pred_horizon]
        target = target.transpose(1, 2)

        # 计算MSE
        mse = (pred - target) ** 2  # [B, 10, pred_horizon]

        # 应用权重
        weighted_mse = mse * self.weight_vector.view(1, -1, 1)

        # 平均
        loss = weighted_mse.mean()

        return loss


class Trainer:
    """训练器"""

    def __init__(self, config, model, train_loader, val_loader, device, start_epoch=0):
        self.config = config
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.start_epoch = start_epoch

        # 优化器
        opt_config = config['training']['optimizer']
        if opt_config['type'] == 'adam':
            self.optimizer = optim.Adam(
                model.parameters(),
                lr=opt_config['lr'],
                weight_decay=opt_config['weight_decay'],
                betas=opt_config.get('betas', [0.9, 0.999])
            )
        else:
            raise ValueError(f"不支持的优化器: {opt_config['type']}")

        # 学习率调度器
        sched_config = config['training']['scheduler']
        if sched_config['type'] == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=sched_config['T_max'],
                eta_min=sched_config['min_lr']
            )
        else:
            raise ValueError(f"不支持的调度器: {sched_config['type']}")

        # 损失函数
        self.criterion = WeightedMSELoss(config).to(device)

        # TensorBoard
        log_dir = Path(config['project']['root']) / config['logging']['tensorboard']['log_dir']
        self.writer = SummaryWriter(log_dir)

        # 训练配置
        self.epochs = config['training']['epochs']
        self.grad_clip = config['training']['gradient_clip']
        self.use_amp = config['training']['mixed_precision']

        # Early Stopping
        self.early_stopping = config['training']['early_stopping']['enabled']
        self.patience = config['training']['early_stopping']['patience']
        self.best_val_loss = float('inf')
        self.patience_counter = 0

        # 模型保存路径
        self.checkpoint_dir = Path(config['project']['root']) / config['project']['checkpoint_dir']
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Mixed Precision
        self.scaler = torch.cuda.amp.GradScaler() if self.use_amp else None

        # 统计
        self.global_step = 0

    def train_epoch(self, epoch):
        """训练一个epoch"""
        self.model.train()
        epoch_loss = 0.0

        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}/{self.epochs} [Train]')

        for batch_idx, batch in enumerate(pbar):
            # 数据移至设备
            camera = batch['camera'].to(self.device)
            sonar = batch['sonar'].to(self.device)
            navigation = batch['navigation'].to(self.device)
            target = batch['target'].to(self.device)

            # 前向传播
            self.optimizer.zero_grad()

            if self.use_amp:
                with torch.cuda.amp.autocast():
                    pred = self.model(camera, sonar, navigation)
                    loss = self.criterion(pred, target)

                # 反向传播
                self.scaler.scale(loss).backward()

                # 梯度裁剪
                if self.grad_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                pred = self.model(camera, sonar, navigation)
                loss = self.criterion(pred, target)

                # 反向传播
                loss.backward()

                # 梯度裁剪
                if self.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

                self.optimizer.step()

            # 统计
            epoch_loss += loss.item()
            self.global_step += 1

            # 更新进度条
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

            # TensorBoard记录
            if self.global_step % 10 == 0:
                self.writer.add_scalar('Train/Loss', loss.item(), self.global_step)
                self.writer.add_scalar('Train/LR', self.optimizer.param_groups[0]['lr'], self.global_step)

        avg_loss = epoch_loss / len(self.train_loader)
        return avg_loss

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

                # 前向传播（与训练保持一致的混合精度设置）
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

    def save_checkpoint(self, epoch, is_best=False):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }

        # 保存最新checkpoint
        latest_path = self.checkpoint_dir / 'latest_model.pth'
        torch.save(checkpoint, latest_path)

        # 保存最优模型
        if is_best:
            best_path = self.checkpoint_dir / 'best_model.pth'
            torch.save(checkpoint, best_path)
            print(f"  最优模型已保存: {best_path}")

    def train(self):
        """完整训练流程"""
        print(f"\n开始训练，共{self.epochs}个epoch")
        print(f"设备: {self.device}")
        print(f"模型参数量: {count_parameters(self.model)}")

        if self.start_epoch > 0:
            print(f"从 Epoch {self.start_epoch} 恢复训练")

        try:
            for epoch in range(self.start_epoch, self.epochs):
                # 训练
                train_loss = self.train_epoch(epoch)

                # 清理CUDA缓存（防止验证时显存不足）
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                # 验证
                val_loss = self.validate(epoch)

                # 学习率调度
                self.scheduler.step()

                # 打印信息
                print(f"\nEpoch {epoch+1}/{self.epochs}:")
                print(f"  Train Loss: {train_loss:.6f}")
                print(f"  Val Loss:   {val_loss:.6f}")
                print(f"  LR:         {self.optimizer.param_groups[0]['lr']:.6f}")

                # 保存检查点
                is_best = val_loss < self.best_val_loss
                if is_best:
                    self.best_val_loss = val_loss
                    self.patience_counter = 0
                else:
                    self.patience_counter += 1

                self.save_checkpoint(epoch, is_best)

                # Early Stopping
                if self.early_stopping and self.patience_counter >= self.patience:
                    print(f"\nEarly Stopping! 验证集loss已{self.patience}个epoch未改善")
                    break

            print("\n训练完成!")
            print(f"最优验证Loss: {self.best_val_loss:.6f}")
            self.writer.close()

        except KeyboardInterrupt:
            print("\n\n" + "="*60)
            print("⚠️ 用户中断了训练 (Ctrl + C)")
            print(f"当前 Epoch: {epoch+1}/{self.epochs}")
            print("正在保存最后的模型状态...")
            print("="*60)

            # 保存当前状态
            self.save_checkpoint(epoch, is_best=False)
            print("✅ 模型已保存到 checkpoints/latest_model.pth")
            print("下次运行 python train.py 时将自动恢复训练")
            print("="*60)

            self.writer.close()
            raise


def main():
    """主函数"""
    # 加载配置
    config_path = Path(__file__).parent / 'configs' / 'config.yaml'
    config = load_config(config_path)

    # 设置随机种子
    set_seed(config['seed'])

    # 获取设备
    device = get_device(config['device']['type'])

    print("="*60)
    print("水下多模态时序预测 - 模型训练")
    print("="*60)

    # 创建DataLoader
    print("\n[1/3] 加载数据...")
    dataloaders = create_dataloaders(config, num_workers=config['training']['num_workers'])

    # 构建模型
    print("\n[2/3] 构建模型...")
    model = build_model(config)
    print(f"模型: {config['model']['name']}")

    # 检查是否有checkpoint需要恢复
    checkpoint_dir = Path(config['project']['root']) / config['project']['checkpoint_dir']
    latest_ckpt_path = checkpoint_dir / 'latest_model.pth'
    start_epoch = 0

    if latest_ckpt_path.exists():
        print(f"\n找到检查点: {latest_ckpt_path}")
        print("尝试加载检查点以恢复训练...")

        try:
            checkpoint = torch.load(latest_ckpt_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            print(f"✅ 检查点加载成功")
            print(f"  从 Epoch {start_epoch} 恢复训练")
            print(f"  最佳验证Loss: {checkpoint.get('best_val_loss', 'N/A')}")
        except Exception as e:
            print(f"⚠️ 检查点加载失败: {e}")
            print("  将从头开始训练")
            start_epoch = 0

    # 训练
    print("\n[3/3] 开始训练...")
    trainer = Trainer(config, model, dataloaders['train'], dataloaders['val'], device, start_epoch=start_epoch)
    trainer.train()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n" + "="*60)
        print("⚠️ 检测到用户中断 (Ctrl + C)")
        print("正在保存最后的模型状态...")
        print("="*60)
        import sys
        sys.exit(0)
