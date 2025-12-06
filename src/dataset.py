"""
PyTorch Dataset类
用于加载预处理后的序列数据
"""
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path


class UnderwaterSequenceDataset(Dataset):
    """水下多模态时序数据集"""

    def __init__(self, data_dir, split='train'):
        """
        Args:
            data_dir: 预处理数据根目录
            split: 'train', 'val', 或 'test'
        """
        self.data_dir = Path(data_dir) / split
        self.split = split

        # 获取所有序列文件
        self.sequence_files = sorted(list(self.data_dir.glob('seq_*.npz')))

        if len(self.sequence_files) == 0:
            raise ValueError(f"在 {self.data_dir} 中未找到序列文件")

        print(f"加载 {split} 数据集: {len(self.sequence_files)} 个序列")

    def __len__(self):
        return len(self.sequence_files)

    def __getitem__(self, idx):
        """
        Args:
            idx: 序列索引

        Returns:
            dict: {
                'camera': [T, 3, 224, 224],
                'sonar': [T, 3, 320, 320],
                'navigation': [T, 10],
                'target': [pred_horizon, 10]
            }
        """
        # 加载.npz文件
        data = np.load(self.sequence_files[idx])

        # 转换为tensor
        sample = {
            'camera': torch.from_numpy(data['camera']).float(),
            'sonar': torch.from_numpy(data['sonar']).float(),
            'navigation': torch.from_numpy(data['navigation']).float(),
            'target': torch.from_numpy(data['target']).float()
        }

        return sample


def create_dataloaders(config, num_workers=2):
    """
    创建训练、验证和测试DataLoader

    Args:
        config: 配置字典
        num_workers: DataLoader工作进程数

    Returns:
        dict: {'train': DataLoader, 'val': DataLoader, 'test': DataLoader}
    """
    data_dir = Path(config['project']['root']) / 'data' / 'processed'
    batch_size = config['training']['batch_size']
    pin_memory = config['training'].get('pin_memory', True)

    # 创建Dataset
    train_dataset = UnderwaterSequenceDataset(data_dir, 'train')
    val_dataset = UnderwaterSequenceDataset(data_dir, 'val')
    test_dataset = UnderwaterSequenceDataset(data_dir, 'test')

    # 创建DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    return {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader
    }
