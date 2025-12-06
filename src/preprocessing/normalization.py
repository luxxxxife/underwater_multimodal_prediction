"""
数据归一化模块
对navigation数据进行归一化和反归一化
"""
import numpy as np
import json
from pathlib import Path


class Normalizer:
    """数据归一化器"""

    def __init__(self, config=None):
        """
        Args:
            config: 配置字典，包含normalization部分
        """
        self.config = config
        self.stats = None  # 统计信息 {mean, std, min, max}

        # 参数分组（基于config中的target_columns顺序）
        # 0:latitude, 1:longitude, 2:depth, 3:yaw, 4:pitch, 5:roll,
        # 6:velocity_x, 7:velocity_y, 8:velocity_z, 9:altitude
        self.param_groups = {
            'position': [0, 1, 2],      # lat, lon, depth
            'orientation': [3, 4, 5],   # yaw, pitch, roll
            'velocity': [6, 7, 8],      # vx, vy, vz
            'altitude': [9]
        }

    def compute_stats(self, data):
        """
        计算数据统计信息

        Args:
            data: np.ndarray, shape [N, 10] 所有训练集navigation数据

        Returns:
            dict: 统计信息
        """
        self.stats = {
            'mean': np.mean(data, axis=0).tolist(),
            'std': np.std(data, axis=0).tolist(),
            'min': np.min(data, axis=0).tolist(),
            'max': np.max(data, axis=0).tolist(),
            'n_samples': len(data)
        }
        return self.stats

    def normalize(self, data):
        """
        归一化数据

        Args:
            data: np.ndarray, shape [..., 10]

        Returns:
            np.ndarray: 归一化后的数据，相同shape
        """
        if self.stats is None:
            raise ValueError("请先调用compute_stats计算统计信息")

        normalized = np.copy(data)
        mean = np.array(self.stats['mean'])
        std = np.array(self.stats['std'])
        min_val = np.array(self.stats['min'])
        max_val = np.array(self.stats['max'])

        # 根据不同参数类型使用不同归一化方法
        # Position: Min-Max归一化
        for idx in self.param_groups['position']:
            normalized[..., idx] = (data[..., idx] - min_val[idx]) / (max_val[idx] - min_val[idx] + 1e-8)

        # Orientation: 转换为弧度后Min-Max归一化到[-1, 1]
        for idx in self.param_groups['orientation']:
            # 假设输入已经是弧度
            normalized[..., idx] = (data[..., idx] - min_val[idx]) / (max_val[idx] - min_val[idx] + 1e-8) * 2 - 1

        # Velocity: Z-score标准化
        for idx in self.param_groups['velocity']:
            normalized[..., idx] = (data[..., idx] - mean[idx]) / (std[idx] + 1e-8)

        # Altitude: Min-Max归一化
        for idx in self.param_groups['altitude']:
            normalized[..., idx] = (data[..., idx] - min_val[idx]) / (max_val[idx] - min_val[idx] + 1e-8)

        return normalized

    def denormalize(self, data):
        """
        反归一化数据

        Args:
            data: np.ndarray, shape [..., 10] 归一化后的数据

        Returns:
            np.ndarray: 原始尺度的数据
        """
        if self.stats is None:
            raise ValueError("请先调用compute_stats计算统计信息")

        denormalized = np.copy(data)
        mean = np.array(self.stats['mean'])
        std = np.array(self.stats['std'])
        min_val = np.array(self.stats['min'])
        max_val = np.array(self.stats['max'])

        # Position: 反Min-Max
        for idx in self.param_groups['position']:
            denormalized[..., idx] = data[..., idx] * (max_val[idx] - min_val[idx]) + min_val[idx]

        # Orientation: 反Min-Max
        for idx in self.param_groups['orientation']:
            denormalized[..., idx] = (data[..., idx] + 1) / 2 * (max_val[idx] - min_val[idx]) + min_val[idx]

        # Velocity: 反Z-score
        for idx in self.param_groups['velocity']:
            denormalized[..., idx] = data[..., idx] * std[idx] + mean[idx]

        # Altitude: 反Min-Max
        for idx in self.param_groups['altitude']:
            denormalized[..., idx] = data[..., idx] * (max_val[idx] - min_val[idx]) + min_val[idx]

        return denormalized

    def save_stats(self, filepath):
        """保存统计信息到JSON文件"""
        if self.stats is None:
            raise ValueError("没有统计信息可保存")

        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, 'w') as f:
            json.dump(self.stats, f, indent=2)

        print(f"统计信息已保存至: {filepath}")

    def load_stats(self, filepath):
        """从JSON文件加载统计信息"""
        with open(filepath, 'r') as f:
            self.stats = json.load(f)

        print(f"统计信息已加载: {filepath}")
        return self.stats
