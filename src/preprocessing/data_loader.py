"""
数据加载模块
加载CSV文件和samples.json，提供数据索引接口
"""
import pandas as pd
import json
from pathlib import Path


class DataLoader:
    """数据加载器"""

    def __init__(self, data_root):
        """
        Args:
            data_root: 数据根目录路径
        """
        self.data_root = Path(data_root)

        # 加载CSV文件
        self.camera_csv = pd.read_csv(self.data_root / 'camera' / 'camera.csv')
        self.sonar_csv = pd.read_csv(self.data_root / 'sonar' / 'sonar.csv')
        self.nav_csv = pd.read_csv(self.data_root / 'navigation' / 'navigation.csv')

        # 加载samples.json
        with open(self.data_root / 'samples.json', 'r') as f:
            self.samples = json.load(f)['samples']

        print(f"数据加载完成:")
        print(f"  Camera CSV: {len(self.camera_csv)} 条")
        print(f"  Sonar CSV: {len(self.sonar_csv)} 条")
        print(f"  Navigation CSV: {len(self.nav_csv)} 条")
        print(f"  Samples: {len(self.samples)} 个")

    def get_sample(self, idx):
        """
        获取第idx个样本的数据

        Args:
            idx: 样本索引（0 - 3827）

        Returns:
            dict: {
                'camera_path': Path对象,
                'sonar_path': Path对象,
                'navigation': np.ndarray [10个参数],
                'timestamp': float
            }
        """
        # 从samples.json获取对应索引
        sample = self.samples[idx]
        cam_idx = sample['camera'][0]
        son_idx = sample['sonar'][0]
        nav_idx = sample['navigation'][0]

        # 获取文件名和路径
        cam_filename = self.camera_csv.iloc[cam_idx]['filename']
        son_filename = self.sonar_csv.iloc[son_idx]['filename']
        cam_path = self.data_root / 'camera' / cam_filename
        son_path = self.data_root / 'sonar' / son_filename

        # 获取navigation数据（去掉timestamp列）
        nav_row = self.nav_csv.iloc[nav_idx]
        nav_data = nav_row.drop('timestamp').values.astype('float32')

        # 获取timestamp
        timestamp = nav_row['timestamp']

        return {
            'camera_path': cam_path,
            'sonar_path': son_path,
            'navigation': nav_data,
            'timestamp': timestamp
        }

    def get_batch_samples(self, indices):
        """
        批量获取样本

        Args:
            indices: 样本索引列表

        Returns:
            list: 样本字典列表
        """
        return [self.get_sample(idx) for idx in indices]

    def __len__(self):
        """返回样本总数"""
        return len(self.samples)
