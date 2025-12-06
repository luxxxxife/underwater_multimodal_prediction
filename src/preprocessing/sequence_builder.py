"""
序列构建模块
根据滑动窗口构建时序序列
"""
import numpy as np
from pathlib import Path
from tqdm import tqdm


class SequenceBuilder:
    """时序序列构建器"""

    def __init__(self, data_loader, image_processor, normalizer, config):
        """
        Args:
            data_loader: DataLoader实例
            image_processor: ImageProcessor实例
            normalizer: Normalizer实例
            config: 配置字典
        """
        self.data_loader = data_loader
        self.image_processor = image_processor
        self.normalizer = normalizer
        self.config = config

        self.seq_length = config['data']['seq_length']
        self.pred_horizon = config['data']['pred_horizon']
        self.stride = config['data']['stride']

    def build_sequences(self, indices, output_dir, split_name):
        """
        构建序列并保存

        Args:
            indices: 样本索引列表
            output_dir: 输出目录
            split_name: 数据集名称（train/val/test）

        Returns:
            int: 生成的序列数量
        """
        output_dir = Path(output_dir) / split_name
        output_dir.mkdir(parents=True, exist_ok=True)

        # 计算可以构建的序列数量
        max_start_idx = len(indices) - self.seq_length - self.pred_horizon + 1
        if max_start_idx <= 0:
            raise ValueError(f"{split_name}集样本数不足以构建序列")

        n_sequences = (max_start_idx - 1) // self.stride + 1

        print(f"\n构建 {split_name} 序列:")
        print(f"  样本索引范围: {indices[0]} - {indices[-1]}")
        print(f"  序列数量: {n_sequences}")
        print(f"  序列长度: {self.seq_length}")
        print(f"  预测步长: {self.pred_horizon}")

        for seq_idx in tqdm(range(n_sequences), desc=f"Processing {split_name}"):
            start_idx = seq_idx * self.stride
            end_idx = start_idx + self.seq_length

            # 输入序列索引
            input_indices = indices[start_idx:end_idx]

            # 目标序列索引（紧接着输入序列之后）
            target_start = end_idx
            target_end = target_start + self.pred_horizon
            target_indices = indices[target_start:target_end]

            # 构建输入序列
            input_samples = self.data_loader.get_batch_samples(input_indices)

            # 处理图像
            camera_paths = [s['camera_path'] for s in input_samples]
            sonar_paths = [s['sonar_path'] for s in input_samples]
            camera_seq, sonar_seq = self.image_processor.batch_process(camera_paths, sonar_paths)

            # 处理navigation数据
            nav_seq = np.stack([s['navigation'] for s in input_samples], axis=0)  # [T, 10]
            nav_seq_norm = self.normalizer.normalize(nav_seq)

            # 构建目标序列（只需要navigation）
            target_samples = self.data_loader.get_batch_samples(target_indices)
            target_nav = np.stack([s['navigation'] for s in target_samples], axis=0)  # [pred_horizon, 10]
            target_nav_norm = self.normalizer.normalize(target_nav)

            # 保存为.npz文件
            seq_filename = output_dir / f"seq_{seq_idx:05d}.npz"
            np.savez_compressed(
                seq_filename,
                camera=camera_seq,        # [T, 3, 224, 224]
                sonar=sonar_seq,          # [T, 3, 320, 320]
                navigation=nav_seq_norm,  # [T, 10]
                target=target_nav_norm,   # [pred_horizon, 10]
                input_indices=np.array(input_indices),
                target_indices=np.array(target_indices)
            )

        print(f"  完成! 保存至: {output_dir}")
        return n_sequences
