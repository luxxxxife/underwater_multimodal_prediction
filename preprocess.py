"""
数据预处理主脚本
加载原始数据，构建时序序列，保存为.npz文件
"""
import sys
from pathlib import Path
import numpy as np

# 添加src到路径
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from utils import load_config, save_json, set_seed
from preprocessing.data_loader import DataLoader
from preprocessing.image_processor import ImageProcessor
from preprocessing.normalization import Normalizer
from preprocessing.sequence_builder import SequenceBuilder


def split_dataset(total_samples, train_ratio, val_ratio, test_ratio):
    """
    划分数据集（按时间顺序）

    Args:
        total_samples: 总样本数
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        test_ratio: 测试集比例

    Returns:
        dict: {'train': [...], 'val': [...], 'test': [...]}
    """
    indices = np.arange(total_samples)

    train_size = int(total_samples * train_ratio)
    val_size = int(total_samples * val_ratio)

    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]

    print(f"\n数据集划分:")
    print(f"  训练集: {len(train_indices)} 样本 (索引 {train_indices[0]}-{train_indices[-1]})")
    print(f"  验证集: {len(val_indices)} 样本 (索引 {val_indices[0]}-{val_indices[-1]})")
    print(f"  测试集: {len(test_indices)} 样本 (索引 {test_indices[0]}-{test_indices[-1]})")

    return {
        'train': train_indices.tolist(),
        'val': val_indices.tolist(),
        'test': test_indices.tolist()
    }


def main():
    """主函数"""
    # 加载配置
    config_path = Path(__file__).parent / 'configs' / 'config.yaml'
    config = load_config(config_path)

    # 设置随机种子
    set_seed(config['seed'])

    print("="*60)
    print("水下多模态时序预测 - 数据预处理")
    print("="*60)

    # 初始化数据加载器
    print("\n[1/6] 加载原始数据...")
    data_loader = DataLoader(config['project']['data_root'])

    # 划分数据集
    print("\n[2/6] 划分数据集...")
    split_config = config['data']['split']
    splits = split_dataset(
        total_samples=config['data']['total_samples'],
        train_ratio=split_config['train_ratio'],
        val_ratio=split_config['val_ratio'],
        test_ratio=split_config['test_ratio']
    )

    # 保存划分索引
    split_file = Path(config['project']['root']) / 'data' / 'split_indices.json'
    save_json(splits, split_file)
    print(f"  划分索引已保存: {split_file}")

    # 计算归一化统计信息（仅使用训练集）
    print("\n[3/6] 计算归一化统计信息（训练集）...")
    train_samples = data_loader.get_batch_samples(splits['train'])
    train_nav_data = np.stack([s['navigation'] for s in train_samples], axis=0)

    normalizer = Normalizer(config)
    stats = normalizer.compute_stats(train_nav_data)
    print(f"  样本数: {stats['n_samples']}")
    print(f"  参数范围:")
    param_names = config['data']['target_columns']
    for i, name in enumerate(param_names):
        print(f"    {name:12s}: [{stats['min'][i]:10.4f}, {stats['max'][i]:10.4f}]")

    # 保存统计信息
    stats_file = Path(config['project']['root']) / config['normalization']['stats_file']
    normalizer.save_stats(stats_file)

    # 初始化图像处理器
    print("\n[4/6] 初始化图像处理器...")
    image_processor = ImageProcessor(config)
    print(f"  Camera目标尺寸: {config['image']['camera']['target_size']}")
    print(f"  Sonar目标尺寸: {config['image']['sonar']['target_size']}")
    print(f"  Sonar ROI: {config['image']['sonar']['roi_coords']}")

    # 初始化序列构建器
    print("\n[5/6] 初始化序列构建器...")
    sequence_builder = SequenceBuilder(data_loader, image_processor, normalizer, config)

    # 构建序列
    print("\n[6/6] 构建时序序列...")
    output_dir = Path(config['project']['root']) / 'data' / 'processed'

    n_train = sequence_builder.build_sequences(splits['train'], output_dir, 'train')
    n_val = sequence_builder.build_sequences(splits['val'], output_dir, 'val')
    n_test = sequence_builder.build_sequences(splits['test'], output_dir, 'test')

    # 总结
    print("\n" + "="*60)
    print("预处理完成!")
    print("="*60)
    print(f"训练序列: {n_train}")
    print(f"验证序列: {n_val}")
    print(f"测试序列: {n_test}")
    print(f"总序列数: {n_train + n_val + n_test}")
    print(f"\n数据保存至: {output_dir}")
    print(f"统计信息: {stats_file}")
    print(f"划分索引: {split_file}")


if __name__ == "__main__":
    main()
