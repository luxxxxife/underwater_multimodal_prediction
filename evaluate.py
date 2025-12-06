"""
评估脚本
计算模型在测试集上的各项指标
"""
import sys
from pathlib import Path
import torch
import numpy as np
from tqdm import tqdm
import json

# 添加src到路径
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from utils import load_config, get_device, load_json
from dataset import create_dataloaders
from models.multimodal_model import build_model
from preprocessing.normalization import Normalizer


class Evaluator:
    """评估器"""

    def __init__(self, config, model, test_loader, normalizer, device):
        self.config = config
        self.model = model.to(device)
        self.test_loader = test_loader
        self.normalizer = normalizer
        self.device = device

        # 参数名称
        self.param_names = config['data']['target_columns']

    def compute_metrics(self, pred, target):
        """
        计算评估指标

        Args:
            pred: [N, 10, pred_horizon] 预测值（归一化后）
            target: [N, pred_horizon, 10] 真实值（归一化后）

        Returns:
            dict: 各项指标
        """
        # 转换为numpy
        pred = pred.cpu().numpy()
        target = target.cpu().numpy()

        # 反归一化
        pred_denorm = self.normalizer.denormalize(pred.transpose(0, 2, 1))  # [N, pred_horizon, 10]
        target_denorm = self.normalizer.denormalize(target)  # [N, pred_horizon, 10]

        metrics = {}

        # 对每个参数计算指标
        for param_idx, param_name in enumerate(self.param_names):
            pred_param = pred_denorm[:, :, param_idx]  # [N, pred_horizon]
            target_param = target_denorm[:, :, param_idx]

            # MAE (Mean Absolute Error)
            mae = np.mean(np.abs(pred_param - target_param))

            # RMSE (Root Mean Square Error)
            rmse = np.sqrt(np.mean((pred_param - target_param) ** 2))

            # MAPE (Mean Absolute Percentage Error) - 避免除零
            epsilon = 1e-8
            mape = np.mean(np.abs((pred_param - target_param) / (target_param + epsilon))) * 100

            metrics[param_name] = {
                'mae': float(mae),
                'rmse': float(rmse),
                'mape': float(mape)
            }

        # 计算horizon-wise指标（每个时间步的平均误差）
        horizon_metrics = []
        for t in range(pred_denorm.shape[1]):  # pred_horizon
            pred_t = pred_denorm[:, t, :]  # [N, 10]
            target_t = target_denorm[:, t, :]

            mae_t = np.mean(np.abs(pred_t - target_t))
            rmse_t = np.sqrt(np.mean((pred_t - target_t) ** 2))

            horizon_metrics.append({
                'step': t + 1,
                'mae': float(mae_t),
                'rmse': float(rmse_t)
            })

        metrics['horizon_wise'] = horizon_metrics

        return metrics, pred_denorm, target_denorm

    def evaluate(self):
        """在测试集上评估"""
        self.model.eval()

        all_predictions = []
        all_targets = []

        print("\n评估测试集...")
        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc="Evaluating"):
                camera = batch['camera'].to(self.device)
                sonar = batch['sonar'].to(self.device)
                navigation = batch['navigation'].to(self.device)
                target = batch['target'].to(self.device)

                # 前向传播
                pred = self.model(camera, sonar, navigation)  # [B, 10, pred_horizon]

                all_predictions.append(pred)
                all_targets.append(target)

        # 合并所有batch
        all_predictions = torch.cat(all_predictions, dim=0)  # [N, 10, pred_horizon]
        all_targets = torch.cat(all_targets, dim=0)  # [N, pred_horizon, 10]

        # 计算指标
        metrics, pred_denorm, target_denorm = self.compute_metrics(all_predictions, all_targets)

        return metrics, pred_denorm, target_denorm

    def print_metrics(self, metrics):
        """打印指标"""
        print("\n" + "="*60)
        print("测试集评估结果")
        print("="*60)

        print("\n【按参数统计】")
        for param_name in self.param_names:
            m = metrics[param_name]
            print(f"\n{param_name:12s}:")
            print(f"  MAE:  {m['mae']:.6f}")
            print(f"  RMSE: {m['rmse']:.6f}")
            print(f"  MAPE: {m['mape']:.2f}%")

        print("\n【按预测步长统计】")
        print(f"{'Step':>6s} {'MAE':>12s} {'RMSE':>12s}")
        print("-" * 32)
        for h in metrics['horizon_wise']:
            print(f"{h['step']:>6d} {h['mae']:>12.6f} {h['rmse']:>12.6f}")

        # 检查关键指标是否达标
        print("\n【关键指标达标检查】")
        thresholds = self.config['evaluation']['thresholds']
        checks = {
            'latitude_rmse': (metrics['latitude']['rmse'], thresholds['latitude_rmse']),
            'longitude_rmse': (metrics['longitude']['rmse'], thresholds['longitude_rmse']),
            'depth_rmse': (metrics['depth']['rmse'], thresholds['depth_rmse']),
            'yaw_rmse': (metrics['yaw']['rmse'], thresholds['yaw_rmse'])
        }

        for name, (value, threshold) in checks.items():
            status = "✓" if value < threshold else "✗"
            print(f"  {status} {name:20s}: {value:.6f} (阈值: {threshold:.6f})")


def main():
    """主函数"""
    # 加载配置
    config_path = Path(__file__).parent / 'configs' / 'config.yaml'
    config = load_config(config_path)

    # 获取设备
    device = get_device(config['device']['type'])

    print("="*60)
    print("水下多模态时序预测 - 模型评估")
    print("="*60)

    # 加载归一化器
    print("\n[1/4] 加载归一化统计信息...")
    normalizer = Normalizer(config)
    stats_file = Path(config['project']['root']) / config['normalization']['stats_file']
    normalizer.load_stats(stats_file)

    # 创建DataLoader
    print("\n[2/4] 加载测试数据...")
    dataloaders = create_dataloaders(config, num_workers=config['training']['num_workers'])
    test_loader = dataloaders['test']

    # 构建模型
    print("\n[3/4] 加载模型...")
    model = build_model(config)

    # 加载最优模型权重
    checkpoint_path = Path(config['project']['root']) / config['project']['checkpoint_dir'] / 'best_model.pth'
    if not checkpoint_path.exists():
        print(f"错误: 未找到模型权重文件 {checkpoint_path}")
        return

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"已加载模型: {checkpoint_path}")
    print(f"训练epoch: {checkpoint['epoch'] + 1}")
    print(f"最优验证loss: {checkpoint['best_val_loss']:.6f}")

    # 评估
    print("\n[4/4] 开始评估...")
    evaluator = Evaluator(config, model, test_loader, normalizer, device)
    metrics, predictions, targets = evaluator.evaluate()

    # 打印结果
    evaluator.print_metrics(metrics)

    # 保存结果
    results_dir = Path(config['project']['root']) / 'results'
    results_dir.mkdir(parents=True, exist_ok=True)

    # 保存指标
    metrics_file = results_dir / 'metrics.json'
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\n指标已保存至: {metrics_file}")

    # 保存预测结果
    predictions_file = results_dir / 'predictions' / 'test_predictions.npz'
    predictions_file.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        predictions_file,
        predictions=predictions,
        targets=targets,
        param_names=config['data']['target_columns']
    )
    print(f"预测结果已保存至: {predictions_file}")

    print("\n评估完成!")


if __name__ == "__main__":
    main()
