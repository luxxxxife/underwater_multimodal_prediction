"""
可视化脚本
生成预测结果的各类可视化图表
"""
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json

# 添加src到路径
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from utils import load_config

# 设置matplotlib中文字体（避免中文显示问题）
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class Visualizer:
    """可视化器"""

    def __init__(self, config, predictions_file):
        self.config = config
        self.output_dir = Path(config['project']['root']) / config['visualization']['save_dir']
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 加载预测结果
        data = np.load(predictions_file)
        self.predictions = data['predictions']  # [N, pred_horizon, 10]
        self.targets = data['targets']  # [N, pred_horizon, 10]
        self.param_names = list(data['param_names'])

        self.pred_horizon = self.predictions.shape[1]
        self.n_samples = self.predictions.shape[0]

        print(f"加载预测结果: {self.n_samples}个样本, 预测步长={self.pred_horizon}")

    def plot_position_prediction(self):
        """图1: 位置预测对比 (Latitude, Longitude, Depth)"""
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # 选择一个样本序列进行展示
        sample_idx = 0
        positions = ['latitude', 'longitude', 'depth']

        for i, param in enumerate(positions):
            param_idx = self.param_names.index(param)

            pred_seq = self.predictions[sample_idx, :, param_idx]
            target_seq = self.targets[sample_idx, :, param_idx]

            time_steps = np.arange(1, self.pred_horizon + 1)

            axes[i].plot(time_steps, target_seq, 'b-o', label='True', linewidth=2, markersize=8)
            axes[i].plot(time_steps, pred_seq, 'r--s', label='Predicted', linewidth=2, markersize=6)
            axes[i].set_xlabel('Prediction Step', fontsize=12)
            axes[i].set_ylabel(param.capitalize(), fontsize=12)
            axes[i].set_title(f'{param.capitalize()} Prediction', fontsize=14, fontweight='bold')
            axes[i].legend(fontsize=10)
            axes[i].grid(True, alpha=0.3)

        plt.tight_layout()
        output_path = self.output_dir / 'position_prediction.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"已保存: {output_path}")

    def plot_orientation_prediction(self):
        """图2: 姿态预测对比 (Yaw, Pitch, Roll)"""
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        sample_idx = 0
        orientations = ['yaw', 'pitch', 'roll']

        for i, param in enumerate(orientations):
            param_idx = self.param_names.index(param)

            pred_seq = self.predictions[sample_idx, :, param_idx]
            target_seq = self.targets[sample_idx, :, param_idx]

            time_steps = np.arange(1, self.pred_horizon + 1)

            axes[i].plot(time_steps, target_seq, 'b-o', label='True', linewidth=2, markersize=8)
            axes[i].plot(time_steps, pred_seq, 'r--s', label='Predicted', linewidth=2, markersize=6)
            axes[i].set_xlabel('Prediction Step', fontsize=12)
            axes[i].set_ylabel(f'{param.capitalize()} (rad)', fontsize=12)
            axes[i].set_title(f'{param.capitalize()} Prediction', fontsize=14, fontweight='bold')
            axes[i].legend(fontsize=10)
            axes[i].grid(True, alpha=0.3)

        plt.tight_layout()
        output_path = self.output_dir / 'orientation_prediction.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"已保存: {output_path}")

    def plot_velocity_prediction(self):
        """图3: 速度预测对比 (Vx, Vy, Vz)"""
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        sample_idx = 0
        velocities = ['velocity_x', 'velocity_y', 'velocity_z']

        for i, param in enumerate(velocities):
            param_idx = self.param_names.index(param)

            pred_seq = self.predictions[sample_idx, :, param_idx]
            target_seq = self.targets[sample_idx, :, param_idx]

            time_steps = np.arange(1, self.pred_horizon + 1)

            axes[i].plot(time_steps, target_seq, 'b-o', label='True', linewidth=2, markersize=8)
            axes[i].plot(time_steps, pred_seq, 'r--s', label='Predicted', linewidth=2, markersize=6)
            axes[i].set_xlabel('Prediction Step', fontsize=12)
            axes[i].set_ylabel(f'{param} (m/s)', fontsize=12)
            axes[i].set_title(f'{param.replace("_", " ").title()} Prediction', fontsize=14, fontweight='bold')
            axes[i].legend(fontsize=10)
            axes[i].grid(True, alpha=0.3)

        plt.tight_layout()
        output_path = self.output_dir / 'velocity_prediction.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"已保存: {output_path}")

    def plot_altitude_prediction(self):
        """图4: 高度预测对比"""
        fig, ax = plt.subplots(figsize=(8, 5))

        sample_idx = 0
        param_idx = self.param_names.index('altitude')

        pred_seq = self.predictions[sample_idx, :, param_idx]
        target_seq = self.targets[sample_idx, :, param_idx]

        time_steps = np.arange(1, self.pred_horizon + 1)

        ax.plot(time_steps, target_seq, 'b-o', label='True', linewidth=2, markersize=8)
        ax.plot(time_steps, pred_seq, 'r--s', label='Predicted', linewidth=2, markersize=6)
        ax.set_xlabel('Prediction Step', fontsize=12)
        ax.set_ylabel('Altitude (m)', fontsize=12)
        ax.set_title('Altitude Prediction', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        output_path = self.output_dir / 'altitude_prediction.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"已保存: {output_path}")

    def plot_trajectory_map(self):
        """图5: 2D轨迹地图 (经纬度平面)"""
        fig, ax = plt.subplots(figsize=(10, 8))

        # 累积多个样本的轨迹
        n_show = min(10, self.n_samples)  # 显示前10个样本

        lat_idx = self.param_names.index('latitude')
        lon_idx = self.param_names.index('longitude')

        for i in range(n_show):
            true_lats = self.targets[i, :, lat_idx]
            true_lons = self.targets[i, :, lon_idx]
            pred_lats = self.predictions[i, :, lat_idx]
            pred_lons = self.predictions[i, :, lon_idx]

            # 绘制真实路径
            if i == 0:
                ax.plot(true_lons, true_lats, 'b-', linewidth=2, alpha=0.7, label='True Path')
                ax.plot(pred_lons, pred_lats, 'r--', linewidth=2, alpha=0.7, label='Predicted Path')
            else:
                ax.plot(true_lons, true_lats, 'b-', linewidth=1, alpha=0.3)
                ax.plot(pred_lons, pred_lats, 'r--', linewidth=1, alpha=0.3)

            # 标记起点和终点
            ax.scatter(true_lons[0], true_lats[0], c='green', marker='o', s=100, zorder=5, edgecolors='black')
            ax.scatter(true_lons[-1], true_lats[-1], c='red', marker='x', s=100, zorder=5, linewidths=2)

        ax.set_xlabel('Longitude', fontsize=12)
        ax.set_ylabel('Latitude', fontsize=12)
        ax.set_title('Trajectory Map (Lat-Lon Plane)', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        output_path = self.output_dir / 'trajectory_map.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"已保存: {output_path}")

    def plot_error_heatmap(self):
        """误差热力图 (10个参数 × pred_horizon个时间步)"""
        fig, ax = plt.subplots(figsize=(10, 8))

        # 计算每个参数在每个时间步的平均绝对误差
        errors = np.abs(self.predictions - self.targets)  # [N, pred_horizon, 10]
        mean_errors = np.mean(errors, axis=0)  # [pred_horizon, 10]

        # 转置为 [10, pred_horizon]
        mean_errors = mean_errors.T

        # 绘制热力图
        sns.heatmap(mean_errors, annot=True, fmt='.4f', cmap='YlOrRd',
                   xticklabels=[f'Step {i+1}' for i in range(self.pred_horizon)],
                   yticklabels=self.param_names,
                   cbar_kws={'label': 'Mean Absolute Error'},
                   ax=ax)

        ax.set_title('Error Heatmap (Parameters × Prediction Steps)', fontsize=14, fontweight='bold')
        ax.set_xlabel('Prediction Step', fontsize=12)
        ax.set_ylabel('Parameters', fontsize=12)

        plt.tight_layout()
        output_path = self.output_dir / 'error_heatmap.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"已保存: {output_path}")

    def plot_horizon_error_curve(self):
        """预测步长误差曲线"""
        fig, ax = plt.subplots(figsize=(10, 6))

        # 计算每个时间步的平均MAE和RMSE
        errors = self.predictions - self.targets  # [N, pred_horizon, 10]

        mae_per_step = []
        rmse_per_step = []

        for t in range(self.pred_horizon):
            errors_t = errors[:, t, :]  # [N, 10]
            mae_t = np.mean(np.abs(errors_t))
            rmse_t = np.sqrt(np.mean(errors_t ** 2))

            mae_per_step.append(mae_t)
            rmse_per_step.append(rmse_t)

        time_steps = np.arange(1, self.pred_horizon + 1)

        ax.plot(time_steps, mae_per_step, 'b-o', label='MAE', linewidth=2, markersize=8)
        ax.plot(time_steps, rmse_per_step, 'r-s', label='RMSE', linewidth=2, markersize=8)

        ax.set_xlabel('Prediction Step', fontsize=12)
        ax.set_ylabel('Error', fontsize=12)
        ax.set_title('Error vs Prediction Horizon', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        output_path = self.output_dir / 'horizon_error_curve.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"已保存: {output_path}")

    def generate_all(self):
        """生成所有可视化图表"""
        print("\n生成可视化图表...")
        print("="*60)

        self.plot_position_prediction()
        self.plot_orientation_prediction()
        self.plot_velocity_prediction()
        self.plot_altitude_prediction()
        self.plot_trajectory_map()
        self.plot_error_heatmap()
        self.plot_horizon_error_curve()

        print("="*60)
        print(f"所有图表已保存至: {self.output_dir}")


def main():
    """主函数"""
    # 加载配置
    config_path = Path(__file__).parent / 'configs' / 'config.yaml'
    config = load_config(config_path)

    print("="*60)
    print("水下多模态时序预测 - 结果可视化")
    print("="*60)

    # 预测结果文件
    predictions_file = Path(config['project']['root']) / 'results' / 'predictions' / 'test_predictions.npz'

    if not predictions_file.exists():
        print(f"\n错误: 未找到预测结果文件 {predictions_file}")
        print("请先运行 evaluate.py 生成预测结果")
        return

    # 创建可视化器
    visualizer = Visualizer(config, predictions_file)

    # 生成所有图表
    visualizer.generate_all()

    print("\n可视化完成!")


if __name__ == "__main__":
    main()
