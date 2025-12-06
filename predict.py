"""
推理脚本
加载训练好的模型，对新数据进行预测
"""
import sys
from pathlib import Path
import torch
import numpy as np

# 添加src到路径
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from utils import load_config, get_device
from models.multimodal_model import build_model
from preprocessing.normalization import Normalizer
from preprocessing.image_processor import ImageProcessor


class Predictor:
    """预测器"""

    def __init__(self, config, checkpoint_path):
        self.config = config
        self.device = get_device(config['device']['type'])

        # 加载模型
        self.model = build_model(config)
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()

        # 加载归一化器
        self.normalizer = Normalizer(config)
        stats_file = Path(config['project']['root']) / config['normalization']['stats_file']
        self.normalizer.load_stats(stats_file)

        # 图像处理器
        self.image_processor = ImageProcessor(config)

        print(f"模型已加载: {checkpoint_path}")
        print(f"设备: {self.device}")

    def predict(self, camera_paths, sonar_paths, navigation_data):
        """
        预测

        Args:
            camera_paths: list of Path, Camera图像路径列表 (长度=seq_length)
            sonar_paths: list of Path, Sonar图像路径列表 (长度=seq_length)
            navigation_data: np.ndarray [seq_length, 10], 归一化前的navigation数据

        Returns:
            np.ndarray: [pred_horizon, 10] 预测结果（反归一化后）
        """
        # 处理图像
        camera_seq, sonar_seq = self.image_processor.batch_process(camera_paths, sonar_paths)
        # camera_seq: [T, 3, H, W], sonar_seq: [T, 3, H, W]

        # 归一化navigation数据
        nav_seq_norm = self.normalizer.normalize(navigation_data)  # [T, 10]

        # 转换为tensor并添加batch维度
        camera = torch.from_numpy(camera_seq).unsqueeze(0).float().to(self.device)  # [1, T, 3, H, W]
        sonar = torch.from_numpy(sonar_seq).unsqueeze(0).float().to(self.device)
        navigation = torch.from_numpy(nav_seq_norm).unsqueeze(0).float().to(self.device)  # [1, T, 10]

        # 预测
        with torch.no_grad():
            pred = self.model(camera, sonar, navigation)  # [1, 10, pred_horizon]

        # 转换为numpy
        pred_np = pred.squeeze(0).cpu().numpy()  # [10, pred_horizon]
        pred_np = pred_np.T  # [pred_horizon, 10]

        # 反归一化
        pred_denorm = self.normalizer.denormalize(pred_np)

        return pred_denorm


def main():
    """示例用法"""
    # 加载配置
    config_path = Path(__file__).parent / 'configs' / 'config.yaml'
    config = load_config(config_path)

    # 模型权重路径
    checkpoint_path = Path(config['project']['root']) / 'checkpoints' / 'best_model.pth'

    if not checkpoint_path.exists():
        print(f"错误: 未找到模型权重 {checkpoint_path}")
        print("请先完成训练")
        return

    # 创建预测器
    predictor = Predictor(config, checkpoint_path)

    print("\n" + "="*60)
    print("推理示例")
    print("="*60)
    print("\n使用方法:")
    print("""
# 准备输入数据
camera_paths = [Path('camera/00001.png'), ..., Path('camera/00020.png')]  # 20张
sonar_paths = [Path('sonar/00001.png'), ..., Path('sonar/00020.png')]     # 20张
navigation_data = np.array([...])  # shape: [20, 10]

# 预测
predictions = predictor.predict(camera_paths, sonar_paths, navigation_data)

# predictions shape: [5, 10] - 5个未来时间步 × 10个参数
# 参数顺序: latitude, longitude, depth, yaw, pitch, roll, velocity_x, velocity_y, velocity_z, altitude
""")

    print("\n预测器已就绪，可以开始推理！")


if __name__ == "__main__":
    main()
