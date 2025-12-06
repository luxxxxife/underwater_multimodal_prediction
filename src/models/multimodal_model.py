"""
完整的多模态LSTM预测模型
整合CNN、Navigation编码器、融合模块和LSTM预测器
"""
import torch
import torch.nn as nn

from .cnn_encoder import DualCNNEncoder
from .nav_encoder import NavigationEncoder
from .fusion_module import FusionModule
from .lstm_predictor import LSTMPredictor


class MultimodalLSTMPredictor(nn.Module):
    """多模态LSTM时序预测模型"""

    def __init__(self, config):
        """
        Args:
            config: 配置字典
        """
        super().__init__()

        self.config = config

        # 双分支CNN编码器（Camera + Sonar）
        self.cnn_encoder = DualCNNEncoder(config)

        # Navigation编码器
        self.nav_encoder = NavigationEncoder(config)

        # 多模态融合模块
        self.fusion = FusionModule(config)

        # LSTM时序预测器
        self.lstm_predictor = LSTMPredictor(config)

    def forward(self, camera, sonar, navigation):
        """
        Args:
            camera: [B, T, 3, H, W] Camera图像序列
            sonar: [B, T, 3, H, W] Sonar图像序列
            navigation: [B, T, 10] Navigation数据序列

        Returns:
            predictions: [B, 10, pred_horizon] 预测结果
        """
        # CNN特征提取
        camera_feat, sonar_feat = self.cnn_encoder(camera, sonar)  # 各为 [B, T, 512]

        # Navigation编码
        nav_feat = self.nav_encoder(navigation)  # [B, T, 128]

        # 多模态融合
        fused_feat = self.fusion(camera_feat, sonar_feat, nav_feat)  # [B, T, 512]

        # LSTM时序预测
        predictions = self.lstm_predictor(fused_feat)  # [B, 10, pred_horizon]

        return predictions

    def get_feature_embeddings(self, camera, sonar, navigation):
        """
        获取中间特征表示（用于可视化或分析）

        Returns:
            dict: {
                'camera_feat': [B, T, 512],
                'sonar_feat': [B, T, 512],
                'nav_feat': [B, T, 128],
                'fused_feat': [B, T, 512]
            }
        """
        with torch.no_grad():
            camera_feat, sonar_feat = self.cnn_encoder(camera, sonar)
            nav_feat = self.nav_encoder(navigation)
            fused_feat = self.fusion(camera_feat, sonar_feat, nav_feat)

        return {
            'camera_feat': camera_feat,
            'sonar_feat': sonar_feat,
            'nav_feat': nav_feat,
            'fused_feat': fused_feat
        }


def build_model(config):
    """
    根据配置构建模型

    Args:
        config: 配置字典

    Returns:
        nn.Module: 模型实例
    """
    model_name = config['model']['name']

    if model_name == 'MultimodalLSTMPredictor':
        model = MultimodalLSTMPredictor(config)
    else:
        raise ValueError(f"不支持的模型: {model_name}")

    return model
