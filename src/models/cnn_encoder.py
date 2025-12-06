"""
CNN编码器模块
使用ResNet提取图像特征
"""
import torch
import torch.nn as nn
import torchvision.models as models


class CNNEncoder(nn.Module):
    """CNN特征提取器"""

    def __init__(self, backbone='resnet18', pretrained=True, feature_dim=512, freeze_layers=0):
        """
        Args:
            backbone: backbone名称 (resnet18, resnet34, resnet50)
            pretrained: 是否使用预训练权重
            feature_dim: 输出特征维度
            freeze_layers: 冻结前N层
        """
        super().__init__()

        # 加载backbone
        if backbone == 'resnet18':
            self.backbone = models.resnet18(pretrained=pretrained)
        elif backbone == 'resnet34':
            self.backbone = models.resnet34(pretrained=pretrained)
        elif backbone == 'resnet50':
            self.backbone = models.resnet50(pretrained=pretrained)
        else:
            raise ValueError(f"不支持的backbone: {backbone}")

        # 移除最后的全连接层
        backbone_out_dim = self.backbone.fc.in_features
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])

        # 冻结部分层
        if freeze_layers > 0:
            layers = list(self.backbone.children())
            for layer in layers[:freeze_layers]:
                for param in layer.parameters():
                    param.requires_grad = False

        # 特征映射层（如果需要调整维度）
        if backbone_out_dim != feature_dim:
            self.fc = nn.Linear(backbone_out_dim, feature_dim)
        else:
            self.fc = nn.Identity()

        self.feature_dim = feature_dim

    def forward(self, x):
        """
        Args:
            x: [B, 3, H, W] 或 [B, T, 3, H, W]

        Returns:
            features: [B, feature_dim] 或 [B, T, feature_dim]
        """
        # 处理时序输入
        if x.dim() == 5:  # [B, T, 3, H, W]
            B, T, C, H, W = x.shape
            x = x.view(B * T, C, H, W)
            time_series = True
        else:
            time_series = False

        # CNN特征提取
        features = self.backbone(x)  # [B*T, feature_dim, 1, 1]
        features = features.flatten(1)  # [B*T, feature_dim]
        features = self.fc(features)  # [B*T, feature_dim]

        # 恢复时序形状
        if time_series:
            features = features.view(B, T, self.feature_dim)

        return features


class DualCNNEncoder(nn.Module):
    """双分支CNN编码器（Camera + Sonar）"""

    def __init__(self, config):
        """
        Args:
            config: 配置字典
        """
        super().__init__()

        cnn_config = config['model']['cnn_encoder']

        # Camera编码器
        self.camera_encoder = CNNEncoder(
            backbone=cnn_config['backbone'],
            pretrained=cnn_config['pretrained'],
            feature_dim=cnn_config['feature_dim'],
            freeze_layers=cnn_config.get('freeze_layers', 0)
        )

        # Sonar编码器（共享架构但独立参数）
        self.sonar_encoder = CNNEncoder(
            backbone=cnn_config['backbone'],
            pretrained=cnn_config['pretrained'],
            feature_dim=cnn_config['feature_dim'],
            freeze_layers=cnn_config.get('freeze_layers', 0)
        )

    def forward(self, camera, sonar):
        """
        Args:
            camera: [B, T, 3, H, W]
            sonar: [B, T, 3, H, W]

        Returns:
            tuple: (camera_feat, sonar_feat)
                camera_feat: [B, T, feature_dim]
                sonar_feat: [B, T, feature_dim]
        """
        camera_feat = self.camera_encoder(camera)
        sonar_feat = self.sonar_encoder(sonar)

        return camera_feat, sonar_feat
