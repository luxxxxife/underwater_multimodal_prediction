"""
多模态融合模块
融合Camera、Sonar和Navigation特征
"""
import torch
import torch.nn as nn


class FusionModule(nn.Module):
    """多模态特征融合模块"""

    def __init__(self, config):
        """
        Args:
            config: 配置字典
        """
        super().__init__()

        fusion_config = config['model']['fusion']

        input_dim = fusion_config['input_dim']
        hidden_dim = fusion_config.get('hidden_dim', 512)
        output_dim = fusion_config['output_dim']
        dropout = fusion_config['dropout']
        fusion_type = fusion_config.get('fusion_type', 'concat')

        self.fusion_type = fusion_type

        if fusion_type == 'concat':
            # 简单拼接 + MLP
            self.fusion = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, output_dim)
            )
        else:
            raise NotImplementedError(f"暂不支持融合类型: {fusion_type}")

        self.output_dim = output_dim

    def forward(self, camera_feat, sonar_feat, nav_feat):
        """
        Args:
            camera_feat: [B, T, dim1]
            sonar_feat: [B, T, dim2]
            nav_feat: [B, T, dim3]

        Returns:
            fused_feat: [B, T, output_dim]
        """
        # 拼接所有模态
        fused = torch.cat([camera_feat, sonar_feat, nav_feat], dim=-1)  # [B, T, input_dim]

        # 融合
        fused_feat = self.fusion(fused)  # [B, T, output_dim]

        return fused_feat
