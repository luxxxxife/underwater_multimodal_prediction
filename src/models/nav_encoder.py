"""
Navigation编码器模块
将navigation数据编码为特征向量
"""
import torch
import torch.nn as nn


class NavigationEncoder(nn.Module):
    """Navigation特征编码器"""

    def __init__(self, config):
        """
        Args:
            config: 配置字典
        """
        super().__init__()

        nav_config = config['model']['nav_encoder']

        input_dim = nav_config['input_dim']
        hidden_dims = nav_config['hidden_dims']
        output_dim = nav_config['output_dim']
        dropout = nav_config['dropout']
        activation = nav_config.get('activation', 'relu')

        # 构建MLP
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))

            if activation == 'relu':
                layers.append(nn.ReLU(inplace=True))
            elif activation == 'gelu':
                layers.append(nn.GELU())
            else:
                raise ValueError(f"不支持的激活函数: {activation}")

            if dropout > 0:
                layers.append(nn.Dropout(dropout))

            prev_dim = hidden_dim

        # 输出层
        layers.append(nn.Linear(prev_dim, output_dim))

        self.encoder = nn.Sequential(*layers)
        self.output_dim = output_dim

    def forward(self, x):
        """
        Args:
            x: [B, T, input_dim] 或 [B, input_dim]

        Returns:
            features: [B, T, output_dim] 或 [B, output_dim]
        """
        return self.encoder(x)
