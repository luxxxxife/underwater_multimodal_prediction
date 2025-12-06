"""
LSTM时序预测模块
"""
import torch
import torch.nn as nn


class LSTMPredictor(nn.Module):
    """LSTM时序预测器"""

    def __init__(self, config):
        """
        Args:
            config: 配置字典
        """
        super().__init__()

        lstm_config = config['model']['lstm']
        predictor_config = config['model']['predictor']

        # LSTM层
        self.lstm = nn.LSTM(
            input_size=lstm_config['input_size'],
            hidden_size=lstm_config['hidden_size'],
            num_layers=lstm_config['num_layers'],
            dropout=lstm_config['dropout'] if lstm_config['num_layers'] > 1 else 0,
            bidirectional=lstm_config.get('bidirectional', False),
            batch_first=True
        )

        # LSTM输出维度
        lstm_out_dim = lstm_config['hidden_size'] * (2 if lstm_config.get('bidirectional', False) else 1)

        # 预测头
        self.predictor = nn.Sequential(
            nn.Linear(lstm_out_dim, predictor_config['hidden_dim']),
            nn.ReLU(inplace=True),
            nn.Dropout(predictor_config['dropout']),
            nn.Linear(predictor_config['hidden_dim'],
                     predictor_config['output_dim'] * predictor_config['pred_horizon'])
        )

        self.output_dim = predictor_config['output_dim']
        self.pred_horizon = predictor_config['pred_horizon']

    def forward(self, x):
        """
        Args:
            x: [B, T, input_size] 融合后的特征序列

        Returns:
            predictions: [B, output_dim, pred_horizon]
        """
        # LSTM处理
        lstm_out, _ = self.lstm(x)  # [B, T, hidden_size]

        # 取最后一个时间步的输出
        last_hidden = lstm_out[:, -1, :]  # [B, hidden_size]

        # 预测
        pred = self.predictor(last_hidden)  # [B, output_dim * pred_horizon]

        # Reshape为 [B, output_dim, pred_horizon]
        pred = pred.view(-1, self.output_dim, self.pred_horizon)

        return pred
