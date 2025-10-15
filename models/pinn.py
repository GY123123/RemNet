import torch
import torch.nn as nn
import torch.nn.functional as F

# class RamanPINNClassifier(nn.Module):
#     def __init__(self, in_channels=5, num_classes=2):
#         super(RamanPINNClassifier, self).__init__()
#         self.num_classes = num_classes
#         self.in_channels = in_channels
#
#         # 信号处理分支（每个通道单独使用 1D-CNN 提取特征）
#         self.feature_extractors = nn.ModuleList([
#             nn.Sequential(
#                 nn.Conv1d(1, 32, kernel_size=7, padding=3),
#                 nn.BatchNorm1d(32),
#                 nn.ReLU(),
#                 nn.MaxPool1d(2),
#                 nn.Conv1d(32, 64, kernel_size=5, padding=2),
#                 nn.BatchNorm1d(64),
#                 nn.ReLU(),
#                 nn.AdaptiveAvgPool1d(1)  # 输出 shape: (B, 64, 1)
#             ) for _ in range(in_channels)
#         ])
#
#         # 分类层
#         self.classifier = nn.Sequential(
#             nn.Linear(64 * in_channels, 128),
#             # nn.ReLU(),
#             nn.Linear(128, num_classes)
#         )
#
#     def forward(self, x):
#         # 输入 shape: (B, 5, H, W) -> 拉曼图像 (5通道)
#         B, C, L = x.shape
#         assert C == self.in_channels
#
#         features = []
#         for i in range(self.in_channels):
#             signal = x[:, i].unsqueeze(1)  # (B, 1, L)
#             feat = self.feature_extractors[i](signal)  # (B, 64, 1)
#             features.append(feat.squeeze(-1))  # (B, 64)
#
#         features = torch.cat(features, dim=1)  # (B, 64 * in_channels)
#         out = self.classifier(features)       # (B, num_classes)
#         return out
#
#     def physics_loss(self, x):
#         """
#         构造物理约束损失:模拟先验知识（信号在特定区域的平滑性、能量限制、单调性等）鼓励信号之间的差分不要过大
#         """
#         diff = x[:, :, 1:] - x[:, :, :-1]  # (B, C, L-1)
#         smoothness = diff.pow(2).mean()
#         return smoothness


class RamanPINNClassifier(nn.Module):
    def __init__(self, in_channels=5, signal_length=1024, num_classes=2):
        super(RamanPINNClassifier, self).__init__()
        self.in_channels = in_channels
        self.signal_length = signal_length
        self.num_classes = num_classes

        self.input_dim = in_channels * signal_length  # 5×1024 = 5120

        self.net = nn.Sequential(
            nn.Linear(self.input_dim, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.3),

            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),

            nn.Linear(512, 128),
            nn.ReLU(),

            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # 输入 x: (B, 5, 1024) -> 展平为 (B, 5120)
        x = x.view(x.size(0), -1)
        out = self.net(x)
        return out

    def physics_loss(self, x):
        """
        简单物理约束损失：鼓励信号序列在频率空间中平滑。
        """
        # 差分平滑项：鼓励信号在拉曼位移方向上变化平稳
        diff = x[:, :, 1:] - x[:, :, :-1]  # shape: (B, 5, L-1)
        smoothness = diff.pow(2).mean()
        return smoothness



class RamanTransformerClassifier(nn.Module):
    def __init__(self, in_channels=5, signal_length=1024, num_classes=2, d_model=128, nhead=4, num_encoder_layers=3, dim_feedforward=512, dropout=0.1):
        super(RamanTransformerClassifier, self).__init__()
        self.in_channels = in_channels
        self.signal_length = signal_length
        self.num_classes = num_classes
        self.d_model = d_model
        # Token embedding
        self.embedding = nn.Linear(in_channels, d_model)
        # Positional encoding
        self.positional_encoding = nn.Parameter(torch.zeros(1, signal_length, d_model))
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        # Output layer
        self.fc_out = nn.Linear(d_model * signal_length, num_classes)

    def forward(self, x):
        # x: (B, 5, 1024)
        x = x.permute(0, 2, 1)
        x = self.embedding(x)  # (B, 1024, 128)
        # Add positional encoding
        x = x + self.positional_encoding
        # Pass through Transformer encoder
        x = self.transformer_encoder(x)
        # Flatten and pass through output layer
        x = x.view(x.size(0), -1)  # (B, 1024*128)
        out = self.fc_out(x)
        return out

    def physics_loss(self, x):
        diff = x[:, :, 1:] - x[:, :, :-1]
        smoothness = diff.pow(2).mean()
        return smoothness
