import torch
import torch.nn as nn


class ChordSequenceModel(nn.Module):
    def __init__(
        self,
        input_dim,
        num_classes,
        conv_channels=64,
        hidden_size=128,
        num_layers=2,
        dropout=0.2,
    ):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv1d(input_dim, conv_channels, kernel_size=5, padding=2),
            nn.BatchNorm1d(conv_channels),
            nn.ReLU(),

            nn.Conv1d(conv_channels, conv_channels, kernel_size=5, padding=2),
            nn.BatchNorm1d(conv_channels),
            nn.ReLU(),
        )

        self.gru = nn.GRU(
            input_size=conv_channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.classifier = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        # x: [batch, time, features]
        x = x.transpose(1, 2)          # [batch, features, time]
        x = self.conv(x)               # [batch, conv_channels, time]
        x = x.transpose(1, 2)          # [batch, time, conv_channels]
        x, _ = self.gru(x)             # [batch, time, hidden_size * 2]
        logits = self.classifier(x)    # [batch, time, num_classes]
        return logits