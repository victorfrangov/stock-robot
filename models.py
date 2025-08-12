import torch
import torch.nn as nn

class FeedForward(nn.Module):
    def __init__(self, input_features, hidden_layers=[1024, 512, 256, 128], dropout=0.2, num_classes=3):
        super(FeedForward, self).__init__()
        layers = []
        prev = input_features
        for h in hidden_layers:
            layers.extend([
                nn.Linear(prev, h),
                nn.LayerNorm(h),
                nn.SiLU(),
                nn.Dropout(dropout)
            ])
            prev = h
        layers.append(nn.Linear(prev, num_classes))  # Changed from 1 to num_classes
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)  # Returns logits, not squeezed

class LSTM(nn.Module):
    """
    LSTM with input LayerNorm + attention head.
    Tune hidden_size (256–512), num_layers (1–2).
    """
    def __init__(self, input_features, hidden_size=512, num_layers=2, dropout=0.2, bidirectional=False, num_classes=3):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        self.input_ln = nn.LayerNorm(input_features)
        self.lstm = nn.LSTM(
            input_size=input_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional
        )
        lstm_out_dim = hidden_size * (2 if bidirectional else 1)

        self.attention = nn.Sequential(
            nn.Linear(lstm_out_dim, lstm_out_dim // 2),
            nn.Tanh(),
            nn.Linear(lstm_out_dim // 2, 1),
            nn.Softmax(dim=1)
        )

        self.classifier = nn.Sequential(
            nn.Linear(lstm_out_dim, 128),
            nn.SiLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.SiLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )
        
        hidden_out = hidden_size * (2 if bidirectional else 1)
        self.head = nn.Sequential(
            nn.Linear(hidden_out, max(32, hidden_out // 2)),
            nn.ReLU(inplace=True),
            nn.Linear(max(32, hidden_out // 2), num_classes),
        )
        
    def forward(self, x):
        # x: (batch, seq_len, features)
        x = self.input_ln(x)
        x = x.contiguous()
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden*)
        attn = self.attention(lstm_out)
        context = torch.sum(attn * lstm_out, dim=1)
        out = self.head(context)
        return out

class ResConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, p=1, dropout=0.1, groups=8):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size=k, padding=p)
        self.gn1 = nn.GroupNorm(num_groups=min(groups, out_ch), num_channels=out_ch)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size=k, padding=p)
        self.gn2 = nn.GroupNorm(num_groups=min(groups, out_ch), num_channels=out_ch)
        self.proj = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        residual = self.proj(x)
        x = self.act(self.gn1(self.conv1(x)))
        x = self.dropout(x)
        x = self.gn2(self.conv2(x))
        x = self.act(x + residual)
        return x

class Hybrid(nn.Module):
    """
    Residual CNN + LSTM. Treat features as channels; convs run over time.
    """
    def __init__(self, input_features, cnn_channels=[128, 64], lstm_hidden=256, cnn_dropout=0.1):
        super(Hybrid, self).__init__()
        # Input: (batch, seq_len, features) -> (batch, features, seq_len)
        self.conv_block1 = ResConvBlock(input_features, cnn_channels[0], dropout=cnn_dropout)
        self.conv_block2 = ResConvBlock(cnn_channels[0], cnn_channels[1], dropout=cnn_dropout)

        self.lstm = nn.LSTM(
            input_size=cnn_channels[1],
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=True
        )

        self.fc = nn.Sequential(
            nn.Linear(lstm_hidden, 128),
            nn.SiLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        # x: (batch, seq_len, features)
        x = x.transpose(1, 2)            # (batch, features, seq_len)
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = x.transpose(1, 2)            # (batch, seq_len, channels)
        lstm_out, _ = self.lstm(x)
        x = lstm_out[:, -1, :]
        return self.fc(x)