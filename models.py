import torch
import torch.nn as nn

class FeedForward(nn.Module):
    """
    Feed-Forward Neural Network for Stock Price Prediction
    """
    def __init__(self, input_features=48, hidden_layers=[256, 128, 64], dropout=0.3):
        super(FeedForward, self).__init__()
        
        layers = []
        prev_size = input_features
        
        layers.extend([
            nn.Linear(prev_size, hidden_layers[0]),
            nn.BatchNorm1d(hidden_layers[0]),
            nn.ReLU(),
            nn.Dropout(dropout)
        ])
        prev_size = hidden_layers[0]
        
        for hidden_size in hidden_layers[1:]:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, 1))
        
        self.network = nn.Sequential(*layers)
        
    #     self.apply(self._init_weights)
    
    # def _init_weights(self, module):
    #     if isinstance(module, nn.Linear):
    #         torch.nn.init.xavier_uniform_(module.weight)
    #         module.bias.data.fill_(0.01)
    
    def forward(self, x):
        return self.network(x)

class LSTM(nn.Module):
    """
    LSTM Network for Time Series Stock Prediction
    Uses sequence of past days to predict next day
    """
    def __init__(self, input_features=48, hidden_size=128, num_layers=2, dropout=0.2):
        super(LSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layers with dropout
        self.lstm = nn.LSTM(
            input_features, 
            hidden_size, 
            num_layers, 
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False
        )
        
        # Attention mechanism (optional)
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1),
            nn.Softmax(dim=1)
        )
        
        # Output layers
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1)
        )
    
    def forward(self, x):
        # x shape: (batch_size, sequence_length, features)
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Apply attention to focus on important time steps
        attention_weights = self.attention(lstm_out)
        context_vector = torch.sum(attention_weights * lstm_out, dim=1)
        
        # Final prediction
        output = self.classifier(context_vector)
        return output


class Hybrid(nn.Module):
    """
    Hybrid Architecture combining CNN and LSTM
    CNN extracts patterns, LSTM captures temporal dependencies
    """
    def __init__(self, input_features=48, cnn_channels=[64, 32], lstm_hidden=64):
        super(Hybrid, self).__init__()
        
        # 1D CNN for pattern extraction
        self.conv1 = nn.Conv1d(1, cnn_channels[0], kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(cnn_channels[0], cnn_channels[1], kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool1d(input_features // 2)
        
        # LSTM for temporal modeling
        self.lstm = nn.LSTM(cnn_channels[1], lstm_hidden, batch_first=True)
        
        # Final layers
        self.fc = nn.Sequential(
            nn.Linear(lstm_hidden, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1)
        )
    
    def forward(self, x):
        # x shape: (batch_size, features) -> (batch_size, 1, features)
        x = x.unsqueeze(1)
        
        # CNN feature extraction
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        
        # Reshape for LSTM: (batch_size, seq_len, features)
        x = x.transpose(1, 2)
        
        # LSTM processing
        lstm_out, _ = self.lstm(x)
        
        # Use last output
        x = lstm_out[:, -1, :]
        
        # Final prediction
        return self.fc(x)
