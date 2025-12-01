import torch
import torch.nn as nn
import math

class ADConversionTransformer(nn.Module):
    """
    Transformer model for Alzheimer's Disease conversion prediction.
    Dual-head architecture: classification (conversion risk) + regression (time to conversion)
    """

    def __init__(self, n_features=6, seq_len=7, d_model=64, nhead=8, num_layers=2,
                 dropout=0.1, dim_feedforward=2048):
        super(ADConversionTransformer, self).__init__()

        self.n_features = n_features
        self.seq_len = seq_len
        self.d_model = d_model

        # Feature projection layer
        self.feature_projection = nn.Linear(n_features, d_model)

        # Learned positional encoding (matches saved model: [1, 7, 64])
        self.pos_encoding = nn.Parameter(torch.zeros(1, seq_len, d_model))
        nn.init.normal_(self.pos_encoding, mean=0, std=0.02)

        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Global average pooling
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)

        # Dual-head outputs with sequential layers (matches saved model: 2 layers)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 32),    # 0
            nn.ReLU(),                 # 1 (no params)
            nn.Dropout(dropout),       # 2 (no params in state_dict but let's try)
            nn.Linear(32, 1)          # 3
        )

        self.time_regressor = nn.Sequential(
            nn.Linear(d_model, 32),    # 0
            nn.ReLU(),                 # 1 (no params)
            nn.Dropout(dropout),       # 2 (no params in state_dict but let's try)
            nn.Linear(32, 1)          # 3
        )

    def forward(self, x):
        """
        Forward pass
        Args:
            x: (batch_size, seq_len, n_features)
        Returns:
            conv_logits: (batch_size,) - conversion probability logits
            time_pred: (batch_size,) - time to conversion prediction
        """
        batch_size = x.size(0)

        # Project features to d_model dimension
        x = self.feature_projection(x)  # (batch_size, seq_len, d_model)

        # Add positional encoding
        x = x + self.pos_encoding  # (batch_size, seq_len, d_model)

        # Transformer encoding
        x = self.transformer(x)  # (batch_size, seq_len, d_model)

        # Global average pooling across sequence dimension
        x = x.transpose(1, 2)  # (batch_size, d_model, seq_len)
        x = self.global_avg_pool(x)  # (batch_size, d_model, 1)
        x = x.squeeze(-1)  # (batch_size, d_model)

        # Dual-head predictions
        conv_logits = self.classifier(x).squeeze(-1)  # (batch_size,)
        time_pred = self.time_regressor(x).squeeze(-1)  # (batch_size,)

        return conv_logits, time_pred