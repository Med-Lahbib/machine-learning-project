import torch
import torch.nn as nn


class SentimentLSTM(nn.Module):
    """
    Bi-directional LSTM for binary sentiment classification.

    Architecture:
        Embedding → Bi-LSTM (2 layers) → Dropout → FC → ReLU → Dropout → FC → Sigmoid
    """

    def __init__(self, vocab_size, embed_dim=128, hidden_dim=256, num_layers=2,
                 dropout=0.3, pad_idx=0):
        super(SentimentLSTM, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Embedding layer: maps word indices to dense vectors
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)

        # Bi-directional LSTM: captures context from both directions
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # Fully connected layers for classification
        self.dropout1 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(hidden_dim * 2, 128)  # *2 for bidirectional
        self.relu = nn.ReLU()
        self.dropout2 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Tensor of shape (batch_size, seq_len) containing word indices.

        Returns:
            Tensor of shape (batch_size, 1) with sentiment probabilities.
        """
        # Embedding: (batch_size, seq_len) → (batch_size, seq_len, embed_dim)
        embedded = self.embedding(x)

        # LSTM: process the sequence
        # lstm_out shape: (batch_size, seq_len, hidden_dim * 2)
        lstm_out, (hidden, cell) = self.lstm(embedded)

        # Concatenate the final forward and backward hidden states
        # hidden shape: (num_layers * 2, batch_size, hidden_dim)
        # Take the last layer's forward and backward hidden states
        hidden_forward = hidden[-2, :, :]   # Last forward layer
        hidden_backward = hidden[-1, :, :]  # Last backward layer
        combined = torch.cat((hidden_forward, hidden_backward), dim=1)

        # Classification head
        out = self.dropout1(combined)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout2(out)
        out = self.fc2(out)
        out = self.sigmoid(out)

        return out

    def count_parameters(self):
        """Returns the total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
