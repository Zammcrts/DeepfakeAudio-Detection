"""Arquitecturas CNN y LSTM. Se importan tanto desde train.py como desde Django.

Entrada de ambos modelos: espectrograma (n_mels, frames) ya normalizado.
Salida: 2 logits  -> índice 0 = real, índice 1 = fake.
"""
import torch
import torch.nn as nn


class CNN(nn.Module):
    """Trata el espectrograma como una imagen de 1 canal: (B, 1, n_mels, frames)."""

    def __init__(self, n_classes=2, dropout=0.4):
        super().__init__()

        def block(cin, cout):
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, padding=1, bias=False),
                nn.BatchNorm2d(cout),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )

        self.features = nn.Sequential(block(1, 32), block(32, 64), block(64, 128), block(128, 256))
        self.pool = nn.AdaptiveAvgPool2d(1)  # independiente del tamaño de entrada
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(256, 64), nn.ReLU(inplace=True),
                                  nn.Dropout(dropout), nn.Linear(64, n_classes))

    def forward(self, x):          # x: (B, n_mels, frames)
        x = x.unsqueeze(1)
        x = self.pool(self.features(x)).flatten(1)
        return self.head(x)


class LSTMNet(nn.Module):
    """Trata el espectrograma como una secuencia: cada frame es un vector de n_mels."""

    def __init__(self, n_mels=128, hidden=128, layers=2, n_classes=2, dropout=0.4):
        super().__init__()
        self.norm = nn.LayerNorm(n_mels)
        self.lstm = nn.LSTM(n_mels, hidden, num_layers=layers, batch_first=True,
                            bidirectional=True, dropout=dropout)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden * 4, 64), nn.ReLU(inplace=True),
                                  nn.Dropout(dropout), nn.Linear(64, n_classes))

    def forward(self, x):          # x: (B, n_mels, frames)
        x = self.norm(x.transpose(1, 2))          # (B, frames, n_mels)
        out, _ = self.lstm(x)                      # (B, frames, 2*hidden)
        feat = torch.cat([out.mean(1), out.max(1).values], dim=1)  # pooling sobre el tiempo
        return self.head(feat)


def build(name, n_mels):
    return CNN() if name == "cnn" else LSTMNet(n_mels=n_mels)