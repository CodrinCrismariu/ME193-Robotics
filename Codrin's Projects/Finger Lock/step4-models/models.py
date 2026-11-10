"""The letter classifier: a small CNN, built from scratch.

  input            3 x 96 x 96   color crop around the hand
  block 1         24 x 48 x 48   3x3 conv -> batch norm -> ReLU -> 2x2 max pool
  block 2         48 x 24 x 24
  block 3         96 x 12 x 12
  block 4        128 x  6 x  6
  block 5        160 x  6 x  6   3x3 conv -> batch norm -> ReLU (no pool)
  global average pooling: 160 numbers, one per feature map
  dropout, then 7 scores: C, O, D, R, I, N, none
"""
import torch
from torch import nn

LABELS = ["C", "O", "D", "R", "I", "N", "none"]


def block(c_in, c_out, pool=True):
    layers = [nn.Conv2d(c_in, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True)]
    if pool:
        layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)


class LetterNet(nn.Module):
    def __init__(self, n_classes=len(LABELS), widths=(24, 48, 96, 128, 160), dropout=0.3):
        super().__init__()
        chans = (3,) + tuple(widths)
        self.features = nn.Sequential(*[block(chans[i], chans[i + 1], pool=i < len(widths) - 1) for i in range(len(widths))])
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(dropout), nn.Linear(widths[-1], n_classes))

    def forward(self, x):
        return self.head(self.features(x))


if __name__ == "__main__":
    net = LetterNet()
    x = torch.zeros(1, 3, 96, 96)
    for i, b in enumerate(net.features):
        x = b(x)
        print(f"block {i + 1}: {tuple(x.shape[1:])}")
    print(f"{sum(p.numel() for p in net.parameters()):,} parameters")
