"""A small convolutional neural network that counts fingers in a 64x64 grayscale image.

  input            1 x 64 x 64     (one gray channel)
  conv block 1     8 x 32 x 32     8 learned 3x3 kernels -> ReLU -> 2x2 max pool
  conv block 2    16 x 16 x 16
  conv block 3    32 x  8 x  8
  conv block 4    32 x  4 x  4
  flatten          512 numbers
  dense            64 ReLU neurons (with dropout while training)
  output           5 scores, one per finger count 1..5 (softmax turns them into probabilities)

Each conv block: a 3x3 convolution slides learned kernels over the image (like the Kernels page in
class, except the network learns the kernel numbers), ReLU keeps the positive responses, and max
pooling keeps the largest value in each 2x2 patch, halving the width and height.
"""
import torch
from torch import nn


def conv_block(c_in, c_out):
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, kernel_size=3, padding=1),   # padding=1 keeps the size the same
        nn.ReLU(),
        nn.MaxPool2d(2),                                     # halve width and height
    )


class FingerCNN(nn.Module):
    def __init__(self, n_classes=5, channels=(8, 16, 32, 32), hidden=64, dropout=0.3, img_size=64):
        super().__init__()
        blocks, c_in = [], 1
        for c_out in channels:
            blocks.append(conv_block(c_in, c_out))
            c_in = c_out
        self.features = nn.Sequential(*blocks)
        side = img_size // 2 ** len(channels)                # 64 -> 32 -> 16 -> 8 -> 4
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(c_in * side * side, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),                             # randomly silence neurons while training
            nn.Linear(hidden, n_classes),                    # raw scores ("logits"), no softmax here
        )

    def forward(self, x):
        return self.classifier(self.features(x))


if __name__ == "__main__":
    net = FingerCNN()
    x = torch.zeros(1, 1, 64, 64)
    print(net)
    for i, block in enumerate(net.features):
        x = block(x)
        print(f"after conv block {i + 1}: {tuple(x.shape[1:])}")
    print(f"{sum(p.numel() for p in net.parameters()):,} parameters")
