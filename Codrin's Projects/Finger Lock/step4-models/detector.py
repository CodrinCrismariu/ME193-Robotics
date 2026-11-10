"""The hand detector: a small grid detector (in the style of YOLO), built from scratch.

The whole camera frame, shrunk to 192 x 144, goes in. Four conv + pool blocks shrink it 16 times, to a 12 x 9 grid
of cells, each 16 x 16 pixels of the input. For every cell the network outputs 5 numbers:

  score    "is the center of a hand in this cell?"  (a logit; sigmoid -> probability)
  dx, dy   where the center is inside the cell      (sigmoid -> 0..1)
  tw, th   the box's width and height                (w = 48 px * e^tw, h = 48 px * e^th)

The cell with the highest score gives the box; if even that score is low, there is no hand.

  input         3 x 144 x 192
  block 1      16 x  72 x  96    3x3 conv -> batch norm -> ReLU -> 2x2 max pool
  block 2      32 x  36 x  48
  block 3      64 x  18 x  24
  block 4      96 x   9 x  12
  block 5     128 x   9 x  12    3x3 conv -> batch norm -> ReLU (no pool)
  block 6     128 x   9 x  12    3x3 conv, dilation 2 (sees more of the hand around each cell)
  output        5 x   9 x  12    1x1 conv
"""
import torch
from torch import nn

IN_W, IN_H = 192, 144        # detector input size (the 320 x 240 photo scaled by 0.6)
STRIDE = 16                  # one grid cell = 16 x 16 input pixels
GRID_W, GRID_H = IN_W // STRIDE, IN_H // STRIDE
REF = 48.0                   # box size (px) that tw = th = 0 stands for


def block(c_in, c_out, pool=True, dilation=1):
    layers = [nn.Conv2d(c_in, c_out, 3, padding=dilation, dilation=dilation, bias=False), nn.BatchNorm2d(c_out),
              nn.ReLU(inplace=True)]
    if pool:
        layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)


class HandDetector(nn.Module):
    def __init__(self, widths=(16, 32, 64, 96, 128)):
        super().__init__()
        chans = (3,) + tuple(widths)
        self.features = nn.Sequential(
            *[block(chans[i], chans[i + 1], pool=i < 4) for i in range(len(widths))],
            block(widths[-1], widths[-1], pool=False, dilation=2))
        self.head = nn.Conv2d(widths[-1], 5, 1)

    def forward(self, x):
        return self.head(self.features(x))            # (N, 5, GRID_H, GRID_W)


def encode(boxes, has_hand):
    """Training targets from boxes in input pixels (N,4: x0,y0,x1,y1). Returns obj (N,GH,GW), reg (N,4), cell index (N,)."""
    n = boxes.shape[0]
    cx, cy = (boxes[:, 0] + boxes[:, 2]) / 2, (boxes[:, 1] + boxes[:, 3]) / 2
    w, h = (boxes[:, 2] - boxes[:, 0]).clamp(min=2), (boxes[:, 3] - boxes[:, 1]).clamp(min=2)
    j = (cx / STRIDE).floor().clamp(0, GRID_W - 1).long()
    i = (cy / STRIDE).floor().clamp(0, GRID_H - 1).long()
    obj = torch.zeros(n, GRID_H, GRID_W)
    obj[has_hand, i[has_hand], j[has_hand]] = 1.0
    reg = torch.stack([cx / STRIDE - j, cy / STRIDE - i, torch.log(w / REF), torch.log(h / REF)], 1)
    return obj, reg, i * GRID_W + j


def decode(out):
    """Network output (N,5,GH,GW) -> best box per image in input pixels (N,4), and its hand probability (N,)."""
    n = out.shape[0]
    score = out[:, 0].reshape(n, -1)
    best = score.argmax(1)
    i, j = best // GRID_W, best % GRID_W
    cell = out.reshape(n, 5, -1)[torch.arange(n), :, best]          # (N,5)
    cx = (j.float() + torch.sigmoid(cell[:, 1])) * STRIDE
    cy = (i.float() + torch.sigmoid(cell[:, 2])) * STRIDE
    w, h = REF * torch.exp(cell[:, 3].clamp(-4, 4)), REF * torch.exp(cell[:, 4].clamp(-4, 4))
    boxes = torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], 1)
    return boxes, torch.sigmoid(cell[:, 0])


def box_iou(a, b):
    """IoU of matching rows of two (N,4) box tensors."""
    ix = (torch.min(a[:, 2], b[:, 2]) - torch.max(a[:, 0], b[:, 0])).clamp(min=0)
    iy = (torch.min(a[:, 3], b[:, 3]) - torch.max(a[:, 1], b[:, 1])).clamp(min=0)
    inter = ix * iy
    area = lambda c: (c[:, 2] - c[:, 0]).clamp(min=0) * (c[:, 3] - c[:, 1]).clamp(min=0)
    return inter / (area(a) + area(b) - inter + 1e-9)


if __name__ == "__main__":
    net = HandDetector()
    print(f"output {tuple(net(torch.zeros(1, 3, IN_H, IN_W)).shape[1:])}, {sum(p.numel() for p in net.parameters()):,} parameters")
