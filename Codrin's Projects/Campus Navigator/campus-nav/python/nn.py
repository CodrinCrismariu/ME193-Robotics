"""numpy forward pass of the step counter's network (no torch on the board).

The network is trained on the laptop (step-counter-training/model.py,
train.py) and exported by export.py into model.npz: weights, the input
normalization, the architecture, and a test input with the output PyTorch
gave for it. selftest() reruns that input here, so a broken export or a
numpy difference shows up in the log at startup.

Architecture: 1x1 conv, six causal dilated convs (kernel 5) with residual
connections and ReLU, 1x1 head. A causal conv with dilation d is the sum over
the kernel taps j of W[:, :, j] @ x[t - (k - 1 - j) d], the left side padded
with zeros: k small matrix products per layer.
"""
import json
from pathlib import Path

import numpy as np

MODEL = Path(__file__).with_name("model.npz")


class Net:
    def __init__(self, path=MODEL):
        d = np.load(path, allow_pickle=False)
        self.mean = d["mean"].astype(np.float64)
        self.std = d["std"].astype(np.float64)
        self.kernel = int(d["kernel"])
        self.dilations = [int(v) for v in d["dilations"]]
        self.lookahead = int(d["lookahead"])
        self.w_in, self.b_in = d["inp_w"], d["inp_b"]
        self.convs = [(d[f"conv{i}_w"], d[f"conv{i}_b"]) for i in range(len(self.dilations))]
        self.w_out, self.b_out = d["head_w"], d["head_b"]
        self.test_in, self.test_out = d["test_in"], d["test_out"]
        self.meta = json.loads(str(d["meta"]))
        self.receptive_field = 1 + (self.kernel - 1) * sum(self.dilations)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """(T, 8) inputs, unnormalized -> (T, 6) outputs (see model.py)."""
        h = ((x - self.mean) / self.std).T                      # (8, T)
        T = h.shape[1]
        h = np.maximum(self.w_in @ h + self.b_in[:, None], 0.0)
        for (w, b), d in zip(self.convs, self.dilations):
            pad = (self.kernel - 1) * d
            hp = np.concatenate([np.zeros((h.shape[0], pad)), h], axis=1)
            acc = np.repeat(b[:, None], T, axis=1).astype(np.float64)
            for j in range(self.kernel):
                acc += w[:, :, j] @ hp[:, j * d:j * d + T]
            h = h + np.maximum(acc, 0.0)
        return (self.w_out @ h + self.b_out[:, None]).T

    def selftest(self) -> float:
        """Largest difference from PyTorch's output for the stored test input."""
        return float(np.abs(self.forward(self.test_in) - self.test_out).max())
