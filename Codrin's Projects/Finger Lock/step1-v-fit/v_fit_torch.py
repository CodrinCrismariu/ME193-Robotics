"""The V fit again, this time in PyTorch - the library the finger lock will use.

PyTorch builds the same network from ready-made pieces (nn.Linear = a layer of neurons,
nn.ReLU = the activation) and works out the gradients itself (autograd), so we no longer
write the backward pass by hand.

Run:
  python v_fit_torch.py                     check autograd against the hand derivation, then train
  python v_fit_torch.py --hidden 8          change the number of neurons (width)
  python v_fit_torch.py --layers 2          change the number of hidden layers (depth)
  python v_fit_torch.py --act tanh          change the activation (relu, leaky, tanh, sigmoid, softplus)
  python v_fit_torch.py --sweep             how often training finds the V, for several widths/depths
"""
import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

from common import AQUA, BLUE, DATA_GRAY, INK_2, load_v, save, standardize, style
from v_fit import LR_NET, SEED, init_network, network_loss_and_grads

torch.set_default_dtype(torch.float64)        # same precision as numpy, so results match exactly
ACTIVATIONS = {"relu": nn.ReLU, "leaky": nn.LeakyReLU, "tanh": nn.Tanh,
               "sigmoid": nn.Sigmoid, "softplus": nn.Softplus}


def build(hidden=2, layers=1, act="relu"):
    """1 input -> `layers` hidden layers of `hidden` neurons each -> 1 output (no output activation)."""
    parts, n_in = [], 1
    for _ in range(layers):
        parts += [nn.Linear(n_in, hidden), ACTIVATIONS[act]()]
        n_in = hidden
    parts.append(nn.Linear(n_in, 1))
    return nn.Sequential(*parts)


def randomize(net, seed):
    """Small random weights and zero biases, like AnnaCode: uniform(-1, 1) / sqrt(number of inputs)."""
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for layer in net:
            if isinstance(layer, nn.Linear):
                fan_in = layer.weight.shape[1]
                layer.weight.copy_((torch.rand(layer.weight.shape, generator=g) * 2 - 1) / fan_in ** 0.5)
                layer.bias.zero_()


def train(net, u, t, lr, epochs):
    """Full-batch gradient descent - the same loop as v_fit.py, with PyTorch doing the calculus."""
    opt = torch.optim.SGD(net.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    losses = []
    for _ in range(epochs + 1):
        opt.zero_grad()                       # forget last epoch's gradients
        loss = loss_fn(net(u), t)             # forward pass + mean squared error
        losses.append(loss.item())
        loss.backward()                       # backward pass: autograd fills in .grad for every weight
        opt.step()                            # w = w - lr * dL/dw for every weight
    return np.array(losses)


def check_against_hand_math(u_np, t_np, u, t):
    """Give PyTorch the exact starting weights of v_fit.py and compare its gradients with ours."""
    m, b, w, c = init_network(SEED)
    net = build(2, 1, "relu")
    with torch.no_grad():
        net[0].weight.copy_(torch.tensor(m).reshape(2, 1))
        net[0].bias.copy_(torch.tensor(b))
        net[2].weight.copy_(torch.tensor(w).reshape(1, 2))
        net[2].bias.fill_(c)
    loss = nn.MSELoss()(net(u), t)
    loss.backward()
    hand_loss, (dm, db, dw, dc) = network_loss_and_grads(m, b, w, c, u_np, t_np)
    pairs = [("dL/dm (hidden weights)", net[0].weight.grad.flatten(), dm),
             ("dL/db (hidden biases)", net[0].bias.grad, db),
             ("dL/dw (output weights)", net[2].weight.grad.flatten(), dw),
             ("dL/dc (output bias)", net[2].bias.grad, np.atleast_1d(dc))]
    print(f"Gradient check at the starting weights (seed {SEED}):  loss  autograd {loss.item():.10f}"
          f"   by hand {hand_loss:.10f}")
    for name, auto, hand in pairs:
        print(f"  {name:24s} autograd {np.round(auto.numpy(), 6)}   by hand {np.round(hand, 6)}")
    worst = max(np.abs(auto.numpy() - hand).max() for _, auto, hand in pairs)
    print(f"  largest difference: {worst:.1e}")
    return net


def sweep(u, t, y_std, epochs, lr, seeds=20):
    configs = [(1, 1), (2, 1), (3, 1), (4, 1), (8, 1), (2, 2), (4, 2)]   # (neurons per layer, layers)
    found = []
    for hidden, layers in configs:
        mses = []
        for s in range(seeds):
            torch.manual_seed(s)
            net = build(hidden, layers)
            randomize(net, s)
            mses.append(train(net, u, t, lr, epochs)[-1] * y_std ** 2)
        mses = np.array(mses)
        found.append(int((mses < 10).sum()))
        print(f"  {layers} hidden layer(s) x {hidden:2d} neurons: found the V in {found[-1]:2d}/{seeds} runs"
              f"   (median final MSE {np.median(mses):.1f})")

    style()
    fig, ax = plt.subplots(figsize=(8, 4))
    pos = [0, 1, 2, 3, 4, 5.8, 6.8]
    bars = ax.bar(pos, found, width=0.6, color=BLUE)
    ax.bar_label(bars, labels=[f"{f}/{seeds}" for f in found], padding=3, color=INK_2, fontsize=9)
    ax.set_xticks(pos, [f"{h}" if l == 1 else f"{h} + {h}" for h, l in configs])
    ax.text(2, -0.2, "1 hidden layer: neurons in it", transform=ax.get_xaxis_transform(), ha="center",
            color=INK_2)
    ax.text(6.3, -0.2, "2 hidden layers", transform=ax.get_xaxis_transform(), ha="center", color=INK_2)
    ax.set_ylim(0, seeds + 2)
    ax.set_yticks(range(0, seeds + 1, 5))
    ax.set_ylabel(f"runs that found the V (of {seeds})")
    ax.grid(axis="x", visible=False)
    ax.set_title(f"{seeds} random starts each: wider trains reliably, deeper-but-tiny does not",
                 loc="left")
    save(fig, "width_sweep.png")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--hidden", type=int, default=2, help="neurons per hidden layer (width)")
    p.add_argument("--layers", type=int, default=1, help="number of hidden layers (depth)")
    p.add_argument("--act", default="relu", choices=ACTIVATIONS, help="activation after each hidden layer")
    p.add_argument("--lr", type=float, default=LR_NET)
    p.add_argument("--epochs", type=int, default=3000)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--sweep", action="store_true", help="run the width/depth experiment")
    a = p.parse_args()

    x, y = load_v()
    u_np, x_mean, x_std = standardize(x)
    t_np, y_mean, y_std = standardize(y)
    u = torch.tensor(u_np).reshape(-1, 1)     # PyTorch wants (points, features)
    t = torch.tensor(t_np).reshape(-1, 1)

    if a.sweep:
        sweep(u, t, y_std, a.epochs, a.lr)
        return

    default = (a.hidden, a.layers, a.act, a.seed) == (2, 1, "relu", SEED)
    if default:
        net = check_against_hand_math(u_np, t_np, u, t)
        net.zero_grad()
    else:
        torch.manual_seed(a.seed)
        net = build(a.hidden, a.layers, a.act)
        randomize(net, a.seed)
    print(net)
    n_params = sum(p.numel() for p in net.parameters())
    losses = train(net, u, t, a.lr, a.epochs) * y_std ** 2
    print(f"{n_params} parameters, final MSE after {a.epochs} epochs: {losses[-1]:.3f}")
    if default:
        from v_fit import train_network
        _, hand = train_network(u_np, t_np, a.lr, a.epochs, SEED)
        print(f"numpy-by-hand run, same start: final MSE {hand[-1] * y_std ** 2:.3f}; "
              f"largest loss difference over all epochs {np.abs(hand * y_std ** 2 - losses).max():.1e}")

    style()
    xs = np.linspace(20, 100, 400)
    with torch.no_grad():
        ys = net(torch.tensor((xs - x_mean) / x_std).reshape(-1, 1)).numpy().ravel() * y_std + y_mean
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.8))
    ax1.scatter(x, y, s=14, color=DATA_GRAY, label="Anna's V data")
    ax1.plot(xs, ys, color=AQUA, lw=2.2, label="PyTorch model")
    ax1.set_title(f"{a.layers} hidden layer(s) x {a.hidden} {a.act} neurons", loc="left")
    ax1.set_xlabel("x (input)")
    ax1.set_ylabel("y (output)")
    ax1.legend(loc="upper center")
    ax2.plot(np.arange(1, len(losses) + 1), losses, color=AQUA, lw=2)
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xlabel("epoch (log scale)")
    ax2.set_ylabel("mean squared error (log scale)")
    ax2.set_title(f"Loss vs. epoch (final {losses[-1]:.2f})", loc="left")
    name = "torch_fit.png" if default else f"torch_fit_{a.layers}x{a.hidden}_{a.act}.png"
    save(fig, name)


if __name__ == "__main__":
    main()
