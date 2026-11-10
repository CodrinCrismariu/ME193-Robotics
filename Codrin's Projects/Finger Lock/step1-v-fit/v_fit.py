"""Fit Anna's V data with gradient descent written by hand (numpy only, no PyTorch).

Three models, each trained with plain batch gradient descent on the mean squared error:

  1. one neuron                 y_hat = m*u + b
  2. one neuron + ReLU          y_hat = ReLU(m*u + b)
  3. 2 hidden ReLU neurons      y_hat = w1*ReLU(m1*u + b1) + w2*ReLU(m2*u + b2) + c

u is x centered and scaled: u = (x - mean(x)) / std(x).

Run:  python v_fit.py
Prints the first 5 iterations of models 1 and 2 (the Excel file must match these),
then saves figures/fit.png (data + fits) and figures/loss.png (loss vs. epoch).
"""
import matplotlib.pyplot as plt
import numpy as np

from common import (AQUA, BLUE, DATA_GRAY, INK_2, MUTED, ORANGE, load_v, relu, relu_prime,
                    save, standardize, style)

# ── Hyperparameters (chosen by us, not learned) ──────────────────────────────
LR_NEURON = 0.25      # learning rate for models 1 and 2 (same value as the Excel file)
M0, B0 = 1.0, 0.0     # starting guess for m and b in models 1 and 2 (same as the Excel file)
LR_NET = 0.05         # learning rate for model 3
SEED = 1              # random starting weights for model 3 (12 of seeds 0-19 find the V; seed 0 gets stuck)
EPOCHS = 3000


def train_neuron(u, y, m, b, lr, epochs, use_relu):
    """Gradient descent on one neuron. Returns a list of (m, b, loss, dL/dm, dL/db), one per epoch.

    The loss is L = (1/N) * sum (y_i - a_i)^2 with a_i = z_i (no activation) or ReLU(z_i).
    Chain rule:  dL/dm = (1/N) * sum  -2 (y_i - a_i) * (da/dz)_i * u_i
                 dL/db = (1/N) * sum  -2 (y_i - a_i) * (da/dz)_i
    where da/dz = 1 without an activation and ReLU'(z) = (1 if z > 0 else 0) with ReLU.
    """
    history = []
    for _ in range(epochs + 1):
        z = m * u + b                                   # forward: weighted sum
        a = relu(z) if use_relu else z                  # activation
        loss = np.mean((y - a) ** 2)                    # mean squared error
        da_dz = relu_prime(z) if use_relu else 1.0
        dL_dm = np.mean(-2 * (y - a) * da_dz * u)       # backward: chain rule
        dL_db = np.mean(-2 * (y - a) * da_dz)
        history.append((m, b, loss, dL_dm, dL_db))
        m -= lr * dL_dm                                 # step downhill
        b -= lr * dL_db
    return history


def init_network(seed):
    """Small random starting weights, zero biases (the same scheme as AnnaCode's Randomize weights)."""
    rng = np.random.default_rng(seed)
    m = rng.uniform(-1, 1, 2)                # hidden weights  [m1, m2]
    b = np.zeros(2)                          # hidden biases   [b1, b2]
    w = rng.uniform(-1, 1, 2) / np.sqrt(2)   # output weights  [w1, w2]
    c = 0.0                                  # output bias
    return m, b, w, c


def network_loss_and_grads(m, b, w, c, u, t):
    """Forward and backward pass for 1 input -> 2 hidden ReLU neurons -> 1 output neuron.

    Hidden neuron k:  z_k = m_k*u + b_k,   h_k = ReLU(z_k)
    Output:           t_hat = w_1*h_1 + w_2*h_2 + c      (no activation on the output)
    Returns the loss and the gradient of the loss with respect to (m, b, w, c).
    """
    N = len(u)
    # forward pass, all N points at once; the two columns are the two hidden neurons
    Z = np.outer(u, m) + b                   # (N, 2) hidden pre-activations
    H = relu(Z)                              # (N, 2) hidden outputs
    t_hat = H @ w + c                        # (N,)   prediction
    loss = np.mean((t - t_hat) ** 2)

    # backward pass (chain rule), averaged over the N points
    d_out = -2 * (t - t_hat) / N                    # dL/dt_hat for each point
    dw = H.T @ d_out                                # dL/dw_k = sum d_out * h_k
    dc = d_out.sum()                                # dL/dc   = sum d_out
    d_hidden = np.outer(d_out, w) * relu_prime(Z)   # dL/dz_k = d_out * w_k * ReLU'(z_k)
    dm = d_hidden.T @ u                             # dL/dm_k = sum dL/dz_k * u
    db = d_hidden.sum(axis=0)                       # dL/db_k = sum dL/dz_k
    return loss, (dm, db, dw, dc)


def train_network(u, t, lr, epochs, seed):
    """Gradient descent on the 2-hidden-neuron network. t is y scaled to mean 0 and spread 1.
    Returns (final parameters, loss per epoch in scaled units)."""
    m, b, w, c = init_network(seed)
    losses = []
    for _ in range(epochs + 1):
        loss, (dm, db, dw, dc) = network_loss_and_grads(m, b, w, c, u, t)
        losses.append(loss)
        m = m - lr * dm
        b = b - lr * db
        w = w - lr * dw
        c = c - lr * dc
    return (m, b, w, c), np.array(losses)


def print_iterations(name, history, n=5):
    print(f"\n{name}: first {n} iterations (learning rate {LR_NEURON}, start m={M0}, b={B0})")
    print("| iteration | m | b | MSE | dL/dm | dL/db |")
    print("|---:|---:|---:|---:|---:|---:|")
    for k, (m, b, loss, gm, gb) in enumerate(history[:n + 1]):
        grads = f"{gm:.4f} | {gb:.4f}" if k < n else "- | -"
        print(f"| {k} | {m:.4f} | {b:.4f} | {loss:.2f} | {grads} |")


def main():
    x, y = load_v()
    u, x_mean, x_std = standardize(x)
    t, y_mean, y_std = standardize(y)
    print(f"{len(x)} points   mean(x) = {x_mean:.4f}   std(x) = {x_std:.4f}   "
          f"mean(y) = {y_mean:.4f}   std(y) = {y_std:.4f}")

    # 1. one neuron
    lin = train_neuron(u, y, M0, B0, LR_NEURON, EPOCHS, use_relu=False)
    print_iterations("Model 1 (one neuron)", lin)
    m_u, b_u = lin[-1][0], lin[-1][1]
    # same line in raw x units:  m*u + b = (m/std) x + (b - m*mean/std)
    m_x, b_x = m_u / x_std, b_u - m_u * x_mean / x_std
    # the calculus answer from the Loss minimization page: m = (mean(xy) - mean(x)mean(y)) / (mean(xx) - mean(x)^2)
    m_cf = (np.mean(x * y) - x.mean() * y.mean()) / (np.mean(x * x) - x.mean() ** 2)
    b_cf = y.mean() - m_cf * x.mean()
    print(f"after {EPOCHS} epochs: y = {m_x:.4f} x + {b_x:.4f}   (calculus formula: y = {m_cf:.4f} x + {b_cf:.4f})"
          f"   MSE = {lin[-1][2]:.2f}")

    # 2. one neuron + ReLU
    rel = train_neuron(u, y, M0, B0, LR_NEURON, EPOCHS, use_relu=True)
    print_iterations("Model 2 (one neuron + ReLU)", rel)
    m2, b2 = rel[-1][0], rel[-1][1]
    active = int((m2 * u + b2 > 0).sum())
    print(f"after {EPOCHS} epochs: m = {m2:.4f}, b = {b2:.4f}, MSE = {rel[-1][2]:.2f}, "
          f"ReLU on for {active}/{len(x)} points")

    # 3. two hidden ReLU neurons + output neuron
    (m3, b3, w3, c3), net_loss_t = train_network(u, t, LR_NET, EPOCHS, SEED)
    net_loss = net_loss_t * y_std ** 2        # back to raw y units, comparable with models 1 and 2
    print(f"\nModel 3 (2 hidden ReLU neurons), learning rate {LR_NET}, seed {SEED}: "
          f"MSE = {net_loss[-1]:.2f} after {EPOCHS} epochs")
    print("| hidden neuron | kink at x = | on for | slope it adds to y (raw units) |")
    print("|---:|---:|:---|---:|")
    for k in range(2):
        kink = x_mean - b3[k] * x_std / m3[k]            # where m_k*u + b_k = 0
        side = "x > kink" if m3[k] > 0 else "x < kink"   # ReLU passes the side where m_k*u + b_k > 0
        slope = y_std * w3[k] * m3[k] / x_std            # dy/dx it contributes on that side
        print(f"| {k + 1} | {kink:.2f} | {side} | {slope:+.3f} |")
    print(f"output bias: y = {y_mean + y_std * c3:.3f} + (sum of the two hidden contributions)")
    print("(the V was made with y = 2|x - 60|, so the ideal answer is kinks at 60 and slopes of -2 and +2)")

    # ── plots ────────────────────────────────────────────────────────────────
    style()
    xs = np.linspace(20, 100, 400)
    us = (xs - x_mean) / x_std
    fits = [
        ("1 neuron", m_u * us + b_u, lin[-1][2], BLUE),
        ("1 neuron + ReLU", relu(m2 * us + b2), rel[-1][2], ORANGE),
        ("2 hidden ReLU neurons", y_mean + y_std * (relu(np.outer(us, m3) + b3) @ w3 + c3), net_loss[-1], AQUA),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharey=True)
    for ax, (title, ys, mse, color) in zip(axes, fits):
        ax.scatter(x, y, s=14, color=DATA_GRAY, label="Anna's V data", zorder=2)
        if color == AQUA:
            ax.plot(xs, 2 * np.abs(xs - 60), color=INK_2, lw=1.2, ls=(0, (4, 3)),
                    label="true curve 2|x − 60|", zorder=3)
        ax.plot(xs, ys, color=color, lw=2.2, label="model", zorder=4)
        ax.set_title(title, loc="left")
        ax.text(0.03, 0.04, f"MSE = {mse:.1f}", transform=ax.transAxes, color=INK_2)
        ax.set_xlabel("x (input)")
    axes[0].set_ylabel("y (output)")
    axes[2].legend(loc="upper center", fontsize=9)
    fig.suptitle("Fitting Anna's V: a single neuron can't bend, two ReLU neurons can", x=0.07, y=1.04,
                 ha="left", fontsize=12, fontweight="bold")
    save(fig, "fit.png")

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    epochs = np.arange(1, EPOCHS + 2)
    ax.plot(epochs, [h[2] for h in lin], color=BLUE, lw=2, label="1 neuron")
    ax.plot(epochs, [h[2] for h in rel], color=ORANGE, lw=2, ls=(0, (5, 2)), label="1 neuron + ReLU")
    ax.plot(epochs, net_loss, color=AQUA, lw=2, label="2 hidden ReLU neurons")
    ax.axhline(4.0, color=MUTED, lw=1, ls=(0, (2, 2)))
    ax.text(1.2, 4.6, "noise floor: σ² = 2² = 4", color=INK_2, fontsize=9)
    ax.text(EPOCHS * 1.08, lin[-1][2], f"{lin[-1][2]:.0f}", color=INK_2, va="center", fontsize=9)
    ax.text(EPOCHS * 1.08, net_loss[-1] * 1.12, f"{net_loss[-1]:.1f}", color=INK_2, va="bottom", fontsize=9)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("epoch (log scale)")
    ax.set_ylabel("mean squared error (log scale)")
    ax.set_title("Loss vs. epoch", loc="left")
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 0.12))
    save(fig, "loss.png")


if __name__ == "__main__":
    main()
