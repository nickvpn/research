"""
Complete experiment script for SINA vs INNA comparison.

Reproduces the experiments from:
  - Castera et al. (2021) "An Inertial Newton Algorithm for Deep Learning" (INNA paper)
  - Chadli et al. (2025) "A smoothing approximation approach to dynamical inertial
    newton systems..." (SINA paper)

What this script runs (matching the INNA paper's Section 5):
  1. INNA (alpha,beta) sensitivity study — Figure 2
  2. INNA vs SGD vs ADAM vs ADAGRAD comparison — Figure 3
  3. INNA step-size decay exponent comparison — Figure 4
  4. SINA (Algorithm 4.1 with adaptive epsilon) vs INNA
  5. SINA hyperparameter search (grid + Bayesian)

All experiments use:
  - Network in Network (NiN) architecture with BatchNorm
  - MNIST, CIFAR-10, CIFAR-100 datasets
  - 200 epochs, 5 random seeds
  - Batch size 32 (matching the INNA paper; configurable)
  - Step-size gamma_k = gamma_0 / sqrt(k+1) per ITERATION for INNA/SGD
  - ADAM and ADAGRAD use their own adaptive rates
  - gamma_0 selected by grid search on TRAINING LOSS after 15 epochs

Usage:
  python run_experiments.py                          # Run everything
  python run_experiments.py --datasets CIFAR10       # Single dataset
  python run_experiments.py --skip-baselines         # Skip SGD/ADAM/ADAGRAD
  python run_experiments.py --batch-size 256 --gpu   # Override for fast GPU
"""

import matplotlib
matplotlib.use('Agg')

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import time
import json
import argparse
import warnings

warnings.filterwarnings("ignore")

# Import our fixed optimizers (in same directory)
from inna_optimizer import INNA as INNAOptimizer
from sina_optimizer import (
    zang_plus, sina_step_fn, sina_update_epsilon,
    compute_grad_norm, initialize_phi, initialize_phi_from_grad,
)

# ============================================================
# 0. CONFIGURATION
# ============================================================
def parse_args():
    parser = argparse.ArgumentParser(description="SINA/INNA Experiments")
    parser.add_argument('--datasets', nargs='+',
                        default=['MNIST', 'CIFAR10', 'CIFAR100'],
                        choices=['MNIST', 'CIFAR10', 'CIFAR100'])
    parser.add_argument('--full-epochs', type=int, default=200,
                        help='Epochs for full training runs (default: 200)')
    parser.add_argument('--search-epochs', type=int, default=15,
                        help='Epochs for hyperparameter search (default: 15)')
    parser.add_argument('--num-seeds', type=int, default=5,
                        help='Number of random seeds (default: 5)')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size (default: 32, matching INNA paper)')
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--gpu', action='store_true',
                        help='Use GPU if available')
    parser.add_argument('--skip-baselines', action='store_true',
                        help='Skip SGD/ADAM/ADAGRAD runs')
    parser.add_argument('--skip-sensitivity', action='store_true',
                        help='Skip (alpha,beta) sensitivity study')
    parser.add_argument('--skip-decay-study', action='store_true',
                        help='Skip step-size decay exponent study')
    parser.add_argument('--skip-sina', action='store_true',
                        help='Skip SINA experiments')
    parser.add_argument('--output-dir', type=str, default='results',
                        help='Directory to save results')
    parser.add_argument('--compile', action='store_true',
                        help='Use torch.compile for speed (requires PyTorch 2+)')
    return parser.parse_args()


# ============================================================
# 1. NETWORK ARCHITECTURE — Network in Network (NiN)
# ============================================================
# Matches the INNA paper Section 5.2.1: "a slightly modified version of the
# popular Network in Network (NiN) (Lin et al., 2014). It is a reasonably
# large convolutional network with P ~ 10^6 parameters. We use ReLU
# activation functions."
#
# For SINA: the forward pass takes an eps argument; when eps > 0, we replace
# ReLU with the Zang smoothing function P_{rho_Z}(t, eps).

class NiNBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_planes)

    def forward(self, x, act_fn):
        return act_fn(self.bn(self.conv(x)))


class NiNNet(nn.Module):
    """Network in Network for image classification.

    Architecture: 3 blocks of (Conv -> 1x1Conv -> 1x1Conv -> Pool),
    followed by global average pooling.
    """
    def __init__(self, in_channels=3, num_classes=10):
        super().__init__()
        # Block 1
        self.block1_conv  = NiNBlock(in_channels, 192, kernel_size=5, padding=2)
        self.block1_cccp1 = NiNBlock(192, 160, kernel_size=1)
        self.block1_cccp2 = NiNBlock(160, 96, kernel_size=1)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Block 2
        self.block2_conv  = NiNBlock(96, 192, kernel_size=5, padding=2)
        self.block2_cccp3 = NiNBlock(192, 192, kernel_size=1)
        self.block2_cccp4 = NiNBlock(192, 192, kernel_size=1)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Block 3
        self.block3_conv  = NiNBlock(192, 192, kernel_size=3, padding=1)
        self.block3_cccp5 = NiNBlock(192, 192, kernel_size=1)
        self.block3_cccp6 = nn.Conv2d(192, num_classes, kernel_size=1)
        self.pool3 = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x, eps=None):
        if eps is not None and eps > 0.0:
            act = lambda z: zang_plus(z, eps)
        else:
            act = F.relu

        x = self.block1_conv(x, act)
        x = self.block1_cccp1(x, act)
        x = self.block1_cccp2(x, act)
        x = self.pool1(x)

        x = self.block2_conv(x, act)
        x = self.block2_cccp3(x, act)
        x = self.block2_cccp4(x, act)
        x = self.pool2(x)

        x = self.block3_conv(x, act)
        x = self.block3_cccp5(x, act)
        x = self.block3_cccp6(x)

        x = self.pool3(x)
        x = x.view(x.size(0), -1)
        return x


# ============================================================
# 2. DATA LOADING
# ============================================================
def get_loaders(dataset_name, batch_size, num_workers=4):
    if dataset_name == 'MNIST':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])
        trainset = torchvision.datasets.MNIST(
            root='./data', train=True, download=True, transform=transform)
        testset = torchvision.datasets.MNIST(
            root='./data', train=False, download=True, transform=transform)
        in_channels, num_classes = 1, 10
    else:
        # INNA paper uses simple normalization, no aggressive augmentation
        stats = ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        transform_train = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(*stats),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(*stats),
        ])

        if dataset_name == 'CIFAR10':
            trainset = torchvision.datasets.CIFAR10(
                root='./data', train=True, download=True, transform=transform_train)
            testset = torchvision.datasets.CIFAR10(
                root='./data', train=False, download=True, transform=transform_test)
            in_channels, num_classes = 3, 10
        elif dataset_name == 'CIFAR100':
            trainset = torchvision.datasets.CIFAR100(
                root='./data', train=True, download=True, transform=transform_train)
            testset = torchvision.datasets.CIFAR100(
                root='./data', train=False, download=True, transform=transform_test)
            in_channels, num_classes = 3, 100
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True, drop_last=False)

    return trainloader, testloader, in_channels, num_classes


# ============================================================
# 3. EVALUATION
# ============================================================
@torch.no_grad()
def evaluate(model, loader, device, eps=None):
    model.eval()
    correct = 0
    total = 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        out = model(x, eps=eps)
        _, pred = out.max(1)
        total += y.size(0)
        correct += (pred == y).sum().item()
    model.train()
    return correct / total


# ============================================================
# 4. TRAINING ROUTINES
# ============================================================

# ---- 4a. INNA Training ----
def train_inna(device, epochs, dataset, alpha, beta, lr, decaypower=0.5,
               batch_size=32, num_workers=4, use_compile=False):
    """Train NiN with the INNA optimizer.

    Step-size schedule: gamma_k = lr / (k+1)^decaypower, per iteration.
    This matches the INNA paper's Assumption 1.
    """
    train_l, test_l, in_c, n_cls = get_loaders(dataset, batch_size, num_workers)
    model = NiNNet(in_channels=in_c, num_classes=n_cls).to(device)
    opt = INNAOptimizer(model.parameters(), lr=lr, alpha=alpha, beta=beta,
                        decaypower=decaypower)
    crit = nn.CrossEntropyLoss()

    if use_compile:
        model = torch.compile(model)

    acc_hist, loss_hist = [], []

    for ep in range(epochs):
        epoch_loss = 0.0
        num_batches = 0
        model.train()
        for x, y in train_l:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            out = model(x, eps=None)  # INNA uses standard ReLU
            loss = crit(out, y)
            loss.backward()
            opt.step()
            epoch_loss += loss.item()
            num_batches += 1

        acc = evaluate(model, test_l, device, eps=None)
        acc_hist.append(acc)
        loss_hist.append(epoch_loss / num_batches)

        if (ep + 1) % 10 == 0:
            print(f"    INNA(a={alpha},b={beta}) Ep {ep+1}: "
                  f"Acc={acc:.4f} Loss={loss_hist[-1]:.4f}")

    return acc_hist, loss_hist


# ---- 4b. Baseline Training (SGD, ADAM, ADAGRAD) ----
def train_baseline(device, epochs, dataset, opt_name, lr, decaypower=0.5,
                   batch_size=32, num_workers=4, use_compile=False):
    """Train NiN with a standard optimizer.

    For SGD: uses the same gamma_k = lr/(k+1)^decaypower schedule as INNA,
             implemented via a LambdaLR scheduler stepping per iteration.
    For ADAM/ADAGRAD: uses their built-in adaptive rates, no external schedule.
    This matches the INNA paper Section 5.2.1.
    """
    train_l, test_l, in_c, n_cls = get_loaders(dataset, batch_size, num_workers)
    model = NiNNet(in_channels=in_c, num_classes=n_cls).to(device)
    crit = nn.CrossEntropyLoss()

    if opt_name == 'SGD':
        opt = torch.optim.SGD(model.parameters(), lr=lr)
    elif opt_name == 'ADAM':
        opt = torch.optim.Adam(model.parameters(), lr=lr)
    elif opt_name == 'ADAGRAD':
        opt = torch.optim.Adagrad(model.parameters(), lr=lr)
    else:
        raise ValueError(f"Unknown optimizer: {opt_name}")

    # For SGD: apply the same decaying step-size schedule as INNA per iteration.
    # For ADAM/ADAGRAD: the INNA paper says they "use an adaptive procedure
    # based on past gradients" — no external schedule.
    if opt_name == 'SGD' and decaypower > 0:
        # gamma_k = lr / (k+1)^q, where k is the iteration count.
        # LambdaLR: new_lr = lr * lambda(step)
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            opt, lr_lambda=lambda step: 1.0 / ((step + 1) ** decaypower))
    else:
        scheduler = None

    if use_compile:
        model = torch.compile(model)

    acc_hist, loss_hist = [], []

    for ep in range(epochs):
        epoch_loss = 0.0
        num_batches = 0
        model.train()
        for x, y in train_l:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            out = model(x, eps=None)
            loss = crit(out, y)
            loss.backward()
            opt.step()
            if scheduler is not None:
                scheduler.step()
            epoch_loss += loss.item()
            num_batches += 1

        acc = evaluate(model, test_l, device, eps=None)
        acc_hist.append(acc)
        loss_hist.append(epoch_loss / num_batches)

        if (ep + 1) % 10 == 0:
            print(f"    {opt_name} Ep {ep+1}: Acc={acc:.4f} Loss={loss_hist[-1]:.4f}")

    return acc_hist, loss_hist


# ---- 4c. SINA Training (Algorithm 4.1 with adaptive epsilon) ----
def train_sina(device, epochs, dataset, alpha, beta, gamma, eps0,
               sigma=0.5, delta=1.0, batch_size=32, num_workers=4,
               use_compile=False, prune=False, init_phi_from_grad=True):
    """Train NiN with the SINA algorithm (Algorithm 4.1).

    Key properties matching the paper:
      - gamma is CONSTANT (Assumption 2(iii): lim inf gamma_k > 0)
      - epsilon is adapted via Step 4 gradient-norm criterion
      - phi_0 optionally initialized from -grad S(theta_0, eps_0)

    Args:
        gamma: constant step size (must satisfy 0 < gamma <= min{beta, 1/alpha, alpha/L})
        eps0: initial smoothing parameter
        sigma: eps reduction factor (0 < sigma < 1)
        delta: gradient-norm threshold (delta > 0)
        init_phi_from_grad: if True, do one forward/backward to init phi_0
    """
    train_l, test_l, in_c, n_cls = get_loaders(dataset, batch_size, num_workers)
    model = NiNNet(in_channels=in_c, num_classes=n_cls).to(device)
    crit = nn.CrossEntropyLoss()

    # ---- Initialize phi ----
    if init_phi_from_grad:
        # Do one forward/backward pass to get initial gradients
        model.train()
        x0, y0 = next(iter(train_l))
        x0, y0 = x0.to(device), y0.to(device)
        out0 = model(x0, eps=eps0)
        loss0 = crit(out0, y0)
        loss0.backward()
        phi = initialize_phi_from_grad(model)
        model.zero_grad(set_to_none=True)
    else:
        phi = initialize_phi(model)

    eps_current = eps0

    if use_compile:
        model = torch.compile(model)

    acc_hist, loss_hist, eps_hist = [], [], []

    for ep in range(epochs):
        epoch_loss = 0.0
        num_batches = 0
        model.train()

        for x, y in train_l:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            model.zero_grad(set_to_none=True)

            # Step 2: Compute gradient g_k = grad_theta S(theta_k, eps_k)
            out = model(x, eps=eps_current)
            loss = crit(out, y)
            loss.backward()

            # Step 3: SINA parameter update (constant gamma)
            sina_step_fn(model, phi, alpha, beta, gamma)

            epoch_loss += loss.item()
            num_batches += 1

        # ---- Step 4: Adaptive epsilon update ----
        # We need ||grad S(theta_{k+1}, eps_k)||. Do one forward/backward
        # on a representative batch with the NEW theta and current eps.
        model.zero_grad(set_to_none=True)
        x_check, y_check = next(iter(train_l))
        x_check = x_check.to(device, non_blocking=True)
        y_check = y_check.to(device, non_blocking=True)
        out_check = model(x_check, eps=eps_current)
        loss_check = crit(out_check, y_check)
        loss_check.backward()
        post_grad_norm = compute_grad_norm(model)
        model.zero_grad(set_to_none=True)

        eps_current = sina_update_epsilon(
            post_grad_norm, eps_current, delta, sigma)

        eps_hist.append(eps_current)
        acc = evaluate(model, test_l, device, eps=eps_current)
        acc_hist.append(acc)
        loss_hist.append(epoch_loss / num_batches)

        if prune and (np.isnan(acc) or acc < 0.10):
            return acc_hist, loss_hist, eps_hist

        if (ep + 1) % 10 == 0:
            print(f"    SINA Ep {ep+1}: Acc={acc:.4f} Loss={loss_hist[-1]:.4f} "
                  f"eps={eps_current:.6f}")

    return acc_hist, loss_hist, eps_hist


# ============================================================
# 5. GAMMA_0 GRID SEARCH
# ============================================================
# INNA paper Section 5.2.1: "for each algorithm we select the initial
# step-size that most decreases the training error J after fifteen epochs"

GAMMA0_GRID = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]

def grid_search_lr(device, dataset, train_fn, search_epochs, batch_size,
                   num_workers, extra_kwargs=None):
    """Find gamma_0 that minimizes training loss after search_epochs.

    Args:
        train_fn: one of train_inna, train_baseline, train_sina
        extra_kwargs: dict of additional kwargs for train_fn (e.g. alpha, beta)

    Returns:
        best_lr (float)
    """
    if extra_kwargs is None:
        extra_kwargs = {}

    best_lr = GAMMA0_GRID[0]
    best_loss = float('inf')

    print(f"    Grid search over gamma_0: {GAMMA0_GRID}")
    for lr in GAMMA0_GRID:
        try:
            result = train_fn(device, search_epochs, dataset, lr=lr,
                              batch_size=batch_size, num_workers=num_workers,
                              **extra_kwargs)
            # train_fn returns (acc_hist, loss_hist) or (acc_hist, loss_hist, eps_hist)
            if isinstance(result, tuple) and len(result) >= 2:
                loss_hist = result[1]
            else:
                continue

            final_loss = loss_hist[-1] if loss_hist else float('inf')

            if np.isfinite(final_loss) and final_loss < best_loss:
                best_loss = final_loss
                best_lr = lr

            print(f"      gamma_0={lr:.4f} -> train_loss={final_loss:.4f}")
        except Exception as e:
            print(f"      gamma_0={lr:.4f} -> FAILED: {e}")

    print(f"    Best gamma_0 = {best_lr} (loss={best_loss:.4f})")
    return best_lr


# ============================================================
# 6. MULTI-SEED RUNNERS
# ============================================================
def run_multi_seed(train_fn, device, dataset, num_seeds, epochs,
                   batch_size, num_workers, method_label, extra_kwargs=None,
                   use_compile=False):
    """Run a training function across multiple seeds and collect results."""
    if extra_kwargs is None:
        extra_kwargs = {}

    print(f"  > Running {method_label} ({num_seeds} seeds, {epochs} epochs)...")
    all_accs, all_losses, all_eps = [], [], []

    for s in range(num_seeds):
        torch.manual_seed(s)
        np.random.seed(s)
        result = train_fn(device, epochs, dataset,
                          batch_size=batch_size, num_workers=num_workers,
                          use_compile=use_compile, **extra_kwargs)

        if len(result) == 3:
            a, l, e = result
            all_eps.append(e)
        else:
            a, l = result

        all_accs.append(a)
        all_losses.append(l)

    acc_np = np.array(all_accs)
    loss_np = np.array(all_losses)
    eps_np = np.array(all_eps[0]) if all_eps else None

    return acc_np, loss_np, eps_np


# ============================================================
# 7. SAVE / LOAD RESULTS
# ============================================================
def save_results(output_dir, dataset, method_name, acc_np, loss_np, eps_np=None):
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.join(output_dir, f"data_{dataset}_{method_name}.json")
    data = {
        "dataset": dataset,
        "method": method_name,
        "epochs": int(acc_np.shape[1]) if acc_np.ndim > 1 else len(acc_np),
        "seeds": int(acc_np.shape[0]) if acc_np.ndim > 1 else 1,
        "accuracy": acc_np.tolist(),
        "loss": loss_np.tolist(),
        "epsilon": eps_np.tolist() if eps_np is not None else None,
    }
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"  [Saved {filename}]")


def load_results(output_dir, dataset, method_name):
    filename = os.path.join(output_dir, f"data_{dataset}_{method_name}.json")
    if not os.path.exists(filename):
        return None
    with open(filename, 'r') as f:
        d = json.load(f)
    acc = np.array(d['accuracy'])
    loss = np.array(d['loss'])
    eps = np.array(d['epsilon']) if d.get('epsilon') is not None else None
    return acc, loss, eps


# ============================================================
# 8. PLOTTING — matches INNA paper format
# ============================================================
# INNA paper: "solid lines represent mean values and pale surfaces
# represent the best and worst runs" — min/max shading, not std.

COLORS = {
    'INNA_(0.1,0.1)': '#1f77b4',    # blue
    'INNA_(0.5,0.1)': '#2ca02c',    # green
    'INNA_(0.5,0.5)': '#17becf',    # cyan
    'INNA_(0.5,1.0)': '#9467bd',    # purple
    'INNA':           '#2ca02c',    # green (default INNA config)
    'SGD':            '#d62728',    # red
    'ADAM':           '#ff7f0e',    # orange
    'ADAGRAD':        '#e377c2',    # pink
    'SINA':           '#1f77b4',    # blue
    'SINA_Grid':      '#1f77b4',    # blue
    'SINA_Bayes':     '#17becf',    # cyan
}

LINESTYLES = {
    'INNA_(0.1,0.1)': '--',
    'INNA_(0.5,0.1)': '-',
    'INNA_(0.5,0.5)': '-',
    'INNA_(0.5,1.0)': '-.',
    'INNA':           '-',
    'SGD':            '-',
    'ADAM':           '-',
    'ADAGRAD':        '-',
    'SINA':           '-',
    'SINA_Grid':      '--',
    'SINA_Bayes':     '-',
}

MARKERS = {
    'INNA_(0.5,1.0)': 'd',
    'SGD':            's',
    'ADAM':           '^',
    'ADAGRAD':        'o',
}


def shade_plot(ax, epochs, data, label, color, linestyle='-', marker=None):
    """Plot mean line with min/max shading (matching INNA paper)."""
    mean = np.mean(data, axis=0)
    lo = np.min(data, axis=0)
    hi = np.max(data, axis=0)

    marker_every = max(1, len(epochs) // 8)  # ~8 markers across plot
    ax.plot(epochs, mean, label=label, color=color, linewidth=2,
            linestyle=linestyle, marker=marker, markevery=marker_every,
            markersize=6)
    ax.fill_between(epochs, lo, hi, color=color, alpha=0.15)


def plot_figure2(output_dir, dataset, results_dict, full_epochs):
    """Reproduce INNA paper Figure 2: (alpha,beta) sensitivity.

    Top row: log10(training loss) vs epochs
    Bottom row: test accuracy vs epochs
    """
    epochs = np.arange(1, full_epochs + 1)

    fig, (ax_loss, ax_acc) = plt.subplots(2, 1, figsize=(8, 10))

    for label, (acc, loss, _) in sorted(results_dict.items()):
        color = COLORS.get(label, 'gray')
        ls = LINESTYLES.get(label, '-')
        mk = MARKERS.get(label, None)

        # Training loss in log scale
        log_loss = np.log10(np.clip(loss, 1e-10, None))
        shade_plot(ax_loss, epochs, log_loss, label, color, ls, mk)

        # Test accuracy
        shade_plot(ax_acc, epochs, acc, label, color, ls, mk)

    ax_loss.set_ylabel(r'Training $\log_{10}(\mathcal{J}(\theta))$')
    ax_loss.set_xlabel('Epochs')
    ax_loss.set_title(f'{dataset} — INNA Hyperparameter Sensitivity')
    ax_loss.legend(fontsize=8)
    ax_loss.grid(True, alpha=0.3)

    ax_acc.set_ylabel('Test Accuracy')
    ax_acc.set_xlabel('Epochs')
    ax_acc.legend(fontsize=8)
    ax_acc.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'fig2_sensitivity_{dataset}.png'), dpi=150)
    plt.close()


def plot_figure3(output_dir, dataset, results_dict, full_epochs):
    """Reproduce INNA paper Figure 3: INNA vs SGD vs ADAM vs ADAGRAD.

    Top: log10(training loss), Bottom: test accuracy.
    """
    epochs = np.arange(1, full_epochs + 1)
    fig, (ax_loss, ax_acc) = plt.subplots(2, 1, figsize=(8, 10))

    for label, (acc, loss, _) in sorted(results_dict.items()):
        color = COLORS.get(label, 'gray')
        ls = LINESTYLES.get(label, '-')
        mk = MARKERS.get(label, None)

        log_loss = np.log10(np.clip(loss, 1e-10, None))
        shade_plot(ax_loss, epochs, log_loss, label, color, ls, mk)
        shade_plot(ax_acc, epochs, acc, label, color, ls, mk)

    ax_loss.set_ylabel(r'Training $\log_{10}(\mathcal{J}(\theta))$')
    ax_loss.set_xlabel('Epochs')
    ax_loss.set_title(f'{dataset} — INNA vs State-of-the-Art')
    ax_loss.legend(fontsize=8)
    ax_loss.grid(True, alpha=0.3)

    ax_acc.set_ylabel('Test Accuracy')
    ax_acc.set_xlabel('Epochs')
    ax_acc.legend(fontsize=8)
    ax_acc.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'fig3_comparison_{dataset}.png'), dpi=150)
    plt.close()


def plot_figure4(output_dir, dataset, results_dict, full_epochs):
    """Reproduce INNA paper Figure 4: step-size decay exponent comparison.

    Top: training loss for various q values.
    Bottom: best INNA (q=1/4) vs ADAM.
    """
    epochs = np.arange(1, full_epochs + 1)

    decay_colors = {
        'INNA_q=0.500': '#1f77b4',
        'INNA_q=0.250': '#d62728',
        'INNA_q=0.125': '#ff7f0e',
        'INNA_q=0.0625': '#9467bd',
    }
    decay_styles = {
        'INNA_q=0.500': '--',
        'INNA_q=0.250': '-',
        'INNA_q=0.125': ':',
        'INNA_q=0.0625': '-.',
    }

    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(8, 10))

    # Top: all decay exponents
    for label, (acc, loss, _) in sorted(results_dict.items()):
        if label.startswith('INNA_q='):
            color = decay_colors.get(label, 'gray')
            ls = decay_styles.get(label, '-')
            q_val = label.split('=')[1]
            log_loss = np.log10(np.clip(loss, 1e-10, None))
            shade_plot(ax_top, epochs, log_loss, f'$k^{{-{q_val}}}$',
                       color, ls)

    ax_top.set_ylabel(r'Training $\log_{10}(\mathcal{J}(\theta))$')
    ax_top.set_xlabel('Epochs')
    ax_top.set_title(f'{dataset} — Step-size Decay Comparison')
    ax_top.legend(fontsize=8)
    ax_top.grid(True, alpha=0.3)

    # Bottom: best INNA decay vs ADAM
    best_inna_key = 'INNA_q=0.250'  # paper shows q=1/4 is best
    if best_inna_key in results_dict:
        acc, loss, _ = results_dict[best_inna_key]
        log_loss = np.log10(np.clip(loss, 1e-10, None))
        shade_plot(ax_bot, epochs, log_loss,
                   r'INNA $\propto k^{-1/4}$', '#d62728', '-')
    if 'ADAM' in results_dict:
        acc, loss, _ = results_dict['ADAM']
        log_loss = np.log10(np.clip(loss, 1e-10, None))
        shade_plot(ax_bot, epochs, log_loss, 'ADAM', '#ff7f0e', '-', '^')

    ax_bot.set_ylabel(r'Training $\log_{10}(\mathcal{J}(\theta))$')
    ax_bot.set_xlabel('Epochs')
    ax_bot.set_title(f'{dataset} — INNA (best decay) vs ADAM')
    ax_bot.legend(fontsize=8)
    ax_bot.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'fig4_decay_{dataset}.png'), dpi=150)
    plt.close()


def plot_sina_comparison(output_dir, dataset, results_dict, full_epochs):
    """Plot SINA vs INNA comparison with epsilon trajectory."""
    epochs = np.arange(1, full_epochs + 1)
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    ax_acc, ax_loss, ax_eps = axes

    for label, (acc, loss, eps) in sorted(results_dict.items()):
        color = COLORS.get(label, 'gray')
        ls = LINESTYLES.get(label, '-')

        shade_plot(ax_acc, epochs, acc, label, color, ls)

        log_loss = np.log10(np.clip(loss, 1e-10, None))
        shade_plot(ax_loss, epochs, log_loss, label, color, ls)

        if eps is not None:
            ax_eps.plot(epochs, eps, label=label, color=color, linestyle='--')

    ax_acc.set_title(f'{dataset} — Test Accuracy')
    ax_acc.set_xlabel('Epochs')
    ax_acc.set_ylabel('Test Accuracy')
    ax_acc.legend(fontsize=8)
    ax_acc.grid(True, alpha=0.3)

    ax_loss.set_title(f'{dataset} — Training Loss')
    ax_loss.set_xlabel('Epochs')
    ax_loss.set_ylabel(r'$\log_{10}(\mathcal{J})$')
    ax_loss.legend(fontsize=8)
    ax_loss.grid(True, alpha=0.3)

    ax_eps.set_title(f'{dataset} — Epsilon Schedule')
    ax_eps.set_xlabel('Epochs')
    ax_eps.set_ylabel(r'$\varepsilon$')
    ax_eps.set_yscale('log')
    ax_eps.legend(fontsize=8)
    ax_eps.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'sina_comparison_{dataset}.png'), dpi=150)
    plt.close()


# ============================================================
# 9. SINA HYPERPARAMETER SEARCH
# ============================================================
def sina_grid_search(device, dataset, search_epochs, batch_size, num_workers):
    """Grid search for SINA hyperparameters.

    Searches over alpha, beta, gamma, eps0, sigma, delta.
    Selects based on TRAINING LOSS (not test accuracy).
    """
    print(f"\n--- SINA Grid Search ({dataset}) ---")

    alphas = [0.1, 0.5]
    betas = [0.05, 0.1]
    gammas = [0.005, 0.01, 0.05]
    # eps0, sigma, delta are less sensitive — fix reasonable defaults
    eps0 = 0.5
    sigma = 0.5
    delta = 1.0

    best_loss = float('inf')
    best_params = dict(alpha=0.5, beta=0.1, gamma=0.01, eps0=0.5,
                       sigma=0.5, delta=1.0)

    for a in alphas:
        for b in betas:
            for g in gammas:
                try:
                    _, loss_hist, _ = train_sina(
                        device, search_epochs, dataset,
                        alpha=a, beta=b, gamma=g, eps0=eps0,
                        sigma=sigma, delta=delta,
                        batch_size=batch_size, num_workers=num_workers,
                        prune=True)
                    final_loss = loss_hist[-1] if loss_hist else float('inf')
                    if np.isfinite(final_loss) and final_loss < best_loss:
                        best_loss = final_loss
                        best_params = dict(alpha=a, beta=b, gamma=g,
                                           eps0=eps0, sigma=sigma, delta=delta)
                    print(f"    a={a} b={b} g={g} -> loss={final_loss:.4f}")
                except Exception as e:
                    print(f"    a={a} b={b} g={g} -> FAILED: {e}")

    print(f"  Best SINA params: {best_params} (loss={best_loss:.4f})")
    return best_params


def sina_bayes_search(device, dataset, search_epochs, batch_size, num_workers):
    """Bayesian optimization for SINA hyperparameters using Optuna."""
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        print("  Optuna not installed, skipping Bayesian search")
        return None

    print(f"\n--- SINA Bayesian Search ({dataset}) ---")

    def objective(trial):
        alpha = trial.suggest_float("alpha", 0.1, 1.0)
        beta = trial.suggest_float("beta", 0.01, 0.5)
        gamma = trial.suggest_float("gamma", 0.001, 0.1, log=True)
        eps0 = trial.suggest_float("eps0", 0.1, 1.0)
        sigma = trial.suggest_float("sigma", 0.3, 0.9)
        delta = trial.suggest_float("delta", 0.1, 5.0)

        try:
            _, loss_hist, _ = train_sina(
                device, search_epochs, dataset,
                alpha=alpha, beta=beta, gamma=gamma, eps0=eps0,
                sigma=sigma, delta=delta,
                batch_size=batch_size, num_workers=num_workers,
                prune=True)
            final_loss = loss_hist[-1] if loss_hist else float('inf')
            return final_loss
        except Exception:
            return float('inf')

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=20, show_progress_bar=False)
    print(f"  Best Bayes params: {study.best_params} (loss={study.best_value:.4f})")
    return study.best_params


# ============================================================
# 10. MAIN ORCHESTRATION
# ============================================================
def main():
    args = parse_args()

    if args.gpu and torch.cuda.is_available():
        device = torch.device('cuda')
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision('high')
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("Using CPU")

    print(f"Config: epochs={args.full_epochs}, seeds={args.num_seeds}, "
          f"batch={args.batch_size}, datasets={args.datasets}")

    os.makedirs(args.output_dir, exist_ok=True)
    start_time = time.time()

    for dataset in args.datasets:
        print(f"\n{'='*60}")
        print(f"  DATASET: {dataset}")
        print(f"{'='*60}")

        all_results = {}  # label -> (acc_np, loss_np, eps_np)

        # ========================================
        # EXPERIMENT 1: INNA (alpha,beta) Sensitivity (Figure 2)
        # ========================================
        if not args.skip_sensitivity:
            print(f"\n--- Experiment 1: INNA Sensitivity Study ({dataset}) ---")
            configs = [(0.1, 0.1), (0.5, 0.1), (0.5, 0.5), (0.5, 1.0)]
            fig2_results = {}

            for alpha, beta in configs:
                label = f'INNA_({alpha},{beta})'
                cached = load_results(args.output_dir, dataset, label)
                if cached is not None:
                    print(f"  [Loaded cached {label}]")
                    fig2_results[label] = cached
                    continue

                # Grid search gamma_0 for this (alpha, beta) config
                print(f"  Finding gamma_0 for INNA(a={alpha}, b={beta})...")
                best_lr = grid_search_lr(
                    device, dataset, train_inna, args.search_epochs,
                    args.batch_size, args.num_workers,
                    extra_kwargs=dict(alpha=alpha, beta=beta))

                acc, loss, eps = run_multi_seed(
                    train_inna, device, dataset, args.num_seeds,
                    args.full_epochs, args.batch_size, args.num_workers,
                    method_label=label,
                    extra_kwargs=dict(alpha=alpha, beta=beta, lr=best_lr),
                    use_compile=args.compile)

                fig2_results[label] = (acc, loss, eps)
                save_results(args.output_dir, dataset, label, acc, loss, eps)

            plot_figure2(args.output_dir, dataset, fig2_results, args.full_epochs)

            # Use the best INNA config for subsequent comparisons
            # INNA paper says (0.5, 0.1) is a good default
            best_inna_label = 'INNA_(0.5,0.1)'
            if best_inna_label in fig2_results:
                all_results['INNA'] = fig2_results[best_inna_label]
        else:
            # Just run INNA with default (0.5, 0.1)
            label = 'INNA'
            cached = load_results(args.output_dir, dataset, label)
            if cached is not None:
                print(f"  [Loaded cached INNA]")
                all_results['INNA'] = cached
            else:
                print(f"\n--- Running INNA (default config) ({dataset}) ---")
                best_lr = grid_search_lr(
                    device, dataset, train_inna, args.search_epochs,
                    args.batch_size, args.num_workers,
                    extra_kwargs=dict(alpha=0.5, beta=0.1))

                acc, loss, eps = run_multi_seed(
                    train_inna, device, dataset, args.num_seeds,
                    args.full_epochs, args.batch_size, args.num_workers,
                    method_label=label,
                    extra_kwargs=dict(alpha=0.5, beta=0.1, lr=best_lr),
                    use_compile=args.compile)

                all_results['INNA'] = (acc, loss, eps)
                save_results(args.output_dir, dataset, label, acc, loss, eps)

        # ========================================
        # EXPERIMENT 2: Baselines — SGD, ADAM, ADAGRAD (Figure 3)
        # ========================================
        if not args.skip_baselines:
            print(f"\n--- Experiment 2: Baselines ({dataset}) ---")
            for opt_name in ['SGD', 'ADAM', 'ADAGRAD']:
                cached = load_results(args.output_dir, dataset, opt_name)
                if cached is not None:
                    print(f"  [Loaded cached {opt_name}]")
                    all_results[opt_name] = cached
                    continue

                print(f"  Finding gamma_0 for {opt_name}...")
                best_lr = grid_search_lr(
                    device, dataset, train_baseline, args.search_epochs,
                    args.batch_size, args.num_workers,
                    extra_kwargs=dict(opt_name=opt_name))

                acc, loss, eps = run_multi_seed(
                    train_baseline, device, dataset, args.num_seeds,
                    args.full_epochs, args.batch_size, args.num_workers,
                    method_label=opt_name,
                    extra_kwargs=dict(opt_name=opt_name, lr=best_lr),
                    use_compile=args.compile)

                all_results[opt_name] = (acc, loss, eps)
                save_results(args.output_dir, dataset, opt_name, acc, loss, eps)

            # Plot Figure 3: INNA vs baselines
            fig3_data = {}
            for k in ['INNA', 'SGD', 'ADAM', 'ADAGRAD']:
                if k in all_results:
                    fig3_data[k] = all_results[k]
            if fig3_data:
                plot_figure3(args.output_dir, dataset, fig3_data, args.full_epochs)

        # ========================================
        # EXPERIMENT 3: Step-size Decay Exponents (Figure 4)
        # ========================================
        if not args.skip_decay_study:
            print(f"\n--- Experiment 3: Decay Exponent Study ({dataset}) ---")
            decay_results = {}
            q_values = [0.5, 0.25, 0.125, 0.0625]

            for q in q_values:
                label = f'INNA_q={q:.4g}'
                cached = load_results(args.output_dir, dataset, label)
                if cached is not None:
                    print(f"  [Loaded cached {label}]")
                    decay_results[label] = cached
                    continue

                # Grid search gamma_0 for this decay exponent
                print(f"  Finding gamma_0 for INNA(q={q})...")
                best_lr = grid_search_lr(
                    device, dataset, train_inna, args.search_epochs,
                    args.batch_size, args.num_workers,
                    extra_kwargs=dict(alpha=0.5, beta=0.1, decaypower=q))

                acc, loss, eps = run_multi_seed(
                    train_inna, device, dataset, args.num_seeds,
                    args.full_epochs, args.batch_size, args.num_workers,
                    method_label=label,
                    extra_kwargs=dict(alpha=0.5, beta=0.1, lr=best_lr,
                                     decaypower=q),
                    use_compile=args.compile)

                decay_results[label] = (acc, loss, eps)
                save_results(args.output_dir, dataset, label, acc, loss, eps)

            # Add ADAM for the bottom panel of Figure 4
            if 'ADAM' in all_results:
                decay_results['ADAM'] = all_results['ADAM']

            plot_figure4(args.output_dir, dataset, decay_results, args.full_epochs)

        # ========================================
        # EXPERIMENT 4: SINA with Adaptive Epsilon
        # ========================================
        if not args.skip_sina:
            print(f"\n--- Experiment 4: SINA ({dataset}) ---")

            # Grid search
            grid_params = sina_grid_search(
                device, dataset, args.search_epochs,
                args.batch_size, args.num_workers)

            label_grid = 'SINA_Grid'
            acc, loss, eps = run_multi_seed(
                train_sina, device, dataset, args.num_seeds,
                args.full_epochs, args.batch_size, args.num_workers,
                method_label=label_grid,
                extra_kwargs=grid_params,
                use_compile=args.compile)
            all_results[label_grid] = (acc, loss, eps)
            save_results(args.output_dir, dataset, label_grid, acc, loss, eps)

            # Bayesian search
            bayes_params = sina_bayes_search(
                device, dataset, args.search_epochs,
                args.batch_size, args.num_workers)
            if bayes_params is not None:
                label_bayes = 'SINA_Bayes'
                acc, loss, eps = run_multi_seed(
                    train_sina, device, dataset, args.num_seeds,
                    args.full_epochs, args.batch_size, args.num_workers,
                    method_label=label_bayes,
                    extra_kwargs=bayes_params,
                    use_compile=args.compile)
                all_results[label_bayes] = (acc, loss, eps)
                save_results(args.output_dir, dataset, label_bayes, acc, loss, eps)

            # Plot SINA comparison
            sina_plot_data = {}
            for k in ['INNA', 'SINA_Grid', 'SINA_Bayes']:
                if k in all_results:
                    sina_plot_data[k] = all_results[k]
            if sina_plot_data:
                plot_sina_comparison(args.output_dir, dataset, sina_plot_data,
                                    args.full_epochs)

    elapsed = (time.time() - start_time) / 3600
    print(f"\nAll experiments finished. Total time: {elapsed:.2f} hours.")


if __name__ == "__main__":
    main()
