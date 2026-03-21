"""
Complete experiment script for SINA vs INNA comparison.

Phase 1: Reproduce INNA paper (Castera et al. 2021, JMLR)
  - Figure 2: INNA (alpha,beta) sensitivity study
  - Figure 3: INNA vs SGD vs ADAM vs ADAGRAD
  - Figure 4: Step-size decay exponent comparison

Phase 2: SINA comparison (Chadli et al. 2025, Optimization)
  - Grid search constant gamma for SINA
  - Delta-sigma sensitivity study
  - Full runs with dual evaluation (smoothed + ReLU)
  - SINA vs INNA comparison plots + epsilon trajectory

All experiments use:
  - Network in Network (NiN) architecture with BatchNorm (~10^6 params)
  - MNIST, CIFAR-10, CIFAR-100
  - 200 epochs, 5 random seeds, batch size 32 (matching INNA paper)
  - gamma_k = gamma_0 / sqrt(k+1) per ITERATION for INNA/SGD
  - gamma_0 selected by grid search on TRAINING LOSS after 15 epochs
  - Min/max shading on plots (matching INNA paper style)

Usage:
  python run_experiments.py                            # Run everything
  python run_experiments.py --phases 1                 # Phase 1 only
  python run_experiments.py --phases 2                 # Phase 2 only (needs Phase 1 cached)
  python run_experiments.py --datasets CIFAR10         # Single dataset
  python run_experiments.py --batch-size 256 --compile # GPU-optimized
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
import time
import json
import argparse
import warnings

warnings.filterwarnings("ignore")

from inna_optimizer import INNA as INNAOptimizer
from research.fixes.sina_optimizer_copy import (
    zang_plus, sina_step_fn, sina_update_epsilon,
    compute_grad_norm, initialize_phi, initialize_phi_from_grad,
)


# ============================================================
# 0. CONFIGURATION
# ============================================================
GAMMA0_GRID = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]


def parse_args():
    p = argparse.ArgumentParser(description="SINA/INNA Experiments")
    p.add_argument('--phases', nargs='+', type=int, default=[1, 2],
                   choices=[1, 2], help='Phases to run (default: 1 2)')
    p.add_argument('--datasets', nargs='+',
                   default=['MNIST', 'CIFAR10', 'CIFAR100'],
                   choices=['MNIST', 'CIFAR10', 'CIFAR100'])
    p.add_argument('--full-epochs', type=int, default=200)
    p.add_argument('--search-epochs', type=int, default=15)
    p.add_argument('--num-seeds', type=int, default=5)
    p.add_argument('--batch-size', type=int, default=32,
                   help='Batch size (default: 32, matching INNA paper)')
    p.add_argument('--num-workers', type=int, default=4)
    p.add_argument('--output-dir', type=str, default='results')
    p.add_argument('--compile', action='store_true',
                   help='Use torch.compile (PyTorch 2+)')
    return p.parse_args()


# ============================================================
# 1. NETWORK ARCHITECTURE - Network in Network (NiN)
# ============================================================
class NiNBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size,
                 stride=1, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size,
                              stride, padding)
        self.bn = nn.BatchNorm2d(out_planes)

    def forward(self, x, act_fn):
        return act_fn(self.bn(self.conv(x)))


class NiNNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=10):
        super().__init__()
        self.block1_conv  = NiNBlock(in_channels, 192, 5, padding=2)
        self.block1_cccp1 = NiNBlock(192, 160, 1)
        self.block1_cccp2 = NiNBlock(160, 96, 1)
        self.pool1 = nn.MaxPool2d(3, stride=2, padding=1)

        self.block2_conv  = NiNBlock(96, 192, 5, padding=2)
        self.block2_cccp3 = NiNBlock(192, 192, 1)
        self.block2_cccp4 = NiNBlock(192, 192, 1)
        self.pool2 = nn.MaxPool2d(3, stride=2, padding=1)

        self.block3_conv  = NiNBlock(192, 192, 3, padding=1)
        self.block3_cccp5 = NiNBlock(192, 192, 1)
        self.block3_cccp6 = nn.Conv2d(192, num_classes, 1)
        self.pool3 = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x, eps=None):
        act = F.relu if (eps is None or eps <= 0.0) else \
              lambda z: zang_plus(z, eps)

        x = self.pool1(self.block1_cccp2(
            self.block1_cccp1(self.block1_conv(x, act), act), act))
        x = self.pool2(self.block2_cccp4(
            self.block2_cccp3(self.block2_conv(x, act), act), act))
        x = self.block3_cccp6(
            self.block3_cccp5(self.block3_conv(x, act), act))
        x = self.pool3(x)
        return x.view(x.size(0), -1)


# ============================================================
# 2. DATA LOADING
# ============================================================
def get_loaders(dataset_name, batch_size, num_workers=4):
    if dataset_name == 'MNIST':
        tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])
        trainset = torchvision.datasets.MNIST(
            root='./data', train=True, download=True, transform=tf)
        testset = torchvision.datasets.MNIST(
            root='./data', train=False, download=True, transform=tf)
        in_ch, n_cls = 1, 10
    else:
        stats = ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        tf_train = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(*stats),
        ])
        tf_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(*stats),
        ])
        if dataset_name == 'CIFAR10':
            trainset = torchvision.datasets.CIFAR10(
                root='./data', train=True, download=True, transform=tf_train)
            testset = torchvision.datasets.CIFAR10(
                root='./data', train=False, download=True, transform=tf_test)
            in_ch, n_cls = 3, 10
        else:
            trainset = torchvision.datasets.CIFAR100(
                root='./data', train=True, download=True, transform=tf_train)
            testset = torchvision.datasets.CIFAR100(
                root='./data', train=False, download=True, transform=tf_test)
            in_ch, n_cls = 3, 100

    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True)
    test_loader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True, drop_last=False)
    return train_loader, test_loader, in_ch, n_cls


# ============================================================
# 3. EVALUATION
# ============================================================
@torch.no_grad()
def evaluate(model, loader, device, eps=None):
    model.eval()
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        out = model(x, eps=eps)
        correct += (out.argmax(1) == y).sum().item()
        total += y.size(0)
    model.train()
    return correct / total


# ============================================================
# 4. RESULT CACHING
# ============================================================
def _cache_path(output_dir, key):
    return os.path.join(output_dir, f"{key}.json")


def result_exists(output_dir, key):
    return os.path.exists(_cache_path(output_dir, key))


def save_result(output_dir, key, data):
    os.makedirs(output_dir, exist_ok=True)
    with open(_cache_path(output_dir, key), 'w') as f:
        json.dump(data, f, indent=2)
    print(f"  [Saved {key}]")


def load_result(output_dir, key):
    with open(_cache_path(output_dir, key), 'r') as f:
        d = json.load(f)
    # Convert lists back to numpy
    for k in ('accuracy', 'loss', 'accuracy_relu', 'epsilon'):
        if k in d and d[k] is not None:
            d[k] = np.array(d[k])
    return d


def _to_list(x):
    if isinstance(x, np.ndarray):
        return x.tolist()
    return x


# ============================================================
# 5. TRAINING ROUTINES
# ============================================================

# ---- 5a. INNA ----
def train_inna(device, epochs, dataset, alpha, beta, lr, decaypower=0.5,
               batch_size=32, num_workers=4, use_compile=False, seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_l, test_l, in_ch, n_cls = get_loaders(dataset, batch_size, num_workers)
    model = NiNNet(in_channels=in_ch, num_classes=n_cls).to(device)
    opt = INNAOptimizer(model.parameters(), lr=lr, alpha=alpha, beta=beta,
                        decaypower=decaypower)
    crit = nn.CrossEntropyLoss()
    if use_compile:
        model = torch.compile(model)

    acc_hist, loss_hist = [], []
    for ep in range(epochs):
        epoch_loss = n_batch = 0
        model.train()
        for x, y in train_l:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            loss = crit(model(x), y)
            loss.backward()
            opt.step()
            epoch_loss += loss.item()
            n_batch += 1

        acc = evaluate(model, test_l, device)
        acc_hist.append(acc)
        loss_hist.append(epoch_loss / n_batch)

        if (ep + 1) % 20 == 0:
            print(f"      [seed {seed}] INNA Ep {ep+1}: "
                  f"Acc={acc:.4f} Loss={loss_hist[-1]:.4f}")

    return acc_hist, loss_hist


# ---- 5b. Baselines (SGD, ADAM, ADAGRAD) ----
def train_baseline(device, epochs, dataset, opt_name, lr, decaypower=0.5,
                   batch_size=32, num_workers=4, use_compile=False, seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_l, test_l, in_ch, n_cls = get_loaders(dataset, batch_size, num_workers)
    model = NiNNet(in_channels=in_ch, num_classes=n_cls).to(device)
    crit = nn.CrossEntropyLoss()

    if opt_name == 'SGD':
        opt = torch.optim.SGD(model.parameters(), lr=lr)
    elif opt_name == 'ADAM':
        opt = torch.optim.Adam(model.parameters(), lr=lr)
    elif opt_name == 'ADAGRAD':
        opt = torch.optim.Adagrad(model.parameters(), lr=lr)
    else:
        raise ValueError(f"Unknown optimizer: {opt_name}")

    # SGD uses same per-iteration decay as INNA; ADAM/ADAGRAD use their own
    scheduler = None
    if opt_name == 'SGD' and decaypower > 0:
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            opt, lr_lambda=lambda step: 1.0 / ((step + 1) ** decaypower))

    if use_compile:
        model = torch.compile(model)

    acc_hist, loss_hist = [], []
    for ep in range(epochs):
        epoch_loss = n_batch = 0
        model.train()
        for x, y in train_l:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            loss = crit(model(x), y)
            loss.backward()
            opt.step()
            if scheduler is not None:
                scheduler.step()
            epoch_loss += loss.item()
            n_batch += 1

        acc = evaluate(model, test_l, device)
        acc_hist.append(acc)
        loss_hist.append(epoch_loss / n_batch)

        if (ep + 1) % 20 == 0:
            print(f"      [seed {seed}] {opt_name} Ep {ep+1}: "
                  f"Acc={acc:.4f} Loss={loss_hist[-1]:.4f}")

    return acc_hist, loss_hist


# ---- 5c. SINA (Algorithm 4.1 with adaptive epsilon) ----
def train_sina(device, epochs, dataset, alpha, beta, gamma, eps0,
               sigma=0.5, delta=1.0, batch_size=32, num_workers=4,
               use_compile=False, seed=0, init_phi_from_grad=True):
    """Train with SINA. Returns 4-tuple: (acc_smoothed, acc_relu, loss, eps)."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_l, test_l, in_ch, n_cls = get_loaders(dataset, batch_size, num_workers)
    model = NiNNet(in_channels=in_ch, num_classes=n_cls).to(device)
    crit = nn.CrossEntropyLoss()

    # Initialize phi
    if init_phi_from_grad:
        model.train()
        x0, y0 = next(iter(train_l))
        x0, y0 = x0.to(device), y0.to(device)
        loss0 = crit(model(x0, eps=eps0), y0)
        loss0.backward()
        phi = initialize_phi_from_grad(model)
        model.zero_grad(set_to_none=True)
    else:
        phi = initialize_phi(model)

    eps_current = eps0

    if use_compile:
        model = torch.compile(model)

    acc_smooth_hist, acc_relu_hist, loss_hist, eps_hist = [], [], [], []

    for ep in range(epochs):
        epoch_loss = n_batch = 0
        model.train()

        for x, y in train_l:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            model.zero_grad(set_to_none=True)

            # Step 2: gradient of smoothed loss
            out = model(x, eps=eps_current)
            loss = crit(out, y)
            loss.backward()

            # Step 3: SINA parameter update (constant gamma)
            sina_step_fn(model, phi, alpha, beta, gamma)

            epoch_loss += loss.item()
            n_batch += 1

        # Step 4: adaptive epsilon - check gradient norm on a batch
        model.zero_grad(set_to_none=True)
        x_chk, y_chk = next(iter(train_l))
        x_chk, y_chk = x_chk.to(device, non_blocking=True), \
                        y_chk.to(device, non_blocking=True)
        loss_chk = crit(model(x_chk, eps=eps_current), y_chk)
        loss_chk.backward()
        grad_norm = compute_grad_norm(model)
        model.zero_grad(set_to_none=True)

        eps_current = sina_update_epsilon(grad_norm, eps_current, delta, sigma)
        eps_hist.append(eps_current)

        # Dual evaluation: smoothed AND ReLU (eps=0)
        acc_smooth = evaluate(model, test_l, device, eps=eps_current)
        acc_relu = evaluate(model, test_l, device, eps=0)
        acc_smooth_hist.append(acc_smooth)
        acc_relu_hist.append(acc_relu)
        loss_hist.append(epoch_loss / n_batch)

        if (ep + 1) % 20 == 0:
            print(f"      [seed {seed}] SINA Ep {ep+1}: "
                  f"Acc(smooth)={acc_smooth:.4f} Acc(relu)={acc_relu:.4f} "
                  f"Loss={loss_hist[-1]:.4f} eps={eps_current:.6f}")

    return acc_smooth_hist, acc_relu_hist, loss_hist, eps_hist


# ============================================================
# 6. GAMMA_0 GRID SEARCH
# ============================================================
def grid_search_gamma0(device, dataset, optimizer_type, search_epochs,
                       batch_size, num_workers, extra_kwargs=None):
    """Find gamma_0 minimizing training loss after search_epochs.

    optimizer_type: 'inna', 'sgd', 'adam', 'adagrad', 'sina'
    Returns: (best_gamma0, search_log)
    """
    if extra_kwargs is None:
        extra_kwargs = {}
    best_lr = GAMMA0_GRID[0]
    best_loss = float('inf')
    log = []

    print(f"    Grid search gamma_0 for {optimizer_type.upper()}: {GAMMA0_GRID}")
    for lr in GAMMA0_GRID:
        try:
            if optimizer_type == 'inna':
                _, loss_h = train_inna(
                    device, search_epochs, dataset, lr=lr,
                    batch_size=batch_size, num_workers=num_workers,
                    seed=0, **extra_kwargs)
                final_loss = loss_h[-1]
            elif optimizer_type == 'sina':
                _, _, loss_h, _ = train_sina(
                    device, search_epochs, dataset, gamma=lr,
                    batch_size=batch_size, num_workers=num_workers,
                    seed=0, **extra_kwargs)
                final_loss = loss_h[-1]
            else:  # sgd, adam, adagrad
                _, loss_h = train_baseline(
                    device, search_epochs, dataset,
                    opt_name=optimizer_type.upper(), lr=lr,
                    batch_size=batch_size, num_workers=num_workers,
                    seed=0, **extra_kwargs)
                final_loss = loss_h[-1]

            if np.isfinite(final_loss) and final_loss < best_loss:
                best_loss = final_loss
                best_lr = lr

            log.append((lr, final_loss))
            print(f"      gamma_0={lr:.4f} -> loss={final_loss:.4f}")
        except Exception as e:
            log.append((lr, float('inf')))
            print(f"      gamma_0={lr:.4f} -> FAILED: {e}")

    print(f"    => Best gamma_0={best_lr} (loss={best_loss:.4f})")
    return best_lr, log


# ============================================================
# 7. MULTI-SEED RUNNER
# ============================================================
def run_multi_seed_inna(device, dataset, seeds, epochs, batch_size,
                        num_workers, use_compile, label, **train_kwargs):
    print(f"  > {label} ({len(seeds)} seeds, {epochs} epochs)...")
    all_acc, all_loss = [], []
    for s in seeds:
        a, l = train_inna(device, epochs, dataset, batch_size=batch_size,
                          num_workers=num_workers, use_compile=use_compile,
                          seed=s, **train_kwargs)
        all_acc.append(a)
        all_loss.append(l)
    return {'accuracy': np.array(all_acc), 'loss': np.array(all_loss)}


def run_multi_seed_baseline(device, dataset, seeds, epochs, batch_size,
                            num_workers, use_compile, label, **train_kwargs):
    print(f"  > {label} ({len(seeds)} seeds, {epochs} epochs)...")
    all_acc, all_loss = [], []
    for s in seeds:
        a, l = train_baseline(device, epochs, dataset, batch_size=batch_size,
                              num_workers=num_workers, use_compile=use_compile,
                              seed=s, **train_kwargs)
        all_acc.append(a)
        all_loss.append(l)
    return {'accuracy': np.array(all_acc), 'loss': np.array(all_loss)}


def run_multi_seed_sina(device, dataset, seeds, epochs, batch_size,
                        num_workers, use_compile, label, **train_kwargs):
    print(f"  > {label} ({len(seeds)} seeds, {epochs} epochs)...")
    all_acc_s, all_acc_r, all_loss, all_eps = [], [], [], []
    for s in seeds:
        a_s, a_r, l, e = train_sina(
            device, epochs, dataset, batch_size=batch_size,
            num_workers=num_workers, use_compile=use_compile,
            seed=s, **train_kwargs)
        all_acc_s.append(a_s)
        all_acc_r.append(a_r)
        all_loss.append(l)
        all_eps.append(e)
    return {
        'accuracy': np.array(all_acc_s),
        'accuracy_relu': np.array(all_acc_r),
        'loss': np.array(all_loss),
        'epsilon': np.array(all_eps),
    }


# ============================================================
# 8. PLOTTING - matches INNA paper style
# ============================================================
# INNA paper: "solid lines represent mean values and pale surfaces
# represent the best and worst runs" → min/max shading

COLORS = {
    'INNA_(0.1,0.1)': '#1f77b4',
    'INNA_(0.5,0.1)': '#2ca02c',
    'INNA_(0.5,0.5)': '#17becf',
    'INNA_(0.5,1.0)': '#9467bd',
    'INNA':           '#2ca02c',
    'SGD':            '#d62728',
    'ADAM':           '#ff7f0e',
    'ADAGRAD':        '#e377c2',
    'SINA':           '#1f77b4',
    'SINA (relu)':    '#8c564b',
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
    'SINA (relu)':    '--',
}

MARKERS = {
    'INNA_(0.5,1.0)': 'd',
    'SGD':            's',
    'ADAM':           '^',
    'ADAGRAD':        'o',
}


def _shade(ax, epochs, data, label, color, ls='-', marker=None):
    """Mean line with min/max shading."""
    mean = np.mean(data, axis=0)
    lo = np.min(data, axis=0)
    hi = np.max(data, axis=0)
    me = max(1, len(epochs) // 8)
    ax.plot(epochs, mean, label=label, color=color, linewidth=2,
            linestyle=ls, marker=marker, markevery=me, markersize=5)
    ax.fill_between(epochs, lo, hi, color=color, alpha=0.15)


def plot_figure2(output_dir, dataset, results, full_epochs):
    """INNA (alpha,beta) sensitivity - top: log loss, bottom: accuracy."""
    epochs = np.arange(1, full_epochs + 1)
    fig, (ax_l, ax_a) = plt.subplots(2, 1, figsize=(8, 10))

    for label in sorted(results):
        r = results[label]
        c = COLORS.get(label, 'gray')
        ls = LINESTYLES.get(label, '-')
        mk = MARKERS.get(label, None)
        _shade(ax_l, epochs, np.log10(np.clip(r['loss'], 1e-10, None)),
               label, c, ls, mk)
        _shade(ax_a, epochs, r['accuracy'], label, c, ls, mk)

    ax_l.set_ylabel(r'$\log_{10}(\mathcal{J}(\theta))$')
    ax_l.set_xlabel('Epochs')
    ax_l.set_title(f'{dataset} - INNA Hyperparameter Sensitivity')
    ax_l.legend(fontsize=8)
    ax_l.grid(True, alpha=0.3)
    ax_a.set_ylabel('Test Accuracy')
    ax_a.set_xlabel('Epochs')
    ax_a.legend(fontsize=8)
    ax_a.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'fig2_sensitivity_{dataset}.png'),
                dpi=150)
    plt.close()


def plot_figure3(output_dir, dataset, results, full_epochs):
    """INNA vs SGD vs ADAM vs ADAGRAD."""
    epochs = np.arange(1, full_epochs + 1)
    fig, (ax_l, ax_a) = plt.subplots(2, 1, figsize=(8, 10))

    for label in sorted(results):
        r = results[label]
        c = COLORS.get(label, 'gray')
        ls = LINESTYLES.get(label, '-')
        mk = MARKERS.get(label, None)
        _shade(ax_l, epochs, np.log10(np.clip(r['loss'], 1e-10, None)),
               label, c, ls, mk)
        _shade(ax_a, epochs, r['accuracy'], label, c, ls, mk)

    ax_l.set_ylabel(r'$\log_{10}(\mathcal{J}(\theta))$')
    ax_l.set_xlabel('Epochs')
    ax_l.set_title(f'{dataset} - INNA vs State-of-the-Art')
    ax_l.legend(fontsize=8)
    ax_l.grid(True, alpha=0.3)
    ax_a.set_ylabel('Test Accuracy')
    ax_a.set_xlabel('Epochs')
    ax_a.legend(fontsize=8)
    ax_a.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'fig3_comparison_{dataset}.png'),
                dpi=150)
    plt.close()


def plot_figure4(output_dir, dataset, results, full_epochs):
    """Step-size decay exponents + best INNA vs ADAM."""
    epochs = np.arange(1, full_epochs + 1)

    decay_colors = {
        'INNA_q=0.500':  '#1f77b4',
        'INNA_q=0.250':  '#d62728',
        'INNA_q=0.125':  '#ff7f0e',
        'INNA_q=0.0625': '#9467bd',
    }
    decay_styles = {
        'INNA_q=0.500':  '--',
        'INNA_q=0.250':  '-',
        'INNA_q=0.125':  ':',
        'INNA_q=0.0625': '-.',
    }

    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(8, 10))

    # Top: all decay exponents
    for label in sorted(results):
        if label.startswith('INNA_q='):
            r = results[label]
            c = decay_colors.get(label, 'gray')
            ls = decay_styles.get(label, '-')
            q_str = label.split('=')[1]
            _shade(ax_top, epochs,
                   np.log10(np.clip(r['loss'], 1e-10, None)),
                   fr'$k^{{-{q_str}}}$', c, ls)

    ax_top.set_ylabel(r'$\log_{10}(\mathcal{J}(\theta))$')
    ax_top.set_xlabel('Epochs')
    ax_top.set_title(f'{dataset} - Step-size Decay Comparison')
    ax_top.legend(fontsize=8)
    ax_top.grid(True, alpha=0.3)

    # Bottom: best INNA (q=1/4) vs ADAM
    if 'INNA_q=0.250' in results:
        r = results['INNA_q=0.250']
        _shade(ax_bot, epochs,
               np.log10(np.clip(r['loss'], 1e-10, None)),
               r'INNA $\propto k^{-1/4}$', '#d62728', '-')
    if 'ADAM' in results:
        r = results['ADAM']
        _shade(ax_bot, epochs,
               np.log10(np.clip(r['loss'], 1e-10, None)),
               'ADAM', '#ff7f0e', '-', '^')

    ax_bot.set_ylabel(r'$\log_{10}(\mathcal{J}(\theta))$')
    ax_bot.set_xlabel('Epochs')
    ax_bot.set_title(f'{dataset} - INNA (best decay) vs ADAM')
    ax_bot.legend(fontsize=8)
    ax_bot.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'fig4_decay_{dataset}.png'),
                dpi=150)
    plt.close()


def plot_sina_vs_inna(output_dir, dataset, sina_res, inna_res, full_epochs):
    """SINA vs INNA: loss, accuracy (smoothed), accuracy (ReLU), epsilon."""
    epochs = np.arange(1, full_epochs + 1)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    (ax_loss, ax_acc_s), (ax_acc_r, ax_eps) = axes

    # Training loss
    _shade(ax_loss, epochs,
           np.log10(np.clip(inna_res['loss'], 1e-10, None)),
           'INNA', COLORS['INNA'])
    _shade(ax_loss, epochs,
           np.log10(np.clip(sina_res['loss'], 1e-10, None)),
           'SINA', COLORS['SINA'])
    ax_loss.set_title(f'{dataset} - Training Loss')
    ax_loss.set_ylabel(r'$\log_{10}(\mathcal{J})$')
    ax_loss.set_xlabel('Epochs')
    ax_loss.legend(fontsize=8)
    ax_loss.grid(True, alpha=0.3)

    # Test accuracy (smoothed eval)
    _shade(ax_acc_s, epochs, inna_res['accuracy'], 'INNA', COLORS['INNA'])
    _shade(ax_acc_s, epochs, sina_res['accuracy'], 'SINA (smoothed)',
           COLORS['SINA'])
    ax_acc_s.set_title(f'{dataset} - Test Accuracy (smoothed eval)')
    ax_acc_s.set_ylabel('Test Accuracy')
    ax_acc_s.set_xlabel('Epochs')
    ax_acc_s.legend(fontsize=8)
    ax_acc_s.grid(True, alpha=0.3)

    # Test accuracy (ReLU eval) - the fair comparison
    _shade(ax_acc_r, epochs, inna_res['accuracy'], 'INNA (ReLU)',
           COLORS['INNA'])
    _shade(ax_acc_r, epochs, sina_res['accuracy_relu'],
           'SINA (ReLU eval)', COLORS['SINA (relu)'], '--')
    ax_acc_r.set_title(f'{dataset} - Test Accuracy (ReLU eval, fair comparison)')
    ax_acc_r.set_ylabel('Test Accuracy')
    ax_acc_r.set_xlabel('Epochs')
    ax_acc_r.legend(fontsize=8)
    ax_acc_r.grid(True, alpha=0.3)

    # Epsilon trajectory
    eps_data = sina_res['epsilon']  # (n_seeds, epochs)
    eps_mean = np.mean(eps_data, axis=0)
    eps_lo = np.min(eps_data, axis=0)
    eps_hi = np.max(eps_data, axis=0)
    ax_eps.semilogy(epochs, eps_mean, color=COLORS['SINA'], linewidth=2,
                    label=r'$\varepsilon$ (mean)')
    ax_eps.fill_between(epochs, eps_lo, eps_hi, color=COLORS['SINA'],
                        alpha=0.15)
    ax_eps.set_title(f'{dataset} - Epsilon Trajectory')
    ax_eps.set_ylabel(r'$\varepsilon$')
    ax_eps.set_xlabel('Epochs')
    ax_eps.legend(fontsize=8)
    ax_eps.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'sina_vs_inna_{dataset}.png'),
                dpi=150)
    plt.close()


def plot_delta_sigma_heatmap(output_dir, dataset, heatmap, deltas, sigmas,
                             best_delta, best_sigma):
    """Heatmap of training loss over (delta, sigma) grid."""
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(heatmap, cmap='viridis_r', aspect='auto')
    ax.set_xticks(range(len(sigmas)))
    ax.set_xticklabels([f'{s:.1f}' for s in sigmas])
    ax.set_yticks(range(len(deltas)))
    ax.set_yticklabels([f'{d:.1f}' for d in deltas])
    ax.set_xlabel(r'$\sigma$ (eps reduction factor)')
    ax.set_ylabel(r'$\delta$ (gradient-norm threshold)')
    ax.set_title(f'{dataset} - SINA (delta, sigma) Sensitivity\n'
                 f'Best: delta={best_delta}, sigma={best_sigma}')

    # Annotate cells
    for i in range(len(deltas)):
        for j in range(len(sigmas)):
            val = heatmap[i, j]
            if np.isfinite(val):
                ax.text(j, i, f'{val:.3f}', ha='center', va='center',
                        fontsize=8, color='white' if val > np.nanmedian(heatmap)
                        else 'black')

    plt.colorbar(im, ax=ax, label='Training Loss (15 epochs)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'heatmap_delta_sigma_{dataset}.png'),
                dpi=150)
    plt.close()


# ============================================================
# 9. PHASE 1 - Reproduce INNA Paper (Figures 2, 3, 4)
# ============================================================
def run_phase1(args, device):
    seeds = list(range(args.num_seeds))

    for dataset in args.datasets:
        print(f"\n{'='*60}")
        print(f"  PHASE 1 - {dataset}")
        print(f"{'='*60}")

        # ---- Figure 2: INNA (alpha,beta) sensitivity ----
        print(f"\n--- Figure 2: INNA sensitivity ({dataset}) ---")
        configs = [(0.1, 0.1), (0.5, 0.1), (0.5, 0.5), (0.5, 1.0)]
        fig2 = {}

        for alpha, beta in configs:
            label = f'INNA_({alpha},{beta})'
            key = f'p1_fig2_{dataset}_{label}'

            if result_exists(args.output_dir, key):
                print(f"  [Cached] {label}")
                fig2[label] = load_result(args.output_dir, key)
                continue

            best_lr, _ = grid_search_gamma0(
                device, dataset, 'inna', args.search_epochs,
                args.batch_size, args.num_workers,
                extra_kwargs={'alpha': alpha, 'beta': beta})

            res = run_multi_seed_inna(
                device, dataset, seeds, args.full_epochs,
                args.batch_size, args.num_workers, args.compile,
                label, alpha=alpha, beta=beta, lr=best_lr)

            res['best_gamma0'] = best_lr
            res['params'] = {'alpha': alpha, 'beta': beta}
            save_result(args.output_dir, key, {
                k: _to_list(v) for k, v in res.items()})
            fig2[label] = res

        plot_figure2(args.output_dir, dataset, fig2, args.full_epochs)

        # ---- Figure 3: INNA vs baselines ----
        print(f"\n--- Figure 3: INNA vs baselines ({dataset}) ---")
        fig3 = {}

        # Reuse best INNA config (0.5, 0.1) from Figure 2
        fig3['INNA'] = fig2['INNA_(0.5,0.1)']

        for opt_name in ['SGD', 'ADAM', 'ADAGRAD']:
            key = f'p1_fig3_{dataset}_{opt_name}'

            if result_exists(args.output_dir, key):
                print(f"  [Cached] {opt_name}")
                fig3[opt_name] = load_result(args.output_dir, key)
                continue

            best_lr, _ = grid_search_gamma0(
                device, dataset, opt_name.lower(), args.search_epochs,
                args.batch_size, args.num_workers)

            res = run_multi_seed_baseline(
                device, dataset, seeds, args.full_epochs,
                args.batch_size, args.num_workers, args.compile,
                opt_name, opt_name=opt_name, lr=best_lr)

            res['best_gamma0'] = best_lr
            save_result(args.output_dir, key, {
                k: _to_list(v) for k, v in res.items()})
            fig3[opt_name] = res

        plot_figure3(args.output_dir, dataset, fig3, args.full_epochs)

        # ---- Figure 4: decay exponents ----
        print(f"\n--- Figure 4: decay exponents ({dataset}) ---")
        fig4 = {}

        for q in [0.5, 0.25, 0.125, 0.0625]:
            label = f'INNA_q={q:.4g}'
            key = f'p1_fig4_{dataset}_{label}'

            if result_exists(args.output_dir, key):
                print(f"  [Cached] {label}")
                fig4[label] = load_result(args.output_dir, key)
                continue

            best_lr, _ = grid_search_gamma0(
                device, dataset, 'inna', args.search_epochs,
                args.batch_size, args.num_workers,
                extra_kwargs={'alpha': 0.5, 'beta': 0.1, 'decaypower': q})

            res = run_multi_seed_inna(
                device, dataset, seeds, args.full_epochs,
                args.batch_size, args.num_workers, args.compile,
                label, alpha=0.5, beta=0.1, lr=best_lr, decaypower=q)

            res['best_gamma0'] = best_lr
            res['params'] = {'alpha': 0.5, 'beta': 0.1, 'decaypower': q}
            save_result(args.output_dir, key, {
                k: _to_list(v) for k, v in res.items()})
            fig4[label] = res

        # Add ADAM from fig3 for bottom panel
        if 'ADAM' in fig3:
            fig4['ADAM'] = fig3['ADAM']

        plot_figure4(args.output_dir, dataset, fig4, args.full_epochs)


# ============================================================
# 10. PHASE 2 - SINA Comparison
# ============================================================
def run_phase2(args, device):
    seeds = list(range(args.num_seeds))
    alpha, beta = 0.5, 0.1  # INNA's best
    eps0 = 0.5

    delta_values = [0.1, 0.5, 1.0, 5.0]
    sigma_values = [0.3, 0.5, 0.7, 0.9]

    for dataset in args.datasets:
        print(f"\n{'='*60}")
        print(f"  PHASE 2 - {dataset}")
        print(f"{'='*60}")

        # Load INNA baseline from Phase 1
        inna_key = f'p1_fig2_{dataset}_INNA_(0.5,0.1)'
        if not result_exists(args.output_dir, inna_key):
            print(f"  WARNING: Phase 1 INNA result not found for {dataset}.")
            print(f"  Run --phases 1 first. Skipping {dataset}.")
            continue
        inna_res = load_result(args.output_dir, inna_key)

        # Step 1: Grid search constant gamma for SINA
        print(f"\n--- Step 1: SINA gamma grid search ({dataset}) ---")
        gamma_key = f'p2_gamma_{dataset}'
        if result_exists(args.output_dir, gamma_key):
            cached = load_result(args.output_dir, gamma_key)
            best_gamma = cached['best_gamma']
            print(f"  [Cached] best_gamma={best_gamma}")
        else:
            best_gamma, gamma_log = grid_search_gamma0(
                device, dataset, 'sina', args.search_epochs,
                args.batch_size, args.num_workers,
                extra_kwargs={'alpha': alpha, 'beta': beta,
                              'eps0': eps0, 'sigma': 0.5, 'delta': 1.0})
            save_result(args.output_dir, gamma_key, {
                'best_gamma': best_gamma,
                'search_log': gamma_log,
            })

        # Step 2: Delta-sigma sensitivity grid
        print(f"\n--- Step 2: (delta, sigma) sensitivity ({dataset}) ---")
        heatmap_key = f'p2_heatmap_{dataset}'
        if result_exists(args.output_dir, heatmap_key):
            cached = load_result(args.output_dir, heatmap_key)
            heatmap = cached['heatmap']
            best_delta = cached['best_delta']
            best_sigma = cached['best_sigma']
            print(f"  [Cached] best_delta={best_delta}, best_sigma={best_sigma}")
        else:
            heatmap = np.full((len(delta_values), len(sigma_values)), np.inf)
            best_loss = float('inf')
            best_delta, best_sigma = 1.0, 0.5

            for i, delta in enumerate(delta_values):
                for j, sigma in enumerate(sigma_values):
                    try:
                        _, _, loss_h, _ = train_sina(
                            device, args.search_epochs, dataset,
                            alpha=alpha, beta=beta, gamma=best_gamma,
                            eps0=eps0, sigma=sigma, delta=delta,
                            batch_size=args.batch_size,
                            num_workers=args.num_workers, seed=0)
                        fl = loss_h[-1] if loss_h else float('inf')
                        heatmap[i, j] = fl
                        if np.isfinite(fl) and fl < best_loss:
                            best_loss = fl
                            best_delta, best_sigma = delta, sigma
                        print(f"    delta={delta:.1f} sigma={sigma:.1f} "
                              f"-> loss={fl:.4f}")
                    except Exception as e:
                        print(f"    delta={delta:.1f} sigma={sigma:.1f} "
                              f"-> FAILED: {e}")

            print(f"  => Best: delta={best_delta}, sigma={best_sigma} "
                  f"(loss={best_loss:.4f})")
            save_result(args.output_dir, heatmap_key, {
                'heatmap': _to_list(heatmap),
                'best_delta': best_delta,
                'best_sigma': best_sigma,
                'delta_values': delta_values,
                'sigma_values': sigma_values,
            })

        plot_delta_sigma_heatmap(args.output_dir, dataset, heatmap,
                                delta_values, sigma_values,
                                best_delta, best_sigma)

        # Step 3: Full SINA run with best config
        print(f"\n--- Step 3: Full SINA run ({dataset}) ---")
        sina_key = f'p2_sina_{dataset}'
        if result_exists(args.output_dir, sina_key):
            print(f"  [Cached] SINA full run")
            sina_res = load_result(args.output_dir, sina_key)
        else:
            sina_res = run_multi_seed_sina(
                device, dataset, seeds, args.full_epochs,
                args.batch_size, args.num_workers, args.compile,
                'SINA',
                alpha=alpha, beta=beta, gamma=best_gamma,
                eps0=eps0, sigma=best_sigma, delta=best_delta)

            sina_res['params'] = {
                'alpha': alpha, 'beta': beta, 'gamma': best_gamma,
                'eps0': eps0, 'sigma': best_sigma, 'delta': best_delta,
            }
            save_result(args.output_dir, sina_key, {
                k: _to_list(v) for k, v in sina_res.items()})

        # Step 4: Plots
        print(f"\n--- Step 4: Plots ({dataset}) ---")
        plot_sina_vs_inna(args.output_dir, dataset, sina_res, inna_res,
                          args.full_epochs)

        print(f"  Done with {dataset}.")


# ============================================================
# 11. MAIN
# ============================================================
def main():
    args = parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision('high')
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("WARNING: Running on CPU - this will be very slow.")

    print(f"Config: phases={args.phases}, datasets={args.datasets}, "
          f"epochs={args.full_epochs}, seeds={args.num_seeds}, "
          f"batch={args.batch_size}")
    os.makedirs(args.output_dir, exist_ok=True)

    start = time.time()

    if 1 in args.phases:
        print("\n" + "=" * 60)
        print("  PHASE 1: Reproduce INNA Paper (Figures 2, 3, 4)")
        print("=" * 60)
        run_phase1(args, device)

    if 2 in args.phases:
        print("\n" + "=" * 60)
        print("  PHASE 2: SINA Comparison")
        print("=" * 60)
        run_phase2(args, device)

    elapsed = (time.time() - start) / 3600
    print(f"\nAll phases complete. Total time: {elapsed:.2f} hours.")


if __name__ == "__main__":
    main()
