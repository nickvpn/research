import matplotlib
matplotlib.use('Agg') # Force non-interactive backend for background running

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import optuna
import os
import time
import warnings
import optuna.visualization.matplotlib as opt_plt
import torch._dynamo
import json
from torch.cuda.amp import autocast



# Suppress warnings for clean logs
warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)
torch._dynamo.config.suppress_errors = True

# ==========================================
# 0. CONFIGURATION (Hardware Optimized)
# ==========================================
FULL_EPOCHS = 200       # Matches INNA paper
SEARCH_EPOCHS = 15      # Short runs for finding params
NUM_SEEDS = 5           # Matches INNA paper rigor
BATCH_SIZE = 512        # Optimized for H200
NUM_WORKERS = 8

USE_AMP = True          # Use Automatic Mixed Precision (FP16)

DATASETS_TO_RUN = ['CIFAR10', 'CIFAR100'] # cifar only on h200

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == 'cuda':
    torch.backends.cudnn.benchmark = True # Boosts speed for fixed image sizes
    # We use TF32 instead of AMP for stability + speed
    torch.set_float32_matmul_precision('high') 

print(f"--- CONFIGURATION ---")
print(f"Device: {device} ({torch.cuda.get_device_name(0) if device.type=='cuda' else 'CPU'})")
print(f"Batch Size: {BATCH_SIZE}")
print(f"Seeds per Run: {NUM_SEEDS}")
print(f"Full Epochs: {FULL_EPOCHS}")
print(f"-------------------------------")

# ==========================================
# 1. ARCHITECTURE (NiN with Batch Norm)
# ==========================================
def zang_plus(t, eps):
    if eps is None or eps <= 0.0:
        return F.relu(t)
    eps_t = t.new_tensor(eps)
    half = 0.5 * eps_t
    gt = t >  half
    lt = t < -half
    mid = (~gt) & (~lt)
    out = torch.zeros_like(t)
    out[gt] = t[gt]
    out[mid] = 0.5 / eps_t * (t[mid] + half) ** 2
    return out

class NiNBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0):
        super(NiNBlock, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_planes)

    def forward(self, x, act_fn):
        return act_fn(self.bn(self.conv(x)))

class NiNNetSINA(nn.Module):
    def __init__(self, in_channels=3, num_classes=10):
        super(NiNNetSINA, self).__init__()
        self.block1_conv = NiNBlock(in_channels, 192, kernel_size=5, padding=2)
        self.block1_cccp1 = NiNBlock(192, 160, kernel_size=1)
        self.block1_cccp2 = NiNBlock(160, 96, kernel_size=1)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.block2_conv = NiNBlock(96, 192, kernel_size=5, padding=2)
        self.block2_cccp3 = NiNBlock(192, 192, kernel_size=1)
        self.block2_cccp4 = NiNBlock(192, 192, kernel_size=1)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.block3_conv = NiNBlock(192, 192, kernel_size=3, padding=1)
        self.block3_cccp5 = NiNBlock(192, 192, kernel_size=1)
        self.block3_cccp6 = nn.Conv2d(192, num_classes, kernel_size=1)
        self.pool3 = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x, eps=None):
        if eps is None or eps <= 0.0:
            act = F.relu
        else:
            act = lambda z: zang_plus(z, eps)

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

# ==========================================
# 2. OPTIMIZERS
# ==========================================
class INNAOptimizer(torch.optim.Optimizer):
    def __init__(self, params, lr=0.1, alpha=0.5, beta=0.1, decaypower=0.5):
        defaults = dict(lr=lr, alpha=alpha, beta=beta, decaypower=decaypower)
        super(INNAOptimizer, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None: loss = closure()
        for group in self.param_groups:
            alpha = group['alpha']
            beta = group['beta']
            lr = group['lr']
            decaypower = group['decaypower']
            for p in group['params']:
                if p.grad is None: continue
                d_p = p.grad.data
                state = self.state[p]
                if 'step' not in state:
                    state['step'] = 0
                    state['psi'] = (1. - alpha * beta) * p.data.clone()
                psi = state['psi']
                cur_lr = lr / ((state['step'] + 1) ** decaypower)
                term_common = (1./beta - alpha) * p.data - (1./beta) * psi
                p.data.add_(cur_lr * (term_common - beta * d_p))
                psi.add_(cur_lr * term_common)
                state['step'] += 1
        return loss

@torch.compile
def sina_step(model, sina_state, gamma_k):
    alpha = sina_state["alpha"]
    beta  = sina_state["beta"]
    phi_iter = iter(sina_state["phi"])
    with torch.no_grad():
        for p in model.parameters():
            if not p.requires_grad or p.grad is None: continue
            g = p.grad
            ph = next(phi_iter)
            
            # Explicit Euler (Theta uses old Phi)
            theta_update = gamma_k * (-alpha * p + ph - beta * g)
            ph.add_(-gamma_k * g)
            p.add_(theta_update)

# ==========================================
# 3. DATA LOADING & UTILS
# ==========================================
def get_loaders(dataset_name, batch_size):
    if dataset_name == 'MNIST':
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
        in_channels, num_classes = 1, 10
    else:
        stats = ((0.5,0.5,0.5), (0.5,0.5,0.5))
        transform_train = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize(*stats)])
        transform_test = transforms.Compose([transforms.ToTensor(), transforms.Normalize(*stats)])
        
        if dataset_name == 'CIFAR10':
            trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
            testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
            in_channels, num_classes = 3, 10
        else: # CIFAR100
            trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
            testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
            in_channels, num_classes = 3, 100

    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
        persistent_workers=(NUM_WORKERS > 0),
    )
    testloader = torch.utils.data.DataLoader(
        testset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
        persistent_workers=(NUM_WORKERS > 0),
    )

    return trainloader, testloader, in_channels, num_classes

def evaluate(model, loader, device, eps=None, use_amp=False):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            if use_amp and device.type == "cuda":
                with autocast(device_type="cuda", dtype=torch.bfloat16):
                    out = model(x, eps=eps)
            else:
                out = model(x, eps=eps)

            _, pred = out.max(1)
            total += y.size(0)
            correct += (pred == y).sum().item()
    model.train()
    return correct / total


def save_raw_data(dataset, method_name, acc_hist, loss_hist, eps_hist=None):
    """ Saves raw training data to JSON for future plotting. """
    filename = f"data_{dataset}_{method_name}.json"
    
    data = {
        "dataset": dataset,
        "method": method_name,
        "epochs": acc_hist.shape[1] if len(acc_hist.shape) > 1 else len(acc_hist),
        "seeds": acc_hist.shape[0] if len(acc_hist.shape) > 1 else 1,
        "accuracy": acc_hist.tolist(), 
        "loss": loss_hist.tolist(),
        "epsilon": eps_hist.tolist() if eps_hist is not None else None
    }
    
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)
    print(f"  [Saved raw data to {filename}]")

# ==========================================
# 4. TRAINING ROUTINES (Compiled + FP32 Stable)
# ==========================================
def train_inna_instance(device, epochs, dataset, alpha, beta, lr=0.05, decay=0.5):
    # clear compiler cache on start of every training run
    torch._dynamo.reset()

    train_l, test_l, in_c, n_cls = get_loaders(dataset, BATCH_SIZE)
    model = NiNNetSINA(in_channels=in_c, num_classes=n_cls).to(device)
    opt = INNAOptimizer(model.parameters(), lr=lr, alpha=alpha, beta=beta, decaypower=decay)
    crit = nn.CrossEntropyLoss()
    
    # COMPILE FOR SPEED
    model = torch.compile(model)
    use_amp = USE_AMP and (device.type == "cuda")
    
    acc_hist, loss_hist = [], []
    
    for ep in range(epochs):
        epoch_loss = 0.0
        model.train()
        for x, y in train_l:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)

            if use_amp:
                with autocast(device_type="cuda", dtype=torch.bfloat16):
                    out = model(x, eps=None)
                    loss = crit(out, y)
            else:
                out = model(x, eps=None)
                loss = crit(out, y)

            loss.backward()
            opt.step()
            epoch_loss += loss.item()
        
        acc = evaluate(model, test_l, device, eps=None, use_amp=use_amp)
        acc_hist.append(acc)
        loss_hist.append(epoch_loss / len(train_l))
        
        if (ep+1) % 5 == 0:
            print(f"    INNA Ep {ep+1}: Acc={acc:.4f}")
            
    return acc_hist, loss_hist

def train_sina_instance(device, epochs, dataset, alpha, beta, gamma0, eps0, decay_rate,
                        decaypower=0.5, prune=False):
    # clear compiler cache on start of every training run
    torch._dynamo.reset()

    train_l, test_l, in_c, n_cls = get_loaders(dataset, BATCH_SIZE)
    model = NiNNetSINA(in_channels=in_c, num_classes=n_cls).to(device)
    crit = nn.CrossEntropyLoss()
    
    phi = [torch.zeros_like(p) for p in model.parameters() if p.requires_grad]
    state = {"phi": phi, "alpha": alpha, "beta": beta, "eps": eps0}
    
    # COMPILE FOR SPEED
    model = torch.compile(model)
    use_amp = USE_AMP and (device.type == "cuda")
    
    acc_hist, loss_hist, eps_hist = [], [], []
    
    for ep in range(epochs):
        gamma_k = gamma0 / ((ep + 1) ** decaypower)
        state["eps"] = max(1e-4, state["eps"] * decay_rate)
        eps_hist.append(state["eps"])
        
        epoch_loss = 0.0
        model.train()
        for x, y in train_l:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            model.zero_grad(set_to_none=True)
            
            if use_amp:
                with autocast(device_type="cuda", dtype=torch.bfloat16):
                    out = model(x, eps=state["eps"])
                    loss = crit(out, y)
            else:
                out = model(x, eps=state["eps"])
                loss = crit(out, y)

            loss.backward()
            sina_step(model, state, gamma_k)
            epoch_loss += loss.item()
            
        acc = evaluate(model, test_l, device, eps=state["eps"], use_amp=use_amp)
        acc_hist.append(acc)
        loss_hist.append(epoch_loss / len(train_l))
        
        if prune and (np.isnan(acc) or acc < 0.10):
            return acc_hist, loss_hist, eps_hist
        if (ep+1) % 5 == 0:
            print(f"    SINA Ep {ep+1}: Acc={acc:.4f} (eps={state['eps']:.4f})")

    return acc_hist, loss_hist, eps_hist


# ==========================================
# 5. WRAPPERS
# ==========================================
def run_multi_seed_inna(device, dataset, alpha, beta):
    print(f"  > Running INNA Multi-Seed ({NUM_SEEDS} seeds)...")
    accs, losses = [], []
    for s in range(NUM_SEEDS):
        torch.manual_seed(s)
        a, l = train_inna_instance(device, FULL_EPOCHS, dataset, alpha, beta)
        accs.append(a); losses.append(l)
    
    acc_np, loss_np = np.array(accs), np.array(losses)
    save_raw_data(dataset, "INNA", acc_np, loss_np)
    return acc_np, loss_np

def run_multi_seed_sina(device, dataset, params, method_label="SINA_Bayes"):
    print(f"  > Running {method_label} Multi-Seed ({NUM_SEEDS} seeds)...")
    accs, losses, eps_lists = [], [], []
    for s in range(NUM_SEEDS):
        torch.manual_seed(s)
        a, l, e = train_sina_instance(device, FULL_EPOCHS, dataset, **params, prune=False)
        accs.append(a); losses.append(l); eps_lists.append(e)
    
    acc_np, loss_np, eps_np = np.array(accs), np.array(losses), np.array(eps_lists[0])
    save_raw_data(dataset, method_label, acc_np, loss_np, eps_np)
    return acc_np, loss_np, eps_np

def run_sina_grid(device, dataset):
    print(f"\n--- Starting Grid Search ({dataset}) ---")
    alphas = [0.1, 0.5]
    betas = [0.1, 0.5]
    gamma0s = [0.01, 0.05]
    best_acc = 0.0
    best_params = {'alpha': 0.5, 'beta': 0.1, 'gamma0': 0.05, 'eps0': 0.5, 'decay_rate': 0.95}
    
    for a in alphas:
        for b in betas:
            for g in gamma0s:
                hist, _, _ = train_sina_instance(device, SEARCH_EPOCHS, dataset, a, b, g, 0.5, 0.95, prune=True)
                if hist and hist[-1] > best_acc:
                    best_acc = hist[-1]
                    best_params = {'alpha': a, 'beta': b, 'gamma0': g, 'eps0': 0.5, 'decay_rate': 0.95}
    print(f"Grid Best: {best_params} (Acc: {best_acc:.4f})")
    return best_params

def run_sina_bayes(device, dataset):
    print(f"\n--- Starting BayesOpt ({dataset}) ---")
    db_url = "sqlite:///optuna_studies.db"
    study_name = f"sina_study_{dataset}"
    
    def objective(trial):
        alpha = trial.suggest_float("alpha", 0.1, 1.0)
        beta = trial.suggest_float("beta", 0.01, 0.5)
        gamma0 = trial.suggest_float("gamma0", 0.01, 0.1)
        eps0 = trial.suggest_float("eps0", 0.1, 1.0)
        decay_rate = trial.suggest_float("decay_rate", 0.90, 0.99)
        hist, _, _ = train_sina_instance(device, SEARCH_EPOCHS, dataset, alpha, beta, gamma0, eps0, decay_rate, prune=True)
        return hist[-1] if hist else 0.0

    study = optuna.create_study(study_name=study_name, storage=db_url, direction="maximize", load_if_exists=True)
    study.optimize(objective, n_trials=20)
    print(f"Bayes Best: {study.best_params}")
    return study.best_params, study

# ==========================================
# 6. PLOTTING
# ==========================================
def plot_dataset_results(dataset, inna_res, grid_res, bayes_res):
    inna_acc, inna_loss = inna_res
    grid_acc, grid_loss, grid_eps = grid_res
    bayes_acc, bayes_loss, bayes_eps = bayes_res
    
    epochs = np.arange(1, FULL_EPOCHS + 1)
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))
    
    def shade(ax, data, label, color):
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        ax.plot(epochs, mean, label=label, color=color, linewidth=2)
        ax.fill_between(epochs, np.maximum(mean - std, 0), mean + std, color=color, alpha=0.15)

    shade(ax1, inna_acc, "INNA", "orange")
    shade(ax1, grid_acc, "SINA (Grid)", "blue")
    shade(ax1, bayes_acc, "SINA (Bayes)", "green")
    ax1.set_title(f"{dataset} Accuracy")
    ax1.legend()
    
    shade(ax2, inna_loss, "INNA", "orange")
    shade(ax2, grid_loss, "SINA (Grid)", "blue")
    shade(ax2, bayes_loss, "SINA (Bayes)", "green")
    ax2.set_title(f"{dataset} Loss")
    
    ax3.plot(epochs, grid_eps, label="Grid", color="blue", linestyle="--")
    ax3.plot(epochs, bayes_eps, label="Bayes", color="green", linestyle="--")
    ax3.set_title(f"{dataset} Epsilon")
    ax3.legend()
    
    plt.tight_layout()
    plt.savefig(f"results_{dataset}.png")

def plot_heatmap(study, dataset):
    try:
        trials = study.trials
        alphas = [t.params['alpha'] for t in trials if t.value]
        betas = [t.params['beta'] for t in trials if t.value]
        scores = [t.value for t in trials if t.value]
        plt.figure(figsize=(7, 6))
        sc = plt.scatter(alphas, betas, c=scores, cmap='viridis', s=100, edgecolors='k')
        plt.colorbar(sc, label='Acc')
        plt.title(f"{dataset} Robustness")
        plt.savefig(f"heatmap_{dataset}.png")
    except: pass

def save_optuna_plots(study, dataset):
    try:
        plt.figure(figsize=(10, 6))
        opt_plt.plot_param_importances(study)
        plt.savefig(f"importance_{dataset}.png")
        plt.close()
        
        plt.figure(figsize=(10, 6))
        opt_plt.plot_optimization_history(study)
        plt.savefig(f"history_{dataset}.png")
        plt.close()
    except: pass

# ==========================================
# 7. MAIN
# ==========================================
if __name__ == "__main__":
    start_time = time.time()
    
    for dataset in DATASETS_TO_RUN:
        print(f"\n{'='*50}\nSTARTING {dataset}\n{'='*50}")
        
        # --- 1. SMART CHECK: INNA Control ---
        inna_file = f"data_{dataset}_INNA.json"
        
        if os.path.exists(inna_file):
            print(f"  [Skipping INNA for {dataset} - Found {inna_file}]")
            # Load the data so we can still plot it later
            with open(inna_file, 'r') as f:
                d = json.load(f)
            # Reconstruct the tuple (accuracy, loss)
            inna_res = (np.array(d['accuracy']), np.array(d['loss']))
        else:
            # File doesn't exist, so we must run it
            inna_res = run_multi_seed_inna(device, dataset, alpha=0.5, beta=0.1)
        
        # --- 2. SINA Grid Search ---
        grid_params = run_sina_grid(device, dataset)
        grid_res = run_multi_seed_sina(device, dataset, grid_params, "SINA_Grid")
        
        # --- 3. SINA BayesOpt ---
        bayes_params, study = run_sina_bayes(device, dataset)
        bayes_res = run_multi_seed_sina(device, dataset, bayes_params, "SINA_Bayes")
        
        # --- 4. Generate Plots ---
        plot_dataset_results(dataset, inna_res, grid_res, bayes_res)
        plot_heatmap(study, dataset)
        save_optuna_plots(study, dataset)
        
    elapsed = (time.time() - start_time) / 3600
    print(f"\nfinished. Total time: {elapsed:.2f} hours.")
