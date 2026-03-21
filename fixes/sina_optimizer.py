"""
SINA -- Smoothing Inertial Newton Algorithm (Chadli et al. 2025)
"A smoothing approximation approach to dynamical inertial newton systems
 for non-smooth and non-convex optimization: the deterministic case"
Implements Algorithm 4.1 faithfully:

  Step 1: Choose alpha > 0, beta > 0, sigma in (0,1), gamma_k > 0,
          eps_0 in (0,1), delta > 0, initial (theta_0, phi_0).
  Step 2: g_k = grad_theta S(theta_k, eps_k)
  Step 3: theta_{k+1} = theta_k + gamma_k * (-a*theta_k + phi_k - beta*g_k)
           phi_{k+1}   = phi_k   - gamma_k * g_k
  Step 4: If ||grad_theta S(theta_{k+1}, eps_k)|| >= delta*eps_k:
               eps_{k+1} = eps_k
           else:
               eps_{k+1} = sigma * eps_k
The Zang smoothing P_{rho_Z}(t, eps) replaces ReLU:
  P(t, eps) = t                         if t > eps/2
            = (1/(2*eps)) * (t + eps/2)^2  if -eps/2 <= t <= eps/2
            = 0                          if t < -eps/2
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Zang Plus Smoothing Function (SINA paper Eq. 18)
# ============================================================
def zang_plus(t, eps):
    """Zang smoothing of the plus function (ReLU).

    P_{rho_Z}(t, eps):
      t                          if t > eps/2
      (1/(2*eps))*(t + eps/2)^2  if -eps/2 <= t <= eps/2
      0                          if t < -eps/2

    When eps <= 0, falls back to standard ReLU.
    """
    if eps is None or eps <= 0.0:
        return F.relu(t)
    half = 0.5 * eps
    gt = t > half
    lt = t < -half
    mid = (~gt) & (~lt)
    out = torch.zeros_like(t)
    out[gt] = t[gt]
    out[mid] = (0.5 / eps) * (t[mid] + half) ** 2
    return out


# ============================================================
# SINA Step - Explicit Euler discretization of Eq. 19
# ============================================================
def sina_step_fn(model, phi_list, alpha, beta, gamma_k):
    """One SINA parameter update (Algorithm 4.1, Step 3).

    theta_{k+1} = theta_k + gamma_k * (-alpha*theta_k + phi_k - beta*g_k)
    phi_{k+1}   = phi_k   - gamma_k * g_k

    Args:
        model: nn.Module with .parameters() having .grad populated
        phi_list: list of tensors, same shapes as model parameters
        alpha, beta: SINA hyperparameters
        gamma_k: step size (should be constant per Assumption 2)

    Note: theta_update is computed before phi is mutated, so the update
    uses the OLD phi_k as required by the explicit Euler scheme.
    """
    phi_iter = iter(phi_list)
    with torch.no_grad():
        for p in model.parameters():
            if not p.requires_grad or p.grad is None:
                continue
            g = p.grad
            ph = next(phi_iter)

            # Compute theta update using OLD phi
            theta_delta = gamma_k * (-alpha * p.data + ph - beta * g)

            # Update phi: phi_{k+1} = phi_k - gamma_k * g_k
            ph.sub_(gamma_k * g)

            # Update theta: theta_{k+1} = theta_k + delta
            p.data.add_(theta_delta)


# ============================================================
# SINA Adaptive Epsilon Schedule (Algorithm 4.1, Step 4)
# ============================================================
def sina_update_epsilon(grad_norm, eps_current, delta, sigma, eps_min=1e-8):
    """Algorithm 4.1, Step 4: adaptive epsilon reduction.

    If ||grad_theta S(theta_{k+1}, eps_k)|| >= delta * eps_k:
        eps_{k+1} = eps_k           (keep current smoothing)
    else:
        eps_{k+1} = sigma * eps_k   (reduce smoothing)

    Args:
        grad_norm: ||grad_theta S(theta_{k+1}, eps_k)||
        eps_current: current epsilon
        delta: threshold parameter (> 0)
        sigma: reduction factor (0 < sigma < 1)
        eps_min: floor to prevent numerical issues

    Returns:
        new epsilon value
    """
    if grad_norm >= delta * eps_current:
        return eps_current
    else:
        return max(eps_min, sigma * eps_current)


# ============================================================
# Gradient norm computation
# ============================================================
def compute_grad_norm(model):
    """Compute the L2 norm of all gradients concatenated."""
    total_norm_sq = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total_norm_sq += p.grad.data.norm(2).item() ** 2
    return total_norm_sq ** 0.5


# ============================================================
# Phi initialization
# ============================================================
def initialize_phi(model, gamma_k_init=None):
    """Initialize phi_0 for SINA.

    The continuous system (Eq. 19) has phi_dot = -grad S(theta, eps), so
    phi tracks the negative accumulated gradient. Two options:

    (a) phi_0 = 0  (simple, but first iteration has no phi contribution)
    (b) phi_0 = -grad S(theta_0, eps_0) (requires one forward/backward pass)

    This function creates the zero-initialized list. To use option (b),
    call this after one forward/backward pass and then run:
        for ph, p in zip(phi_list, model.parameters()):
            if p.grad is not None:
                ph.copy_(-p.grad.data)

    Returns:
        list of zero tensors matching trainable parameters
    """
    return [torch.zeros_like(p) for p in model.parameters() if p.requires_grad]


def initialize_phi_from_grad(model):
    """Initialize phi_0 = -grad S(theta_0, eps_0).

    Call AFTER one forward/backward pass with the initial epsilon.

    Returns:
        list of tensors: phi_0 = -grad for each trainable parameter
    """
    phi = []
    for p in model.parameters():
        if p.requires_grad:
            if p.grad is not None:
                phi.append(-p.grad.data.clone())
            else:
                phi.append(torch.zeros_like(p))
    return phi
