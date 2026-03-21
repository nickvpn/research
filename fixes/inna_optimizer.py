"""
Fixed INNA Optimizer - matches Castera et al. (2021) "An Inertial Newton Algorithm
for Deep Learning", Table 1 / Eq. 12.

Fixes vs. original inna.py:
  1. Deprecated two-arg .sub_(scalar, tensor) replaced with .sub_(scalar * tensor)
  2. psi_0 initialization includes the gradient term from Section 5.2.1 p.22:
       psi_0 = (1 - alpha*beta)*theta_0 - (beta^2 - beta)*grad(theta_0)
     The gradient term is incorporated on the first .step() call when p.grad is available.
  3. weight_decay branch uses the same modern API.

Update equations (INNA paper Eq. 12, Table 1):
  v_k   in  D J_{B_k}(theta_k)              [mini-batch subgradient]
  theta_{k+1} = theta_k + gamma_k * ((1/beta - alpha)*theta_k - (1/beta)*psi_k - beta*v_k)
  psi_{k+1}   = psi_k   + gamma_k * ((1/beta - alpha)*theta_k - (1/beta)*psi_k)

  psi_0 = (1 - alpha*beta)*theta_0 - (beta^2 - beta)*grad_J(theta_0)
  gamma_k = gamma_0 / sqrt(k+1)   [per-iteration, Assumption 1]
"""

import torch
from torch.optim.optimizer import Optimizer


class INNA(Optimizer):
    """Inertial Newton Algorithm for Deep Learning (Castera et al. 2021).

    Args:
        params: iterable of parameters to optimize
        lr (float): base step-size gamma_0 (default: 0.1)
        alpha (float): viscous damping coefficient (default: 0.5)
        beta (float): Newton damping coefficient (default: 0.1)
        decaypower (float): exponent q in gamma_k = gamma_0/(k+1)^q.
            Set to 0 if using an external PyTorch scheduler. (default: 0.0)
        weight_decay (float): L2 penalty coefficient (default: 0.0)
    """

    def __init__(self, params, lr=0.1, alpha=0.5, beta=0.1,
                 decaypower=0., weight_decay=0.):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if alpha <= 0.0:
            raise ValueError(f"Invalid alpha: {alpha}")
        if beta <= 0.0:
            raise ValueError(f"Invalid beta: {beta}")

        if decaypower > 0:
            print('Warning: Do not combine the decaypower parameter with a '
                  'PyTorch scheduler — they will compound.')

        defaults = dict(lr=lr, alpha=alpha, beta=beta,
                        decaypower=decaypower, weight_decay=weight_decay)
        super(INNA, self).__init__(params, defaults)

        # Pre-initialize step counters (psi is lazily initialized in step()
        # because we need the gradient for the full formula).
        for group in self.param_groups:
            for p in group['params']:
                self.state[p]['step'] = 0

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            alpha = group['alpha']
            beta = group['beta']
            lr = group['lr']
            dc = group['decaypower']
            weight_decay = group['weight_decay']

            for p in group['params']:
                if p.grad is None:
                    continue

                d_p = p.grad

                param_state = self.state[p]

                # ---- Lazy psi initialization (first call to step) ----
                # INNA paper Section 5.2.1 p.22:
                #   psi_0 = (1 - alpha*beta)*theta_0 - (beta^2 - beta)*grad_J(theta_0)
                # This sets the initial velocity in the -grad direction.
                if 'psi' not in param_state:
                    param_state['psi'] = (
                        (1. - alpha * beta) * p.data.clone()
                        - (beta ** 2 - beta) * d_p.clone()
                    )

                psi = param_state['psi']

                # ---- Prepare the common term ----
                # common = (1/beta - alpha)*theta - (1/beta)*psi
                #        = -[(alpha - 1/beta)*theta + (1/beta)*psi]
                # We compute it as in the paper's positive form:
                phase_update = (alpha - 1. / beta) * p.data + (1. / beta) * psi
                geom_update = beta * d_p

                # ---- Step-size with optional power decay ----
                if dc > 0:
                    lr_t = lr / ((1 + param_state['step']) ** dc)
                else:
                    lr_t = lr

                # ---- Weight decay (decoupled, added to gradient direction) ----
                if weight_decay > 0:
                    wd_term = weight_decay * p.data
                else:
                    wd_term = None

                # ---- Update psi ----
                # psi_{k+1} = psi_k - lr_t * phase_update
                #           = psi_k + lr_t * ((1/beta - alpha)*theta - (1/beta)*psi)
                psi.sub_(lr_t * phase_update)

                # ---- Update theta ----
                # theta_{k+1} = theta_k - lr_t * (phase_update + geom_update)
                #             = theta_k + lr_t * ((1/beta - alpha)*theta - (1/beta)*psi - beta*grad)
                if wd_term is not None:
                    p.data.sub_(lr_t * (phase_update + geom_update + wd_term))
                else:
                    p.data.sub_(lr_t * (phase_update + geom_update))

                param_state['step'] += 1

        return loss
