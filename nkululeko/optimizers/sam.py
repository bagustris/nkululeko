"""Sharpness-Aware Minimization (SAM) optimizer wrapper.

Foret et al., "Sharpness-Aware Minimization for Efficiently Improving
Generalization" (ICLR 2021, arXiv:2010.01412). Instead of descending
toward low loss at the current weights, SAM takes an ascent step to the
worst-case point in a rho-ball around them, computes gradients there,
then descends from the *original* weights using those gradients -- this
biases training toward flat loss-landscape regions, which several
studies report closes much of the cross-domain generalization gap for
AASIST specifically on unseen audio deepfake generalization benchmarks
(Huang et al., Interspeech 2025, arXiv:2506.11532; Shim et al. 2023).

The technique itself has nothing to do with AASIST's architecture: it
only needs a base optimizer and a closure that recomputes the loss, so
it applies identically to any nkululeko neural model's train() loop --
see get_optimizer()'s MODEL.sam handling below, used by both
AasistModel.train() and ADMModel.train().

SAM wraps a base optimizer (Adam, AdamW, SGD, ...) rather than replacing
it: the ascent step is optimizer-agnostic (pure gradient-direction
perturbation), only the final descent step delegates to the wrapped
base optimizer's own update rule.
"""

import torch


class SAM(torch.optim.Optimizer):
    """Wraps `base_optimizer_cls` with SAM's two-step ascent/descent rule.

    Unlike a plain optimizer, SAM.step() needs a *closure* that zeroes
    gradients, recomputes the loss, calls loss.backward(), and returns
    the loss -- because it must be called twice per training step (once
    at the current weights to find the ascent direction, once at the
    perturbed weights to compute the actual descent gradient). See
    sam_training_step() below for the two-forward-pass call pattern this
    implies for a model's train() loop.
    """

    def __init__(
        self,
        params,
        base_optimizer_cls,
        rho: float = 0.05,
        adaptive: bool = False,
        **base_optimizer_kwargs,
    ):
        if rho < 0:
            raise ValueError(f"SAM rho must be non-negative, got {rho}")
        defaults = dict(rho=rho, adaptive=adaptive, **base_optimizer_kwargs)
        super().__init__(params, defaults)
        self.base_optimizer = base_optimizer_cls(self.param_groups, **base_optimizer_kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad: bool = False):
        """Ascend to the worst-case point in the rho-ball (perturbs
        weights in place; the pre-perturbation weights are stashed in
        `self.state[p]["old_p"]` for second_step() to restore)."""
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)
            for p in group["params"]:
                if p.grad is None:
                    continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale
                p.add_(e_w)
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad: bool = False):
        """Restore the pre-ascent weights, then let the base optimizer
        take its real update using the gradient computed at the
        perturbed point (the closure's second call must run, with its
        backward(), between first_step() and second_step())."""
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None or "old_p" not in self.state[p]:
                    continue
                p.data = self.state[p]["old_p"]
        self.base_optimizer.step()
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def step(self, closure):
        if closure is None:
            raise ValueError(
                "SAM.step() requires a closure (zero_grad + forward + "
                "loss.backward() + return loss) -- it must run twice per "
                "step, once at the current weights and once at the "
                "ascent point. Use sam_training_step() for the standard "
                "call pattern."
            )
        closure = torch.enable_grad()(closure)
        loss = closure()
        self.first_step(zero_grad=True)
        closure()
        self.second_step()
        return loss

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device
        return torch.norm(
            torch.stack(
                [
                    ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad)
                    .norm(p=2)
                    .to(shared_device)
                    for group in self.param_groups
                    for p in group["params"]
                    if p.grad is not None
                ]
            ),
            p=2,
        )

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups


def is_sam_optimizer(optimizer) -> bool:
    """True if `get_optimizer()` wrapped the base optimizer in SAM
    (MODEL.sam = True) -- a model's train() loop checks this once to
    pick between the plain zero_grad/backward/step path and a SAM
    closure calling `optimizer.step(closure)` (see AasistModel.train()
    and ADMModel.train() for the two call sites)."""
    return isinstance(optimizer, SAM)
