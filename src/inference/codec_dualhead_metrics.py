from __future__ import annotations

import math
from typing import Tuple

import torch


def probs_entropy_margin(probs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute entropy and margin from probabilities.

    Args:
        probs: Tensor (B, C) row-normalized probabilities.

    Returns:
        entropy: (B,)
        margin : (B,) top1 - top2
    """
    eps = 1e-12
    p = torch.clamp(probs, min=eps, max=1.0)
    p = p / p.sum(dim=1, keepdim=True).clamp_min(eps)

    entropy = -(p * torch.log(p)).sum(dim=1)
    top2 = torch.topk(p, k=2, dim=1).values
    margin = top2[:, 0] - top2[:, 1]
    return entropy, margin


def kl_divergence(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """KL(p||q) per row.

    Args:
        p, q: (B, C) probabilities (not necessarily perfectly normalized).

    Returns:
        (B,) KL divergence.
    """
    p = torch.clamp(p, min=eps, max=1.0)
    q = torch.clamp(q, min=eps, max=1.0)
    p = p / p.sum(dim=1, keepdim=True).clamp_min(eps)
    q = q / q.sum(dim=1, keepdim=True).clamp_min(eps)
    return (p * (torch.log(p) - torch.log(q))).sum(dim=1)


def js_divergence(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Jensen–Shannon divergence per row.

    JS(p,q) = 0.5*KL(p||m) + 0.5*KL(q||m), where m = 0.5*(p+q).

    Returns:
        (B,) JS divergence in nats.
    """
    p = torch.clamp(p, min=eps, max=1.0)
    q = torch.clamp(q, min=eps, max=1.0)
    p = p / p.sum(dim=1, keepdim=True).clamp_min(eps)
    q = q / q.sum(dim=1, keepdim=True).clamp_min(eps)

    m = 0.5 * (p + q)
    return 0.5 * kl_divergence(p, m, eps=eps) + 0.5 * kl_divergence(q, m, eps=eps)


def agreement_from_js(js: torch.Tensor, *, alpha: float = 8.0) -> torch.Tensor:
    """Map JS divergence to agreement score in (0,1].

    agreement = exp(-alpha * js)

    Args:
        js: (B,) JS divergence
        alpha: positive scale

    Returns:
        (B,) agreement
    """
    a = float(alpha)
    if not math.isfinite(a) or a <= 0:
        a = 8.0
    return torch.exp(-a * js)
