from __future__ import annotations

import random

import numpy as np
import torch
import torch.nn.functional as F


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def classification_metrics(logits: torch.Tensor, labels: torch.Tensor, bins: int = 15) -> dict[str, float]:
    probs = F.softmax(logits, dim=-1)
    conf, pred = probs.max(dim=-1)
    correct = pred.eq(labels)
    acc = correct.float().mean().item()
    nll = F.cross_entropy(logits, labels).item()

    ece = torch.zeros((), device=logits.device)
    edges = torch.linspace(0, 1, bins + 1, device=logits.device)
    for i in range(bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (conf > lo) & (conf <= hi)
        if mask.any():
            bin_acc = correct[mask].float().mean()
            bin_conf = conf[mask].mean()
            ece += mask.float().mean() * torch.abs(bin_acc - bin_conf)

    return {"accuracy": acc, "nll": nll, "ece": ece.item()}

