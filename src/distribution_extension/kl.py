"""KL divergence between two custom distributions."""

from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar

import torch
import torch.distributions as td
from einops import repeat
from torch import Tensor

from .base import Distribution, Independent

if TYPE_CHECKING:
    from .continuous import GMM, Normal

T = TypeVar("T", Independent, Distribution)


def kl_divergence(
    q: Independent,
    p: Independent,
    use_balancing: bool,  # noqa: FBT001
) -> Tensor:
    """Calculate KL divergence between q and p."""
    if use_balancing:
        return kl_balancing(q=q, p=p)
    return td.kl_divergence(q=q, p=p).mean()


def kl_balancing(
    q: Independent,
    p: Independent,
    q_factor: float = 0.5,
    p_factor: float = 0.1,
) -> Tensor:
    """KL Balancing.

    ```
    kl_loss =       alpha * compute_kl(stop_grad(posterior), prior)
            + (1 - alpha) * compute_kl(posterior, stop_grad(prior))
    ```

    References
    ----------
    * https://arxiv.org/abs/2010.02193 [Dreamer V2]
    * https://arxiv.org/abs/2301.04104 [Dreamer V3]

    """
    dyn = td.kl_divergence(q.detach(), p).mean()
    rep = td.kl_divergence(q, p.detach()).mean()
    dyn = torch.clamp(dyn, min=1) * q_factor
    rep = torch.clamp(rep, min=1) * p_factor
    return dyn + rep


def mc_kl_divergence(
    q: Distribution,
    p: Distribution,
    num_samples: int = 100,
) -> Tensor:
    """
    Calculate KL divergence using Monte Carlo sampling.

    Parameters
    ----------
    q : Independent
        Poserior Distribution q.
    p : Independent
        Prior Distribution p.
    num_samples : int, optional
        Number of samples, by default 100.

    Returns
    -------
    Tensor
        KL divergence.

    """
    p_samples = p.rsample(torch.Size([num_samples]))
    return q.log_prob(p_samples).mean() - p.log_prob(p_samples).mean()


def gmm_loss(gmm: GMM, normal: Normal) -> Tensor:
    """Compute the gmm loss.

    Compute minus the log probability of batch under the GMM model described
    by mus, sigmas, pi. Precisely, with bs1, bs2, ... the sizes of the batch
    dimensions (several batch dimension are useful when you have both a batch
    axis and a time step axis), gs the number of mixtures and fs the number of
    features.

    loss(batch) = - mean_{*batch} log(
    sum_{k=1..gs} pi[i1, i2, ..., k] * N(
    batch[i1, i2, ..., :] | mus[i1, i2, ..., k, :], sigmas[i1, i2, ..., k, :]))

    Note
    ----
    This function is deprecated. Use `mc_kl_divergence` instead.

    References
    ----------
    * https://github.com/ctallec/world-models/blob/master/models/mdrnn.py

    Parameters
    ----------
    gmm : GMM
        mean, scale : [batch_size, seq_len, dim, num_mixture]
        probs : [batch_size, seq_len, num_mixture]

    normal : td.Normal
        mean, scale : [batch_size, seq_len, dim]

    """
    batch = repeat(normal.rsample(), "b t d -> b t d m", m=gmm.num_mixture)
    g_log_probs = td.Normal(gmm.mean, gmm.scale).log_prob(batch)
    g_log_probs = gmm.probs + torch.sum(g_log_probs, dim=-2)
    max_log_probs = torch.max(g_log_probs, dim=-2, keepdim=True)[0]
    g_log_probs -= max_log_probs

    g_probs = torch.exp(g_log_probs)
    probs = torch.sum(g_probs, dim=-2)

    log_prob = max_log_probs.squeeze() + torch.log(probs)
    return -log_prob.mean()
