"""Track-level temporal JEPA adapter.

This module deliberately operates on a fixed-length history of *past* track
observations.  A current detection is not an input to :meth:`forward`; it is
only encoded separately by the association code after the prediction has
been made.  Keeping that boundary explicit makes the online protocol easy to
audit and prevents candidate/label leakage into the predictor.
"""

from __future__ import absolute_import, division, print_function

import torch
import torch.nn as nn
import torch.nn.functional as F


# geometry (4), velocity (4), score (1), age (1), missing-gap proxy (1), and
# visibility proxy (1).  The last two are observability proxies, not GT
# occlusion labels.
TOKEN_META_DIM = 12


def _identity_or_xavier(linear):
    """Initialize a square projection close to the detector feature space."""
    with torch.no_grad():
        if linear.weight.shape[0] == linear.weight.shape[1]:
            linear.weight.zero_()
            linear.weight.copy_(torch.eye(linear.weight.shape[0]))
        else:
            nn.init.xavier_uniform_(linear.weight)
        if linear.bias is not None:
            linear.bias.zero_()


class FrozenTargetEncoder(nn.Module):
    """The stop-gradient target encoder used by offline and online JEPA.

    It starts as an identity map when the latent and detector dimensions are
    equal.  The streaming tracker may update it with an EMA of the trainable
    context projection, but gradients never flow through this module.
    """

    def __init__(self, embed_dim, latent_dim):
        super(FrozenTargetEncoder, self).__init__()
        self.projection = nn.Linear(embed_dim, latent_dim, bias=False)
        _identity_or_xavier(self.projection)
        for parameter in self.parameters():
            parameter.requires_grad = False

    def forward(self, embeds):
        return F.normalize(self.projection(embeds), p=2, dim=-1, eps=1e-6)


class TemporalJEPAAdapter(nn.Module):
    """Predict a future/reappearance latent from past track observations.

    Args:
        embed_dim (int): Detector/track-head appearance feature dimension.
        history_len (int): Number of fixed history slots.
        meta_dim (int): Number of non-appearance token features.
        hidden_dim (int): Hidden width of the student predictor.
        latent_dim (int): Output and target latent dimension.

    The model has no candidate-box argument.  ``candidate_latent`` is
    produced by :meth:`encode_target` only at association time, after the
    history prediction and its gate have been computed.
    """

    def __init__(self,
                 embed_dim=256,
                 history_len=4,
                 meta_dim=TOKEN_META_DIM,
                 hidden_dim=128,
                 latent_dim=None):
        super(TemporalJEPAAdapter, self).__init__()
        if latent_dim is None:
            latent_dim = embed_dim
        self.embed_dim = int(embed_dim)
        self.history_len = int(history_len)
        self.meta_dim = int(meta_dim)
        self.hidden_dim = int(hidden_dim)
        self.latent_dim = int(latent_dim)

        # A trainable student context projection and a stop-gradient target
        # projection are kept separate so that EMA teacher updates are
        # explicit and auditable.
        self.context_projection = nn.Linear(
            self.embed_dim, self.latent_dim, bias=False)
        _identity_or_xavier(self.context_projection)
        self.target_encoder = FrozenTargetEncoder(
            self.embed_dim, self.latent_dim)

        token_dim = self.latent_dim + self.meta_dim + 1
        self.predictor = nn.Sequential(
            nn.Linear(self.history_len * token_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU())
        self.latent_head = nn.Linear(self.hidden_dim, self.latent_dim)
        self.quality_head = nn.Linear(self.hidden_dim, 1)
        self.uncertainty_head = nn.Linear(self.hidden_dim, 1)
        self.motion_head = nn.Linear(self.hidden_dim, 4)

    def encode_context(self, embeds):
        """Encode student-side history or no-teacher targets."""
        return F.normalize(
            self.context_projection(embeds), p=2, dim=-1, eps=1e-6)

    def encode_target(self, embeds):
        """Encode a clean target/candidate without exposing it to forward."""
        return self.target_encoder(embeds)

    def forward(self, history_embeds, history_meta, history_mask):
        """Predict from past history only.

        ``history_embeds`` is ``[B, K, E]``, ``history_meta`` is
        ``[B, K, meta_dim]``, and ``history_mask`` is ``[B, K]``.  Padded
        slots are zeroed before flattening.
        """
        if history_embeds.dim() != 3:
            raise ValueError('history_embeds must have shape [B, K, E]')
        if history_meta.dim() != 3 or history_mask.dim() != 2:
            raise ValueError('history_meta/mask must have shape [B, K, ...]')
        if history_embeds.size(1) != self.history_len:
            raise ValueError('history length does not match adapter config')
        if history_meta.size(2) != self.meta_dim:
            raise ValueError('history metadata width does not match config')

        mask = history_mask.float().unsqueeze(-1)
        context = self.encode_context(history_embeds) * mask
        tokens = torch.cat([context, history_meta.float(), mask], dim=-1)
        hidden = self.predictor(tokens.reshape(tokens.size(0), -1))
        predicted = self.latent_head(hidden)
        quality = torch.sigmoid(self.quality_head(hidden)).squeeze(-1)
        # This is a positive standard-deviation-like uncertainty.  Returning
        # it directly makes the gate readable; the loss turns it into a
        # Gaussian NLL with a clamp for numerical stability.
        uncertainty = F.softplus(self.uncertainty_head(hidden)).squeeze(-1)
        uncertainty = uncertainty + 1e-4
        motion = self.motion_head(hidden)
        return dict(
            latent=predicted,
            quality=quality,
            uncertainty=uncertainty,
            motion=motion,
            hidden=hidden)

    @torch.no_grad()
    def update_teacher(self, ema=0.99):
        """EMA-update the frozen target projection from the student."""
        ema = float(ema)
        ema = min(max(ema, 0.0), 1.0)
        target = self.target_encoder.projection.weight
        source = self.context_projection.weight
        target.mul_(ema).add_(source, alpha=1.0 - ema)


def _off_diagonal_covariance_loss(predicted, target):
    """Match covariance away from the diagonal when a batch permits it."""
    if predicted.size(0) < 2:
        return predicted.new_zeros(())
    p = predicted - predicted.mean(dim=0, keepdim=True)
    t = target - target.mean(dim=0, keepdim=True)
    denom = float(max(predicted.size(0) - 1, 1))
    p_cov = p.transpose(0, 1).mm(p) / denom
    t_cov = t.transpose(0, 1).mm(t) / denom
    off_diag = ~torch.eye(
        p_cov.size(0), dtype=torch.bool, device=p_cov.device)
    return (p_cov[off_diag] - t_cov[off_diag]).pow(2).mean()


def temporal_jepa_loss(adapter,
                       batch,
                       use_teacher=True,
                       latent_weight=1.0,
                       quality_weight=0.25,
                       uncertainty_weight=0.10,
                       motion_weight=0.20,
                       covariance_weight=0.05,
                       ranking_weight=0.20,
                       ranking_margin=0.10):
    """Compute the MOT-specific JEPA objective on a prepared batch.

    The target embed in ``batch`` is a delayed observation.  In online mode
    the caller only passes samples whose delay has elapsed; in offline mode
    future/current source targets are allowed and recorded as training-only
    provenance by the cache/trainer scripts.
    """
    outputs = adapter(batch['history_embeds'], batch['history_meta'],
                      batch['history_mask'])
    predicted = F.normalize(outputs['latent'], p=2, dim=-1, eps=1e-6)
    with torch.no_grad():
        if use_teacher:
            target = adapter.encode_target(batch['target_embed'])
        else:
            target = adapter.encode_context(batch['target_embed'])
    target = target.detach()

    latent_error = (predicted - target).pow(2).mean(dim=-1)
    latent_loss = latent_error.mean()
    quality_target = batch.get(
        'quality_target',
        torch.ones_like(outputs['quality'])).float().clamp(0.0, 1.0)
    quality_loss = F.binary_cross_entropy(outputs['quality'], quality_target)

    uncertainty = outputs['uncertainty'].clamp(min=1e-3, max=10.0)
    uncertainty_loss = 0.5 * (latent_error / uncertainty.pow(2) +
                              2.0 * uncertainty.log()).mean()

    motion_target = batch.get('target_motion')
    if motion_target is None:
        motion_loss = outputs['motion'].new_zeros(())
    else:
        motion_loss = F.smooth_l1_loss(outputs['motion'], motion_target)
    covariance_loss = _off_diagonal_covariance_loss(predicted, target)

    # In-batch hard-negative ranking.  Samples are drawn from different
    # temporal positions/identities by the cache sampler in normal training;
    # the diagonal remains the positive pair and the hardest off-diagonal
    # target is used as a compact association surrogate.
    if predicted.size(0) > 1:
        pairwise = predicted.mm(target.transpose(0, 1))
        positive = pairwise.diag()
        negative_mask = ~torch.eye(
            pairwise.size(0), dtype=torch.bool, device=pairwise.device)
        hard_negative = pairwise.masked_fill(~negative_mask, -1e4).max(dim=1)[0]
        ranking_loss = F.relu(
            float(ranking_margin) - positive + hard_negative).mean()
    else:
        ranking_loss = predicted.new_zeros(())

    total = (latent_weight * latent_loss + quality_weight * quality_loss +
             uncertainty_weight * uncertainty_loss + motion_weight *
             motion_loss + covariance_weight * covariance_loss +
             ranking_weight * ranking_loss)
    return dict(
        total=total,
        latent=latent_loss,
        quality=quality_loss,
        uncertainty=uncertainty_loss,
        motion=motion_loss,
        covariance=covariance_loss,
        ranking=ranking_loss,
        predictions=outputs)
