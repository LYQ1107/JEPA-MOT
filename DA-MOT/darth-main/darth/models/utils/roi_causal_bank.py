"""TACT-style RoI causal direction bank — extended for V1/V2/V3 PCA modes.

PCA modes (causal_pca_mode):
  "global_pc_ema"        — original: per-batch PCs, sign-aligned PC EMA (default)
  "cov_ema"              — V1: EMA covariance matrix, eigh each step
  "car_weighted_cov_ema" — V2: same but only class-filtered, optionally weighted
  "pairwise_residual"    — V3: residuals of matched key-ref pairs, cov EMA

Design invariants (all modes):
  * PCA update uses DETACHED features only. No gradient through update path.
  * trim() gradients flow through feats; pcs/cov are detached buffers.
  * alpha=0  -> trim returns feats unchanged (identity).
  * N < min_samples -> update skipped, trim skipped (return feats).
  * Any eigh/SVD/shape error -> graceful fallback, no crash.
  * Old causal_only (global_pc_ema, default) is bit-exact with previous code.
"""
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalDirectionBank(nn.Module):
    """Running bank of top principal directions of RoI embeddings.

    Args:
        feat_dim (int): Embedding dimensionality D.
        num_pcs (int): Number of principal components to store/remove.
        momentum (float): EMA momentum for PC vectors (global_pc_ema mode).
        cov_momentum (float): EMA momentum for covariance (cov_ema modes).
        min_samples (int): Minimum #features required to update/trim.
        normalize (bool): If True, L2-normalize features before PCA.
        pca_mode (str): one of the four modes above.
        qr_orthogonalize (bool): enforce orthogonality of PCs after eigh.
        log_interval (int): print debug stats every N update calls.
        debug_stats (bool): enable extended stat logging.
    """

    def __init__(self, feat_dim, num_pcs=1, momentum=0.9, cov_momentum=0.95,
                 min_samples=32, normalize=True,
                 pca_mode="global_pc_ema", qr_orthogonalize=True,
                 log_interval=50, debug_stats=False,
                 covariance_shrinkage=0.0):
        super().__init__()
        self.feat_dim = int(feat_dim)
        self.num_pcs = int(num_pcs)
        self.momentum = float(momentum)
        self.cov_momentum = float(cov_momentum)
        self.min_samples = int(min_samples)
        self.normalize = bool(normalize)
        self.pca_mode = str(pca_mode)
        self.qr_orthogonalize = bool(qr_orthogonalize)
        self.log_interval = int(log_interval)
        self.debug_stats = bool(debug_stats)
        self.covariance_shrinkage = float(covariance_shrinkage)

        D = self.feat_dim
        # shared: stored PCs [num_pcs, D]
        self.register_buffer("pcs", torch.zeros(self.num_pcs, D))
        self.register_buffer("initialized", torch.zeros(1, dtype=torch.bool))
        # cov-EMA modes: running covariance [D, D]
        self.register_buffer("cov_bank", torch.zeros(D, D))
        self.register_buffer("cov_initialized", torch.zeros(1, dtype=torch.bool))
        # diagnostics
        self.register_buffer("num_updates", torch.zeros(1, dtype=torch.long))
        self.register_buffer("last_eig1", torch.zeros(1))
        self.register_buffer("last_eig_ratio", torch.zeros(1))
        self.register_buffer("last_pc_drift", torch.zeros(1))
        self.register_buffer("last_num_samples", torch.zeros(1))
        self.register_buffer("last_update_ok", torch.ones(1, dtype=torch.bool))

    # ─────────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ─────────────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def _normalize_if_needed(self, feats):
        if self.normalize:
            return F.normalize(feats, dim=-1)
        return feats

    @torch.no_grad()
    def compute_covariance(self, feats, weights=None):
        """Compute (optionally weighted) covariance of detached feats.

        Args:
            feats   : [N, D] detached tensor
            weights : [N] positive weights or None (uniform)
        Returns:
            Tensor [D, D] or None on failure / too few samples.
        """
        if feats is None or feats.numel() == 0:
            return None
        N = feats.shape[0]
        if N < max(self.min_samples, 2):
            return None
        x = self._normalize_if_needed(feats).float()
        try:
            if weights is None:
                mu = x.mean(dim=0)
                xc = x - mu
                cov = (xc.t() @ xc) / max(N - 1, 1)
            else:
                w = weights.float().to(x.device)
                w = w / w.sum().clamp_min(1e-6)
                mu = (w.unsqueeze(1) * x).sum(dim=0)
                xc = x - mu
                cov = (xc.t() * w.unsqueeze(0)) @ xc
            if not torch.isfinite(cov).all():
                return None
            shrink = max(0.0, min(float(self.covariance_shrinkage), 1.0))
            if shrink > 0:
                diag_cov = torch.diag(torch.diagonal(cov))
                cov = (1.0 - shrink) * cov + shrink * diag_cov
            return cov.detach()
        except Exception as e:
            warnings.warn(f"[CausalDirectionBank] compute_covariance error: {e}")
            return None

    @torch.no_grad()
    def _eigh_top_pcs(self, cov):
        """Solve eigh on cov, return top num_pcs unit PCs or None."""
        try:
            evals, evecs = torch.linalg.eigh(cov)
            # eigh returns ascending eigenvalues
            D = cov.shape[0]
            idx = torch.arange(D - 1, D - 1 - self.num_pcs, -1,
                               device=cov.device)
            pcs = evecs[:, idx].t().contiguous()  # [num_pcs, D]
            if self.qr_orthogonalize and self.num_pcs > 1:
                pcs, _ = torch.linalg.qr(pcs.t())
                pcs = pcs.t().contiguous()
            pcs = F.normalize(pcs, dim=-1)
            # record diagnostics
            top_evals = evals[idx]
            total_var = evals.sum().clamp_min(1e-6)
            self.last_eig1.fill_(float(top_evals[0].item()))
            self.last_eig_ratio.fill_(float((top_evals.sum() / total_var).item()))
            return pcs
        except Exception as e:
            warnings.warn(f"[CausalDirectionBank] eigh failed: {e}; skipping.")
            return None

    @torch.no_grad()
    def _update_pcs_from_batch(self, pcs_new):
        """Sign-aligned PC EMA (original global_pc_ema logic)."""
        if not bool(self.initialized.item()):
            self.pcs.copy_(pcs_new)
            self.initialized.fill_(True)
            self.last_pc_drift.fill_(0.0)
        else:
            dots = (self.pcs * pcs_new).sum(dim=-1, keepdim=True)
            signs = torch.where(dots < 0,
                                torch.full_like(dots, -1.0),
                                torch.full_like(dots, 1.0))
            aligned = pcs_new * signs
            drift = (self.pcs - aligned).norm(dim=-1).mean().item()
            self.last_pc_drift.fill_(drift)
            updated = self.momentum * self.pcs + (1 - self.momentum) * aligned
            self.pcs.copy_(F.normalize(updated, dim=-1))

    @torch.no_grad()
    def update_from_covariance(self, cov_batch):
        """EMA-update cov_bank with cov_batch, then solve eigh to get PCs.

        Returns:
            bool: True if update succeeded.
        """
        if cov_batch is None:
            self.last_update_ok.fill_(False)
            return False
        try:
            if not bool(self.cov_initialized.item()):
                self.cov_bank.copy_(cov_batch)
                self.cov_initialized.fill_(True)
            else:
                m = self.cov_momentum
                self.cov_bank.copy_(m * self.cov_bank + (1 - m) * cov_batch)
            pcs_new = self._eigh_top_pcs(self.cov_bank)
            if pcs_new is None:
                self.last_update_ok.fill_(False)
                return False
            # sign-align against existing PCs (or init)
            self._update_pcs_from_batch(pcs_new)
            self.last_update_ok.fill_(True)
            return True
        except Exception as e:
            warnings.warn(f"[CausalDirectionBank] update_from_covariance error: {e}")
            self.last_update_ok.fill_(False)
            return False

    # ─────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def compute_pcs(self, feats):
        """Legacy: compute top PCs from feats directly (global_pc_ema mode).

        Args:
            feats (Tensor): [N, D] already detached.
        Returns:
            Tensor [num_pcs, D] or None.
        """
        N = feats.shape[0]
        if N < max(self.min_samples, 2):
            return None
        x = self._normalize_if_needed(feats)
        x = x - x.mean(dim=0, keepdim=True)
        cov = (x.t() @ x) / max(N - 1, 1)
        return self._eigh_top_pcs(cov)

    @torch.no_grad()
    def update(self, feats, domain=None, weights=None, mode_override=None):
        """Dispatch to correct update strategy based on pca_mode.

        Args:
            feats         : [N, D] detached features
            domain        : optional domain tag (for future per-domain bank)
            weights       : [N] optional sample weights (V2 weighted cov)
            mode_override : override self.pca_mode for this call
        Returns:
            bool: whether an update happened.
        """
        mode = mode_override if mode_override is not None else self.pca_mode
        if feats is None or feats.numel() == 0:
            return False
        N = feats.shape[0]
        self.last_num_samples.fill_(N)

        if N < self.min_samples:
            self.last_update_ok.fill_(False)
            return False

        self.num_updates.add_(1)

        if mode == "global_pc_ema":
            # original logic
            pcs_batch = self.compute_pcs(feats)
            if pcs_batch is None:
                self.last_update_ok.fill_(False)
                return False
            self._update_pcs_from_batch(pcs_batch)
            if not bool(self.initialized.item()):
                self.initialized.fill_(True)
            self.last_update_ok.fill_(True)
            return True

        elif mode in ("cov_ema", "car_weighted_cov_ema", "pairwise_residual",
                      "shrinkage_cov_ema", "prototype_centered_cov_ema",
                      "proto_shrinkage_cov_ema"):
            cov_batch = self.compute_covariance(feats, weights=weights)
            return self.update_from_covariance(cov_batch)

        else:
            warnings.warn(f"[CausalDirectionBank] unknown pca_mode={mode}; "
                          f"falling back to global_pc_ema.")
            pcs_batch = self.compute_pcs(feats)
            if pcs_batch is None:
                self.last_update_ok.fill_(False)
                return False
            self._update_pcs_from_batch(pcs_batch)
            self.initialized.fill_(True)
            self.last_update_ok.fill_(True)
            return True

    def trim(self, feats, domain=None, alpha=0.25, start_pc=0, num_pcs=None):
        """Remove alpha-weighted projections onto stored PCs.

        Gradients flow through feats; pcs buffer is detached.
        alpha=0 -> identity. Not initialized -> identity.
        N < min_samples -> identity.
        """
        if feats is None or feats.numel() == 0:
            return feats
        if alpha == 0:
            return feats
        if not bool(self.initialized.item()):
            return feats
        if feats.shape[0] < self.min_samples:
            return feats
        n = self.num_pcs if num_pcs is None else int(num_pcs)
        end = min(start_pc + n, self.num_pcs)
        out = feats
        for i in range(start_pc, end):
            pc = self.pcs[i].detach().to(feats.device, feats.dtype)
            coeff = (out * pc.unsqueeze(0)).sum(dim=-1, keepdim=True)
            out = out - alpha * coeff * pc.unsqueeze(0)
        return out

    @torch.no_grad()
    def pc_norm(self):
        if not bool(self.initialized.item()):
            return 0.0
        return float(self.pcs.norm(dim=-1).mean().item())

    @torch.no_grad()
    def cov_trace(self):
        if not bool(self.cov_initialized.item()):
            return 0.0
        return float(self.cov_bank.diagonal().sum().item())
