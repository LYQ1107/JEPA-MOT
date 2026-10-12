"""Causal QuasiDense track head — extended for V1/V2/V3 PCA modes.

New parameters (all default OFF / backward-compatible with original causal_only):
  causal_pca_mode          : "global_pc_ema" | "cov_ema" | "car_weighted_cov_ema"
                             | "pairwise_residual"
  causal_update_by_covariance : convenience bool (sets cov_ema if True and mode
                             is still default)
  causal_cov_momentum      : EMA momentum for covariance bank
  causal_qr_orthogonalize  : enforce PC orthogonality after eigh
  causal_log_interval      : print [CAUSAL_PCA_STATS] every N forward calls
  causal_debug_stats       : enable extended stats
  causal_class_ids         : tuple/list of class ids to filter for PCA update
  causal_score_thr         : confidence threshold for V2 weighted cov
  causal_use_scores        : whether to use detection scores as weights (V2)
  causal_residual_source   : "key_ref_match" (V3)
  causal_residual_normalize_before_delta : normalize before computing delta

Original parameters unchanged (same defaults, same behavior when
causal_pca_mode=="global_pc_ema").
"""
import logging
import warnings

import torch
import torch.nn.functional as F
from mmdet.models import HEADS

from mmtrack.models.track_heads import QuasiDenseTrackHead

from darth.models.utils.roi_causal_bank import CausalDirectionBank
from darth.models.utils.roi_prototype_memory import RoIPrototypeMemory as FIFORoIPrototypeMemory
from darth.models.utils.roi_proto_memory import RoIPrototypeMemory as StageRoIPrototypeMemory

logger = logging.getLogger("mmtrack")


def _rank():
    try:
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            return dist.get_rank()
    except Exception:
        pass
    return 0


@HEADS.register_module()
class CausalQuasiDenseTrackHead(QuasiDenseTrackHead):

    def __init__(self,
                 use_causal_direction_bank=False,
                 causal_num_pcs=1,
                 causal_start_pc=0,
                 causal_trim_alpha=0.25,
                 causal_loss_weight=0.1,
                 causal_min_samples=32,
                 causal_update_momentum=0.9,
                 causal_normalize_feats=True,
                 causal_bank_scope='global',
                 causal_feat_dim=256,
                 use_roi_proto_memory=False,
                 roi_memory_loss_weight=0.0,
                 roi_memory_class_weights=None,
                 roi_memory_max_per_class=100,
                 roi_memory_temperature=0.07,
                 roi_proto_memory_mode='fifo',
                 roi_proto_class_ids=(1,),
                 roi_proto_score_thr=0.70,
                 roi_proto_k_per_domain_class=100,
                 roi_proto_max_candidates_per_domain=1000,
                 roi_proto_topk=3,
                 roi_proto_loss_weight=0.01,
                 roi_proto_max_rois_per_batch=128,
                 roi_proto_diverse_select=False,
                 roi_proto_diverse_preselect_top=500,
                 roi_proto_diverse_alpha=0.5,
                 roi_proto_feature_source='student',
                 roi_proto_log_interval=50,
                 roi_proto_use_sim_gate=False,
                 roi_proto_sim_gate_thr=0.50,
                 current_domain=None,
                 causal_pca_mode="global_pc_ema",
                 causal_update_by_covariance=False,
                 causal_cov_momentum=0.95,
                 causal_qr_orthogonalize=True,
                 causal_log_interval=50,
                 causal_debug_stats=True,
                 causal_class_ids=None,
                 causal_score_thr=0.70,
                 causal_use_scores=False,
                 causal_residual_source="key_ref_match",
                 causal_residual_normalize_before_delta=True,
                 causal_covariance_shrinkage=0.0,
                 causal_proto_center_momentum=0.95,
                 causal_proto_center_min_count=4,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)

        self.use_causal_direction_bank = use_causal_direction_bank
        self.causal_num_pcs = causal_num_pcs
        self.causal_start_pc = causal_start_pc
        self.causal_trim_alpha = causal_trim_alpha
        self.causal_loss_weight = causal_loss_weight
        self.causal_min_samples = causal_min_samples
        self.causal_normalize_feats = causal_normalize_feats
        self.causal_bank_scope = causal_bank_scope
        self.causal_feat_dim = causal_feat_dim
        self.use_roi_proto_memory = use_roi_proto_memory
        self.roi_memory_loss_weight = roi_memory_loss_weight
        self.roi_memory_temperature = roi_memory_temperature
        self.roi_proto_memory_mode = str(roi_proto_memory_mode)
        self.roi_proto_loss_weight = float(roi_proto_loss_weight)
        self.roi_proto_log_interval = int(roi_proto_log_interval)
        self._proto_iter = 0
        self.current_domain = current_domain

        if causal_update_by_covariance and causal_pca_mode == "global_pc_ema":
            causal_pca_mode = "cov_ema"
        self.causal_pca_mode = str(causal_pca_mode)
        self.causal_cov_momentum = float(causal_cov_momentum)
        self.causal_qr_orthogonalize = bool(causal_qr_orthogonalize)
        self.causal_log_interval = int(causal_log_interval)
        self.causal_debug_stats = bool(causal_debug_stats)
        self.causal_class_ids = (list(causal_class_ids)
                                 if causal_class_ids is not None else None)
        self.causal_score_thr = float(causal_score_thr)
        self.causal_use_scores = bool(causal_use_scores)
        self.causal_residual_source = str(causal_residual_source)
        self.causal_residual_normalize_before_delta = bool(
            causal_residual_normalize_before_delta)
        self.causal_covariance_shrinkage = float(causal_covariance_shrinkage)
        self.causal_proto_center_momentum = float(causal_proto_center_momentum)
        self.causal_proto_center_min_count = int(causal_proto_center_min_count)

        self._causal_logged_init = False
        self._causal_forward_count = 0

        if self.use_causal_direction_bank:
            self.causal_bank = CausalDirectionBank(
                feat_dim=causal_feat_dim,
                num_pcs=causal_num_pcs,
                momentum=causal_update_momentum,
                cov_momentum=causal_cov_momentum,
                min_samples=causal_min_samples,
                normalize=causal_normalize_feats,
                pca_mode=self.causal_pca_mode,
                qr_orthogonalize=causal_qr_orthogonalize,
                log_interval=causal_log_interval,
                debug_stats=causal_debug_stats,
                covariance_shrinkage=self.causal_covariance_shrinkage)
        else:
            self.causal_bank = None

        self.register_buffer(
            "causal_proto_centers",
            torch.zeros(6, int(causal_feat_dim)))
        self.register_buffer(
            "causal_proto_center_counts",
            torch.zeros(6, dtype=torch.long))

        self.roi_memory = None
        self.roi_proto_memory = None
        if self.use_roi_proto_memory:
            if self.roi_proto_memory_mode == 'stage':
                self.roi_proto_memory = StageRoIPrototypeMemory(
                    class_ids=roi_proto_class_ids,
                    score_thr=roi_proto_score_thr,
                    max_candidates_per_domain=roi_proto_max_candidates_per_domain,
                    k_per_domain_class=roi_proto_k_per_domain_class,
                    topk=roi_proto_topk,
                    max_rois_per_batch=roi_proto_max_rois_per_batch,
                    diverse_select=roi_proto_diverse_select,
                    diverse_preselect_top=roi_proto_diverse_preselect_top,
                    diverse_alpha=roi_proto_diverse_alpha,
                    feature_source=roi_proto_feature_source,
                    use_sim_gate=roi_proto_use_sim_gate,
                    sim_gate_thr=roi_proto_sim_gate_thr)
                logger.info('[ROI_PROTO] enabled=True mode=stage history_only=True stage_end_update=True')
                logger.info('[ROI_PROTO] class_ids=%s score_thr=%.3f K=%d topk=%d loss_weight=%.4f',
                            list(roi_proto_class_ids), roi_proto_score_thr,
                            roi_proto_k_per_domain_class, roi_proto_topk,
                            roi_proto_loss_weight)
                logger.info('[ROI_PROTO] max_candidates_per_domain=%d max_rois_per_batch=%d diverse_select=%s sim_gate=%s thr=%.3f',
                            roi_proto_max_candidates_per_domain,
                            roi_proto_max_rois_per_batch,
                            roi_proto_diverse_select,
                            roi_proto_use_sim_gate,
                            roi_proto_sim_gate_thr)
            else:
                self.roi_memory = FIFORoIPrototypeMemory(
                    feat_dim=causal_feat_dim,
                    max_prototypes_per_class=roi_memory_max_per_class,
                    class_weights=roi_memory_class_weights,
                    normalize=True)
                logger.info('[ROI_PROTO] enabled=True mode=fifo loss_weight=%.4f max_per_class=%d',
                            roi_memory_loss_weight, roi_memory_max_per_class)

    def set_current_domain(self, domain):
        self.current_domain = domain
        if self.roi_proto_memory is not None:
            self.roi_proto_memory.current_domain = domain

    def set_current_target_domain(self, domain_name):
        self.set_current_domain(domain_name)

    def roi_proto_start_stage(self, domain_name):
        if (self.use_roi_proto_memory and self.roi_proto_memory is not None
                and self.roi_proto_memory_mode == 'stage'):
            self.roi_proto_memory.start_stage(domain_name)
            logger.info('[ROI_PROTO][rank=%d] Start stage %s, history_domains=%s',
                        _rank(), domain_name,
                        list(self.roi_proto_memory._history.keys()))

    def roi_proto_finalize_stage(self, domain_name):
        if (self.use_roi_proto_memory and self.roi_proto_memory is not None
                and self.roi_proto_memory_mode == 'stage'):
            self.roi_proto_memory.finalize_stage(domain_name)
            s = self.roi_proto_memory.summary()
            protos = s['prototypes_per_domain_class'].get(domain_name, {})
            logger.info('[ROI_PROTO][rank=%d] Finalize domain %s: prototypes=%s mean_norm=%.4f',
                        _rank(), domain_name, protos, s['mean_feature_norm'])

    def roi_proto_memory_summary(self):
        if self.roi_proto_memory is not None:
            return self.roi_proto_memory.summary()
        return None

    # ── logging ───────────────────────────────────────────────────────────────

    def _log_init(self):
        if self._causal_logged_init:
            return
        logger.info("[CAUSAL_PCA] enabled=True")
        logger.info(f"[CAUSAL_PCA] mode={self.causal_pca_mode}")
        logger.info(f"[CAUSAL_PCA] cov_momentum={self.causal_cov_momentum}")
        logger.info(f"[CAUSAL_PCA] qr_orthogonalize={self.causal_qr_orthogonalize}")
        logger.info(f"[CAUSAL_PCA] num_pcs={self.causal_num_pcs}")
        logger.info(f"[CAUSAL_PCA] trim_alpha={self.causal_trim_alpha}")
        logger.info(f"[CAUSAL_PCA] loss_weight={self.causal_loss_weight}")
        logger.info(f"[CAUSAL_PCA] min_samples={self.causal_min_samples}")
        logger.info(f"[CAUSAL_PCA] normalize_feats={self.causal_normalize_feats}")
        logger.info(f"[CAUSAL_PCA] class_ids={self.causal_class_ids}")
        logger.info(f"[CAUSAL_PCA] score_thr={self.causal_score_thr}")
        logger.info(f"[CAUSAL_PCA] use_scores={self.causal_use_scores}")
        logger.info("[CAUSAL_PCA] normalized_tensor_spa=False")
        logger.info(f"[CAUSAL_PCA] use_roi_proto_memory={self.use_roi_proto_memory} mode={self.roi_proto_memory_mode}")
        self._causal_logged_init = True

    def _log_stats(self, mode, num_feats, update_ok, skip_reason="",
                   num_pairs=0, num_car=0, loss_trim=None, loss_aux_trim=None):
        if not self.causal_debug_stats:
            return
        if self._causal_forward_count % self.causal_log_interval != 0:
            return
        bank = self.causal_bank
        eig1  = float(bank.last_eig1.item())      if bank else 0.0
        er    = float(bank.last_eig_ratio.item())  if bank else 0.0
        drift = float(bank.last_pc_drift.item())   if bank else 0.0
        pcn   = bank.pc_norm()  if bank else 0.0
        ctr   = bank.cov_trace() if bank else 0.0
        lt    = f"{float(loss_trim.item()):.5f}"   if loss_trim  is not None else "N/A"
        lta   = f"{float(loss_aux_trim.item()):.5f}" if loss_aux_trim is not None else "N/A"
        logger.info(
            f"[CAUSAL_PCA_STATS] step={self._causal_forward_count} "
            f"mode={mode} domain={self.current_domain} num_feats={num_feats} "
            f"update_ok={update_ok} skip_reason={skip_reason!r} "
            f"eig1={eig1:.4f} eig_ratio={er:.4f} cov_trace={ctr:.4f} "
            f"pc_norm={pcn:.4f} pc_drift={drift:.4f} "
            f"loss_track_trim={lt} loss_track_aux_trim={lta}")
        if mode == "car_weighted_cov_ema":
            logger.info(
                f"[CAUSAL_PCA_V2] key_pos_total={num_feats} "
                f"car_pos_count={num_car} "
                f"score_available={self.causal_use_scores} "
                f"weighted={self.causal_use_scores}")
        if mode == "pairwise_residual":
            logger.info(
                f"[CAUSAL_PCA_V3] matched_pairs={num_pairs} car_pairs={num_car}")

    # ── helpers ───────────────────────────────────────────────────────────────

    def _get_key_pos_labels(self, key_sampling_results, gt_labels):
        """GT labels for key positive RoIs (row-aligned with key_feats)."""
        try:
            parts = []
            for i, res in enumerate(key_sampling_results):
                if gt_labels is None or i >= len(gt_labels):
                    return None
                inds = res.pos_assigned_gt_inds
                if inds is None or len(inds) == 0:
                    continue
                parts.append(gt_labels[i][inds])
            return torch.cat(parts, dim=0) if parts else None
        except Exception as e:
            warnings.warn(f"[CausalQuasiDenseTrackHead] _get_key_pos_labels: {e}")
            return None

    @torch.no_grad()
    def _build_proto_centered_feats(self, feats, labels):
        """Build residual features by subtracting running class centers.

        Vectorised: no Python loop over RoIs.
        """
        if feats is None or feats.numel() == 0 or labels is None:
            return None, 0
        try:
            x = feats.detach()
            y = labels.detach().long().to(x.device)
            if x.shape[0] != y.shape[0]:
                return None, 0

            if self.causal_class_ids is not None:
                mask = torch.zeros(len(y), dtype=torch.bool, device=y.device)
                for cid in self.causal_class_ids:
                    mask |= (y == int(cid))
                x = x[mask]
                y = y[mask]

            if x.shape[0] < self.causal_min_samples:
                return None, int(x.shape[0])

            max_cls = self.causal_proto_centers.shape[0]
            valid_cls = (y >= 0) & (y < max_cls)
            x = x[valid_cls]
            y = y[valid_cls]
            if x.shape[0] == 0:
                return None, 0

            # update running class centers for all unique classes
            for cid_t in y.unique():
                cid = int(cid_t.item())
                cls_mask = (y == cid)
                center = F.normalize(x[cls_mask].mean(dim=0), dim=0)
                if int(self.causal_proto_center_counts[cid].item()) == 0:
                    self.causal_proto_centers[cid].copy_(center)
                else:
                    m = self.causal_proto_center_momentum
                    old = self.causal_proto_centers[cid].to(center.device)
                    self.causal_proto_centers[cid].copy_(
                        F.normalize(m * old + (1.0 - m) * center, dim=0))
                self.causal_proto_center_counts[cid] += int(cls_mask.sum().item())

            # vectorised residual: gather class centers for every RoI at once
            min_count = self.causal_proto_center_min_count
            counts = self.causal_proto_center_counts[y]          # [N]
            ready  = counts >= min_count                          # [N] bool
            if not ready.any():
                return None, 0

            x_ready = x[ready]
            y_ready = y[ready]
            centers = self.causal_proto_centers.to(x.device)[y_ready]  # [N', D]
            out = x_ready - centers
            if out.shape[0] < self.causal_min_samples:
                return None, int(out.shape[0])
            return out.detach(), out.shape[0]
        except Exception as e:
            warnings.warn(f"[CausalQuasiDenseTrackHead] _build_proto_centered_feats: {e}")
            return None, 0

    def _build_pairwise_residuals(self, key_feats, ref_feats,
                                  key_sampling_results, ref_sampling_results,
                                  gt_match_indices, gt_labels):
        """Matched key-ref residuals for V3 PCA.

        gt_match_indices[i][k] = ref_gt_idx matched to key_gt k (-1 = unmatched).
        Returns (Tensor[M,D] detached, int num_pairs).
        """
        try:
            key_sizes = [len(r.pos_bboxes) for r in key_sampling_results]
            ref_sizes = [len(r.bboxes)     for r in ref_sampling_results]
            key_splits = key_feats.detach().split(key_sizes, dim=0)
            ref_splits = ref_feats.detach().split(ref_sizes, dim=0)

            deltas, labels_out = [], []
            for i, (ks, rs) in enumerate(zip(key_splits, ref_splits)):
                if ks.shape[0] == 0 or rs.shape[0] == 0:
                    continue
                key_res  = key_sampling_results[i]
                ref_res  = ref_sampling_results[i]
                match_inds = gt_match_indices[i]       # [num_key_gt]
                key_gt_inds = key_res.pos_assigned_gt_inds   # [Nk]
                ref_gt_inds = ref_res.pos_assigned_gt_inds   # [Nr]

                # ref_gt_idx -> first ref_feat row
                ref_gt_to_row = {}
                for row_r, rgt in enumerate(ref_gt_inds.tolist()):
                    if rgt not in ref_gt_to_row:
                        ref_gt_to_row[rgt] = row_r

                for row_k, kgt in enumerate(key_gt_inds.tolist()):
                    kgt = int(kgt)
                    if kgt < 0 or kgt >= len(match_inds):
                        continue
                    ref_gt = int(match_inds[kgt])
                    if ref_gt < 0 or ref_gt not in ref_gt_to_row:
                        continue
                    kf = ks[row_k]
                    rf = rs[ref_gt_to_row[ref_gt]]
                    if self.causal_residual_normalize_before_delta:
                        kf = F.normalize(kf, dim=-1)
                        rf = F.normalize(rf, dim=-1)
                    deltas.append((kf - rf).unsqueeze(0))
                    if (self.causal_class_ids is not None
                            and gt_labels is not None
                            and i < len(gt_labels)):
                        labels_out.append(int(gt_labels[i][kgt].item()))

            if not deltas:
                return None, 0

            residuals = torch.cat(deltas, dim=0)  # [M, D]

            if self.causal_class_ids is not None and len(labels_out) == len(deltas):
                lbl_t = torch.tensor(labels_out, device=residuals.device)
                mask  = torch.zeros(len(lbl_t), dtype=torch.bool,
                                    device=residuals.device)
                for cid in self.causal_class_ids:
                    mask |= (lbl_t == cid)
                residuals = residuals[mask]

            num_pairs = residuals.shape[0]
            if num_pairs < self.causal_min_samples:
                return None, num_pairs
            return residuals.detach(), num_pairs
        except Exception as e:
            warnings.warn(f"[CausalQuasiDenseTrackHead] _build_pairwise_residuals: {e}")
            return None, 0

    # ── forward_train ─────────────────────────────────────────────────────────

    def forward_train(self,
                      x, img_metas, proposal_list,
                      gt_bboxes, gt_labels, gt_match_indices,
                      ref_x, ref_img_metas, ref_proposals,
                      ref_gt_bboxes, ref_gt_labels,
                      gt_bboxes_ignore=None, gt_masks=None,
                      ref_gt_bboxes_ignore=None, ref_gt_mask=None,
                      gt_scores=None, ref_gt_scores=None,
                      *args, **kwargs):
        assert self.with_track
        num_imgs = len(img_metas)
        if gt_bboxes_ignore     is None: gt_bboxes_ignore     = [None] * num_imgs
        if ref_gt_bboxes_ignore is None: ref_gt_bboxes_ignore = [None] * num_imgs

        key_sampling_results, ref_sampling_results = [], []
        for i in range(num_imgs):
            ar = self.bbox_assigner.assign(
                proposal_list[i], gt_bboxes[i], gt_bboxes_ignore[i], gt_labels[i])
            sr = self.bbox_sampler.sample(
                ar, proposal_list[i], gt_bboxes[i], gt_labels[i],
                feats=[lvl[i][None] for lvl in x])
            key_sampling_results.append(sr)

            ref_ar = self.bbox_assigner.assign(
                ref_proposals[i], ref_gt_bboxes[i],
                ref_gt_bboxes_ignore[i], ref_gt_labels[i])
            ref_sr = self.bbox_sampler.sample(
                ref_ar, ref_proposals[i], ref_gt_bboxes[i], ref_gt_labels[i],
                feats=[lvl[i][None] for lvl in ref_x])
            ref_sampling_results.append(ref_sr)

        key_feats = self.extract_bbox_feats(
            x,     [r.pos_bboxes for r in key_sampling_results])
        ref_feats = self.extract_bbox_feats(
            ref_x, [r.bboxes     for r in ref_sampling_results])

        # ── original PCL (always preserved) ──────────────────────────────────
        match_feats  = self.embed_head.match(
            key_feats, ref_feats, key_sampling_results, ref_sampling_results)
        asso_targets = self.embed_head.get_targets(
            gt_match_indices, key_sampling_results, ref_sampling_results)
        losses = self.embed_head.loss(*match_feats, *asso_targets)

        # ── trimmed PCL (additional, gated) ──────────────────────────────────
        self._causal_forward_count += 1
        if (self.use_causal_direction_bank and self.causal_bank is not None
                and key_feats is not None and ref_feats is not None
                and key_feats.numel() > 0 and ref_feats.numel() > 0):
            if not self._causal_logged_init:
                self._log_init()
            self._add_trimmed_pcl(
                losses, key_feats, ref_feats,
                key_sampling_results, ref_sampling_results,
                asso_targets, gt_labels, gt_match_indices)

        # ── RoI prototype memory (default OFF) ───────────────────────────────
        if self.use_roi_proto_memory:
            if self.roi_proto_memory is not None:
                self._add_stage_proto_loss(losses, key_feats,
                                           key_sampling_results, gt_labels,
                                           gt_scores=gt_scores)
            elif self.roi_memory is not None:
                self._add_memory_loss(losses, key_feats, key_sampling_results,
                                      gt_labels)
        return losses

    # ── _add_trimmed_pcl ──────────────────────────────────────────────────────

    def _add_trimmed_pcl(self, losses, key_feats, ref_feats,
                         key_sampling_results, ref_sampling_results,
                         asso_targets, gt_labels, gt_match_indices):
        try:
            mode = self.causal_pca_mode
            update_ok, skip_reason = False, ""
            num_feats = key_feats.shape[0] + ref_feats.shape[0]
            num_pairs = num_car = 0

            if mode == "global_pc_ema":
                pca_feats = torch.cat(
                    [key_feats.detach(), ref_feats.detach()], dim=0)
                update_ok = self.causal_bank.update(pca_feats,
                                                    domain=self.current_domain)

            elif mode in ("cov_ema", "shrinkage_cov_ema"):
                pca_feats = torch.cat(
                    [key_feats.detach(), ref_feats.detach()], dim=0)
                update_ok = self.causal_bank.update(pca_feats,
                                                    domain=self.current_domain)

            elif mode in ("prototype_centered_cov_ema", "proto_shrinkage_cov_ema"):
                key_labels = self._get_key_pos_labels(
                    key_sampling_results, gt_labels)
                num_feats = key_feats.shape[0]
                pca_feats, num_proto = self._build_proto_centered_feats(
                    key_feats, key_labels)
                num_car = num_proto
                if pca_feats is None:
                    skip_reason = f"insufficient_proto_centered({num_proto})"
                else:
                    update_ok = self.causal_bank.update(
                        pca_feats, domain=self.current_domain)

            elif mode == "car_weighted_cov_ema":
                key_labels = self._get_key_pos_labels(
                    key_sampling_results, gt_labels)
                num_feats = key_feats.shape[0]
                if key_labels is None:
                    skip_reason = "key_labels_unavailable"
                    pca_feats   = None
                else:
                    if self.causal_class_ids is not None:
                        mask = torch.zeros(len(key_labels), dtype=torch.bool,
                                           device=key_labels.device)
                        for cid in self.causal_class_ids:
                            mask |= (key_labels == cid)
                        pca_feats = key_feats.detach()[mask]
                        num_car   = int(mask.sum().item())
                    else:
                        pca_feats = key_feats.detach()
                        num_car   = num_feats
                    if self.causal_use_scores:
                        logger.info("[CAUSAL_PCA_V2] scores unavailable; "
                                    "using car-only unweighted cov")
                    update_ok = self.causal_bank.update(
                        pca_feats, domain=self.current_domain, weights=None)

            elif mode == "pairwise_residual":
                residuals, num_pairs = self._build_pairwise_residuals(
                    key_feats, ref_feats,
                    key_sampling_results, ref_sampling_results,
                    gt_match_indices, gt_labels)
                num_car = num_pairs
                if residuals is None:
                    skip_reason = f"insufficient_pairs({num_pairs})"
                    pca_feats   = None
                else:
                    pca_feats = residuals
                    update_ok = self.causal_bank.update(
                        pca_feats, domain=self.current_domain)
            else:
                warnings.warn(f"[CausalQuasiDenseTrackHead] unknown "
                              f"causal_pca_mode={mode}; skipping trimmed PCL.")
                return

            key_trim = self.causal_bank.trim(
                key_feats, domain=self.current_domain,
                alpha=self.causal_trim_alpha,
                start_pc=self.causal_start_pc, num_pcs=self.causal_num_pcs)
            ref_trim = self.causal_bank.trim(
                ref_feats, domain=self.current_domain,
                alpha=self.causal_trim_alpha,
                start_pc=self.causal_start_pc, num_pcs=self.causal_num_pcs)

            trim_match  = self.embed_head.match(
                key_trim, ref_trim, key_sampling_results, ref_sampling_results)
            trim_losses = self.embed_head.loss(*trim_match, *asso_targets)

            lt  = trim_losses.get('loss_track',     None)
            lta = trim_losses.get('loss_track_aux', None)
            if lt  is not None and torch.isfinite(lt).all():
                losses['loss_track_trim']     = self.causal_loss_weight * lt
            if lta is not None and torch.isfinite(lta).all():
                losses['loss_track_aux_trim'] = self.causal_loss_weight * lta

            self._log_stats(mode, num_feats, update_ok, skip_reason,
                            num_pairs, num_car,
                            losses.get('loss_track_trim'),
                            losses.get('loss_track_aux_trim'))

        except Exception as e:
            warnings.warn(f"[CausalQuasiDenseTrackHead] trimmed PCL failed "
                          f"({e}); skipping this step.")

    def _add_stage_proto_loss(self, losses, key_feats, key_sampling_results,
                              gt_labels, gt_scores=None):
        try:
            if key_feats is None or key_feats.numel() == 0:
                return
            pos_labels = []
            pos_scores = []
            for i, res in enumerate(key_sampling_results):
                if gt_labels is not None and i < len(gt_labels):
                    inds = res.pos_assigned_gt_inds
                    pos_labels.append(gt_labels[i][inds])
                    if gt_scores is not None and i < len(gt_scores):
                        pos_scores.append(gt_scores[i][inds])
            if not pos_labels:
                return
            key_labels = torch.cat(pos_labels, dim=0)
            key_scores = torch.cat(pos_scores, dim=0) if pos_scores else None
            scores_available = key_scores is not None and key_scores.numel() == key_labels.numel()
            mem = self.roi_proto_memory

            proto_loss = mem.compute_loss(
                feats=key_feats, labels=key_labels,
                scores=key_scores if scores_available else None)
            if proto_loss is not None:
                losses['loss_roi_proto'] = self.roi_proto_loss_weight * proto_loss

            mem.collect(
                domain_name=self.current_domain,
                feats=key_feats, labels=key_labels,
                scores=key_scores if scores_available else None)

            self._proto_iter += 1
            if (proto_loss is not None
                    and self._proto_iter % self.roi_proto_log_interval == 0):
                st = mem.last_stats
                logger.info(
                    '[ROI_PROTO_LOSS][rank=%d] scores_available=%s domain=%s '
                    'history_domains=%s prototypes_per_domain_class=%s '
                    'current_candidates=%s selected_rois=%s prototypes=%s '
                    'mean_topk_sim=%.4f loss=%.4f weighted_loss=%.4f '
                    'gate_enabled=%s sim_gate_thr=%s gate_selected_rois=%s '
                    'gate_total_rois=%s gate_ratio=%s',
                    _rank(), scores_available, self.current_domain,
                    list(mem._history.keys()),
                    mem.summary().get('prototypes_per_domain_class'),
                    mem.summary().get('current_candidates'),
                    st.get('selected_rois'), st.get('prototypes'),
                    st.get('mean_topk_sim', 0.0), st.get('loss', 0.0),
                    self.roi_proto_loss_weight * st.get('loss', 0.0),
                    st.get('gate_enabled', False), st.get('sim_gate_thr'),
                    st.get('gate_selected_rois'), st.get('gate_total_rois'),
                    st.get('gate_ratio'))
        except Exception as e:
            warnings.warn(f"[CausalQuasiDenseTrackHead] stage roi proto loss "
                          f"failed ({e}); skipping.")

    def _add_memory_loss(self, losses, key_feats, key_sampling_results,
                         gt_labels):
        try:
            pos_labels = []
            for i, res in enumerate(key_sampling_results):
                if gt_labels is not None and i < len(gt_labels):
                    pos_labels.append(gt_labels[i][res.pos_assigned_gt_inds])
            if not pos_labels:
                losses['loss_roi_memory'] = key_feats.sum() * 0.0
                return
            key_labels = torch.cat(pos_labels, dim=0)
            loss_mem = self.roi_memory.memory_contrastive_loss(
                key_feats, key_labels, domain=self.current_domain,
                temperature=self.roi_memory_temperature)
            losses['loss_roi_memory'] = self.roi_memory_loss_weight * loss_mem
            with torch.no_grad():
                self.roi_memory.update(key_feats.detach(), key_labels,
                                       domain=self.current_domain)
        except Exception as e:
            warnings.warn(f"[CausalQuasiDenseTrackHead] memory loss failed "
                          f"({e}); skipping.")
