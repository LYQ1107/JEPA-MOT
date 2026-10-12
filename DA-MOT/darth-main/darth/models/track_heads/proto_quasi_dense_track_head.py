"""Prototype QuasiDense track head (independent from the causal head).

Subclasses mmtrack's `QuasiDenseTrackHead` and adds, behind a default-OFF
switch, a History-guided RoI Prototype Memory loss on the key-view RoI
association embedding. Baseline PCL is preserved unchanged.

This head does NOT touch causal logic and does NOT mix with SPA.
"""
import logging

import torch
from mmdet.models import HEADS

from mmtrack.models.track_heads import QuasiDenseTrackHead

from darth.models.utils.roi_proto_memory import RoIPrototypeMemory


@HEADS.register_module()
class ProtoQuasiDenseTrackHead(QuasiDenseTrackHead):
    """QuasiDense track head with optional RoI prototype memory.

    Args:
        use_roi_proto_memory (bool): master switch (default OFF).
        roi_proto_class_ids (list[int]): numeric class ids (e.g. [1] for car).
        roi_proto_score_thr (float): score threshold for reliable RoIs.
        roi_proto_k_per_domain_class (int): prototypes per domain/class.
        roi_proto_max_candidates_per_domain (int): candidate cap.
        roi_proto_topk (int): top-k prototypes averaged in the loss.
        roi_proto_loss_weight (float): weight of loss_roi_proto (keep small).
        roi_proto_max_rois_per_batch (int): RoI cap per loss call.
        roi_proto_diverse_select (bool): diversity selection at finalize.
        roi_proto_diverse_preselect_top (int): pre-select pool size.
        roi_proto_diverse_alpha (float): score vs diversity trade-off.
        roi_proto_log_interval (int): iters between loss logs.
    """

    def __init__(self,
                 use_roi_proto_memory=False,
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
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.use_roi_proto_memory = use_roi_proto_memory
        self.roi_proto_loss_weight = float(roi_proto_loss_weight)
        self.roi_proto_log_interval = int(roi_proto_log_interval)
        self.current_target_domain = None
        self._proto_iter = 0
        if self.use_roi_proto_memory:
            self.roi_proto_memory = RoIPrototypeMemory(
                class_ids=roi_proto_class_ids,
                score_thr=roi_proto_score_thr,
                max_candidates_per_domain=roi_proto_max_candidates_per_domain,
                k_per_domain_class=roi_proto_k_per_domain_class,
                topk=roi_proto_topk,
                max_rois_per_batch=roi_proto_max_rois_per_batch,
                diverse_select=roi_proto_diverse_select,
                diverse_preselect_top=roi_proto_diverse_preselect_top,
                diverse_alpha=roi_proto_diverse_alpha,
                feature_source=roi_proto_feature_source)
        else:
            self.roi_proto_memory = None
        if self.use_roi_proto_memory:
            lg = logging.getLogger('mmtrack')
            lg.info('[ROI_PROTO] enabled=True')
            lg.info('[ROI_PROTO] class_ids=%s', list(roi_proto_class_ids))
            lg.info('[ROI_PROTO] feature_source=%s', roi_proto_feature_source)
            lg.info('[ROI_PROTO] score_thr=%.3f (scores NOT wired to track '
                    'head; reliable RoIs = conf_thr-filtered pseudo boxes of '
                    'target class, scores=None)', roi_proto_score_thr)
            lg.info('[ROI_PROTO] K=%d topk=%d loss_weight=%.4f',
                    roi_proto_k_per_domain_class, roi_proto_topk,
                    roi_proto_loss_weight)
            lg.info('[ROI_PROTO] max_candidates_per_domain=%d '
                    'max_rois_per_batch=%d', roi_proto_max_candidates_per_domain,
                    roi_proto_max_rois_per_batch)
            lg.info('[ROI_PROTO] history_only=True stage_end_update=True')
            lg.info('[ROI_PROTO] diverse_select=%s preselect_top=%d alpha=%.2f',
                    roi_proto_diverse_select, roi_proto_diverse_preselect_top,
                    roi_proto_diverse_alpha)
            lg.info('[ROI_PROTO] local_rank_memory=True')
        else:
            logging.getLogger('mmtrack').info('[ROI_PROTO] enabled=False')

    # ------------------------------------------------------------------
    # Stage hooks (driven by the episode trainer / adapter)
    # ------------------------------------------------------------------
    def set_current_target_domain(self, domain_name):
        self.current_target_domain = domain_name
        if self.roi_proto_memory is not None:
            self.roi_proto_memory.current_domain = domain_name

    def roi_proto_start_stage(self, domain_name):
        if self.use_roi_proto_memory and self.roi_proto_memory is not None:
            self.roi_proto_memory.start_stage(domain_name)
            rank = _rank()
            logging.getLogger('mmtrack').info(
                '[ROI_PROTO][rank=%d] Start stage %s, history_domains=%s',
                rank, domain_name,
                list(self.roi_proto_memory._history.keys()))

    def roi_proto_finalize_stage(self, domain_name):
        if self.use_roi_proto_memory and self.roi_proto_memory is not None:
            self.roi_proto_memory.finalize_stage(domain_name)
            s = self.roi_proto_memory.summary()
            rank = _rank()
            protos = s['prototypes_per_domain_class'].get(domain_name, {})
            logging.getLogger('mmtrack').info(
                '[ROI_PROTO][rank=%d] Finalize domain %s: prototypes=%s '
                'mean_norm=%.4f', rank, domain_name, protos,
                s['mean_feature_norm'])

    def roi_proto_memory_summary(self):
        if self.use_roi_proto_memory and self.roi_proto_memory is not None:
            return self.roi_proto_memory.summary()
        return None

    def forward_train(self,
                      x,
                      img_metas,
                      proposal_list,
                      gt_bboxes,
                      gt_labels,
                      gt_match_indices,
                      ref_x,
                      ref_img_metas,
                      ref_proposals,
                      ref_gt_bboxes,
                      ref_gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      ref_gt_bboxes_ignore=None,
                      ref_gt_mask=None,
                      *args,
                      **kwargs):
        """Baseline QuasiDenseTrackHead.forward_train + RoI prototype memory."""
        assert self.with_track
        num_imgs = len(img_metas)
        if gt_bboxes_ignore is None:
            gt_bboxes_ignore = [None for _ in range(num_imgs)]
        if ref_gt_bboxes_ignore is None:
            ref_gt_bboxes_ignore = [None for _ in range(num_imgs)]
        key_sampling_results, ref_sampling_results = [], []
        for i in range(num_imgs):
            assign_result = self.bbox_assigner.assign(
                proposal_list[i], gt_bboxes[i], gt_bboxes_ignore[i],
                gt_labels[i])
            sampling_result = self.bbox_sampler.sample(
                assign_result, proposal_list[i], gt_bboxes[i], gt_labels[i],
                feats=[lvl_feat[i][None] for lvl_feat in x])
            key_sampling_results.append(sampling_result)

            ref_assign_result = self.bbox_assigner.assign(
                ref_proposals[i], ref_gt_bboxes[i], ref_gt_bboxes_ignore[i],
                ref_gt_labels[i])
            ref_sampling_result = self.bbox_sampler.sample(
                ref_assign_result, ref_proposals[i], ref_gt_bboxes[i],
                ref_gt_labels[i],
                feats=[lvl_feat[i][None] for lvl_feat in ref_x])
            ref_sampling_results.append(ref_sampling_result)

        key_bboxes = [res.pos_bboxes for res in key_sampling_results]
        key_feats = self.extract_bbox_feats(x, key_bboxes)
        ref_bboxes = [res.bboxes for res in ref_sampling_results]
        ref_feats = self.extract_bbox_feats(ref_x, ref_bboxes)

        # ---- original PCL (preserved, unchanged) ----
        match_feats = self.embed_head.match(
            key_feats, ref_feats, key_sampling_results, ref_sampling_results)
        asso_targets = self.embed_head.get_targets(
            gt_match_indices, key_sampling_results, ref_sampling_results)
        losses = self.embed_head.loss(*match_feats, *asso_targets)

        # ---- RoI prototype memory (default OFF) ----
        if self.use_roi_proto_memory and self.roi_proto_memory is not None:
            self._add_proto_loss(losses, key_feats, key_sampling_results,
                                 gt_labels)

        return losses

    def _pos_labels(self, key_sampling_results, gt_labels):
        """Pseudo class ids aligned with key_feats (pos RoIs) ordering."""
        pos_labels = []
        for i, res in enumerate(key_sampling_results):
            inds = res.pos_assigned_gt_inds
            if gt_labels is not None and len(gt_labels) > i:
                pos_labels.append(gt_labels[i][inds])
        if not pos_labels:
            return None
        return torch.cat(pos_labels, dim=0)

    def _add_proto_loss(self, losses, key_feats, key_sampling_results,
                        gt_labels):
        """History-guided proto loss first, then collect current candidates.

        scores=None here: teacher pseudo confidence is not wired into the
        track head, so reliable RoIs are those that survived the upstream
        conf_thr filter and the requested class filter. Logged accordingly.
        """
        if key_feats is None or key_feats.numel() == 0:
            return
        key_labels = self._pos_labels(key_sampling_results, gt_labels)
        if key_labels is None or key_labels.numel() == 0:
            return

        mem = self.roi_proto_memory
        # 1) history-guided loss (current candidates do NOT participate)
        proto_loss = mem.compute_loss(
            feats=key_feats, labels=key_labels, scores=None)
        if proto_loss is not None:
            losses['loss_roi_proto'] = self.roi_proto_loss_weight * proto_loss

        # 2) collect current-stage candidates (used only by later stages)
        mem.collect(
            domain_name=self.current_target_domain,
            feats=key_feats, labels=key_labels, scores=None)

        # periodic logging
        self._proto_iter += 1
        if (proto_loss is not None
                and self._proto_iter % self.roi_proto_log_interval == 0):
            st = mem.last_stats
            logging.getLogger('mmtrack').info(
                '[ROI_PROTO_LOSS][rank=%d] domain=%s history_domains=%s '
                'selected_rois=%s prototypes=%s mean_topk_sim=%.4f loss=%.4f '
                'weighted_loss=%.4f', _rank(), self.current_target_domain,
                list(mem._history.keys()), st.get('selected_rois'),
                st.get('prototypes'), st.get('mean_topk_sim', 0.0),
                st.get('loss', 0.0),
                self.roi_proto_loss_weight * st.get('loss', 0.0))


def _rank():
    try:
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            return dist.get_rank()
    except Exception:  # noqa: BLE001
        pass
    return 0
