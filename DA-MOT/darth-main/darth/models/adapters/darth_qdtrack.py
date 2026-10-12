import torch
import torchvision
from copy import deepcopy

from mmcv.parallel import collate
from mmdet.datasets.pipelines import Compose
from mmtrack.models.builder import MODELS

from darth.models.utils.transforms import (
    detections2bboxes,
    filter_bboxes_by_confidence,
    bboxes_to_tensor,
)

from .kd_qdtrack import KDQDTrack


def to_results(
    img, bboxes, labels, img_metas, bboxes_ignore=None, instance_ids=None,
    scores=None
):
    if instance_ids is None:
        instance_ids = [
            torch.arange(1, len(label)+1).to(label.device) for label in labels]
    if scores is None:
        scores = [torch.ones_like(label, dtype=torch.float32) for label in labels]
    results = []
    for i, (im, box, label, id, score) in enumerate(zip(
        img, bboxes, labels, instance_ids, scores)):
        im = im.cpu().detach().numpy().transpose(1, 2, 0)
        result = dict(
            img=im,
            img_shape=tuple(im.shape),
            img_info={},
            gt_bboxes=box.cpu().detach().numpy(),
            gt_bboxes_ignore=(
                bboxes_ignore[i] if bboxes_ignore else None),
            gt_labels=label.cpu().detach().numpy(),
            gt_scores=score.cpu().detach().numpy(),
            gt_instance_ids=id,
            img_metas=img_metas[i],
            img_fields=['img'],
            bbox_fields=(
                ['gt_bboxes', 'gt_bboxes_ignore'] if bboxes_ignore
                else ['gt_bboxes']),
            label_fields=['gt_labels'],
        )
        results.append([result])
    return results


def concatenate_results(results, references):
    for result, reference in zip(results, references):
        result.extend(reference)
    return results


def apply_pipeline(results, pipeline):
    outs = []
    for result in results:
        out = pipeline(result)
        outs.append(out)
    return outs


def format_results(results, samples_per_gpu=1, ref_prefix='c'):
    from mmtrack.datasets.pipelines import SeqDefaultFormatBundle
    from mmcv.parallel import DataContainer as DC

    class _Bundle(SeqDefaultFormatBundle):
        def default_format_bundle(self, results):
            results = super().default_format_bundle(results)
            if 'gt_scores' in results and not isinstance(
                    results['gt_scores'], DC):
                results['gt_scores'] = DC(torch.as_tensor(results['gt_scores']))
            return results

    formatter_cls = _Bundle if any(
        'gt_scores' in item for result in results for item in result
    ) else SeqDefaultFormatBundle
    formatter = formatter_cls(ref_prefix=ref_prefix)
    outs = []
    for result in results:
        out = formatter(result)
        outs.append(out)
    outs = collate(outs, samples_per_gpu=samples_per_gpu)
    return outs


def parse_results(results, device, keys=[]):
    fields_to_return = []
    for key in keys:
        if key in results:
            data = results[key].data[0]
            if 'img_metas' in key:
                pass
            elif isinstance(data, list):
                data = [d.to(device) for d in data]
            else:
                data = data.to(device)
            fields_to_return.append(data)
        else:
            fields_to_return.append(None)
    return fields_to_return


def drop_empty_images(
    img, bboxes, labels, scores, match_indices, img_metas, bboxes_ignore,
    ref_img, ref_bboxes, ref_labels, ref_scores, ref_match_indices,
    ref_img_metas, ref_bboxes_ignore,
):
    """Drop images without bboxes."""
    keep_img = []
    keep_bboxes = []
    keep_labels = []
    keep_scores = [] if scores is not None else None
    keep_match_indices = []
    keep_img_metas = []
    keep_bboxes_ignore = [] if bboxes_ignore is not None else None
    ref_keep_img = []
    ref_keep_bboxes = []
    ref_keep_labels = []
    ref_keep_scores = [] if ref_scores is not None else None
    ref_keep_match_indices = []
    ref_keep_img_metas = []
    ref_keep_bboxes_ignore = [] if ref_bboxes_ignore is not None else None
    for i, bbox in enumerate(bboxes):
        if len(bbox) > 0:
            keep_img.append(img[i])
            keep_bboxes.append(bboxes[i])
            keep_labels.append(labels[i])
            if keep_scores is not None:
                keep_scores.append(scores[i])
            keep_match_indices.append(match_indices[i])
            keep_img_metas.append(img_metas[i])
            if bboxes_ignore is not None:
                keep_bboxes_ignore.append(bboxes_ignore[i])
            ref_keep_img.append(ref_img[i])
            ref_keep_bboxes.append(ref_bboxes[i])
            ref_keep_labels.append(ref_labels[i])
            if ref_keep_scores is not None:
                ref_keep_scores.append(ref_scores[i])
            ref_keep_match_indices.append(ref_match_indices[i])
            ref_keep_img_metas.append(ref_img_metas[i])
            if ref_bboxes_ignore is not None:
                ref_keep_bboxes_ignore.append(ref_bboxes_ignore[i])
    
    keep_img = torch.stack(keep_img) if len(keep_img) > 0 else None
    ref_keep_img = torch.stack(ref_keep_img) if len(ref_keep_img) > 0 else None
    return (
        keep_img, keep_bboxes, keep_labels, keep_scores, keep_match_indices,
        keep_img_metas, keep_bboxes_ignore,
        ref_keep_img, ref_keep_bboxes, ref_keep_labels, ref_keep_scores,
        ref_keep_match_indices, ref_keep_img_metas, ref_keep_bboxes_ignore,
    )


@MODELS.register_module()
class DARTHQDTrack(KDQDTrack):
    """Implementation of `DARTH <https://arxiv.org/abs/2310.01926>`_, a domain adaptive MOT framework based on `QDTrack
    <https://arxiv.org/abs/2006.06664>`_.
    Args:
        transforms list(dict): Configuration of transforms for data augmentation
            of keyframe into reference frame.
    """

    def __init__(self,
                 teacher_pipeline,
                 student_pipeline,
                 contrastive_pipeline,
                 out_pipeline,
                 conf_thr=0.0,
                 use_spa_frequency_aug=False,
                 spa_apply_to_student=True,
                 spa_apply_to_contrastive=False,
                 spa_low_ratio=0.2,
                 spa_keep_prob=0.8,
                 spa_noise_ratio=0.0,
                 spa_apply_noise=False,
                 use_spa_conf_gate=False,
                 spa_conf_thr=0.70,
                 spa_detection_only=False,
                 spa_apply_to_detection=True,
                 spa_apply_to_tracking=False,
                 log_view_stats=False,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)

        # build augmentation pipeline
        self.teacher_pipeline = Compose(teacher_pipeline)
        self.student_pipeline = Compose(student_pipeline)
        self.contrastive_pipeline = Compose(contrastive_pipeline)
        self.out_pipeline = Compose(out_pipeline)
        
        # confidence threshold for teacher bboxes selection
        self.conf_thr = conf_thr

        # SPA geometry-preserving frequency augmentation (default OFF).
        # Port of SPA (ICML 2025) Prompt.masking: operates on already-normalized
        # GPU tensors inside forward_train using torch.fft — same as the paper.
        # Teacher view is never augmented to keep pseudo-labels stable.
        self.use_spa_frequency_aug = use_spa_frequency_aug
        self.spa_apply_to_student = spa_apply_to_student
        self.spa_apply_to_contrastive = spa_apply_to_contrastive
        # Scheme 3: SPA-specific higher pseudo-label confidence gate (default OFF).
        self.use_spa_conf_gate = use_spa_conf_gate
        self.spa_conf_thr = float(spa_conf_thr)
        # Scheme 4: detection/tracking decoupling (default OFF).
        self.spa_detection_only = spa_detection_only
        self.spa_apply_to_detection = spa_apply_to_detection
        self.spa_apply_to_tracking = spa_apply_to_tracking
        self._spa_conf_gate_logged = False
        self._spa_det_only_logged = False
        self.log_view_stats = log_view_stats
        self._view_stats_logged = False
        if self.use_spa_frequency_aug:
            from darth.models.utils.spa_frequency import SPAFrequencyAugmentor
            self.spa_augmentor = SPAFrequencyAugmentor(
                low_ratio=spa_low_ratio,
                keep_prob=spa_keep_prob,
                noise_ratio=spa_noise_ratio,
                apply_noise=spa_apply_noise)
        else:
            self.spa_augmentor = None

    # ------------------------------------------------------------------
    # RoI prototype memory delegation (forwarded to the track head).
    # No-ops unless the configured track head implements them and the
    # memory is enabled. Never touches causal / SPA.
    # ------------------------------------------------------------------
    def set_current_target_domain(self, domain_name):
        if hasattr(self.track_head, 'set_current_target_domain'):
            self.track_head.set_current_target_domain(domain_name)

    def roi_proto_start_stage(self, domain_name):
        if hasattr(self.track_head, 'roi_proto_start_stage'):
            self.track_head.roi_proto_start_stage(domain_name)

    def roi_proto_finalize_stage(self, domain_name):
        if hasattr(self.track_head, 'roi_proto_finalize_stage'):
            self.track_head.roi_proto_finalize_stage(domain_name)

    def roi_proto_memory_summary(self):
        if hasattr(self.track_head, 'roi_proto_memory_summary'):
            return self.track_head.roi_proto_memory_summary()
        return None

    def _prepare_views(self,
                       img,
                       img_metas,
                       gt_bboxes,
                       gt_labels,
                       gt_bboxes_ignore=None,
                       gt_masks=None):
        """Prepare teacher, student and contrastive views for the self-supervised tracking framework."""
        keys = [
            'img', 'gt_bboxes', 'gt_labels', 'gt_scores', 'img_metas',
            'gt_instance_ids', 'gt_bboxes_ignore']

        # process teacher
        t_results = to_results(
            img,
            gt_bboxes,
            gt_labels,
            img_metas,
            bboxes_ignore=gt_bboxes_ignore,
            instance_ids=None)
        t_t_processed = concatenate_results(deepcopy(t_results), deepcopy(t_results))
        t_t_processed = apply_pipeline(t_t_processed, self.out_pipeline)
        t_t_results = format_results(
            t_t_processed, samples_per_gpu=img.shape[0], ref_prefix='t'
        )
        (
            gt_img, gt_bboxes, gt_labels, gt_scores, img_metas,
            gt_instance_ids, gt_bboxes_ignore
        ) = parse_results(t_t_results, img.device, keys=keys)

        # teacher forward pass to generate pseudo labels
        t_x = self.teacher_detector.extract_feat(gt_img)
        t_proposal_list = self.teacher_detector.rpn_head.simple_test_rpn(
            t_x, img_metas)
        t_det_results = self.teacher_detector.roi_head.simple_test(
            t_x, t_proposal_list, img_metas, rescale=False)

        t_bboxes, t_labels, t_confidences = detections2bboxes(
            t_det_results)
        # Scheme 3: SPA-specific reliability gate. When enabled, use a higher
        # pseudo-label confidence threshold so SPA-perturbed student does not
        # over-learn low-confidence teacher boxes. Default OFF (baseline intact).
        effective_conf_thr = self.conf_thr
        if self.use_spa_frequency_aug and self.use_spa_conf_gate:
            effective_conf_thr = max(float(self.conf_thr), float(self.spa_conf_thr))
            if not self._spa_conf_gate_logged:
                import logging
                logging.getLogger('mmtrack').info(
                    '[SPA_CONF_GATE] enabled=True conf_thr=%.3f spa_conf_thr=%.3f '
                    'effective_conf_thr=%.3f', float(self.conf_thr),
                    float(self.spa_conf_thr), float(effective_conf_thr))
                self._spa_conf_gate_logged = True
        t_bboxes, t_labels, t_confidences = filter_bboxes_by_confidence(
            t_bboxes, t_labels, t_confidences, conf_thr=effective_conf_thr)
        t_bboxes, t_labels, t_confidences = bboxes_to_tensor(
            t_bboxes, t_labels, t_confidences, img.device)
        
        # augment teacher view for harder distillation
        t_results = to_results(
            img,
            t_bboxes,
            t_labels,
            img_metas,
            bboxes_ignore=gt_bboxes_ignore,
            instance_ids=None,
            scores=t_confidences)
        t_processed = apply_pipeline(deepcopy(t_results), self.teacher_pipeline)
        t_t_processed = concatenate_results(deepcopy(t_processed), deepcopy(t_processed))
        t_t_processed = apply_pipeline(t_t_processed, self.out_pipeline)
        t_t_results = format_results(
            t_t_processed, samples_per_gpu=img.shape[0], ref_prefix='t'
        )
        (
            t_img, t_bboxes, t_labels, t_scores, t_img_metas,
            t_instance_ids, t_bboxes_ignore
        ) = parse_results(t_t_results, img.device, keys=keys)

        # cleaning conflicting fields (if any)
        s_input = deepcopy(t_processed)
        [t[0].pop('scale', 0) for t in s_input]
        [t[0].pop('scale_factor', 0) for t in s_input]
        # augment teacher view to student view
        s_processed = apply_pipeline(s_input, self.student_pipeline)
        # cleaning conflicting fields (if any)
        # start from the view before teacher augmentation to get an independent view
        c_input = deepcopy(t_results)  
        [t[0].pop('scale', 0) for t in c_input]
        [t[0].pop('scale_factor', 0) for t in c_input]
        # augment student view to contrastive view
        c_processed = apply_pipeline(c_input, self.contrastive_pipeline)
        # c_processed = apply_pipeline(s_processed, self.contrastive_pipeline)
        s_c_processed = concatenate_results(deepcopy(s_processed), c_processed)
        s_c_processed = apply_pipeline(s_c_processed, self.out_pipeline)
        s_c_results = format_results(
            s_c_processed, samples_per_gpu=img.shape[0], ref_prefix='c'
        )
        (
            s_img, s_bboxes, s_labels, s_scores, s_img_metas,
            s_instance_ids, s_bboxes_ignore
        ) = parse_results(s_c_results, img.device, keys=keys)
        (
            c_img, c_bboxes, c_labels, c_scores, c_img_metas,
            c_instance_ids, c_bboxes_ignore
        ) = parse_results(s_c_results, img.device, keys=['c_' + key for key in keys])
        s_match_indices, c_match_indices = parse_results(
            s_c_results, img.device, keys=['gt_match_indices', 'c_gt_match_indices'])

        # pad img to same size as s_img for correct distillation
        import torch.nn.functional as F
        t_img = F.pad(
            t_img, pad=(
                0, s_img.shape[3] - t_img.shape[3],
                0, s_img.shape[2] - t_img.shape[2],
            ), mode='constant', value=0
        )

        # visualization (useful when debugging)
        if False:
            import matplotlib.pyplot as plt
            from ..utils import get_det_im
            imgs = []
            columns = 4
            # rows = 1
            rows = img.shape[0]
            for i in range(0, rows):
                imgs.append(get_det_im(gt_img, gt_bboxes, gt_labels, i, img_metas[0]['img_norm_cfg']))
                imgs.append(get_det_im(t_img, t_bboxes, t_labels, i, t_img_metas[0]['img_norm_cfg']))
                imgs.append(get_det_im(s_img, s_bboxes, s_labels, i, s_img_metas[0]['img_norm_cfg']))
                imgs.append(get_det_im(c_img, c_bboxes, c_labels, i, c_img_metas[0]['img_norm_cfg']))
            fig = plt.figure(figsize=(8, 8))
            for i in range(0, rows*columns):
                fig.add_subplot(rows, columns, i+1)
                plt.imshow(imgs[i])
                if False:
                    plt.imsave(f'tmp/img_{i}.jpg', imgs[i])
                plt.axis('off')
            plt.subplots_adjust(wspace=0, hspace=0)
            plt.show()

        t_view = (t_img, t_bboxes, t_labels, t_scores, None,
                  t_img_metas, t_bboxes_ignore)
        s_view = (s_img, s_bboxes, s_labels, s_scores, s_match_indices,
                  s_img_metas, s_bboxes_ignore)
        c_view = (c_img, c_bboxes, c_labels, c_scores, c_match_indices,
                  c_img_metas, c_bboxes_ignore)
        return t_view, s_view, c_view

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_match_indices,
                      ref_img,
                      ref_img_metas,
                      ref_gt_bboxes,
                      ref_gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      ref_gt_bboxes_ignore=None,
                      ref_gt_masks=None,
                      **kwargs):
        """Forward function during training.

         Args:
            img (Tensor): of shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
            gt_bboxes (list[Tensor]): Ground truth bboxes of the image,
                each item has a shape (num_gts, 4).
            gt_labels (list[Tensor]): Ground truth labels of all images.
                each has a shape (num_gts,).
            gt_match_indices (list(Tensor)): Mapping from gt_instance_ids to
                ref_gt_instance_ids of the same tracklet in a pair of images.
            ref_img (Tensor): of shape (N, C, H, W) encoding input reference
                images. Typically these should be mean centered and std scaled.
            ref_img_metas (list[dict]): list of reference image info dict where
                each dict has: 'img_shape', 'scale_factor', 'flip', and may
                also contain 'filename', 'ori_shape', 'pad_shape',
                and 'img_norm_cfg'.
            ref_gt_bboxes (list[Tensor]): Ground truth bboxes of the
                reference image, each item has a shape (num_gts, 4).
            ref_gt_labels (list[Tensor]): Ground truth labels of all
                reference images, each has a shape (num_gts,).
            gt_masks (list[Tensor]) : Masks for each bbox, has a shape
                (num_gts, h , w).
            gt_bboxes_ignore (list[Tensor], None): Ground truth bboxes to be
                ignored, each item has a shape (num_ignored_gts, 4).
            ref_gt_bboxes_ignore (list[Tensor], None): Ground truth bboxes
                of reference images to be ignored,
                each item has a shape (num_ignored_gts, 4).
            ref_gt_masks (list[Tensor]) : Masks for each reference bbox,
                has a shape (num_gts, h , w).

        Returns:
            dict[str : TensorNone]: All losses.
        """
        # basic assertions
        assert self.detector.with_rpn, "two-stage KDQDT must have rpn"
        assert self.with_teacher_detector, "teacher_detector must exist"
        assert gt_masks is None, "gt_masks is not supported"

        with torch.no_grad():
            t_view, s_view, c_view = self._prepare_views(
                img, img_metas, gt_bboxes, gt_labels, gt_bboxes_ignore, gt_masks)

        img, gt_bboxes, gt_labels, gt_scores, _, img_metas, gt_bboxes_ignore = t_view
        (s_img, s_bboxes, s_labels, s_scores, s_match_indices,
         s_img_metas, s_bboxes_ignore) = s_view
        (c_img, c_bboxes, c_labels, c_scores, c_match_indices,
         c_img_metas, c_bboxes_ignore) = c_view

        # SPA geometry-preserving frequency augmentation (default OFF).
        # Port of SPA (ICML 2025) Prompt.masking: torch.fft on normalized GPU
        # tensors, same as the original paper. Teacher view never augmented.
        # Scheme 4: when spa_detection_only=True, SPA feeds ONLY the detection
        # branch; the tracking/association branch uses the clean student image.
        s_img_clean = s_img
        s_img_det = s_img
        if self.use_spa_frequency_aug and self.spa_augmentor is not None:
            with torch.no_grad():
                if self.spa_detection_only:
                    if self.spa_apply_to_detection and s_img is not None:
                        s_img_det = self.spa_augmentor(s_img_clean)
                    # tracking branch stays clean; contrastive stays clean
                    if not self._spa_det_only_logged:
                        import logging
                        lg = logging.getLogger('mmtrack')
                        lg.info('[SPA_DET_ONLY] enabled=True')
                        lg.info('[SPA_DET_ONLY] detection_branch=SPA')
                        lg.info('[SPA_DET_ONLY] tracking_branch=clean')
                        lg.info('[SPA_DET_ONLY] contrastive_branch=clean')
                        self._spa_det_only_logged = True
                    # detection branch drives s_img downstream (distillation);
                    # tracking uses the clean copy via s_x_track below.
                    s_img = s_img_det
                else:
                    if self.spa_apply_to_student and s_img is not None:
                        s_img = self.spa_augmentor(s_img)
                    if self.spa_apply_to_contrastive and c_img is not None:
                        c_img = self.spa_augmentor(c_img)

        # one-time post-Normalize view stats (debug; confirms SPA did NOT push
        # the normalized tensors off zero-mean/unit-std).
        if self.log_view_stats and not self._view_stats_logged:
            import logging
            lg = logging.getLogger('mmtrack')
            for nm, v in (('teacher', img), ('student', s_img), ('contrastive', c_img)):
                if v is not None:
                    lg.info('[VIEW STATS post-Normalize] %-11s mean=%.4f std=%.4f '
                            'min=%.4f max=%.4f', nm, float(v.mean()), float(v.std()),
                            float(v.min()), float(v.max()))
            self._view_stats_logged = True

        # student training
        losses = dict()

        # teacher forward pass on padded teacher view
        with torch.no_grad():
            t_x = self.teacher_detector.extract_feat(img)
            t_proposal_list = self.teacher_detector.rpn_head.simple_test_rpn(
                t_x, img_metas)

        # key frame forward.
        # Scheme 4 (spa_detection_only): run backbone ONCE on clean image for
        # tracking, then ONCE on SPA-augmented image for detection — same total
        # count as the standard two-view path, avoiding OOM from a third pass.
        if self.use_spa_frequency_aug and self.spa_detection_only:
            s_x_clean = self.detector.extract_feat(s_img_clean)
            s_x = self.detector.extract_feat(s_img)   # s_img is s_img_det here
        else:
            s_x_clean = None
            s_x = self.detector.extract_feat(s_img)
        s_proposal_list = self.detector.rpn_head.simple_test_rpn(s_x, s_img_metas)

        # rpn distillation loss
        if (
            self.rpn_distillation_loss is not None and
            self.rpn_distillation_loss.loss_weight > 0
        ):
            rpn_outs = self.get_rpn_displacement(self.detector, s_x)
            with torch.no_grad():
                teacher_rpn_outs = self.get_rpn_displacement(
                    self.teacher_detector, t_x)
            rpn_distillation_loss = self.rpn_distillation_loss(
                rpn_outs, teacher_rpn_outs)
            losses.update({"rpn_distillation_loss": rpn_distillation_loss})

        # roi distillation loss
        if (
            self.roi_distillation_loss is not None and
            self.roi_distillation_loss.loss_weight > 0
        ):
            roi_outs = self.get_roi_displacement(
                self.detector, s_x, t_proposal_list)
            with torch.no_grad():
                teacher_roi_outs = self.get_roi_displacement(
                    self.teacher_detector, t_x, t_proposal_list)
            roi_distillation_loss = self.roi_distillation_loss(
                roi_outs, teacher_roi_outs)
            losses.update({"roi_distillation_loss": roi_distillation_loss})

        (
            s_img, s_bboxes, s_labels, s_scores, s_match_indices,
            s_img_metas, s_bboxes_ignore,
            c_img, c_bboxes, c_labels, c_scores, c_match_indices,
            c_img_metas, c_bboxes_ignore,
        ) = drop_empty_images(
            s_img, s_bboxes, s_labels, s_scores, s_match_indices,
            s_img_metas, s_bboxes_ignore,
            c_img, c_bboxes, c_labels, c_scores, c_match_indices,
            c_img_metas, c_bboxes_ignore,
        )
        
        if (
            s_img is not None and (
                self.track_head.embed_head.loss_track.loss_weight > 0 or
                self.track_head.embed_head.loss_track_aux.loss_weight > 0)
        ):
            # ref frame forward
            c_x = self.detector.extract_feat(c_img)
            c_proposal_list = self.detector.rpn_head.simple_test_rpn(
                c_x, c_img_metas)
            # Scheme 4: tracking embedding uses clean features (s_x_clean,
            # pre-computed above); detection branch used the SPA-aug features
            # (s_x). No extra backbone pass here — OOM fix.
            if self.use_spa_frequency_aug and self.spa_detection_only:
                s_x_track = s_x_clean
            else:
                s_x_track = s_x
            track_losses = self.track_head.forward_train(
                s_x_track, s_img_metas, s_proposal_list, s_bboxes, s_labels,
                s_match_indices, c_x, c_img_metas, c_proposal_list,
                c_bboxes, c_labels, s_bboxes_ignore, gt_masks,
                c_bboxes_ignore, gt_scores=s_scores, ref_gt_scores=c_scores)
        else:
            track_losses = {
                'loss_track': torch.tensor(0.0).to(img.device),
                'loss_track_aux': torch.tensor(0.0).to(img.device)}
        losses.update(track_losses)

        # print(losses)
        return losses
