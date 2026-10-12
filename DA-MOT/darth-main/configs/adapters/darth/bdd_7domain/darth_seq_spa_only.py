# Experiment A: SPA frequency augmentation only.
# use_spa_frequency_aug=True, causal bank OFF, memory OFF.
_base_ = [
    './darth_seq_ablation_base.py',
]

model = dict(
    use_spa_frequency_aug=True,
    spa_apply_to_student=True,
    spa_apply_to_contrastive=False,
    spa_low_ratio=0.2,
    spa_keep_prob=0.8,
    spa_noise_ratio=0.0,
    track_head=dict(
        use_causal_direction_bank=False,
        use_roi_proto_memory=False,
    ),
)
