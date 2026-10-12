#!/usr/bin/env python3
"""Sequential 4-target DARTH episode trainer with student reset per stage.
Task3 ablation: teacher weights are carried across stages; student weights are
re-initialised from the source checkpoint at the start of every new stage.
"""
import argparse, copy, json, os, os.path as osp, time, yaml
from pathlib import Path
import mmcv, torch, torch.distributed as dist
from mmcv import Config, DictAction
from mmcv.parallel import is_module_wrapper
from mmcv.runner import get_dist_info, init_dist, load_checkpoint, save_checkpoint
from mmdet.apis import set_random_seed
from mmtrack import __version__
from mmtrack.apis import init_random_seed, train_model
from mmtrack.core import setup_multi_processes
from mmtrack.utils import collect_env, get_root_logger
from darth.datasets import build_dataset
from darth.models import build_model


def parse_args():
    p = argparse.ArgumentParser(description='Seq4 DARTH: teacher kept, student reset each stage')
    p.add_argument('--base-config', required=True)
    p.add_argument('--source-ckpt', required=True)
    p.add_argument('--target-order', nargs='+', default=['overcast', 'rainy', 'dawn_dusk', 'night'])
    p.add_argument('--eval-domains', nargs='+', default=['overcast', 'rainy', 'dawn_dusk', 'night', 'clear'])
    p.add_argument('--work-dir', required=True)
    p.add_argument('--epochs-per-stage', type=int, default=10)
    p.add_argument('--samples-per-gpu', type=int, default=10)
    p.add_argument('--workers-per-gpu', type=int, default=2)
    p.add_argument('--launcher', choices=['none', 'pytorch', 'slurm', 'mpi'], default='none')
    p.add_argument('--local_rank', type=int, default=0)
    p.add_argument('--seed', type=int, default=None)
    p.add_argument('--diff_seed', action='store_true')
    p.add_argument('--deterministic', action='store_true')
    p.add_argument('--no-validate', action='store_true')
    p.add_argument('--resume-episode-from-stage', type=int, default=0)
    p.add_argument('--cfg-options', nargs='+', action=DictAction)
    a = p.parse_args()
    os.environ.setdefault('LOCAL_RANK', str(a.local_rank))
    return a


def rank0():
    return get_dist_info()[0] == 0


def unwrap(m):
    return m.module if is_module_wrapper(m) else m


def mkdir(p):
    if rank0():
        mmcv.mkdir_or_exist(osp.abspath(p))


def target_cfg_path(t):
    return f'configs/adapters/darth/bdd_7domain/darth_qdtrack_faster-rcnn_r50_fpn_12e_bdd_{t}.py'


def target_train_cfg(t):
    return Config.fromfile(target_cfg_path(t)).data.train


def _copy_teacher_from_student(model, logger):
    """Copy student weights into teacher (used once after initial source load)."""
    m = unwrap(model)
    cp = cb = 0
    ps = dict(m.named_parameters(recurse=True))
    bs = dict(m.named_buffers(recurse=True))
    for n, tp in ps.items():
        if n.startswith('teacher_'):
            sp = ps.get(n[len('teacher_'):])
            if sp is not None and sp.shape == tp.shape:
                tp.data.copy_(sp.data); cp += 1
    for n, tb in bs.items():
        if n.startswith('teacher_'):
            sb = bs.get(n[len('teacher_'):])
            if sb is not None and sb.shape == tb.shape:
                tb.data.copy_(sb.data); cb += 1
    logger.info('[INIT] Teacher initialised from student. params=%d buffers=%d', cp, cb)


def reset_student_keep_teacher(model, source_ckpt_path, logger):
    """Re-load source checkpoint into student weights only; teacher is unchanged."""
    m = unwrap(model)

    # Snapshot teacher before reload
    teacher_params = {}
    teacher_buffers = {}
    for n, p in m.named_parameters(recurse=True):
        if n.startswith('teacher_'):
            teacher_params[n] = p.data.clone()
    for n, b in m.named_buffers(recurse=True):
        if n.startswith('teacher_'):
            teacher_buffers[n] = b.data.clone()

    # Reload full checkpoint into model (overwrites both student and teacher)
    load_checkpoint(m, source_ckpt_path, map_location='cpu')

    # Restore teacher from snapshot
    restored_p = restored_b = 0
    for n, p in m.named_parameters(recurse=True):
        if n in teacher_params:
            p.data.copy_(teacher_params[n]); restored_p += 1
    for n, b in m.named_buffers(recurse=True):
        if n in teacher_buffers:
            b.data.copy_(teacher_buffers[n]); restored_b += 1

    logger.info('[RESET] Student reset from source ckpt; teacher restored. '
                'teacher_params=%d teacher_buffers=%d', restored_p, restored_b)


def param_norm(model, prefix_exclude='teacher_'):
    s = 0.0; c = 0
    with torch.no_grad():
        for n, p in unwrap(model).named_parameters(recurse=True):
            if not n.startswith(prefix_exclude):
                s += float(torch.sum(p.detach().float() ** 2).cpu()); c += p.numel()
    return s ** 0.5, c


def save_snap(model, path, meta):
    if rank0():
        mmcv.mkdir_or_exist(osp.dirname(path))
        save_checkpoint(unwrap(model), path, meta=meta)


def write_plan(root, src, targets, evals, epochs):
    snaps = {'stage00_source_only': f'{root}/stage_snapshots/stage00_source_only.pth'}
    stages = []
    for i, t in enumerate(targets, 1):
        name = f'stage{i:02d}_after_{t}'
        snaps[name] = f'{root}/stage_snapshots/{name}.pth'
        stages.append(dict(
            stage_idx=i, stage=name, target_domain=t,
            student_init='reset_from_source_ckpt',
            teacher_init='carried_from_previous_stage' if i > 1 else 'copied_from_student_after_source_load',
            snapshot=snaps[name], epochs=epochs))
    plan = dict(
        source_domain='clear', source_ckpt=src,
        target_order=targets, eval_domains=evals,
        epochs_per_stage=epochs, ablation='teacher_kept_student_reset',
        stage00=dict(stage='stage00_source_only', snapshot=snaps['stage00_source_only']),
        stages=stages, snapshots=snaps)
    sd = Path(root) / 'state'
    sd.mkdir(parents=True, exist_ok=True)
    (sd / 'stage_plan.json').write_text(json.dumps(plan, indent=2))
    lines = [
        'BDD seq4 DARTH: teacher kept, student reset each stage',
        f'source_ckpt: {src}',
        f'target_order: {" -> ".join(targets)}',
        'At each stage boundary: student <- source_ckpt; teacher <- carried from previous stage',
    ]
    for st in stages:
        lines.append(f"{st['stage']}: student={st['student_init']} teacher={st['teacher_init']}")
    (sd / 'stage_plan.txt').write_text('\n'.join(lines) + '\n')


def stage_cfg(base, target, wdir, args):
    cfg = copy.deepcopy(base)
    cfg.data.train = target_train_cfg(target)
    cfg.work_dir = wdir
    cfg.total_epochs = args.epochs_per_stage
    cfg.find_unused_parameters = True
    if 'runner' in cfg:
        cfg.runner.max_epochs = args.epochs_per_stage
    cfg.data.samples_per_gpu = args.samples_per_gpu
    cfg.data.workers_per_gpu = args.workers_per_gpu
    cfg.load_from = None
    cfg.resume_from = None
    cfg.workflow = [('train', 1)]
    if cfg.get('checkpoint_config', None) is None:
        cfg.checkpoint_config = dict(interval=1, max_keep_ckpts=-1)
    else:
        cfg.checkpoint_config.interval = 1
        cfg.checkpoint_config.max_keep_ckpts = -1
    if cfg.get('evaluation', None) is not None:
        cfg.evaluation.interval = args.epochs_per_stage + 1
    return cfg


def set_ckpt_meta(cfg, ds):
    cat = ds.cat_name_to_label if hasattr(ds, 'cat_name_to_label') else None
    if cfg.checkpoint_config is not None:
        cfg.checkpoint_config.meta = dict(
            mmtrack_version=__version__,
            config=cfg.pretty_text,
            CLASSES=ds.CLASSES,
            cat_name_to_label=cat)
    return cat


def main():
    args = parse_args()
    cfg = Config.fromfile(args.base_config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    setup_multi_processes(cfg)
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    distributed = args.launcher != 'none'
    if distributed:
        init_dist(args.launcher, **cfg.dist_params)
    ts = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    mkdir(args.work_dir)
    if rank0():
        Path(args.work_dir, 'state').mkdir(parents=True, exist_ok=True)
        Path(args.work_dir, 'stage_snapshots').mkdir(parents=True, exist_ok=True)
    if distributed:
        dist.barrier()

    logger = get_root_logger(
        log_file=osp.join(args.work_dir, f'episode_{ts}.log'),
        log_level=cfg.log_level)
    env = '\n'.join([f'{k}: {v}' for k, v in collect_env().items()])
    logger.info('Environment info:\n%s', env)
    logger.info('[EPISODE] seq4 student-reset trainer')
    logger.info('[EPISODE] Ablation: teacher carried across stages; student reset to source each stage')
    logger.info('[EPISODE] Target order: %s', args.target_order)
    logger.info('[EPISODE] Eval domains: %s', args.eval_domains)

    cfg.seed = args.seed if args.seed is not None else cfg.get('seed', None) or init_random_seed()
    if args.diff_seed and distributed:
        cfg.seed += dist.get_rank()
    det = True if args.deterministic else cfg.get('deterministic', False)
    set_random_seed(cfg.seed, deterministic=det)
    cfg.gpu_ids = range(dist.get_world_size()) if distributed else [0]
    cfg.data.samples_per_gpu = args.samples_per_gpu
    cfg.data.workers_per_gpu = args.workers_per_gpu
    cfg.load_from = None
    cfg.resume_from = None

    if not osp.isfile(args.source_ckpt):
        raise FileNotFoundError(args.source_ckpt)
    if rank0():
        write_plan(args.work_dir, args.source_ckpt, args.target_order, args.eval_domains, args.epochs_per_stage)
    if distributed:
        dist.barrier()

    logger.info('[EPISODE] Build DARTH model')
    model = (build_model(cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg)
             if cfg.get('train_cfg', False) else build_model(cfg.model))
    model.init_weights()

    logger.info('[EPISODE] Load source checkpoint: %s', args.source_ckpt)
    ckpt = load_checkpoint(model, args.source_ckpt, map_location='cpu')
    if 'meta' in ckpt and 'CLASSES' in ckpt['meta']:
        model.CLASSES = ckpt['meta']['CLASSES']

    # Initialise teacher from student once after source load
    _copy_teacher_from_student(model, logger)
    n, c = param_norm(model)
    logger.info('[CHECK] stage00 student norm=%.6f num_params=%d', n, c)

    save_snap(model,
              osp.join(args.work_dir, 'stage_snapshots', 'stage00_source_only.pth'),
              dict(stage='stage00_source_only', source_ckpt=args.source_ckpt))

    meta = dict(env_info=env, seed=cfg.seed, episode='bdd_seq4_student_reset')
    start = max(1, args.resume_episode_from_stage or 1)

    for i, t in enumerate(args.target_order, 1):
        if i < start:
            logger.info('[STAGE %d] skip (resume from stage %d)', i, start)
            continue

        name = f'stage{i:02d}_after_{t}'
        wdir = osp.join(args.work_dir, 'adapt', name)
        mkdir(wdir)
        if distributed:
            dist.barrier()

        logger.info('[STAGE %d] target=%s  RESET student from source; KEEP teacher', i, t)
        reset_student_keep_teacher(model, args.source_ckpt, logger)
        sn, _ = param_norm(model)
        logger.info('[CHECK] %s student norm after reset=%.6f', name, sn)

        scfg = stage_cfg(cfg, t, wdir, args)
        scfg.seed = cfg.seed
        scfg.gpu_ids = cfg.gpu_ids
        if rank0():
            scfg.dump(osp.join(wdir, osp.basename(target_cfg_path(t))))
        ds = build_dataset(scfg.data.train)
        cat = set_ckpt_meta(scfg, ds)
        unwrap(model).CLASSES = ds.CLASSES
        unwrap(model).cat_name_to_label = cat
        if rank0() and scfg.checkpoint_config is not None:
            with open(osp.join(wdir, 'checkpoint_meta.yml'), 'w') as f:
                yaml.dump(scfg.checkpoint_config.meta, f, default_flow_style=False)

        logger.info('[STAGE %d] Train on %s for %d epochs', i, t, args.epochs_per_stage)
        train_model(model, [ds], scfg, distributed=distributed,
                    validate=False, timestamp=ts, meta=meta)

        en, _ = param_norm(model)
        logger.info('[CHECK] %s student norm after train=%.6f', name, en)
        sp = osp.join(args.work_dir, 'stage_snapshots', name + '.pth')
        save_snap(model, sp, dict(
            stage=name, target_domain=t,
            ablation='teacher_kept_student_reset',
            adapted_domains_so_far=args.target_order[:i]))
        logger.info('[STAGE %d] Snapshot saved: %s', i, sp)
        if distributed:
            dist.barrier()

    logger.info('[EPISODE] Training complete.')


if __name__ == '__main__':
    main()
