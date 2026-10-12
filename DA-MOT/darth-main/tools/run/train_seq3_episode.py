#!/usr/bin/env python3
"""Continuous sequential DARTH episode trainer.
Builds one model, loads clear source once, then adapts in-memory through targets.
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
    p=argparse.ArgumentParser(description='Continuous BDD seq3 DARTH episode')
    p.add_argument('--base-config',required=True); p.add_argument('--source-ckpt',required=True)
    p.add_argument('--target-config-template', default='configs/adapters/darth/bdd_7domain/darth_qdtrack_faster-rcnn_r50_fpn_12e_bdd_{target}.py')
    p.add_argument('--source-domain', default='clear')
    p.add_argument('--target-order',nargs='+',default=['dawn_dusk','night','partly_cloudy'])
    p.add_argument('--eval-domains',nargs='+',default=['dawn_dusk','night','partly_cloudy'])
    p.add_argument('--work-dir',required=True); p.add_argument('--epochs-per-stage',type=int,default=20)
    p.add_argument('--samples-per-gpu',type=int,default=10); p.add_argument('--workers-per-gpu',type=int,default=2)
    p.add_argument('--launcher',choices=['none','pytorch','slurm','mpi'],default='none'); p.add_argument('--local_rank',type=int,default=0)
    p.add_argument('--seed',type=int,default=None); p.add_argument('--diff_seed',action='store_true'); p.add_argument('--deterministic',action='store_true')
    p.add_argument('--no-validate',action='store_true')
    p.add_argument('--reset-optimizer-each-stage',action='store_true',default=True)
    p.add_argument('--keep-teacher-across-stages',action='store_true',default=True)
    p.add_argument('--keep-optimizer-across-stages',action='store_true')
    p.add_argument('--resume-episode-from-stage',type=int,default=0)
    p.add_argument('--cfg-options',nargs='+',action=DictAction)
    a=p.parse_args(); os.environ.setdefault('LOCAL_RANK',str(a.local_rank)); return a

def rank0(): return get_dist_info()[0]==0
def unwrap(m): return m.module if is_module_wrapper(m) else m
def mkdir(p):
    if rank0(): mmcv.mkdir_or_exist(osp.abspath(p))
def target_cfg_path(t, template=None):
    template = template or 'configs/adapters/darth/bdd_7domain/darth_qdtrack_faster-rcnn_r50_fpn_12e_bdd_{target}.py'
    return template.format(target=t)
def target_train_cfg(t, template=None): return Config.fromfile(target_cfg_path(t, template)).data.train

def init_teacher_from_student_once(model,logger):
    m=unwrap(model); cp=cb=0; ps=dict(m.named_parameters(recurse=True)); bs=dict(m.named_buffers(recurse=True))
    for n,tp in ps.items():
        if n.startswith('teacher_'):
            sp=ps.get(n[len('teacher_'):])
            if sp is not None and sp.shape==tp.shape: tp.data.copy_(sp.data); cp+=1
    for n,tb in bs.items():
        if n.startswith('teacher_'):
            sb=bs.get(n[len('teacher_'):])
            if sb is not None and sb.shape==tb.shape: tb.data.copy_(sb.data); cb+=1
    logger.info('Teacher initialized from student once before Stage 1. Teacher will be kept across stages. copied_params=%d copied_buffers=%d',cp,cb)

def student_norm(model):
    s=0.0; c=0
    with torch.no_grad():
        for n,p in unwrap(model).named_parameters(recurse=True):
            if not n.startswith('teacher_'):
                s+=float(torch.sum(p.detach().float()**2).cpu()); c+=p.numel()
    return s**0.5,c

def save_snap(model,path,meta):
    if rank0(): mmcv.mkdir_or_exist(osp.dirname(path)); save_checkpoint(unwrap(model),path,meta=meta)

def write_plan(root,src,source_domain,targets,evals,epochs):
    snaps={'stage00_source_only':f'{root}/stage_snapshots/stage00_source_only.pth'}; stages=[]; acc=[]
    for i,t in enumerate(targets,1):
        acc.append(t); name=f'stage{i:02d}_after_{t}'; snaps[name]=f'{root}/stage_snapshots/{name}.pth'
        stages.append(dict(stage_idx=i,stage=name,target_domain=t,adapted_domains_so_far=list(acc),start_model='in_memory_previous_stage' if i>1 else 'in_memory_source_checkpoint',snapshot=snaps[name],epochs=epochs))
    plan=dict(source_domain=source_domain,source_ckpt=src,target_order=targets,eval_domains=evals,epochs_per_stage=epochs,stage00=dict(stage='stage00_source_only',snapshot=snaps['stage00_source_only']),stages=stages,snapshots=snaps)
    sd=Path(root)/'state'; sd.mkdir(parents=True,exist_ok=True); (sd/'stage_plan.json').write_text(json.dumps(plan,indent=2))
    lines=[f'{source_domain} continuous DARTH episode',f'source_ckpt: {src}',f'target_order: {" -> ".join(targets)}','normal flow loads source checkpoint once; stage snapshots are not next-stage inputs.']
    for st in stages: lines.append(f"{st['stage']}: start={st['start_model']} snapshot={st['snapshot']}")
    (sd/'stage_plan.txt').write_text('\n'.join(lines)+'\n')

def stage_cfg(base,target,wdir,args):
    cfg=copy.deepcopy(base); cfg.data.train=target_train_cfg(target, args.target_config_template); cfg.work_dir=wdir
    cfg.total_epochs=args.epochs_per_stage
    cfg.find_unused_parameters=False
    if 'runner' in cfg: cfg.runner.max_epochs=args.epochs_per_stage
    cfg.data.samples_per_gpu=args.samples_per_gpu; cfg.data.workers_per_gpu=args.workers_per_gpu
    cfg.load_from=None; cfg.resume_from=None; cfg.workflow=[('train',1)]
    if cfg.get('checkpoint_config',None) is None: cfg.checkpoint_config=dict(interval=1,max_keep_ckpts=-1)
    else: cfg.checkpoint_config.interval=1; cfg.checkpoint_config.max_keep_ckpts=-1
    if cfg.get('evaluation',None) is not None: cfg.evaluation.interval=args.epochs_per_stage+1
    return cfg

def set_ckpt_meta(cfg,ds):
    cat=ds.cat_name_to_label if hasattr(ds,'cat_name_to_label') else None
    if cfg.checkpoint_config is not None:
        cfg.checkpoint_config.meta=dict(mmtrack_version=__version__,config=cfg.pretty_text,CLASSES=ds.CLASSES,cat_name_to_label=cat)
    return cat

def main():
    args=parse_args(); cfg=Config.fromfile(args.base_config)
    if args.cfg_options is not None: cfg.merge_from_dict(args.cfg_options)
    setup_multi_processes(cfg)
    if cfg.get('cudnn_benchmark',False): torch.backends.cudnn.benchmark=True
    distributed=args.launcher!='none'
    if distributed: init_dist(args.launcher,**cfg.dist_params)
    ts=time.strftime('%Y%m%d_%H%M%S',time.localtime()); mkdir(args.work_dir)
    if rank0(): Path(args.work_dir,'state').mkdir(parents=True,exist_ok=True); Path(args.work_dir,'stage_snapshots').mkdir(parents=True,exist_ok=True)
    if distributed: dist.barrier()
    logger=get_root_logger(log_file=osp.join(args.work_dir,f'episode_{ts}.log'),log_level=cfg.log_level)
    env='\n'.join([f'{k}: {v}' for k,v in collect_env().items()]); logger.info('Environment info:\n%s',env)
    logger.info('[EPISODE] Continuous seq3 trainer: one model, one source load, in-memory stage transitions')
    logger.info('[EPISODE] Target order: %s',args.target_order); logger.info('[EPISODE] Eval domains: %s',args.eval_domains)
    logger.info('[EPISODE] reset_optimizer_each_stage=True; keep_teacher_across_stages=True')
    cfg.seed=args.seed if args.seed is not None else cfg.get('seed',None) or init_random_seed()
    if args.diff_seed and distributed: cfg.seed+=dist.get_rank()
    det=True if args.deterministic else cfg.get('deterministic',False); set_random_seed(cfg.seed,deterministic=det)
    cfg.gpu_ids=range(dist.get_world_size()) if distributed else [0]; cfg.data.samples_per_gpu=args.samples_per_gpu; cfg.data.workers_per_gpu=args.workers_per_gpu
    cfg.load_from=None; cfg.resume_from=None
    if not osp.isfile(args.source_ckpt): raise FileNotFoundError(args.source_ckpt)
    if 'init_cfg' in cfg.model and cfg.model.init_cfg is not None:
        cfg.model.init_cfg.checkpoint = args.source_ckpt
        cfg.model.init_cfg.map_location = 'cpu'
    if rank0(): write_plan(args.work_dir,args.source_ckpt,args.source_domain,args.target_order,args.eval_domains,args.epochs_per_stage)
    if distributed: dist.barrier()
    logger.info('[EPISODE] Build DARTH model once')
    model=build_model(cfg.model,train_cfg=cfg.train_cfg,test_cfg=cfg.test_cfg) if cfg.get('train_cfg',False) else build_model(cfg.model)
    model.init_weights(); logger.info('[EPISODE] Load source checkpoint once: %s',args.source_ckpt)
    ckpt=load_checkpoint(model,args.source_ckpt,map_location='cpu')
    if 'meta' in ckpt and 'CLASSES' in ckpt['meta']: model.CLASSES=ckpt['meta']['CLASSES']
    init_teacher_from_student_once(model,logger)
    n,c=student_norm(model); logger.info('[CHECK] stage00 source-loaded student param norm = %.6f num_params=%d',n,c)
    save_snap(model,osp.join(args.work_dir,'stage_snapshots','stage00_source_only.pth'),dict(stage='stage00_source_only',source_ckpt=args.source_ckpt,source_loaded_once=True))
    meta=dict(env_info=env,seed=cfg.seed,episode='bdd_clear_seq3_continuous')
    start=max(1,args.resume_episode_from_stage or 1)
    if args.resume_episode_from_stage:
        pi=args.resume_episode_from_stage-1; prev='stage00_source_only' if pi==0 else f'stage{pi:02d}_after_{args.target_order[pi-1]}'
        rec=osp.join(args.work_dir,'stage_snapshots',prev+'.pth'); logger.warning('[RECOVERY] load snapshot for abnormal recovery only: %s',rec); load_checkpoint(model,rec,map_location='cpu')
    for i,t in enumerate(args.target_order,1):
        if i<start: logger.info('[STAGE %d] skip due to recovery start=%d',i,start); continue
        name=f'stage{i:02d}_after_{t}'; wdir=osp.join(args.work_dir,'adapt',name); mkdir(wdir)
        if distributed: dist.barrier()
        logger.info('[STAGE %d] %s, target=%s',i,'Start from in-memory model after source checkpoint' if i==1 else f'Continue from in-memory model after Stage {i-1}',t)
        sn,_=student_norm(model); logger.info('[CHECK] %s start student param norm = %.6f',name,sn); logger.info('[CHECK] normal Stage %d does NOT call load_checkpoint/load_from',i)
        scfg=stage_cfg(cfg,t,wdir,args); scfg.seed=cfg.seed; scfg.gpu_ids=cfg.gpu_ids
        if rank0(): scfg.dump(osp.join(wdir,osp.basename(target_cfg_path(t, args.target_config_template))))
        ds=build_dataset(scfg.data.train); cat=set_ckpt_meta(scfg,ds); unwrap(model).CLASSES=ds.CLASSES; unwrap(model).cat_name_to_label=cat
        if rank0() and scfg.checkpoint_config is not None:
            with open(osp.join(wdir,'checkpoint_meta.yml'),'w') as f: yaml.dump(scfg.checkpoint_config.meta,f,default_flow_style=False)
        logger.info('[STAGE %d] Build new optimizer/runner; SAME model object is reused',i)
        mod=unwrap(model)
        if hasattr(mod,'set_current_target_domain'): mod.set_current_target_domain(t)
        if hasattr(mod,'roi_proto_start_stage'): mod.roi_proto_start_stage(t)
        train_model(model,[ds],scfg,distributed=distributed,validate=False,timestamp=ts,meta=meta)
        if hasattr(mod,'roi_proto_finalize_stage'): mod.roi_proto_finalize_stage(t)
        if hasattr(mod,'roi_proto_memory_summary'):
            _summ=mod.roi_proto_memory_summary()
            if _summ is not None: logger.info('[ROI_PROTO] Stage %d summary: %s',i,_summ)
        en,_=student_norm(model); logger.info('[CHECK] %s end student param norm = %.6f',name,en)
        sp=osp.join(args.work_dir,'stage_snapshots',name+'.pth'); save_snap(model,sp,dict(stage=name,target_domain=t,adapted_domains_so_far=args.target_order[:i],normal_next_stage_uses_checkpoint=False))
        logger.info('[STAGE %d] Save snapshot: %s',i,sp); logger.info('[STAGE %d] Teacher kept across stages; no boundary reset',i)
        if distributed: dist.barrier()
    logger.info('[EPISODE] Training complete. Source checkpoint loaded exactly once in normal flow.')
if __name__=='__main__': main()
