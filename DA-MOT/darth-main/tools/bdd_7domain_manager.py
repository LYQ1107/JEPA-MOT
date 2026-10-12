#!/usr/bin/env python3
import argparse, csv, json, os, subprocess, sys, time
from datetime import datetime
from pathlib import Path

ROOT=Path('/data1/LWR/vranlee/SERVER_ONLY/avis/DA-MOT/darth-main')
STATE=ROOT/'work_dirs/bdd_7domain/state'
ERR=ROOT/'work_dirs/bdd_7domain/error_summaries'
SUM=ROOT/'work_dirs/bdd_7domain/summaries'
DOMAINS=['clear','dawn_dusk','night','partly_cloudy','overcast','rainy','snowy']
SOURCE_EPOCHS=[20]  # Only final epoch for source-only eval
TTA_EPOCHS=[6,12]   # Only 2 checkpoints for TTA eval
FIELDS='job_id,stage,source,target,source_epoch,tta_epoch,mode,status,start_time,end_time,gpu_ids,samples_per_gpu,work_dir,checkpoint_path,log_path,result_path,done_marker,running_marker,fail_marker,pid_path,error_summary_path,command'.split(',')

def now(): return datetime.now().strftime('%Y-%m-%d %H:%M:%S')
def mkdirs():
    for p in [STATE,ERR,SUM,STATE/'running',STATE/'done',STATE/'failed',STATE/'pids']:
        p.mkdir(parents=True,exist_ok=True)
def jobs_path(): return STATE/'jobs.csv'
def marker(j,k):
    mp={'running':('running','.running'),'done':('done','.done'),'failed':('failed','.failed'),'pid':('pids','.pid')}[k]
    return str(STATE/mp[0]/f'{j}{mp[1]}')
def job(jid,stage,s='',t='',se='',te=''):
    return dict(job_id=jid,stage=stage,source=s,target=t,source_epoch=str(se),tta_epoch=str(te),mode='full_matrix',status='pending',start_time='',end_time='',gpu_ids='',samples_per_gpu='',work_dir='',checkpoint_path='',log_path='',result_path='',done_marker=marker(jid,'done'),running_marker=marker(jid,'running'),fail_marker=marker(jid,'failed'),pid_path=marker(jid,'pid'),error_summary_path=str(ERR/f'{jid}_error_summary.txt'),command='')
def read_jobs():
    if not jobs_path().exists(): return []
    with jobs_path().open() as f: return list(csv.DictReader(f))
def write_jobs(js):
    mkdirs()
    with jobs_path().open('w',newline='') as f:
        w=csv.DictWriter(f,FIELDS); w.writeheader(); [w.writerow({k:j.get(k,'') for k in FIELDS}) for j in js]
def update(jid,**kw):
    js=read_jobs(); ok=False
    for j in js:
        if j['job_id']==jid: j.update({k:str(v) for k,v in kw.items()}); ok=True
    if not ok: raise SystemExit(f'job not found: {jid}')
    write_jobs(js)
def init(force=False):
    mkdirs(); p=jobs_path()
    if p.exists() and not force:
        print(f'{p} exists; not overwriting'); return
    js=[job('prepare_data_7domains','prepare_data'),job('generate_configs_7domains','generate_config'),job('batch_probe_source','batch_probe_source'),job('batch_probe_tta','batch_probe_tta')]
    for s in DOMAINS: js.append(job(f'source_train_{s}_20e','source_train',s=s))
    for s in DOMAINS:
        for e in SOURCE_EPOCHS:
            for t in DOMAINS:
                # Include same-domain evaluation (e.g. clear_to_clear) for baseline
                js.append(job(f'source_only_{s}_epoch{e}_to_{t}','source_only_eval',s,t,e,''))
    for s in DOMAINS:
        for t in DOMAINS:
            if s==t: continue
            js.append(job(f'tta_{s}_to_{t}_12e','tta_adapt',s,t))
            for e in TTA_EPOCHS: js.append(job(f'tta_eval_{s}_to_{t}_epoch{e}','tta_eval',s,t,'',e))
    js.append(job('summary_parse_all_results','summary'))
    write_jobs(js); write_state(current_stage='initialized',current_job_id='',last_completed_job_id='')
    write_continue(); print(f'initialized {len(js)} jobs')
def write_state(**kw):
    mkdirs(); p=STATE/'pipeline_state.json'; d={}
    if p.exists():
        try: d=json.loads(p.read_text())
        except Exception: d={}
    d.update(dict(project_root=str(ROOT),env_name='DARTH',mode='full_matrix',source_epochs=20,tta_epochs=12,source_eval_epochs=SOURCE_EPOCHS,tta_eval_epochs=TTA_EPOCHS,gpu_num=10,last_update_time=now()))
    d.update(kw); d['last_update_time']=now(); p.write_text(json.dumps(d,indent=2))
def write_continue():
    (STATE/'continue_instructions.md').write_text('If user says “继续”, run:\n\n```bash\nbash scripts/resume_bdd_7domain_pipeline.sh --status-only\n```\n\nIf a job is running, report progress and do not restart. Start next only with `--resume-next`.\n')
def pid_alive(pid):
    try: os.kill(int(pid),0); return True
    except Exception: return False
def summarize():
    js=read_jobs(); counts={}
    for j in js: counts[j['status']]=counts.get(j['status'],0)+1
    running=[j for j in js if j['status']=='running']
    nxt=next((j for j in js if j['status'] in ['pending','interrupted','failed_or_incomplete']),None)
    text=f'# BDD 7-domain status\n- Updated: {now()}\n- Completed: {counts.get("completed",0)}/{len(js)}\n- Counts: {counts}\n- Running: {[j["job_id"] for j in running]}\n- Next: {nxt["job_id"] if nxt else "NONE"}\n- Continue: `bash scripts/resume_bdd_7domain_pipeline.sh --status-only`\n'
    (STATE/'last_cursor_summary.md').write_text(text); return text
def status():
    js=read_jobs(); print(summarize())
    for j in js:
        rp=Path(j['running_marker']); pp=Path(j['pid_path'])
        if rp.exists() or j['status']=='running':
            pid=pp.read_text().strip() if pp.exists() else ''
            print(f'RUNNING_MARKER job={j["job_id"]} pid={pid} alive={pid_alive(pid) if pid else False} log={j.get("log_path","")}')
            lp_s=j.get('log_path','')
            if lp_s:
                lp=Path(lp_s)
                if lp.is_file(): print('\n'.join(lp.read_text(errors='ignore').splitlines()[-80:]))
def get_spg(key,default='12'):
    tuning=SUM/'batch_tuning_result.json'
    if tuning.exists():
        try: return str(json.loads(tuning.read_text()).get(key,default))
        except Exception: pass
    return default
def get_ckpt(source,epoch=20):
    base=ROOT/f'work_dirs/bdd_7domain/source/{source}/qdtrack_20e'
    if not base.exists(): return ''
    for pattern in [f'**/epoch_{epoch}.pth', '**/latest.pth']:
        cands=list(base.glob(pattern))
        if cands: return str(sorted(cands,key=lambda x:x.stat().st_mtime)[-1])
    return ''
def is_valid_ckpt(p, min_size_mb=300):
    """Return True for checkpoints that are large enough to be complete."""
    try:
        return Path(p).is_file() and Path(p).stat().st_size >= min_size_mb * 1024 * 1024
    except Exception:
        return False

def get_resume_ckpt(wd):
    """Find latest valid epoch checkpoint in work_dir for resume training."""
    base=ROOT/wd
    if not base.exists(): return ''
    # Only search in non-archive directories (exclude *_archive), and skip truncated checkpoints.
    cands=sorted([p for p in base.rglob('epoch_*.pth') if '_archive' not in str(p) and is_valid_ckpt(p)],key=lambda x:x.stat().st_mtime)
    return str(cands[-1]) if cands else ''
def verify_source_metrics(log_path,min_mota_pct=5.0):
    """Parse MOTA and mAP from training/eval log. Returns (ok, mota_pct, map_pct)."""
    import re
    try:
        text=Path(log_path).read_text(errors='ignore')
        motas=re.findall(r'track/OVERALL/MOTA:\s*([\d.eE+\-]+)',text)
        maps=re.findall(r'bbox/OVERALL/mAP:\s*([\d.eE+\-]+)',text)
        mota=float(motas[-1])*100 if motas else 0.0
        map_val=float(maps[-1])*100 if maps else 0.0
        return mota>=min_mota_pct,round(mota,2),round(map_val,2)
    except Exception: return False,0.0,0.0

def recover_failed_jobs():
    """Recover jobs that completed successfully but were marked as failed."""
    js=read_jobs()
    recovered=[]
    
    for j in js:
        if j['status'] != 'failed':
            continue
        
        stage=j['stage']
        jid=j['job_id']
        
        # Check for TTA adapt with successful checkpoint
        if stage == 'tta_adapt':
            s,t = j['source'], j['target']
            tta_wd=ROOT/f'work_dirs/bdd_7domain/tta/{s}_to_{t}'
            
            # Check for epoch_12 checkpoint
            epoch_12 = list(tta_wd.rglob('epoch_12.pth')) if tta_wd.exists() else []
            if epoch_12:
                Path(j['done_marker']).write_text(f'{now()}\n')
                Path(j['fail_marker']).unlink() if Path(j['fail_marker']).exists() else None
                update(jid, status='completed', end_time=now())
                recovered.append(f"{jid} (epoch_12 found)")
                continue
            
            # Check for epoch_6 checkpoint - partial success
            epoch_6 = list(tta_wd.rglob('epoch_6.pth')) if tta_wd.exists() else []
            if epoch_6:
                # Mark as interrupted so it can be resumed
                Path(j['fail_marker']).write_text(f'{now()}\npartial: stopped at epoch 6\n')
                update(jid, status='interrupted', end_time=now())
                recovered.append(f"{jid} (partial: epoch_6 found, needs resume)")
                continue
        
        # Check for TTA eval with checkpoint available
        elif stage == 'tta_eval':
            s,t,e = j['source'], j['target'], j['tta_epoch']
            tta_wd=ROOT/f'work_dirs/bdd_7domain/tta/{s}_to_{t}'
            cands = list(tta_wd.rglob(f'epoch_{e}.pth')) if tta_wd.exists() else []
            if cands:
                # Checkpoint exists, just need to rerun eval
                Path(j['fail_marker']).unlink() if Path(j['fail_marker']).exists() else None
                update(jid, status='pending', end_time='')
                recovered.append(f"{jid} (checkpoint exists, reset to pending)")
                continue
    
    return recovered
def build_cmd(j):
    stage,s,t=j['stage'],j.get('source',''),j.get('target','')
    spg_src=get_spg('source_samples_per_gpu')
    spg_tta=get_spg('tta_samples_per_gpu')
    NGPU=8
    if stage=='source_train':
        # val videos: clear=36,dawn_dusk=20,night=71,partly_cloudy=9,overcast=22,rainy=7,snowy=15
        # GPU count must be <= val video count to avoid distributed eval rank starvation
        gpu_map={'rainy':6,'partly_cloudy':6,'snowy':8}
        ngpu=gpu_map.get(s,8)
        wd=f'work_dirs/bdd_7domain/source/{s}/qdtrack_20e'
        cfg=f'configs/mot/qdtrack/bdd_7domain/qdtrack_faster-rcnn_r50_fpn_20e_bdd_{s}.py'
        resume_ckpt=get_resume_ckpt(wd)
        resume_flag=f' --resume-from {resume_ckpt}' if resume_ckpt else ''
        return f"python -m torch.distributed.launch --nproc_per_node={ngpu} --master_addr=127.0.0.1 --master_port=$PORT tools/run/train.py {cfg} --launcher pytorch --work-dir {wd}{resume_flag} --cfg-options data.samples_per_gpu={spg_src} data.workers_per_gpu=4"
    if stage=='source_train_30e':
        # Continue source-domain training from epoch20 to epoch30; save every epoch; disable mid-train eval.
        gpu_map={'rainy':6,'partly_cloudy':6,'snowy':8}
        ngpu=gpu_map.get(s,8)
        wd=f'work_dirs/bdd_7domain/source/{s}/qdtrack_20e'
        cfg=f'configs/mot/qdtrack/bdd_7domain/qdtrack_faster-rcnn_r50_fpn_20e_bdd_{s}.py'
        resume_ckpt=get_resume_ckpt(wd)
        resume_flag=f' --resume-from {resume_ckpt}' if resume_ckpt else ''
        return f"python -m torch.distributed.launch --nproc_per_node={ngpu} --master_addr=127.0.0.1 --master_port=$PORT tools/run/train.py {cfg} --launcher pytorch --work-dir {wd}{resume_flag} --cfg-options data.samples_per_gpu={spg_src} data.workers_per_gpu=4 total_epochs=30 checkpoint_config.interval=1 evaluation.interval=31"
    if stage=='source_only_eval':
        e=j['source_epoch']
        ckpt=get_ckpt(s,int(e))
        wd=f'work_dirs/bdd_7domain/source_only_eval/{s}_epoch{e}_to_{t}'
        cfg=f'configs/mot/qdtrack/bdd_7domain/qdtrack_faster-rcnn_r50_fpn_20e_bdd_{t}.py'
        # val videos: clear=36,dawn_dusk=20,night=71,partly_cloudy=9,overcast=22,rainy=7,snowy=15
        gpu_map={'rainy':6,'partly_cloudy':6,'snowy':8}
        ngpu=gpu_map.get(t,8)
        return f"python -m torch.distributed.launch --nproc_per_node={ngpu} --master_addr=127.0.0.1 --master_port=$PORT tools/run/test.py {cfg} --checkpoint {ckpt} --launcher pytorch --work-dir {wd} --eval bbox track --cfg-options data.workers_per_gpu=4"
    if stage=='tta_adapt':
        ckpt=get_ckpt(s,20)
        wd=f'work_dirs/bdd_7domain/tta/{s}_to_{t}'
        cfg=f'configs/adapters/darth/bdd_7domain/darth_qdtrack_faster-rcnn_r50_fpn_12e_bdd_{t}.py'
        # GPU count must be <= val video count: clear=36,dawn_dusk=20,night=71,partly_cloudy=9,overcast=22,rainy=7,snowy=15
        # evaluation.interval=13 (> total_epochs=12) disables mid-training distributed eval entirely,
        # preventing rank starvation / BrokenPipe on small val sets like rainy(7) and partly_cloudy(9)
        gpu_map={'rainy':6,'partly_cloudy':6,'snowy':8,'dawn_dusk':8,'overcast':8,'clear':8,'night':8}
        ngpu=gpu_map.get(t,8)
        # Resume from latest checkpoint if exists
        resume_ckpt=get_resume_ckpt(wd)
        resume_flag=f' --resume-from {resume_ckpt}' if resume_ckpt else ''
        return f"python -m torch.distributed.launch --nproc_per_node={ngpu} --master_addr=127.0.0.1 --master_port=$PORT tools/run/train.py {cfg} --launcher pytorch --work-dir {wd}{resume_flag} --cfg-options model.init_cfg.checkpoint={ckpt} find_unused_parameters=True data.samples_per_gpu={spg_tta} data.workers_per_gpu=4 total_epochs=12 checkpoint_config.interval=1 evaluation.interval=13 optimizer.lr=0.002 model.loss_rpn_distillation.loss_weight=0.1 model.loss_roi_distillation.loss_weight=0.1 custom_hooks.0.momentum=0.002 model.conf_thr=0.7"
    if stage=='tta_adapt_20e':
        # Resume from epoch_12 and continue to epoch_20; save every epoch; no mid-train eval
        ckpt=get_ckpt(s,20)
        wd=f'work_dirs/bdd_7domain/tta/{s}_to_{t}'
        cfg=f'configs/adapters/darth/bdd_7domain/darth_qdtrack_faster-rcnn_r50_fpn_12e_bdd_{t}.py'
        gpu_map={'rainy':6,'partly_cloudy':6,'snowy':8,'dawn_dusk':8,'overcast':8,'clear':8,'night':8}
        ngpu=gpu_map.get(t,8)
        resume_ckpt=get_resume_ckpt(wd)
        resume_flag=f' --resume-from {resume_ckpt}' if resume_ckpt else ''
        return f"python -m torch.distributed.launch --nproc_per_node={ngpu} --master_addr=127.0.0.1 --master_port=$PORT tools/run/train.py {cfg} --launcher pytorch --work-dir {wd}{resume_flag} --cfg-options model.init_cfg.checkpoint={ckpt} find_unused_parameters=True data.samples_per_gpu={spg_tta} data.workers_per_gpu=4 total_epochs=20 checkpoint_config.interval=1 evaluation.interval=21 optimizer.lr=0.002 model.loss_rpn_distillation.loss_weight=0.1 model.loss_roi_distillation.loss_weight=0.1 custom_hooks.0.momentum=0.002 model.conf_thr=0.7"
    if stage=='tta_eval':
        e=int(j['tta_epoch'])
        tta_wd=ROOT/f'work_dirs/bdd_7domain/tta/{s}_to_{t}'
        cands=[p for p in tta_wd.rglob(f'epoch_{e}.pth') if is_valid_ckpt(p)]
        ckpt=str(sorted(cands,key=lambda x:x.stat().st_mtime)[-1]) if cands else ''
        wd=f'work_dirs/bdd_7domain/tta_eval/{s}_to_{t}_epoch{e}'
        cfg=f'configs/mot/qdtrack/bdd_7domain/qdtrack_faster-rcnn_r50_fpn_20e_bdd_{t}.py'
        # val videos: clear=36,dawn_dusk=20,night=71,partly_cloudy=9,overcast=22,rainy=7,snowy=15
        gpu_map={'rainy':6,'partly_cloudy':6,'snowy':8}
        ngpu=gpu_map.get(t,8)
        return f"python -m torch.distributed.launch --nproc_per_node={ngpu} --master_addr=127.0.0.1 --master_port=$PORT tools/run/test.py {cfg} --checkpoint {ckpt} --launcher pytorch --work-dir {wd} --eval bbox track --cfg-options data.workers_per_gpu=4"
    if stage=='tta_eval_20e':
        # Eval TTA checkpoint at any epoch (13-20) on target domain
        e=int(j['tta_epoch'])
        tta_wd=ROOT/f'work_dirs/bdd_7domain/tta/{s}_to_{t}'
        cands=[p for p in tta_wd.rglob(f'epoch_{e}.pth') if is_valid_ckpt(p)]
        ckpt=str(sorted(cands,key=lambda x:x.stat().st_mtime)[-1]) if cands else ''
        wd=f'work_dirs/bdd_7domain/tta_eval/{s}_to_{t}_epoch{e}'
        cfg=f'configs/mot/qdtrack/bdd_7domain/qdtrack_faster-rcnn_r50_fpn_20e_bdd_{t}.py'
        gpu_map={'rainy':6,'partly_cloudy':6,'snowy':8}
        ngpu=gpu_map.get(t,8)
        return f"python -m torch.distributed.launch --nproc_per_node={ngpu} --master_addr=127.0.0.1 --master_port=$PORT tools/run/test.py {cfg} --checkpoint {ckpt} --launcher pytorch --work-dir {wd} --eval bbox track --cfg-options data.workers_per_gpu=4"
    if stage=='source_self_eval':
        # Evaluate source model on its own domain (e.g. snowy→snowy baseline)
        ckpt=get_ckpt(s,20)
        wd=f'work_dirs/bdd_7domain/source_self_eval/{s}'
        cfg=f'configs/mot/qdtrack/bdd_7domain/qdtrack_faster-rcnn_r50_fpn_20e_bdd_{s}.py'
        gpu_map={'rainy':6,'partly_cloudy':6,'snowy':8}
        ngpu=gpu_map.get(s,8)
        return f"python -m torch.distributed.launch --nproc_per_node={ngpu} --master_addr=127.0.0.1 --master_port=$PORT tools/run/test.py {cfg} --checkpoint {ckpt} --launcher pytorch --work-dir {wd} --eval bbox track --cfg-options data.workers_per_gpu=4"
    if stage=='source_self_eval_extra':
        # Evaluate source model on its own domain at epoch25 or epoch30.
        e=int(j['source_epoch'])
        base=ROOT/f'work_dirs/bdd_7domain/source/{s}/qdtrack_20e'
        cands=[p for p in base.rglob(f'epoch_{e}.pth') if is_valid_ckpt(p)]
        ckpt=str(sorted(cands,key=lambda x:x.stat().st_mtime)[-1]) if cands else ''
        wd=f'work_dirs/bdd_7domain/source_self_eval/{s}_epoch{e}'
        cfg=f'configs/mot/qdtrack/bdd_7domain/qdtrack_faster-rcnn_r50_fpn_20e_bdd_{s}.py'
        gpu_map={'rainy':6,'partly_cloudy':6,'snowy':8}
        ngpu=gpu_map.get(s,8)
        return f"python -m torch.distributed.launch --nproc_per_node={ngpu} --master_addr=127.0.0.1 --master_port=$PORT tools/run/test.py {cfg} --checkpoint {ckpt} --launcher pytorch --work-dir {wd} --eval bbox track --cfg-options data.workers_per_gpu=4"
    if stage=='summary':
        return 'python tools/analysis/parse_bdd_7domain_results.py'
    return ''
def run_job(jid):
    j=next((x for x in read_jobs() if x['job_id']==jid),None)
    if not j: raise SystemExit(f'job not found: {jid}')
    if Path(j['done_marker']).exists():
        if j['status'] != 'completed':
            update(jid,status='completed',end_time=now())
        print(f'{jid} already done')
        return 0
    cmd=build_cmd(j)
    if not cmd: print(f'cannot build command for {jid}'); return 1
    wd=cmd.split('--work-dir')[1].split()[0] if '--work-dir' in cmd else f'work_dirs/bdd_7domain/logs/{j["stage"]}'
    Path(wd).mkdir(parents=True,exist_ok=True)
    log=f'{wd}/{jid}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    Path(j['running_marker']).write_text(f'{now()}\n')
    Path(j['pid_path']).write_text(str(os.getpid()))
    update(jid,status='running',start_time=now(),work_dir=wd,log_path=log,command=cmd)
    port=subprocess.run("python3 -c \"import socket;s=socket.socket();s.bind(('',0));print(s.getsockname()[1]);s.close()\"",shell=True,capture_output=True,text=True).stdout.strip()
    shell_cmd=f"eval \"$(conda shell.bash hook)\" && conda activate DARTH && export MPLBACKEND=Agg CUDA_VISIBLE_DEVICES=9,8,7,6,5,4,3,2 PORT={port} && {cmd}"
    import signal as _signal
    proc=subprocess.Popen(shell_cmd,shell=True,cwd=str(ROOT),stdout=open(log,'w'),stderr=subprocess.STDOUT,preexec_fn=os.setsid)
    try:
        proc.wait()
    except Exception:
        pass
    finally:
        # Kill the entire process group to ensure no GPU workers survive as orphans.
        # shell=True creates a new session via setsid; os.getpgid gives that group id.
        try:
            pgid=os.getpgid(proc.pid)
            os.killpg(pgid,_signal.SIGTERM)
            time.sleep(3)
            os.killpg(pgid,_signal.SIGKILL)
        except ProcessLookupError:
            pass
        except Exception:
            pass
    rc=proc.returncode if proc.returncode is not None else 1
    rm = Path(j['running_marker'])
    if rm.exists():
        rm.unlink()
    if rc==0:
        Path(j['done_marker']).write_text(f'{now()}\n')
        update(jid,status='completed',end_time=now())
    elif rc in (-2, -15, 130, 137, 143):
        Path(j['fail_marker']).write_text(f'{now()}\nrc={rc} (interrupted by signal)\n')
        update(jid,status='interrupted',end_time=now())
        print(f'{jid} interrupted (rc={rc}), will retry next run')
    else:
        Path(j['fail_marker']).write_text(f'{now()}\nrc={rc}\n')
        update(jid,status='failed',end_time=now())
    return rc
def main():
    ap=argparse.ArgumentParser(); sub=ap.add_subparsers(dest='cmd',required=True)
    p=sub.add_parser('init'); p.add_argument('--force',action='store_true')
    sub.add_parser('status')
    p=sub.add_parser('run-job'); p.add_argument('--job-id',required=True)
    sub.add_parser('recover')
    a=ap.parse_args()
    if a.cmd=='init': init(a.force)
    elif a.cmd=='status': status()
    elif a.cmd=='run-job': sys.exit(run_job(a.job_id))
    elif a.cmd=='recover':
        recovered = recover_failed_jobs()
        print(f"\n=== Recovery Report ===")
        print(f"Recovered {len(recovered)} jobs:")
        for r in recovered:
            print(f"  - {r}")
        if recovered:
            print(f"\nRun 'python {sys.argv[0]} status' to see updated status")
if __name__=='__main__': main()
