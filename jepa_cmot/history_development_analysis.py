"""Complete BDD native history-length comparisons against actual source/geometry outputs."""
from pathlib import Path
import json,hashlib
from jepa_cmot.development_analysis import compare_domains
def main():
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--clips',type=Path,required=True)
    parser.add_argument('--baseline',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    root=args.root;clips=args.clips;baseline=args.baseline
    c=json.loads((root/'complete.json').read_text());b=json.loads((baseline/'complete.json').read_text());cc=json.loads((clips/'complete.json').read_text());domains=['clear','overcast','rainy','night']
    def reports(base,method):return {d:json.loads((base/'metrics'/method/(d+'.json')).read_text()) for d in domains}
    comparisons={};parity={}
    for method in ['discriminative','predictive','predictive_ranking']:
     name=method+'_history4';assert c['matrix'][name]==cc['matrix'][method]
     reports4=reports(root,name);checks={}
     for domain in domains:
      for video in reports4[domain]['videos']:
       a=root/'tracks'/name/video/'tracks.txt';z=clips/'tracks'/method/video/'tracks.txt';assert a.read_bytes()==z.read_bytes();checks[video]=hashlib.sha256(a.read_bytes()).hexdigest()
     parity[name]=checks
     comparisons[method+'_history8_vs_history1']=compare_domains(reports(root,method+'_history1'),reports(root,method+'_history8'))
     comparisons[method+'_history8_vs_geometry']=compare_domains(reports(baseline,'geometry_null'),reports(root,method+'_history8'))
     comparisons[method+'_history4_vs_history1']=compare_domains(reports(root,method+'_history1'),reports4)
    result=dict(status='COMPLETE_FIXED_SEED_FULL_FROZEN_DEVELOPMENT_HISTORY_ABLATION',dataset='bdd100k',seed=17,videos=19,native_frames=3844,source='same contiguous source clips, 2000x64 per objective; matched capacity history1/2/4/8',metrics=c['matrix'],equal_domain_average={n:{k:sum(x[k] for x in m.values())/4 for k in ['HOTA','AssA','IDF1']} for n,m in c['matrix'].items()},geometry_reference=b['matrix']['geometry_null'],paired_comparisons=comparisons,actual_history4_tracks_equal_independent_clips_deployment=parity,source_checkpoint_hashes_unchanged=c['checkpoint_hashes_unchanged'],target_GT_in_training=False,sealed_used=False,limitation='fixed-seed development exploration; not selection-adjusted, multi-seed or continual evidence')
    args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(json.dumps(result,indent=2)+'\n');print('FULL_HISTORY_DEVELOPMENT_ANALYSIS_COMPLETE',result['equal_domain_average'],flush=True)


if __name__=="__main__":main()
