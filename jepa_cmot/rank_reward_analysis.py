"""Full BDD affinity-rule comparison and offline assignment diagnosis."""
from pathlib import Path
from collections import Counter,defaultdict
import json,hashlib
from jepa_cmot.development_analysis import compare_domains
from jepa_cmot.audit_counterfactual import audit
def main():
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--reference',type=Path,required=True)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    root=args.root; baseline=args.reference
    c=json.loads((root/'complete.json').read_text());b=json.loads((baseline/'complete.json').read_text());domains=['clear','overcast','rainy','night'];manifest=json.loads(args.manifest.read_text());videos={v['name']:v for v in manifest['videos'] if v['dataset']=='bdd100k' and v['split']=='development'}
    def reports(base,name):return {d:json.loads((base/'metrics'/name/(d+'.json')).read_text()) for d in domains}

    for name in videos:
     for base,key in [(root,'predictive_residual_audit'),(root,'random_rank_reward'),(baseline,'predictive'),(baseline,'geometry_null')]:assert (base/'tracks'/key/name/'tracks.txt').is_file()
    p=reports(root,'predictive_rank_reward');comp={}
    for name in ['predictive_residual_audit','discriminative_rank_reward','frozen_last_rank_reward','random_rank_reward']:
     comp[name]=compare_domains(reports(root,name),p)
    comp['geometry_null']=compare_domains(reports(baseline,'geometry_null'),p)
    (root/'paired_comparisons.json').write_text(json.dumps(comp,indent=2))
    counts={};by_domain={};parity={}
    for name in c['matrix']:
     total=Counter();details=defaultdict(Counter)
     for video,v in videos.items():
      events=root/'videos'/video/name/video/'association_changes.jsonl'; data=[json.loads(line) for line in events.read_text().splitlines()] if events.exists() else []
      summary=audit(json.loads(Path(v['labels']).read_text()),data,'bdd100k');total.update(summary['counts']);details[v['domain']].update(summary['counts'])
     counts[name]=dict(total);by_domain[name]={d:dict(x) for d,x in details.items()}
    for candidate,reference in [('predictive_residual_audit','predictive'),('random_rank_reward','geometry_null')]:
     checks={}
     for name in videos:
      left=root/'tracks'/candidate/name/'tracks.txt';right=baseline/'tracks'/reference/name/'tracks.txt'
      checks[name]=dict(equal=left.read_bytes()==right.read_bytes(),candidate_sha256=hashlib.sha256(left.read_bytes()).hexdigest(),reference_sha256=hashlib.sha256(right.read_bytes()).hexdigest())
     parity[candidate+'_vs_'+reference]=checks
    result=dict(dataset='bdd100k',seed=17,scope='complete frozen BDD development split',videos=len(videos),native_frames=sum(v['frames'] for v in videos.values()),source_checkpoint_sha256=c['protocol']['checkpoint_sha256'],source_checkpoint_hashes_unchanged=c['checkpoint_hashes_unchanged'],target_GT_used_for_training_or_gates=False,sealed_used=False,domain_metrics=c['matrix'],equal_domain_average={n:{k:sum(v[k] for v in m.values())/4 for k in ['HOTA','AssA','IDF1']} for n,m in {**c['matrix'],'geometry_null':b['matrix']['geometry_null']}.items()},paired_comparisons=comp,offline_counterfactual_counts=counts,offline_counterfactual_counts_by_domain=by_domain,track_byte_parity=parity,diagnosis_limitation='one-step counterfactual on the method own state; counts are not official IDSW or independently propagated baseline',conclusion='No practically substantial JEPA-specific benefit from this affinity rule on the frozen full development split; continual-learning efficacy is unresolved.')
    out=args.out;out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2));print(json.dumps({k:result[k] for k in ['equal_domain_average','offline_counterfactual_counts']},indent=2));print('parity', {k:all(x['equal'] for x in v.values()) for k,v in parity.items()})


if __name__=="__main__":main()
