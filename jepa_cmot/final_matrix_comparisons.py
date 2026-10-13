"""Paired raw-count comparisons of actual complete final continual evaluations."""
import argparse
import json
from pathlib import Path

from .development_analysis import compare_domains


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sequence', type=Path, required=True)
    parser.add_argument('--frozen', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    complete = json.loads((args.sequence/'complete.json').read_text())
    domains = complete['protocol']['order']
    final = str(len(domains)-1)
    if any(set(values) != {str(i) for i in range(len(domains))} for values in complete['matrix'].values()):
        raise ValueError('Final comparison requires every actually evaluated stage')
    def reports(name):
        return {d: json.loads((args.sequence/'evaluation'/('stage'+final)/name/(d+'_metrics.json')).read_text())
                for d in domains}
    geometry = {d: json.loads((args.frozen/'metrics/geometry_null'/(d+'.json')).read_text()) for d in domains}
    comparisons = {}
    for candidate, reference in [
        ('predictive_ranking_replay128_none', 'discriminative_replay128_none'),
        ('predictive_ranking_replay0_none', 'discriminative_replay0_none'),
        ('predictive_ranking_replay128_features', 'discriminative_replay128_none'),
        ('predictive_ranking_replay128_relations', 'discriminative_replay128_none')]:
        comparisons[candidate+':'+reference] = compare_domains(reports(reference), reports(candidate))
    for candidate in ['predictive_ranking_replay128_none', 'predictive_ranking_replay128_features']:
        comparisons[candidate+':actual_frozen_geometry'] = compare_domains(geometry, reports(candidate))
    result = dict(scope='complete final native development; fixed seed17, no method-selection correction or multi-seed claim',
        comparisons=comparisons, reference_note='geometry is actual full static frozen development on identical images/detections/frame coverage',
        target_GT_in_training=False, sealed_used=False)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2)+'\n')
    print('FINAL_PAIRED_COMPARISONS_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
