"""Complete official encoder/masked-predictor frozen development comparison."""
import argparse
import json
from pathlib import Path
import sys

from .development_analysis import compare_domains
from .train_source import sha256


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--dataset', choices=['bdd100k', 'ua_detrac'], required=True)
    parser.add_argument('--frozen', type=Path, required=True)
    parser.add_argument('--source-audit', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    source = json.loads(args.source_audit.read_text())
    assert source['dataset'] == args.dataset and not source['target_GT_read']
    baseline = json.loads((args.frozen / 'complete.json').read_text())
    assert baseline['protocol']['dataset'] == args.dataset
    domains = list(baseline['matrix']['geometry_null'])
    expected_videos = baseline['videos']
    matrices, reports, protocols, inputs = {}, {}, {}, {}
    methods = ['frozen_last', 'discriminative', 'predictive', 'predictive_ranking']
    for readout in ['encoder', 'predictor']:
        root = args.root / 'development' / f'{args.dataset}_{readout}_seed17'
        complete = json.loads((root / 'complete.json').read_text())
        assert complete['checkpoint_hashes_unchanged'] and complete['videos'] == expected_videos
        protocol = complete['protocol']
        assert protocol['dataset'] == args.dataset and not protocol['sealed_test_used']
        assert set(complete['matrix']) == set(methods)
        protocols[readout] = protocol
        actual_inputs = {}
        for path in (root / 'inputs').glob('*.json'):
            item = json.loads(path.read_text())[0]
            prepared = Path(item['prepared']).resolve()
            actual_inputs[item['video']] = dict(domain=item['domain'], frames=item['frames'],
                annotation_sha256=sha256(prepared / 'annotations/val.json'))
            original = json.loads((args.frozen / 'inputs' / path.name).read_text())[0]
            assert original['frames'] == item['frames'] and original['domain'] == item['domain']
            assert sha256(Path(original['prepared']) / 'annotations/val.json') == \
                actual_inputs[item['video']]['annotation_sha256']
        assert len(actual_inputs) == expected_videos
        inputs[readout] = actual_inputs
        for method in methods:
            assert set(complete['matrix'][method]) == set(domains)
            if method != 'frozen_last':
                assert protocol['checkpoint_sha256'][method] == \
                    source['source_training'][readout][method]['checkpoint_sha256']
            key = readout + '/' + method
            matrices[key] = complete['matrix'][method]
            reports[key] = {d: json.loads((root / 'metrics' / method / (d + '.json')).read_text())
                            for d in domains}
    assert inputs['encoder'] == inputs['predictor']
    geometry = {d: json.loads((args.frozen / 'metrics/geometry_null' / (d + '.json')).read_text())
                for d in domains}
    comparisons = {}
    for readout in ['encoder', 'predictor']:
        for candidate, reference in [('predictive', 'discriminative'),
                                     ('predictive', 'frozen_last'),
                                     ('predictive_ranking', 'predictive')]:
            a, b = readout + '/' + candidate, readout + '/' + reference
            comparisons[a + ':' + b] = compare_domains(reports[b], reports[a])
        for method in ['predictive', 'discriminative']:
            name = readout + '/' + method
            comparisons[name + ':actual_frozen_geometry'] = compare_domains(geometry, reports[name])
    for method in ['predictive', 'frozen_last']:
        a, b = 'predictor/' + method, 'encoder/' + method
        comparisons[a + ':' + b] = compare_domains(reports[b], reports[a])
    result = dict(status='COMPLETE_OFFICIAL_PAIRED_READOUT_FROZEN_DEVELOPMENT', dataset=args.dataset,
                  seed=17, videos=expected_videos, domain_metrics=matrices,
                  equal_domain_average={name: {metric: sum(v[metric] for v in values.values()) / len(domains)
                                              for metric in ['HOTA', 'AssA', 'IDF1']}
                                        for name, values in matrices.items()},
                  actual_protocols=protocols, identical_prepared_inputs=inputs['encoder'],
                  source_pool_audit=source, paired_comparisons=comparisons,
                  geometry_reference={d: r['metrics'] for d, r in geometry.items()},
                  native_frames=sum(v['frames'] for v in inputs['encoder'].values()),
                  limitation='Same physical causal window/ROI/source sampling and near-matched '
                             'trainable capacity, but encoder sees current pixels while the masked '
                             'predictor does not. Feature dimensions and frozen parameters/compute '
                             'differ. Cross-readout contrasts are descriptive, not equal-information '
                             'or equal-compute causal effects. Fixed-seed development uncertainty, '
                             'no selection adjustment, cross-seed or continual-performance claim.',
                  invocation=[sys.executable, '-m', 'jepa_cmot.paired_readout_development_analysis']
                             + sys.argv[1:], analysis_code_sha256=sha256(__file__),
                  target_GT_in_training=False, sealed_used=False)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + '\n')
    print('COMPLETE_OFFICIAL_READOUT_COMPARISONS', result['equal_domain_average'], flush=True)


if __name__ == '__main__':
    main()
