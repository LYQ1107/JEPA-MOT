"""Compare actual domain-boundary replay state across fixed-teacher runs."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

import torch

from .online import OnlineConfig
from .train_source import sha256


SAMPLE_FIELDS = [0, 1, 2, 3, 4, 7, 8]
SUPPLY_STATS = ['proposed', 'delayed_confirmed', 'reliability_rejected',
                'need_rejected', 'video_resets', 'updates']


def normalized_control(control):
    config = asdict(OnlineConfig(**control['online_config']))
    config.pop('retention')
    return dict(method=control['method'], online_config=config,
                geometry_duplicates=control['geometry_duplicates'],
                checkpoint_sha256=sha256(control['checkpoint']),
                calibration_sha256=sha256(control['calibration']))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sequences', type=Path, nargs='+', required=True)
    parser.add_argument('--stages', type=int, nargs='+', default=[1, 2, 3])
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    runs = []
    common = None
    source = None
    for sequence in args.sequences:
        protocol = json.loads((sequence / 'protocol.json').read_text())
        assert protocol['actual_update_counts_matched_each_frame']
        assert protocol['target_GT_in_training'] is False
        assert protocol['sealed_test_used'] is False
        scope = dict(dataset=protocol['dataset'], order=protocol['order'],
                     teacher_calibration_sha256=sha256(protocol['fixed_teacher_calibration']))
        if common is None:
            common = scope
        assert scope == common, 'Different dataset/order/fixed teacher'
        for control in protocol['controls']:
            normalized = normalized_control(control)
            if source is None:
                source = normalized
            assert normalized == source, 'More than retention differs in control settings'
        runs.append((sequence, protocol, json.loads((sequence / 'snapshots.json').read_text())))
    stages = []
    for stage in args.stages:
        reference = None
        cases = []
        for sequence, protocol, snapshots in runs:
            for control, checkpoint in snapshots[stage].items():
                state = torch.load(checkpoint, map_location='cpu')
                assert state['physical_video_complete'] and not state['pending']
                stats = {k: v for k, v in state['stats'].items() if k != 'losses'}
                counters = {k: state[k] for k in ['frames', 'updates', 'seen']}
                steps = sorted({int(s['step']) for s in state['optimizer']['state'].values()})
                assert steps == [state['updates']], 'Optimizer and recorded updates differ'
                if reference is None:
                    reference = state
                assert state['config'] == reference['config']
                assert counters == {k: reference[k] for k in counters}
                assert {k: stats[k] for k in SUPPLY_STATS} == \
                    {k: reference['stats'][k] for k in SUPPLY_STATS}
                assert state['sampler_state'] == reference['sampler_state'], 'Replay RNG differs'
                for pool in ['memory', 'recent']:
                    assert len(state[pool]) == len(reference[pool])
                    for actual, expected in zip(state[pool], reference[pool]):
                        assert len(actual) == len(expected) == 9
                        assert all(torch.equal(actual[k], expected[k]) for k in SAMPLE_FIELDS), \
                            'Retained physical sample or ordering differs'
                cases.append(dict(sequence=str(sequence), control=control, checkpoint=checkpoint,
                                  checkpoint_sha256=sha256(checkpoint), **counters,
                                  optimizer_steps=steps, stats=stats,
                                  memory=len(state['memory']), recent=len(state['recent']),
                                  sample_fields_and_order_equal=True, replay_rng_equal=True))
        stages.append(dict(stage=stage, controls=cases))
    result = dict(status='VERIFIED_ACTUAL_DOMAIN_BOUNDARY_PARITY_NOT_MOT_RESULT',
                  scope=common, common_control_except_retention=source,
                  compared_sample_fields=SAMPLE_FIELDS,
                  compared_supply_stats=SUPPLY_STATS,
                  excluded_sample_fields={5: 'learner-dependent old candidate response',
                                          6: 'learner-dependent old predicted query'},
                  limitation='Compares stored boundary counters, optimizer steps, RNG, ordered '
                             'reservoir and remaining recent samples. Does not log every update '
                             'batch or prove per-frame parity between separate runs.',
                  target_GT_read=False, stages=stages,
                  invocation=[sys.executable, '-m', 'jepa_cmot.audit_matched_replay_buffers']
                             + sys.argv[1:],
                  protocols={str(s): sha256(s / 'protocol.json') for s in args.sequences},
                  audit_code_sha256=sha256(__file__))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + '\n')
    print('BOUNDARY_REPLAY_PARITY', [(s['stage'], len(s['controls']),
                                     s['controls'][0]['updates']) for s in stages], flush=True)


if __name__ == '__main__':
    main()
