"""Summarize completed native live benchmarks with precise timing scope."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    methods = ['geometry_null', 'frozen_last', 'CE_frozen', 'P_frozen', 'CE_online', 'P_online']
    datasets = {}
    for dataset, expected in [('bdd100k', 202), ('ua_detrac', 1750)]:
        cases = {}
        table = {}
        for method in methods:
            paths = list((args.root/dataset/method).glob('*/complete.json'))
            if len(paths) != 1:
                raise ValueError('Need the registered single complete video per case')
            report = json.loads(paths[0].read_text())
            if (report['frames'] != expected or report['native_frames_available'] != expected
                or not report['native_full_frames'] or report['engineering_smoke_only']
                or report['GT_read'] or report['protocol']['cached_input_used']
                or report['causal_stats']['current_history_violations']):
                raise ValueError('Live case does not satisfy full native label-free benchmark scope')
            label = method.replace('P_', 'prediction_plus_ranking_')
            if method.startswith('P_') and report['protocol']['calibration']['method'] != 'predictive_ranking':
                raise ValueError('P_* path does not refer to the registered joint objective')
            online = method.endswith('_online')
            if bool(report['actual_updates']) != online:
                raise ValueError('Actual updates disagree with benchmark method')
            if report['online_update_call_ms']['count'] != report['actual_updates']:
                raise ValueError('Update timing does not cover all actual updates')
            cases[label] = report
            table[label] = dict(frames=report['frames'], fps_after_warmup=report['fps_after_warmup'],
                end_to_end_latency_ms=report['latency_ms']['end_to_end_ms'],
                actual_updates=report['actual_updates'], update_call_ms=report['online_update_call_ms'],
                query_call_ms=report['predictor_query_call_ms'], adapter_parameters=report['adapter_parameters'],
                parent_peak_allocated_bytes=report['parent_peak_allocated_bytes'],
                detector_peak_allocated_bytes=report['detector_peak_allocated_bytes'],
                combined_peak_reserved_bytes=report['combined_peak_reserved_bytes'],
                other_gpu_jobs_observed_at_start_or_end=report['other_gpu_jobs_present'])
        streams = {v['protocol']['stream_sha256'] for v in cases.values()}
        detectors = {v['protocol']['detector_checkpoint_sha256'] for v in cases.values()}
        if len(streams) != 1 or len(detectors) != 1:
            raise ValueError('Methods used different native videos/detectors')
        datasets[dataset] = dict(cases=cases, table=table)
    result = dict(status='COMPLETE_TWELVE_FULL_VIDEO_CASES', datasets=datasets,
        scope='source-initialized isolated adaptation-video live efficiency, not adaptation accuracy',
        input='actual NAS images, private detector IPC, current official encoder and frozen matcher; no input cache',
        timing='image read/decode, detector IPC/inference, encoder, query, tracking and actual updates; weight loading excluded; warmup10 native frames in the same trajectory',
        update_timing='pseudo-pair processing, fresh/replay batching, gradients and optimizer; all updates including warmup',
        memory='per-process torch allocation peaks and sum of reservation peaks; peaks may not coincide; driver contexts and unrelated jobs excluded',
        label_note='original P_* folder labels are prediction+ranking, not prediction-only',
        reference_note='this full-video launch recipe supplied no parity reference arrays; zero reference-error counters are not evidence of bitwise parity',
        limitation='one full video per dataset, one execution per method; other jobs checked only at start/end; shared CPU/NAS/server; differences are not repeated timing statistics; graph/RKD and FP32 paired-w4 readouts not measured here',
        benchmark_checkpoints_adopted_for_research=False, target_GT_read=False, sealed_used=False)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2)+'\n')
    print('FULL_LIVE_EFFICIENCY_ANALYSIS_COMPLETE', {d: {k: round(v['fps_after_warmup'], 3)
        for k, v in values['table'].items()} for d, values in datasets.items()}, flush=True)


if __name__ == '__main__':
    main()
