"""Train identical history networks with three distinct supervised source losses."""
import argparse
import hashlib
import json
import random
import subprocess
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, default_collate
from torch.nn import functional as F

from .models import HistoryPredictor, PredictorConfig, objective
from .history import pack_history


class SourcePairs(Dataset):
    def __init__(self, path, history, videos, width, height, max_candidates=16, prediction_gaps=(1,)):
        with np.load(path, allow_pickle=False) as data:
            self.features = np.asarray(data['features'], np.float32)
            self.boxes = np.asarray(data['boxes'], np.float32)
            self.frames = np.asarray(data['frame_ids'], np.int64)
            self.videos = np.asarray(data['video_ids'], np.int64)
            self.tracks = np.asarray(data['track_ids'], np.int64)
        self.features /= np.maximum(np.linalg.norm(self.features, axis=1, keepdims=True), 1e-8)
        self.history = history
        self.scale = np.array([width, height, width, height], np.float32)
        self.max_candidates = max_candidates
        groups, by_frame = {}, {}
        for i, (video, frame, track) in enumerate(zip(self.videos, self.frames, self.tracks)):
            if int(video) not in videos:
                continue
            groups.setdefault((int(video), int(track)), []).append(i)
            by_frame.setdefault((int(video), int(frame)), []).append(i)
        self.samples = []
        self.birth_frames = {}
        for indices in groups.values():
            indices.sort(key=lambda i: int(self.frames[i]))
            self.birth_frames[(int(self.videos[indices[0]]), int(self.tracks[indices[0]]))] = int(self.frames[indices[0]])
            for position,gap in [(p,g) for p in range(1,len(indices)) for g in prediction_gaps if p>=g]:
                target = indices[position]
                end = position-gap+1
                previous = indices[max(0,end-history):end]
                candidates = by_frame[(int(self.videos[target]), int(self.frames[target]))]
                negatives = [i for i in candidates if i != target]
                if not negatives:
                    continue
                negatives.sort(key=lambda i: -float(self.features[previous[-1]] @ self.features[i]))
                if not np.all(self.frames[previous] < self.frames[target]):
                    raise ValueError('Current/future observation entered source history')
                self.samples.append((previous, target, [target] + negatives[:max_candidates - 1]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        previous, target, candidates = self.samples[index]
        feature_dim = self.features.shape[1]
        observations = [dict(feature=self.features[row],box=self.boxes[row],frame=int(self.frames[row]),
            birth=self.birth_frames[(int(self.videos[row]),int(self.tracks[row]))],confidence=1.)
            for row in previous]
        features,metadata,mask = pack_history(observations,int(self.frames[target]),
            self.history,feature_dim,self.scale[0],self.scale[1])
        candidate_features = np.zeros((self.max_candidates, feature_dim), np.float32)
        valid = np.zeros(self.max_candidates, np.bool_)
        candidate_features[:len(candidates)] = self.features[candidates]
        valid[:len(candidates)] = True
        return tuple(torch.from_numpy(x) for x in (features, metadata, mask, candidate_features, valid))


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--method', choices=['discriminative','predictive','predictive_ranking',
                                           'relation_predictive','relation_ranking'], required=True)
    parser.add_argument('--source-domain', required=True)
    parser.add_argument('--seed', type=int, default=17)
    parser.add_argument('--history', type=int, default=4)
    parser.add_argument('--hidden', type=int, default=128)
    parser.add_argument('--prediction-gaps', type=int, nargs='+', default=[1])
    parser.add_argument('--steps', type=int, default=1000)
    parser.add_argument('--batch', type=int, default=64)
    parser.add_argument('--width', type=float, default=960)
    parser.add_argument('--height', type=float, default=540)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--engineering-smoke', action='store_true')
    args = parser.parse_args()
    if min(args.prediction_gaps)<1:raise ValueError('Prediction gaps must be positive')
    torch.set_num_threads(4)
    metadata = json.loads(args.cache.with_suffix(args.cache.suffix + '.json').read_text())
    if metadata.get('source_domain') != args.source_domain or metadata.get('target_domains_read'):
        raise ValueError('Cache is not an audited source-only cache')
    if not metadata.get('gt_read'):
        raise ValueError('This entry requires source-supervised cache provenance')
    if metadata.get('engineering_smoke_only') and not args.engineering_smoke:
        raise ValueError('Engineering cache cannot become a formal research run')
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    with np.load(args.cache, allow_pickle=False) as cache:
        videos = sorted(set(int(x) for x in cache['video_ids']))
        dimension = int(cache['features'].shape[1])
    if len(videos) < 2:
        raise ValueError('Need separate fitting and validation source videos')
    # Formal source splits are supplied by audited cache subsets; this split
    # provides an additional video-disjoint source-only development check.
    heldout = videos[-max(1, int(len(videos) * .20)):]
    fitting = sorted(set(videos) - set(heldout))
    train = SourcePairs(args.cache,args.history,fitting,args.width,args.height,prediction_gaps=args.prediction_gaps)
    valid = SourcePairs(args.cache,args.history,heldout,args.width,args.height,prediction_gaps=args.prediction_gaps)
    if not len(train) or not len(valid):
        raise ValueError('No temporal source examples in one of the video-disjoint splits')
    config = PredictorConfig(feature_dim=dimension, history=args.history, hidden_dim=args.hidden)
    model = HistoryPredictor(config).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    args.out.mkdir(parents=True, exist_ok=True)
    final = args.out / 'final.pt'
    experiment = dict(config=asdict(config), method=args.method, seed=args.seed,
                      steps=args.steps, batch=args.batch, source_domain=args.source_domain,
                      width=args.width, height=args.height, cache_sha256=sha256(args.cache),
                      prediction_gaps=args.prediction_gaps,
                      engineering_smoke_only=args.engineering_smoke,
                      sampler='uniform-with-replacement-v1', geometry='shared-causal-history-v2')
    code_paths=[Path(__file__),Path(__file__).with_name('models.py'),Path(__file__).with_name('history.py')]
    experiment['code_sha256']={p.name:sha256(p) for p in code_paths}
    fingerprint = hashlib.sha256(json.dumps(experiment, sort_keys=True).encode()).hexdigest()
    report_path = args.out / 'metrics.json'
    if final.exists() and report_path.exists():
        report = json.loads(report_path.read_text())
        if report.get('experiment_fingerprint') != fingerprint:
            raise ValueError('Existing completed experiment has different configuration')
        print('TRAIN_ALREADY_COMPLETE', str(report_path), flush=True)
        return
    rng = np.random.RandomState(args.seed)
    losses = []; first_step = 0; previous_seconds = 0.
    checkpoints = sorted(args.out.glob('step_*.pt'))
    if checkpoints:
        saved = torch.load(checkpoints[-1], map_location=args.device)
        if saved['experiment_fingerprint'] != fingerprint:
            raise ValueError('Resume configuration differs from existing checkpoint')
        model.load_state_dict(saved['state_dict']); optimizer.load_state_dict(saved['optimizer'])
        rng.set_state(saved['sampler_state']); torch.set_rng_state(saved['torch_rng'].cpu())
        if args.device.startswith('cuda'):
            torch.cuda.set_rng_state_all([state.cpu() for state in saved['cuda_rng']])
        losses = saved['losses']; first_step = saved['step']; previous_seconds = saved['seconds']
        print('TRAIN_RESUME', first_step, str(checkpoints[-1]), flush=True)
    start = time.perf_counter()
    for step in range(first_step, args.steps):
        batch = default_collate([train[int(i)] for i in rng.randint(0, len(train), args.batch)])
        features, geometry, mask, candidates, candidate_mask = [x.to(args.device) for x in batch]
        query = model(features, geometry, mask)
        loss, components = objective(query, candidates,
            torch.zeros(len(query), dtype=torch.long, device=query.device), args.method,
            candidate_mask=candidate_mask)
        if not torch.isfinite(loss):raise FloatingPointError('Non-finite source training objective')
        optimizer.zero_grad(set_to_none=True);loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not torch.isfinite(norm):raise FloatingPointError('Non-finite source gradient')
        optimizer.step()
        losses.append({'step': step + 1, 'prediction': float(components['prediction']),
                       'ranking': float(components['ranking']), 'relation':float(components['relation']),
                       'gradient_norm': float(norm)})
        if (step + 1) % 100 == 0 or step + 1 == args.steps:
            print(args.method, step + 1, losses[-1], flush=True)
            torch.save({'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict(),
                        'config': asdict(config), 'step': step + 1, 'method': args.method,
                        'experiment_fingerprint': fingerprint, 'sampler_state': rng.get_state(),
                        'torch_rng': torch.get_rng_state(),
                        'cuda_rng': torch.cuda.get_rng_state_all() if args.device.startswith('cuda') else [],
                        'losses': losses, 'seconds': previous_seconds + time.perf_counter() - start},
                       args.out / f'step_{step + 1:06d}.pt')
    model.eval();hits = 0;total = 0;prediction_error = 0.
    with torch.no_grad():
        for batch in DataLoader(valid, batch_size=args.batch, num_workers=0):
            features, geometry, mask, candidates, candidate_mask = [x.to(args.device) for x in batch]
            query = model(features, geometry, mask)
            scores = torch.einsum('bd,bnd->bn', query, candidates).masked_fill(~candidate_mask, -1e4)
            hits += int((scores.argmax(1) == 0).sum());total += len(query)
            prediction_error += float((1 - F.cosine_similarity(query, candidates[:, 0])).sum())
    payload = {'state_dict': model.state_dict(), 'config': asdict(config), 'method': args.method,
               'step': args.steps, 'seed': args.seed, 'source_domain': args.source_domain,
               'source_feature_metadata': metadata,
               'training_code_sha256': experiment['code_sha256'],
               'experiment_fingerprint': fingerprint}
    torch.save(payload, final)
    report = {'status': 'complete', 'engineering_smoke_only': args.engineering_smoke,
              'method': args.method, 'config': asdict(config), 'seed': args.seed,
              'steps': args.steps, 'batch_size': args.batch,
              'prediction_gaps': args.prediction_gaps,
              'parameters': sum(p.numel() for p in model.parameters()),
              'fit_videos': fitting, 'validation_videos': heldout,
              'fit_samples': len(train), 'validation_samples': len(valid),
              'source_validation_rank1': hits / total,
              'source_validation_prediction_error': prediction_error / total,
              'loss_first': losses[0], 'loss_last': losses[-1],
              'checkpoint_sha256': sha256(final),
              'cache_sha256': experiment['cache_sha256'], 'experiment_fingerprint': fingerprint,
              'training_code_sha256': experiment['code_sha256'],
              'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=Path(__file__).parents[1]).decode().strip(),
              'seconds': previous_seconds + time.perf_counter() - start,
              'target_gt_used': False, 'candidate_input_to_history_predictor': False}
    (args.out / 'metrics.json').write_text(json.dumps(report, indent=2))
    (args.out / 'losses.json').write_text(json.dumps(losses))
    print('TRAIN_COMPLETE', json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
