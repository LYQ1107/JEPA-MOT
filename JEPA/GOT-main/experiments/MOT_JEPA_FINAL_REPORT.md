# MOT + GOT-JEPA 最终审计报告

日期：2026-08-31。本文是独立的最终汇总，不覆盖此前的审计文档。可复核的机器结果见
[final_cross_protocol_summary.json](/data1/LWR/vranlee/SERVER_ONLY/avis/JEPA/GOT-main/experiments/final_cross_protocol_summary.json)
和
[final_cross_protocol_summary.csv](/data1/LWR/vranlee/SERVER_ONLY/avis/JEPA/GOT-main/experiments/final_cross_protocol_summary.csv)。

## 结论先行

JEPA 最适合放在 MOT 的“已有轨迹重新出现（reappearance）/遮挡恢复”关联分支，作为
短时历史到当前外观的预测成本；不应无条件替换连续可见帧的外观，也不应改检测器或
轨迹生命周期。严格的 GOT-JEPA 256-D pooled 特征实验给出了以下结论：

1. 无条件 temporal replacement 在 DanceTrack 上是 **NO-GO**，HOTA 和 IDF1 都下降。
2. 只做 ambiguity gate 是 **NO-OP**（评估集没有实际应用事件，输出与 raw 完全相同）。
3. lifecycle-invariant 的 reappearance-only 版本在 120 帧、两个固定 seed 上均提高
   HOTA mean 与 IDF1；这是目前最有价值的正反馈，但幅度很小。
4. 延长到 300 帧后 seed17 仍略正、seed23 的 HOTA mean 变负，故不能宣称完整序列
   的稳健提升；已按预注册规则停止继续扫阈值。
5. `no_teacher` 在若干设置与 JEPA 相当或更好，因此目前不能把增益归因于 teacher
   的 JEPA 预测本身。

## 1. 数据、基线与比较边界

### 数据和评估协议

- MOT17：固定公开 FRCNN `det.txt`，五个序列
  `02/04/09/05/11-FRCNN`，120 帧前缀；GT 只在输出写完后交给官方 TrackEval。
- DanceTrack：训练序列 `dancetrack0001,0002,0006`，评估序列
  `dancetrack0004,0007,0014`；分别做 120 帧和 300 帧前缀。
- 指标均来自 vendored 官方 TrackEval：`HOTAmean`、`HOTA@0.5`、`AssA@0.5`、
  `DetA@0.5`、`IDF1`、`MOTA`、`IDSW`。表中 `↑/↓` 表示越大/越小越好。
- 所有 JEPA tracker 都只读取检测行（无检测 ID），`eval_gt_read_by_tracker=false`；
  评估 GT 不参与阈值选择、轨迹修补或 predictor 输入。

### 基线与 harness 边界（必须分开解释）

正式的 MOT17 baseline 是 **OC-SORT public-detection**，结果如下。它是五序列的
formal public-det sanity run：

| 方法 | HOTAmean | HOTA@0.5 | AssA | DetA | IDF1 | MOTA | IDSW |
|---|---:|---:|---:|---:|---:|---:|---:|
| OC-SORT | 0.600354 | 0.647286 | 0.829144 | 0.505315 | 0.656208 | 0.503406 | 16 |

结果文件：[OC-SORT TrackEval CSV](/data1/LWR/vranlee/SERVER_ONLY/avis/JEPA/GOT-main/experiments/results/baseline_ocsort_mot120/trackeval_official/official_metrics.csv)。

官方 GOT-JEPA 256-D 的 DanceTrack proof-of-concept 使用的是隔离的
**SimpleMOT/Hungarian association harness**；MOT17 JEPA gate sweep 也是隔离的
association diagnostic，并不是把 predictor 已无缝插入 OC-SORT 的 leaderboard。因此
下面的 JEPA 数值只做“同一 harness 内相对 raw”的比较，不能把它们与 OC-SORT 的绝对
分数直接比较。下一步才是将 adapter 移植回 OC-SORT/BoT-SORT 后跑完整 benchmark。

## 2. JEPA 如何接入 MOT，以及是否需要训练

当前实现遵循 MOT 特有的遮挡属性：

```text
detector box_t + track history (最近 4 个 appearance/box/速度/age/score/missing)
                 └─> temporal predictor ──> 预测的 track token
当前检测与预测 token 仅在 reappearance 关联成本中进入 Hungarian cost
```

- 官方 GOT-JEPA encoder/head 冻结，输出 `[256,27,27]`，空间平均并 L2-normalize
  成 256-D；checkpoint 为
  `/data3/testdata/vranlee/GOT_JEPA_checkpoints/GOT_JEPA_378.tar`，SHA-256
  `5800f0fdc748708431743c5520e960bb6e45f4a786dacc0dbbc9107c8ae51a98`。
- **需要训练的是 MOT temporal adapter，不是重新训练整个 JEPA encoder**：每个协议用
  训练序列拟合 8 epochs（120 帧 2,841 个 temporal pairs；300 帧 7,085 个 pairs）。
- clean teacher 只在训练序列使用 GT 框生成目标；student 的输入不含 GT 框，评估序列
  完全不读 GT。`no_teacher` 和 `random_frozen` 是必要控制。
- 固定 task-semantic gate：`1 <= misses == frame_gap <= 4`、quality `>=0.35`、
  `sigmoid(uncertainty) <=0.65`。连续可见帧保持 raw appearance+geometry；禁止
  post-hoc track repair。该 invariant 由 preflight/event/lifecycle trace 审计。

## 3. MOT17：JEPA gate sweep（负面/无效证据）

这是 `MOT17-05-FRCNN, MOT17-11-FRCNN` 的 120 帧 held-out association diagnostic，
不是 OC-SORT 结果。A 为原始宽松 gate，B/C/D 为预注册的更严格 gate。

| 方法 | HOTAmean | HOTA@0.5 | AssA | DetA | IDF1 | MOTA | IDSW |
|---|---:|---:|---:|---:|---:|---:|---:|
| raw | 0.517456 | 0.571340 | 0.765677 | 0.426328 | 0.566335 | 0.421822 | 10 |
| A | 0.507571 | 0.560755 | 0.737568 | 0.426328 | 0.547457 | 0.421822 | 10 |
| B | 0.517456 | 0.571340 | 0.765677 | 0.426328 | 0.566335 | 0.421822 | 10 |
| C | 0.517456 | 0.571340 | 0.765677 | 0.426328 | 0.566335 | 0.421822 | 10 |
| D | 0.517456 | 0.571340 | 0.765677 | 0.426328 | 0.566335 | 0.421822 | 10 |

相对 raw 的完整 delta（顺序仍为 HOTAmean / HOTA@0.5 / AssA / DetA / IDF1 /
MOTA / IDSW）为：

| 方法 | ΔHOTAmean | ΔHOTA@0.5 | ΔAssA | ΔDetA | ΔIDF1 | ΔMOTA | ΔIDSW |
|---|---:|---:|---:|---:|---:|---:|---:|
| A | -0.009885 | -0.010585 | -0.028109 | 0.000000 | -0.018878 | 0.000000 | 0 |
| B | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0 |
| C | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0 |
| D | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0 |

机器产物：[gate sweep results](/data2/usr_for_deadline/mot_jepa_adapter_v4_gate_sweep_120/results.json)、
[official audit](/data2/usr_for_deadline/mot_jepa_adapter_v4_gate_sweep_120/trackeval_official/audit.json)。

## 4. DanceTrack：严格 GOT-JEPA 256-D 结果

### 4.1 无条件 temporal replacement：NO-GO

| 方法 | HOTAmean | HOTA@0.5 | AssA | DetA | IDF1 | MOTA | IDSW |
|---|---:|---:|---:|---:|---:|---:|---:|
| raw | 0.727609 | 0.783502 | 0.708034 | 0.867014 | 0.789288 | 0.904367 | 33 |
| jepa_temporal | 0.708561 | 0.762432 | 0.671468 | 0.865718 | 0.770690 | 0.903991 | 34 |
| no_teacher | 0.709049 | 0.762940 | 0.672363 | 0.865718 | 0.770690 | 0.903614 | 35 |
| random_frozen | 0.705667 | 0.775518 | 0.657380 | 0.914886 | 0.762135 | 0.906627 | 35 |

`jepa_temporal - raw = (-0.019047, -0.021070, -0.036566, -0.001296,
-0.018598, -0.000377, +1)`。容易帧被无条件 predictor 扰动，故不采用。

产物：[results](/data2/usr_for_deadline/mot_dance_jepa_got_full120_s17/results.json)、
[GOT protocol audit](/data2/usr_for_deadline/mot_dance_jepa_got_full120_s17/got_protocol_audit.json)、
[TrackEval CSV](/data2/usr_for_deadline/mot_dance_jepa_got_full120_s17/trackeval_official/official_metrics.csv)。

### 4.2 ambiguity-only gate：NO-OP

| 方法 | HOTAmean | HOTA@0.5 | AssA | DetA | IDF1 | MOTA | IDSW |
|---|---:|---:|---:|---:|---:|---:|---:|
| raw | 0.727609 | 0.783502 | 0.708034 | 0.867014 | 0.789288 | 0.904367 | 33 |
| ambiguity_jepa | 0.727609 | 0.783502 | 0.708034 | 0.867014 | 0.789288 | 0.904367 | 33 |
| no_teacher | 0.727609 | 0.783502 | 0.708034 | 0.867014 | 0.789288 | 0.904367 | 33 |
| random_frozen | 0.728215 | 0.783722 | 0.708431 | 0.867014 | 0.789288 | 0.910392 | 31 |

`ambiguity_jepa` 的七项 delta 全为 0；评估 eligible 事件虽被统计，但实际应用事件为
0、改变为 0。该分支没有提供 JEPA 增益证据。

### 4.3 lifecycle-invariant reappearance-only：120 帧（正反馈边界）

下表是官方 TrackEval 的 combined row；每个 seed 的 raw 相同，candidate 只改变
允许的 reappearance 事件。

| seed | 方法 | HOTAmean | HOTA@0.5 | AssA | DetA | IDF1 | MOTA | IDSW |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 17 | raw | 0.727609 | 0.783502 | 0.708034 | 0.867014 | 0.789288 | 0.904367 | 33 |
| 17 | reappearance_jepa | 0.730484 | 0.799334 | 0.707812 | 0.902689 | 0.796727 | 0.904367 | 33 |
| 17 | no_teacher | 0.731840 | 0.801543 | 0.710670 | 0.904037 | 0.797843 | 0.903991 | 34 |
| 17 | random_frozen | 0.713345 | 0.780905 | 0.676056 | 0.902016 | 0.760647 | 0.902485 | 38 |
| 23 | raw | 0.727609 | 0.783502 | 0.708034 | 0.867014 | 0.789288 | 0.904367 | 33 |
| 23 | reappearance_jepa | 0.731840 | 0.801543 | 0.710670 | 0.904037 | 0.797843 | 0.903991 | 34 |
| 23 | no_teacher | 0.731840 | 0.801543 | 0.710670 | 0.904037 | 0.797843 | 0.903991 | 34 |
| 23 | random_frozen | 0.739697 | 0.814554 | 0.727934 | 0.911482 | 0.804166 | 0.903614 | 35 |

`reappearance_jepa - raw` 的完整 delta：

| seed | ΔHOTAmean | ΔHOTA@0.5 | ΔAssA | ΔDetA | ΔIDF1 | ΔMOTA | ΔIDSW |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 17 | +0.002875 | +0.015832 | -0.000222 | +0.035675 | +0.007439 | 0.000000 | 0 |
| 23 | +0.004231 | +0.018041 | +0.002636 | +0.037023 | +0.008555 | -0.000377 | +1 |
| **paired mean ± population std** | **+0.003553 ± 0.000678** | **+0.016937 ± 0.001105** | **+0.001207 ± 0.001429** | **+0.036349 ± 0.000674** | **+0.007997 ± 0.000558** | **-0.000188 ± 0.000188** | **+0.5 ± 0.5** |

因此 120 帧是“可以继续工程化验证”的边界结果，而不是完整 benchmark 的最终
结论。审计产物：[seed17](/data2/usr_for_deadline/mot_dance_jepa_got_reappearance120_s17_invariant)、
[seed23](/data2/usr_for_deadline/mot_dance_jepa_got_reappearance120_s23_invariant)、
[seed-pair audit](/data2/usr_for_deadline/mot_dance_jepa_got_reappearance120_seedpair_invariant/seed_pair_audit.json)。

### 4.4 lifecycle-invariant reappearance-only：300 帧（混合，停止扩展）

| seed | 方法 | HOTAmean | HOTA@0.5 | AssA | DetA | IDF1 | MOTA | IDSW |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 17 | raw | 0.720856 | 0.766523 | 0.697229 | 0.842704 | 0.766873 | 0.923779 | 71 |
| 17 | reappearance_jepa | 0.721063 | 0.770408 | 0.698567 | 0.849637 | 0.779659 | 0.923632 | 72 |
| 17 | no_teacher | 0.716516 | 0.764034 | 0.691088 | 0.844680 | 0.772539 | 0.922454 | 80 |
| 17 | random_frozen | 0.707063 | 0.764110 | 0.655823 | 0.890277 | 0.759753 | 0.921572 | 86 |
| 23 | raw | 0.720856 | 0.766523 | 0.697229 | 0.842704 | 0.766873 | 0.923779 | 71 |
| 23 | reappearance_jepa | 0.716264 | 0.762975 | 0.690385 | 0.843198 | 0.775736 | 0.922896 | 77 |
| 23 | no_teacher | 0.716767 | 0.764357 | 0.690055 | 0.846660 | 0.776026 | 0.922160 | 82 |
| 23 | random_frozen | 0.749113 | 0.812431 | 0.740958 | 0.890797 | 0.812060 | 0.921572 | 86 |

candidate delta：

| seed | ΔHOTAmean | ΔHOTA@0.5 | ΔAssA | ΔDetA | ΔIDF1 | ΔMOTA | ΔIDSW |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 17 | +0.000207 | +0.003885 | +0.001338 | +0.006933 | +0.012786 | -0.000147 | +1 |
| 23 | -0.004592 | -0.003548 | -0.006844 | +0.000493 | +0.008863 | -0.000883 | +6 |
| **paired mean ± population std** | **-0.002193 ± 0.002399** | **+0.000168 ± 0.003716** | **-0.002753 ± 0.004091** | **+0.003713 ± 0.003220** | **+0.010825 ± 0.001961** | **-0.000515 ± 0.000368** | **+3.5 ± 2.5** |

预注册的 primary rule 是两个 seed 都同时有正 HOTA mean 和正 IDF1；300 帧只有
seed17 满足，seed23 不满足。因此结论为 **mixed / stop expansion**，保留 120 帧
作为边界证据，而不是继续做阈值 sweep。审计产物：[300 seed-pair audit](/data2/usr_for_deadline/mot_dance_jepa_got_reappearance300_seedpair_invariant/seed_pair_audit.json)。

## 5. 反思、限制与下一步

- 120 帧的正反馈主要来自 DetA 和 IDF1，HOTA mean 增幅约 0.0036；同时存在很小
  的 MOTA/IDSW 代价，不能包装成大幅提升。
- `no_teacher` 在 120 帧略高于 JEPA candidate，300 帧也没有稳定的 teacher 优势；
  必须保留该 control，当前不能声称“收益由 teacher/JEPA 独有”。
- random control 在 seed23 的 300 帧异常偏高，说明短 prefix/随机性方差不可忽略。
- 旧的 crop80 temporal 实验（例如
  `/data2/usr_for_deadline/real_dance_temporal_repro_300_s17`）属于不同输入表示和
  脚本路径；它们可以作为历史 provisional 证据，但**不能**与本报告的严格 256-D
  结果合并或计算总平均。
- 工程下一步：将 invariant reappearance adapter 移植到 OC-SORT 或 BoT-SORT 的
  public-detection association 代码，保持检测器和 lifecycle 不变，在完整 MOT17 /
  DanceTrack 序列上用两个以上 seed 重跑；随后再考虑 spatial-token/track-level
  pair filter，而不是继续事后调 gate 阈值。

## 6. 审计与可复现性状态

下列最终产物均已完成并可复核：

| 范围 | 关键审计状态 |
|---|---|
| OC-SORT MOT17 baseline | `results.json=complete`，官方 TrackEval `complete` |
| MOT17 v4 gate sweep | `results.json=complete`，官方 TrackEval `complete`，source stress gate=`pass` |
| DanceTrack unconditional | GOT protocol `all_pass=true`，官方 TrackEval `complete` |
| DanceTrack ambiguity | event audit=`complete`，官方 TrackEval `complete` |
| DanceTrack reappearance 120/300（四个 seed run） | preflight=`complete/pass`，protocol=`pass`，官方 TrackEval=`complete` |
| 120/300 seed-pair aggregation | `seed_pair_audit.status=pass` |
| 300 帧 cache | `cache_manifest.status=complete`；manifest SHA-256 `24f794bf28a335b711136ffaf35bfdaf8e073f1d00ffe6f879678aebcf8c9a65` |

这不等于工作区内“每一个历史 JSON”都显示 pass/complete：按
`/data1/LWR/vranlee/SERVER_ONLY/avis/JEPA/AGENTS.md` 的保留失败证据规则，旧的
`failed_attempts/*.json`（包括 pre-fix/lifecycle 错误和环境失败）仍明确标为
`status=failure`，没有删除或覆盖；它们不属于最终有效指标。最终汇总脚本和报告只
引用上述通过审计的 artifact，并保留 no-go/mixed 反思状态。
