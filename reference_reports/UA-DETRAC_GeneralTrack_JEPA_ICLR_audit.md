# UA-DETRAC GeneralTrack + GOT-JEPA 实验审计报告

**日期：** 2026-09-01  
**状态：** 三个 UA-DETRAC 目标域均已完成全量评测；最佳固定组合为 `GeneralTrack + raw center3 GOT-JEPA high_fallback(.90) + GT-free quality filter`。  
**范围：** 单 agent 执行；未开启子 agent；只使用 GPU 1/2/3，未占用 GPU 0 及 4–7 上的既有任务。

## 1. 结论摘要

当前最可靠的 JEPA 方案不是全局替换 GeneralTrack 的关联代价，而是：

1. 继续使用 GeneralTrack 原始 detector、Point2InstanceRelation high-score association、Kalman filter 和 ByteTrack lifecycle。
2. 对 native high-score assignment 后双方仍未匹配的 track/detection，使用 frozen GOT-JEPA center3 特征做恢复；要求 JEPA cosine >= 0.90、预测框 IoU >= 0.05、时间 gap <= 8、候选相对 margin >= 0.03。
3. 只对输出中的短 tracklet 做 GT-free quality filter，保留长度 >= 5 的轨迹；短轨迹还需满足 detector score、相邻 JEPA similarity 和局部运动约束。

该组合的 HOTA 与 MOTA 在 cloudy、night、rainy 三个域均超过 GeneralTrack baseline。HOTA 提升分别为 **+0.4902、+1.8703、+1.2306** 个百分点；MOTA 提升分别为 **+5.2208、+5.3300、+2.5693** 个百分点。三域平均 HOTA 从 33.5983 提升到 34.7953（+1.1970），平均 MOTA 从 18.7448 提升到 23.1181（+4.3733）。

这已经是可复现的正向结果，但还不能声称达到了 ICLR 级别的最终 SOTA：目标域 detector 的 DetA 没有提升，质量过滤通过牺牲部分 FN 换取了大量 FP/IDSW 下降；`.85` floor 版本还使用了首视频探索，必须作为 target-tuned exploratory 结果而不是严格主结果。

## 2. Baseline、训练时间和 JEPA 插入位置

### 2.1 Baseline 是什么

本实验复用的 baseline 是：

- 代码：[`GeneralTrack-main`](</data1/LWR/vranlee/SERVER_ONLY/avis/GeneralTrack-main>)。
- checkpoint：[`generaltrack_bdd.pth`](</data1/LWR/vranlee/SERVER_ONLY/avis/GeneralTrack-main/checkpoints/generaltrack_bdd.pth)，SHA256 `4a0243876a69c7eaccede582817cfb0701b49962d1abe03087a77fef27d384dd`。
- high-score association：GeneralTrack 的 `Point2InstanceRelation` learned relation distance + score fusion。
- lost-track association：原始 ByteTrack IoU distance + score fusion。
- lifecycle：`track_buffer=50`、`match_thresh=0.9`、`track_thresh=0.35`、`det_thresh=0.45`、resize `720x1280`。
- detector 输入：已有的 `source_detector_thr02`，不是 GT；来源为 QDTrack Faster R-CNN R50 sunny 20e 的固定 detector stream，target rows 只作为候选检测。
- baseline prior output：[`outputs/generaltrack_thr02`](</data1/LWR/vranlee/SERVER_ONLY/avis/GeneralTrack-main/outputs/generaltrack_thr02>)。

GeneralTrack 本身确实使用 ByteTrack：[`byte_tracker_bdd.py`](</data1/LWR/vranlee/SERVER_ONLY/avis/GeneralTrack-main/yolox/tracker/byte_tracker_bdd.py) 是原始 tracker 实现，`mot_evaluator.py` 在非 JEPA 模式下实例化它。

### 2.2 训练/提取多久

GeneralTrack baseline checkpoint 在本任务中**没有重新训练**，直接复用已有 checkpoint 和 prior output。GOT-JEPA 也没有重新训练，复用官方 checkpoint [`GOT_JEPA_378.tar`](</data3/testdata/vranlee/GOT_JEPA_checkpoints/GOT_JEPA_378.tar>)，SHA256 `5800f0fdc748708431743c5520e960bb6e45f4a786dacc0dbbc9107c8ae51a98`；该 checkpoint 为 epoch 35 的 `ToMPnet_PT`，输出 head map 为 `256x27x27`。

主要耗时如下：

| 阶段 | cloudy | night | rainy | 说明 |
|---|---:|---:|---:|---|
| GOT-JEPA cache extraction | 6115.2 s | 2647.6 s | 3509.3 s | 每个域独立提取，center3 predictor，context factor 5 |
| high_fallback tracker wall time | 665.9 s | 577.4 s | 634.4 s | 每域一张 GPU，包含 DataLoader/写结果 |
| target-unlabeled adapter training | 6.5 s | 12.6 s | 24.9 s | 20 epochs，CPU；只作探索，不是最终主线 |
| source pair adapter training | 34.9 s | — | — | sunny source，30 epochs，CPU；target runtime 失效 |

Cache 规模为 cloudy `5060 images / 96498 detections`、night `5015 / 43671`、rainy `4999 / 56894`。完整 cache sidecar 分别为 [`cloudy_center3.npz.json`](</data1/LWR/vranlee/SERVER_ONLY/avis/JEPA/outputs/ua_detrac_jepa/cache/cloudy_center3.npz.json>)、[`night_center3.npz.json`](</data1/LWR/vranlee/SERVER_ONLY/avis/JEPA/outputs/ua_detrac_jepa/cache/night_center3.npz.json>)、[`rainy_center3.npz.json`](</data1/LWR/vranlee/SERVER_ONLY/avis/JEPA/outputs/ua_detrac_jepa/cache/rainy_center3.npz.json>)。

### 2.3 JEPA 是怎么插入的

实现位于 [`byte_tracker_bdd_jepa.py`](</data1/LWR/vranlee/SERVER_ONLY/avis/GeneralTrack-main/yolox/tracker/byte_tracker_bdd_jepa.py>) 和 [`track_damot_domain.py`](</data1/LWR/vranlee/SERVER_ONLY/avis/GeneralTrack-main/tools/track_damot_domain.py)：

- cache builder [`build_generaltrack_jepa_cache.py`](</data1/LWR/vranlee/SERVER_ONLY/avis/JEPA/GOT-main/experiments/build_generaltrack_jepa_cache.py) 对每个固定 detector row crop image，运行 frozen official GOT-JEPA，取 predictor map 的中心 `3x3` 空间池化并 L2 normalize，得到 `[N,256]`。
- `PreparedGeneralTrackDataset` 按 `image_id` 和 detector row 顺序把 256-D vector 追加到原始六列 `[x1,y1,x2,y2,score,class]` 后面。
- tracker 将它挂到 `STrack`/detection 上，只把被 native high/lost association 接受的 high-score detection 特征写入 causal history；low-score detection 不写入，避免污染 query。
- `high_fallback` 先完整运行原始 GeneralTrack high assignment；JEPA 只处理 native unmatched track 与 unmatched detection，且不能抢走已匹配 detection、不能创建额外 track、不能改变 Kalman/lifecycle。
- 质量过滤是 tracker 输出后的独立 GT-free 后处理，不修改 detector box，只删除低质量短轨迹。

因此，这不是把 GT、目标 ID 或未来帧送入 JEPA，也不是重新训练 GeneralTrack relation model。

## 3. 完整 UA-DETRAC 结果

### 3.1 GeneralTrack baseline

| target domain | videos | frames | HOTA | DetA | AssA | MOTA | IDF1 | IDSW | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| cloudy | 30 | 5060 | 35.2668 | 31.8860 | 39.5842 | 15.8500 | 48.6058 | 3820 | 18768 | 20416 |
| night | 28 | 5015 | 34.5917 | 30.5086 | 39.7572 | 21.3725 | 46.0185 | 1697 | 8093 | 15657 |
| rainy | 19 | 4999 | 30.9364 | 25.0934 | 39.2198 | 19.0118 | 39.1974 | 2080 | 10072 | 37842 |

原始 metrics 文件：[`cloudy metrics`](</data1/LWR/vranlee/SERVER_ONLY/avis/GeneralTrack-main/outputs/generaltrack_thr02/ua_detrac/cloudy/metrics.json>)、[`night metrics`](</data1/LWR/vranlee/SERVER_ONLY/avis/GeneralTrack-main/outputs/generaltrack_thr02/ua_detrac/night/metrics.json>)、[`rainy metrics`](</data1/LWR/vranlee/SERVER_ONLY/avis/GeneralTrack-main/outputs/generaltrack_thr02/ua_detrac/rainy/metrics.json>)。

### 3.2 当前主候选：raw center3 high_fallback `.90` + GT-free quality filter

| target domain | HOTA | ΔHOTA | DetA | AssA | MOTA | ΔMOTA | IDF1 | IDSW | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| cloudy | 35.7570 | +0.4902 | 31.4347 | 41.2222 | 21.0708 | +5.2208 | 49.7106 | 2594 | 15451 | 22291 |
| night | 36.4620 | +1.8703 | 30.3047 | 44.4476 | 26.7025 | +5.3300 | 48.6432 | 873 | 6349 | 16500 |
| rainy | 32.1670 | +1.2306 | 24.7776 | 42.8097 | 21.5811 | +2.5693 | 41.3226 | 1077 | 8767 | 38564 |

结果文件：[`cloudy highfb+filter`](</data1/LWR/vranlee/SERVER_ONLY/avis/JEPA/outputs/ua_detrac_jepa/eval/cloudy_highfb090_quality_filter/metrics.json>)、[`night highfb+filter`](</data1/LWR/vranlee/SERVER_ONLY/avis/JEPA/outputs/ua_detrac_jepa/eval/night_highfb090_quality_filter/metrics.json>)、[`rainy highfb+filter`](</data1/LWR/vranlee/SERVER_ONLY/avis/JEPA/outputs/ua_detrac_jepa/eval/rainy_highfb090_quality_filter/metrics.json>)。

与 baseline 相比，组合结果的 FN 增加：cloudy +1875、night +843、rainy +722；这是 filter 通过删除大量 FP 换取 MOTA/AssA/IDSW 的代价，必须在论文中如实报告，不能只报 HOTA/MOTA。

### 3.3 high_fallback tracker 本身（不含质量过滤）

| target domain | HOTA | ΔHOTA | DetA | AssA | MOTA | ΔMOTA | IDF1 | IDSW | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| cloudy | 35.6984 | +0.4316 | 31.8486 | 40.6055 | 16.2336 | +0.3836 | 49.2336 | 3606 | 18800 | 20402 |
| night | 36.0758 | +1.4841 | 30.5271 | 43.2275 | 22.8371 | +1.4646 | 48.1040 | 1259 | 8079 | 15635 |
| rainy | 32.1652 | +1.2288 | 25.0702 | 42.3146 | 20.0356 | +1.0238 | 41.3375 | 1436 | 10083 | 37843 |

结果文件：[`cloudy high_fallback`](</data1/LWR/vranlee/SERVER_ONLY/avis/JEPA/outputs/ua_detrac_jepa/eval/cloudy_high_fallback_iou005_full/metrics.json>)、[`night high_fallback`](</data1/LWR/vranlee/SERVER_ONLY/avis/JEPA/outputs/ua_detrac_jepa/eval/night_high_fallback_iou005_full/metrics.json>)、[`rainy high_fallback`](</data1/LWR/vranlee/SERVER_ONLY/avis/JEPA/outputs/ua_detrac_jepa/eval/rainy_high_fallback_iou005_full/metrics.json>)。

它共恢复 cloudy/night/rainy 的 523/605/811 个 high-score unmatched pair；rejection count 为 35/44/61；所有 full runs 的 missing feature pair count 均为 0。详细诊断分别在各输出的 `track_results/tracker_diagnostics.json`。

### 3.4 其他 full-domain 对照

下面列出本轮所有重要的三域全量对照；target-tuned 或 transductive 结果不与严格主结果混写。

| 方法 | cloudy HOTA/MOTA | night HOTA/MOTA | rainy HOTA/MOTA | 备注 |
|---|---|---|---|---|
| raw consensus `.90` | 35.7728 / 17.3157 | 35.1117 / 22.6672 | 30.9652 / 19.5723 | 只 veto native lost match；三域 HOTA 均不负 |
| raw consensus `.96 + gap3` | 35.8151 / 17.9438 | — | 30.8740 / 19.9158 | cloudy/rainy full；night 未重复该组合 |
| target-unlabeled adapter + consensus | 35.8603 / 17.9908 | 35.2341 / 23.5231 | 30.6802 / 19.8834 | transductive；rainy HOTA 回落 |
| high_fallback `.90` | 35.6984 / 16.2336 | 36.0758 / 22.8371 | 32.1652 / 20.0356 | 当前 tracker 主候选 |
| high_fallback `.85` | 35.7223 / 16.2649 | 35.9134 / 23.0225 | 32.2884 / 20.0486 | 首视频探索后选择，target-tuned |
| high_fallback `.90` + quality filter | 35.7570 / 21.0708 | 36.4620 / 26.7025 | 32.1670 / 21.5811 | 当前固定组合 |
| high_fallback `.85` + quality filter | 35.7747 / 21.1021 | 36.2899 / 26.8477 | 32.2723 / 21.5195 | target-tuned exploratory |
| no-JEPA threshold `.40/.45` | 35.6893 / 19.3938 | 35.1494 / 24.3388 | 30.8513 / 20.3823 | target-tuned exploratory，无 JEPA |
| baseline tracklet length >=5 | 35.2679 / 20.8986 | 34.8883 / 25.2781 | 30.9208 / 20.7387 | GT-free output filter diagnostic |
| quality filter default on baseline | 35.3460 / 20.8868 | 34.9193 / 25.3244 | 30.9251 / 20.6383 | 只作 lifecycle/FP 对照 |
| high_fallback `.90` + `min_long=3` | 35.7965 / 17.7559 | 36.2489 / 24.3295 | 32.2188 / 20.6237 | filter sensitivity，整体低于 min_long=5 |
| tracklet relink `.96/g50/c6` | 35.0417 / 15.9087 | 34.5443 / 21.4374 | 30.9646 / 19.0701 | merge 不稳定，未采用 |

完整文件：[`raw/adapter/threshold/filter 汇总目录`](</data1/LWR/vranlee/SERVER_ONLY/avis/JEPA/outputs/ua_detrac_jepa/eval>)。

## 4. 首视频实验与失败证据

首视频为 cloudy 的 `MVI_39501`，80 frames；baseline 为 HOTA 22.8197、MOTA -8.5580、IDF1 32.4022、IDSW 18、FP 317、FN 591。首视频只用于机制筛选和回归，完整域指标才是主要结论。

| 变体 | HOTA | MOTA | AssA | IDF1 | IDSW | 结论 |
|---|---:|---:|---:|---:|---:|---|
| raw lost consensus `.90` | 21.7813 | -8.0891 | 29.1715 | 30.7153 | 20 | lost veto 误伤，未采用 |
| raw lost consensus `.96 + gap3` | 22.5408 | -7.8546 | 31.1673 | 32.4022 | 20 | 较稳但不如 high fallback |
| high_fallback `.90`, IoU=.05 | 23.6698 | -6.9168 | 34.1593 | 33.3801 | 12 | 首个明确正向机制 |
| high_fallback `.85`, margin=.03 | 24.2571 | -6.7995 | 35.7588 | 33.9411 | 11 | 更强但 floor 来自首视频探索 |
| high_fallback `.90`, IoU=.20 | 23.4855 | -7.5029 | 33.9361 | 33.0301 | 12 | IoU gate 过强 |
| high_fallback + lost consensus | 23.3479 | -6.9168 | 33.1623 | 32.3261 | 15 | 组合不如单独 high fallback |
| high_fallback + lost `.96/gap3` | 23.3766 | -6.4478 | 33.1686 | 32.3944 | 14 | 仍不如单独 high fallback |
| center5 lost consensus | 22.5607 | -8.6753 | 31.3831 | 31.5863 | 20 | center3 更好 |
| center7 lost consensus | 22.7837 | -8.7925 | 32.0519 | 32.2401 | 19 | center3 更好 |
| frame-centered high fallback | 19.9804 | -7.6202 | 24.4818 | 31.9104 | 13 | 无监督 frame centering 破坏表示 |
| source pair adapter high | 19.5479 | -12.7784 | 23.2029 | 32.4022 | 39 | 原 pair 分支失败 |
| target-unlabeled adapter consensus | 21.7813 | -8.0891 | 29.1715 | 30.7153 | 20 | 投影未改善 raw query |
| official filter / head-map variants | 约 19.3–21.5 | 更低 | — | — | 增加 | 官方 filter 语义不等于 re-ID descriptor |

另有 `high_selective`、`high_selective_spatial`、`high_jepa_iou`、`birth_confirm`、tracklet relink 和多个 threshold grid。它们的结果与原始 JSON 均保存在 [`JEPA outputs`](</data1/LWR/vranlee/SERVER_ONLY/avis/JEPA/outputs/ua_detrac_jepa>)，不删除失败证据。

### 4.1 为什么没有采用 source pair adapter

source pair adapter 训练于 sunny source GT track IDs，30 epochs，fit samples 162366、held-out samples 98805，held-out F1 0.9139，checkpoint 为 [`pair_adapter.pt`](</data1/LWR/vranlee/SERVER_ONLY/avis/JEPA/outputs/ua_detrac_jepa/pair_adapter_source_center3/pair_adapter.pt>)，SHA256 `d6ff76dbd56e9e58bedaf3de94df024782f4ba3156e39c9a5911c7ea1293537d`。

但在 target runtime 的 high association 中没有超过 source threshold 的 pair probability；pair-relative 分支因此没有改变决策。原 `pair_high` 还曾把 lost branch 错误地走进 raw JEPA fusion，已修复为 native lost association；修复后受控 ambiguous pair 分支仍与 baseline 相同，因此不进入主结果。

## 5. 数据泄漏审计

### 5.1 训练/跟踪阶段允许读取的内容

- target prepared JSON 的 `videos`、`images`、`image_id`、`frame_id`、`file_name`、`detection`。
- 对应 image pixels。
- frozen GOT-JEPA checkpoint。
- 已生成的 target candidate feature cache。
- source-only adapter 使用 sunny source GT；target adapter 只使用 target detector-box temporal pseudo-pairs。

### 5.2 明确禁止且未用于决策的内容

- target `annotations` 中的 GT boxes、track IDs、future labels。
- target GT 评测指标或 per-object correctness。
- 用 target GT 选择 floor、margin、quality-filter 阈值作为严格主配置。

`build_generaltrack_jepa_cache.py`、`uadetrac_jepa_tracklet_relink.py`、`uadetrac_jepa_quality_filter.py` 和 `train_uadetrac_target_unsup_adapter.py` 的 sidecar/metadata 都记录 `gt_read:false`；其中 quality filter 的 provenance 文件为：

- [`cloudy filter_diagnostics.json`](</data1/LWR/vranlee/SERVER_ONLY/avis/GeneralTrack-main/outputs/ua_detrac_jepa_cloudy_highfb090_quality_filter/track_results/filter_diagnostics.json>)
- [`night filter_diagnostics.json`](</data1/LWR/vranlee/SERVER_ONLY/avis/GeneralTrack-main/outputs/ua_detrac_jepa_night_highfb090_quality_filter/track_results/filter_diagnostics.json>)
- [`rainy filter_diagnostics.json`](</data1/LWR/vranlee/SERVER_ONLY/avis/GeneralTrack-main/outputs/ua_detrac_jepa_rainy_highfb090_quality_filter/track_results/filter_diagnostics.json>)

质量过滤输入/输出行数如下：cloudy `49502 -> 44264`、night `24808 -> 22213`、rainy `33970 -> 31933`；输入轨迹不会读取 GT，只按 detector score、tracklet length、JEPA adjacent similarity 和 local motion 做判断。

### 5.3 GT 何时被读取

GT 只在独立的 [`eval_damot_generaltrack.py`](</data1/LWR/vranlee/SERVER_ONLY/avis/GeneralTrack-main/tools/eval_damot_generaltrack.py>) 评测阶段读取，用于计算 HOTA/DetA/AssA/MOTA/IDF1/IDSW/FP/FN。跟踪进程和质量过滤进程与评测进程分离。

早期曾有一次把 80-frame 首视频 track result 对着全域 30-video prepared annotation 评测，得到约 HOTA 3 的无效记录；该记录已排除，所有本报告首视频结果都使用 80-frame temporary annotation，所有 full-domain 结果都使用对应 domain annotation。

## 6. 可复现主命令

下面是 cloudy 主 tracker 的模板；night/rainy 只需替换 domain、prepared dir、cache 和 output dir。它使用严格 fixed `.90` floor，不含首视频 `.85` target-tuned 选择：

```bash
CUDA_VISIBLE_DEVICES=1 \
/home/lwr/anaconda3/envs/mft25ocsort/bin/python \
/data1/LWR/vranlee/SERVER_ONLY/avis/GeneralTrack-main/tools/track_damot_domain.py \
  --dataset ua_detrac \
  --domain cloudy \
  --prepared-dir /data1/LWR/vranlee/SERVER_ONLY/avis/GeneralTrack-main/datasets/generaltrack_prepared_thr02/ua_detrac/cloudy \
  --output-dir /data1/LWR/vranlee/SERVER_ONLY/avis/GeneralTrack-main/outputs/ua_detrac_jepa_cloudy_high_fallback_iou005_full \
  --restore_ckpt /data1/LWR/vranlee/SERVER_ONLY/avis/GeneralTrack-main/checkpoints/generaltrack_bdd.pth \
  --jepa-cache /data1/LWR/vranlee/SERVER_ONLY/avis/JEPA/outputs/ua_detrac_jepa/cache/cloudy_center3.npz \
  --jepa-mode high_fallback \
  --jepa-high-min-sim 0.90 \
  --jepa-high-iou 0.05 \
  --jepa-relative-margin 0.03 \
  --jepa-max-gap 8
```

质量过滤命令模板：

```bash
/home/lwr/anaconda3/envs/mft25ocsort/bin/python \
/data1/LWR/vranlee/SERVER_ONLY/avis/JEPA/GOT-main/experiments/uadetrac_jepa_quality_filter.py \
  --prepared-dir /data1/LWR/vranlee/SERVER_ONLY/avis/GeneralTrack-main/datasets/generaltrack_prepared_thr02/ua_detrac/cloudy \
  --track-results /data1/LWR/vranlee/SERVER_ONLY/avis/GeneralTrack-main/outputs/ua_detrac_jepa_cloudy_high_fallback_iou005_full/track_results \
  --jepa-cache /data1/LWR/vranlee/SERVER_ONLY/avis/JEPA/outputs/ua_detrac_jepa/cache/cloudy_center3.npz \
  --output-dir /data1/LWR/vranlee/SERVER_ONLY/avis/GeneralTrack-main/outputs/ua_detrac_jepa_cloudy_highfb090_quality_filter/track_results
```

质量过滤默认参数为 `min_long=5`、`short_min_score=0.55`、`short_min_sim=0.95`、`single_min_score=0.85`、`max_center_norm=2.5`、`min_iou=0.20`。

## 7. 代码、结果和研究来源

关键代码：

- [`build_generaltrack_jepa_cache.py`](</data1/LWR/vranlee/SERVER_ONLY/avis/JEPA/GOT-main/experiments/build_generaltrack_jepa_cache.py>)
- [`byte_tracker_bdd_jepa.py`](</data1/LWR/vranlee/SERVER_ONLY/avis/GeneralTrack-main/yolox/tracker/byte_tracker_bdd_jepa.py>)
- [`track_damot_domain.py`](</data1/LWR/vranlee/SERVER_ONLY/avis/GeneralTrack-main/tools/track_damot_domain.py>)
- [`uadetrac_jepa_quality_filter.py`](</data1/LWR/vranlee/SERVER_ONLY/avis/JEPA/GOT-main/experiments/uadetrac_jepa_quality_filter.py>)
- [`uadetrac_jepa_tracklet_relink.py`](</data1/LWR/vranlee/SERVER_ONLY/avis/JEPA/GOT-main/experiments/uadetrac_jepa_tracklet_relink.py>)
- [`train_uadetrac_jepa_pair_adapter.py`](</data1/LWR/vranlee/SERVER_ONLY/avis/JEPA/GOT-main/experiments/train_uadetrac_jepa_pair_adapter.py>)
- [`train_uadetrac_target_unsup_adapter.py`](</data1/LWR/vranlee/SERVER_ONLY/avis/JEPA/GOT-main/experiments/train_uadetrac_target_unsup_adapter.py>)

研究/代码检索时记录的候选方法（仅作后续研究入口，未声称本实验复现）：

- [QTrack](https://github.com/gaash-lab/QTrack)
- [OVTR](https://github.com/jinyanglii/OVTR)
- [MASA-OA](https://github.com/gonzalocarreteroh/MASA-OA)
- [Drive-JEPA](https://github.com/linhanwang/Drive-JEPA)
- [SAM3](https://github.com/facebookresearch/sam3/blob/main/RELEASE_SAM3p1.md)
- [LITE](https://github.com/Jumabek/LITE)
- [ST-TrackFormer](https://github.com/se2un/ST-TrackFormer)
- [KPR](https://github.com/vlsomers/keypoint_promptable_reidentification)

## 8. 下一步建议

如果继续冲击 ICLR，最值得做的不是继续扫 tracker 阈值，而是训练一个 source-only、video-disjoint 的 association head，把 GeneralTrack native relation score、JEPA cosine、predicted IoU、score 和 gap 联合建模，并在 source held-out 上选定 operating point；当前 `high_fallback` 已表明 JEPA 对未匹配 high-score candidate 有有效补充信号。其次应把 quality filter 改成可验证的 detection-quality calibration，并报告 FN 增加的代价。当前结果足以作为严谨 ablation 和正向工程结果，但不应包装成 detector-level 或 ICLR SOTA 结论。
