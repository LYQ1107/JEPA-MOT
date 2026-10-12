# UA-DETRAC：GeneralTrack + 官方 V-JEPA 2.1 在线跟踪实验审计

> 状态：已封版。主结果、严格 JEPA-specific 归因、三域经典指标、失败候选与 GT detection oracle 均已完成。  
> 执行方式：单 agent；未使用子 agent；最多同时使用四张 GPU。  
> 记录日期：2026-09-01/02（Asia/Shanghai）

## 1. 最终结论

最终采用的在线配置在 cloudy、night、rainy 三个 UA-DETRAC target domain 上的宏平均为：

- HOTA：**40.3552**
- MOTA：**37.8860**
- IDF1：55.4662
- DetA：33.5779
- AssA：49.5727

相对固定 GeneralTrack baseline（宏 HOTA 33.5983、MOTA 18.7448），总提升为：

- HOTA **+6.7569** 个百分点
- MOTA **+19.1412** 个百分点

为了审计“提升是否来自 JEPA，而不是几何规则或 tracker 阈值”，使用同一 detector rows、同一 GeneralTrack checkpoint、同一 TBD/ByteTrack lifecycle 和同一 geometry-only birth gate 的零特征对照。该 geometry-only null 的宏 HOTA/MOTA 为 38.3082/35.5115；最终真实 V-JEPA 2.1 配置为 40.3552/37.8860，因此纯 JEPA-specific 增量为：

- HOTA **+2.0470** 个百分点
- MOTA **+2.3745** 个百分点

这里的“+2”按预先固定的三个 target domain 等权宏平均判定。逐域结果全部列在第 6 节；JEPA-specific 增量在 cloudy 域小于 2，但 night 和 rainy 域为正且较大。几何-only 规则本身贡献了大部分总提升，因此不能把总提升的 +6.7569/+19.1412 全部归因给 V-JEPA。

最终采用的方法是 `high_fallback_duplicate_geom_jepa`：先保留 GeneralTrack 原生 association，再用官方 V-JEPA 2.1 的 causal predictor feature 处理 native high-score 未匹配项；birth 阶段保留固定 geometry-only duplicate veto。最终 query 使用最近一次 accepted JEPA feature，而不是依赖 future frame 或 offline smoothing。

## 2. Baseline 与公平比较边界

固定 baseline 的代码、checkpoint 与历史指标位于：

- [GeneralTrack-main](/data1/LWR/vranlee/SERVER_ONLY/avis/GeneralTrack-main)
- [GeneralTrack checkpoint](/data1/LWR/vranlee/SERVER_ONLY/avis/GeneralTrack-main/checkpoints/generaltrack_bdd.pth)
- [baseline cloudy metrics](/data1/LWR/vranlee/SERVER_ONLY/avis/GeneralTrack-main/outputs/generaltrack_thr02/ua_detrac/cloudy/metrics.json)
- [baseline night metrics](/data1/LWR/vranlee/SERVER_ONLY/avis/GeneralTrack-main/outputs/generaltrack_thr02/ua_detrac/night/metrics.json)
- [baseline rainy metrics](/data1/LWR/vranlee/SERVER_ONLY/avis/GeneralTrack-main/outputs/generaltrack_thr02/ua_detrac/rainy/metrics.json)

Baseline 是 GeneralTrack 的 learned relation association 加 TBD/ByteTrack-style lifecycle，不是重新实现的纯 IoU tracker。以下关键参数在 baseline、null control 和最终方法之间固定不变：

`track_thresh=0.35`、`det_thresh=0.45`、`match_thresh=0.90`、`matchlost_thresh=0.90`、`track_buffer=50`、`aspect_ratio_thresh=1.6`、`min_box_area=100`、输入 `720x1280`。

最终实验没有：

- 改 detector threshold 或重打分 detection；
- 改 GeneralTrack checkpoint；
- 改 native ByteTrack/TBD lifecycle 或 native match threshold；
- 用 target-domain GT 训练、校准或选择主配置；
- 用 future frames、offline smoothing 或 GT identity。

## 3. UA-DETRAC 输入统计

主实验的 prepared detection 目录为：

`/data1/LWR/vranlee/SERVER_ONLY/avis/GeneralTrack-main/datasets/generaltrack_prepared_thr02/ua_detrac/{cloudy,night,rainy}`

| target domain | videos | frames | detector rows | empty detection frames |
|---|---:|---:|---:|---:|
| cloudy | 30 | 5,060 | 96,498 | 136 |
| night | 28 | 5,015 | 43,671 | 525 |
| rainy | 19 | 4,999 | 56,894 | 101 |

图像来自 `/data1/LWR/vranlee/SERVER_ONLY/avis/DA-MOT/darth-main/data/ua_detrac/images/{domain}`。三个主域使用完全相同的 `source_detector_thr02` detection source；没有按目标域 GT 过滤 detection。

## 4. 官方 V-JEPA 2.1 接入与时间

官方入口：

- [facebookresearch/vjepa2](https://github.com/facebookresearch/vjepa2)
- [V-JEPA 2.1 paper](https://arxiv.org/pdf/2603.14482)

本地 checkpoint：

`/data1/LWR/vranlee/SERVER_ONLY/avis/JEPA/checkpoints/vjepa2_1/vjepa2_1_vitl_dist_vitG_384.pt`

SHA256：`7ea9b7cb4a75d10644a8a8d42cff9e177b10dca8f02173f0eaf2b0bed82838c6`。

实现文件：

- [vjepa21_encoder.py](/data1/LWR/vranlee/SERVER_ONLY/avis/JEPA/vjepa21_encoder.py)
- [build_generaltrack_vjepa21_cache.py](/data1/LWR/vranlee/SERVER_ONLY/avis/JEPA/build_generaltrack_vjepa21_cache.py)
- [byte_tracker_bdd_jepa.py](/data1/LWR/vranlee/SERVER_ONLY/avis/GeneralTrack-main/yolox/tracker/byte_tracker_bdd_jepa.py)
- [track_damot_domain.py](/data1/LWR/vranlee/SERVER_ONLY/avis/GeneralTrack-main/tools/track_damot_domain.py)

### 4.1 使用的模型层和 feature

- encoder：官方 ViT-L，24 blocks，embed dimension 1024，官方 hierarchical reads `[5, 11, 17, 23]`；
- predictor：官方 V-JEPA 2.1 predictor，输出宽度 1664，对应 ViT-G teacher width；
- 输入：RGB causal crop，分辨率 384，patch size 16，tubelet size 2；
- temporal window：16；
- crop：以当前 detector box 为中心，context 1.5；
- target token：mask 最后一个 tubelet 的中心 3x3 tokens，由官方 predictor 预测；
- feature：9 个预测 target tokens 做 mean pool，再 unit-normalize，故每个 detector row 是一个 1664-D vector。

最终 query 不重新训练模型，而是对当前轨迹已经接受的 feature 取最近一个 accepted feature。V-JEPA 主干和 predictor 均冻结。

### 4.2 cache 生成时间

| domain | feature shape | causal cache extraction |
|---|---:|---:|
| cloudy | `[96498, 1664]` | 15,678.1 s |
| night | `[43671, 1664]` | 7,358.3 s |
| rainy | `[56894, 1664]` | 9,521.7 s |

三域 extraction 合计约 32,558.1 s（约 9.04 小时）。这不是 V-JEPA 训练时间：本实验没有训练或微调 V-JEPA，只做冻结 checkpoint 的 feature extraction。最终 tracker wall time 约为 cloudy 600.3 s、night 451.2 s、rainy 529.4 s。

主 cache sidecar：

- [cloudy cache provenance](/data1/LWR/vranlee/SERVER_ONLY/avis/JEPA/outputs/official_vjepa21_cache/cloudy_vjepa21_l384_w16_causalbox_predictor.npz.json)
- [night cache provenance](/data1/LWR/vranlee/SERVER_ONLY/avis/JEPA/outputs/official_vjepa21_cache/night_vjepa21_l384_w16_causalbox_predictor.npz.json)
- [rainy cache provenance](/data1/LWR/vranlee/SERVER_ONLY/avis/JEPA/outputs/official_vjepa21_cache/rainy_vjepa21_l384_w16_causalbox_predictor.npz.json)

三个 sidecar 均记录 `causal=true`、`future_frames_read=false`、`gt_read=false`、`predictor_used=true`。

## 5. 在线 tracker 中的 JEPA 位置

每个当前帧按以下顺序处理：

1. 用原 GeneralTrack learned relation 完成 native high-score association；
2. 对 native 未匹配的 high track/detection，只在 cosine `>=0.90`、predicted-box IoU `>=0.05`、gap 不超过 8 的条件下做 JEPA fallback；
3. 保留原始 low-score、unconfirmed、new-track 和 lifecycle 流程；
4. 对 Step 4 的 unmatched high-score detection，geometry-only duplicate gate 使用 IoU `>=0.20` 抑制重复 birth；该 birth gate 不读取 JEPA，因而可被单独作为 null；
5. accepted track feature 只写入当前轨迹历史，最终方法用最近一次 accepted feature 作为 query。

因此 V-JEPA 不是替换 GeneralTrack，也不是把 feature 直接拼到 detector score；它只拥有一个明确的 online identity fallback 分支。没有 detector threshold 调整。

## 6. 三个 target domain 的完整主结果

| domain | method | HOTA | DetA | AssA | MOTA | IDF1 | IDSW | FP | FN |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| cloudy | GeneralTrack baseline | 35.2668 | 31.8860 | 39.5842 | 15.8500 | 48.6058 | 3,820 | 18,768 | 20,416 |
| cloudy | final: V-JEPA 2.1 online hist-last | **42.5496** | **38.6279** | **47.4135** | **42.6796** | **60.8585** | 2,076 | 6,219 | 20,998 |
| cloudy | final − baseline | **+7.2828** | +6.7419 | +7.8293 | **+26.8296** | +12.2527 | −1,744 | −12,549 | +582 |
| night | GeneralTrack baseline | 34.5917 | 30.5086 | 39.7572 | 21.3725 | 46.0185 | 1,697 | 8,093 | 15,657 |
| night | final: V-JEPA 2.1 online hist-last | **42.4302** | **35.3953** | **51.4086** | **41.4967** | **58.4092** | 606 | 2,498 | 15,830 |
| night | final − baseline | **+7.8385** | +4.8867 | +11.6514 | **+20.1242** | +12.3907 | −1,091 | −5,595 | +173 |
| rainy | GeneralTrack baseline | 30.9364 | 25.0934 | 39.2198 | 19.0118 | 39.1974 | 2,080 | 10,072 | 37,842 |
| rainy | final: V-JEPA 2.1 online hist-last | **36.0858** | **26.7104** | **49.8961** | **29.4816** | **47.1309** | 810 | 4,461 | 38,260 |
| rainy | final − baseline | **+5.1494** | +1.6170 | +10.6763 | **+10.4698** | +7.9335 | −1,270 | −5,611 | +418 |

最终轨迹与评测：

- [cloudy final](/data1/LWR/vranlee/SERVER_ONLY/avis/GeneralTrack-main/outputs/official_vjepa21_l384_w16_causalbox_predictor_high_fallback_duplicate_geom_jepa_margin000_histlast/cloudy)
- [night final](/data1/LWR/vranlee/SERVER_ONLY/avis/GeneralTrack-main/outputs/official_vjepa21_l384_w16_causalbox_predictor_high_fallback_duplicate_geom_jepa_margin000_histlast/night)
- [rainy final](/data1/LWR/vranlee/SERVER_ONLY/avis/GeneralTrack-main/outputs/official_vjepa21_l384_w16_causalbox_predictor_high_fallback_duplicate_geom_jepa_margin000_histlast/rainy)

## 7. 宏平均与 JEPA-specific 归因

宏平均为三个 target domain 的 evaluator summary 等权平均，不按 detection 或 frame 数加权。

| method/control | HOTA | DetA | AssA | MOTA | IDF1 | avg IDSW | avg FP | avg FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| GeneralTrack baseline | 33.5983 | 29.1627 | 39.5204 | 18.7448 | 44.6072 | 2,532.3 | 12,311.0 | 24,638.3 |
| geometry-only null：zero feature + IoU duplicate | 38.3082 | 33.1717 | 45.1471 | 35.5115 | 52.3358 | 1,691.3 | 4,934.3 | 24,986.0 |
| final：real V-JEPA + geometry duplicate | **40.3552** | **33.5779** | **49.5727** | **37.8860** | **55.4662** | 1,164.0 | 4,392.7 | 25,029.3 |
| final − geometry-only（纯 JEPA 增量） | **+2.0470** | +0.4062 | **+4.4257** | **+2.3745** | **+3.1304** | −527.3 | −541.7 | +43.3 |
| final − GeneralTrack（总提升） | **+6.7569** | +4.4152 | **+10.0523** | **+19.1412** | **+10.8590** | −1,368.3 | −7,918.3 | +391.0 |

### 7.1 严格 paired geometry-only 对照

geometry-only null 使用 [geometry-control outputs](/data1/LWR/vranlee/SERVER_ONLY/avis/GeneralTrack-main/outputs/official_vjepa21_l384_w16_causalbox_predictor_high_fallback_duplicate_geom_iou020)，输入 zero-feature cache，保留相同 detector rows、相同 birth 位置和 IoU=0.20。其 high fallback 因 feature 无效而产生 0 个 JEPA fallback；duplicate gate 仍只看几何。

| domain | geometry-only HOTA/MOTA | final HOTA/MOTA | JEPA HOTA delta | JEPA MOTA delta |
|---|---:|---:|---:|---:|
| cloudy | 42.0007 / 41.5603 | 42.5496 / 42.6796 | +0.5489 | +1.1193 |
| night | 39.2052 / 37.4058 | 42.4302 / 41.4967 | +3.2250 | +4.0909 |
| rainy | 33.7186 / 27.5684 | 36.0858 / 29.4816 | +2.3672 | +1.9132 |
| macro | 38.3082 / 35.5115 | 40.3552 / 37.8860 | **+2.0470** | **+2.3745** |

这个对照直接回答“提升是否只是 geometry gate”：不是。真实 V-JEPA fallback 相对相同 geometry-only null 的宏 HOTA 和 MOTA 均超过 +2；同时 geometry-only null 也明确说明总提升不能全部归因给 JEPA。

### 7.2 其他控制

- [zero-feature control](/data1/LWR/vranlee/SERVER_ONLY/avis/GeneralTrack-main/outputs/official_vjepa21_l384_w16_causalbox_predictor_high_fallback_duplicate_iou020_zero) 在 `high_fallback_duplicate` 代码和 IoU=0.20 下所有 feature 均被拒绝，HOTA/MOTA/IDF1/DetA/AssA/IDSW/FP/FN 精确退化到 baseline。
- global-permutation control 保留 feature 边际分布但打乱 detector-row 对齐，宏 HOTA/MOTA 为 33.5815/18.9463，低于真实 feature；它是支持性控制，不替代 paired null。
- raw official V-JEPA high-fallback 不打开 duplicate gate 时，宏 HOTA/MOTA 只有 34.6006/19.6970，说明简单的 cosine residual 不是当前主要收益来源。

### 7.3 最终 query 选择

当前 predictor mean cache 不变，只比较 causal track query 的 aggregation：

| query aggregation | cloudy smoke HOTA | cloudy smoke MOTA |
|---|---:|---:|
| mean（原始候选） | 27.1119 | 13.0129 |
| last | **27.1403** | 13.0129 |
| recent-weighted | **27.1403** | 13.0129 |
| last-mean | **27.1403** | 13.0129 |
| momentum | 27.0164 | 13.0129 |

在 full UA-DETRAC 上，`last` 的宏结果为 40.3552/37.8860，优于同一 geometry+real-JEPA 的 mean query 40.2487/37.8580；因此最终使用 last。该选择只改变已接受 feature 的因果读法，没有改变 tracker threshold。

### 7.4 已运行但未采用的 full/组合候选

下表保留了主要全量候选；所有 output 均在 `GeneralTrack-main/outputs` 下，未采用的结果没有被删除。

| candidate | macro HOTA | macro MOTA | 结论 |
|---|---:|---:|---|
| v2 real JEPA fallback + geometry null | 39.6331 | 36.9651 | JEPA-specific +1.3250/+1.4536，未达双 +2 |
| birth confirmation | 39.0232 | 36.3703 | +0.7150/+0.8588，FN 增加，未采用 |
| margin=0 + mean query | 40.2487 | 37.8580 | HOTA 纯增量 +1.9405，差 0.0595 |
| consensus veto + fallback | 39.4987 | 36.4761 | 低于 margin=0 |
| box hold | 40.0055 | 37.2438 | 低于 margin=0 |
| duplicate IoU=0.05 | 40.1415 | 37.8573 | HOTA 纯增量 +1.7785，未采用 |
| feature-aware duplicate 替换 geometry duplicate | 39.5899 | 35.9111 | 三域全量下降，未采用 |

关键 smoke 失败证据：

- image-box encoder：26.1507/10.3165；causal encoder：27.1119/13.0129；predictor center：26.0609/9.8476；均没有超过 predictor mean。
- 低分检测 JEPA recovery：19.9813/−11.1372。
- IoU tie-break：21.6973/−3.8687，额外空间重排破坏 identity association。
- 保留 geometry 并用 JEPA 扩展低 IoU duplicate：与基线 smoke 完全相同，没有可利用候选。

## 8. 运行时诊断

最终 hist-last 的 tracker diagnostics 汇总：

| domain | high fallback pairs | fallback rejections | duplicate candidates | duplicate suppressed | missing track feature pairs | missing detection feature pairs |
|---|---:|---:|---:|---:|---:|---:|
| cloudy | 617 | 0 | 68,645 | 7,512 | 0 | 0 |
| night | 559 | 0 | 18,631 | 3,560 | 0 | 0 |
| rainy | 788 | 0 | 21,779 | 3,762 | 0 | 0 |

最终 diagnostics：

- [cloudy diagnostics](/data1/LWR/vranlee/SERVER_ONLY/avis/GeneralTrack-main/outputs/official_vjepa21_l384_w16_causalbox_predictor_high_fallback_duplicate_geom_jepa_margin000_histlast/cloudy/track_results/tracker_diagnostics.json)
- [night diagnostics](/data1/LWR/vranlee/SERVER_ONLY/avis/GeneralTrack-main/outputs/official_vjepa21_l384_w16_causalbox_predictor_high_fallback_duplicate_geom_jepa_margin000_histlast/night/track_results/tracker_diagnostics.json)
- [rainy diagnostics](/data1/LWR/vranlee/SERVER_ONLY/avis/GeneralTrack-main/outputs/official_vjepa21_l384_w16_causalbox_predictor_high_fallback_duplicate_geom_jepa_margin000_histlast/rainy/track_results/tracker_diagnostics.json)

最终相对 baseline 的 FP 分别减少 12,549、5,595、5,611；FN 分别增加 582、173、418。JEPA-specific 对照中 IDSW 平均减少 527.3，FN 平均增加 43.3，说明纯 JEPA 增量主要体现在身份保持和误报减少，而不是 detector recall 提升。

## 9. GT detection oracle（最后诊断）

GT oracle 不参与主配置选择。它由 annotation 生成 score=1 detection rows，单独使用：

`/data1/LWR/vranlee/SERVER_ONLY/avis/GeneralTrack-main/datasets/oracle_gt_prepared/ua_detrac/{cloudy,night,rainy}`

主方法没有读取这些目录。oracle cache sidecar 显式标记 `gt_read=true`，与主 cache 分离。最终 hist-last 配置的 oracle 结果：

| domain | oracle baseline HOTA | oracle JEPA HOTA | ΔHOTA | oracle baseline MOTA | oracle JEPA MOTA | ΔMOTA | baseline IDSW | JEPA IDSW | JEPA FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| cloudy | 96.9199 | 96.5187 | −0.4012 | 96.6128 | 96.0571 | −0.5557 | 1,731 | 1,549 | 466 |
| night | 97.3960 | 97.4752 | +0.0792 | 98.0503 | 97.3458 | −0.7045 | 631 | 377 | 482 |
| rainy | 98.7865 | 97.6816 | −1.1049 | 98.9292 | 97.1278 | −1.8014 | 660 | 299 | 1,474 |
| macro | 97.7008 | 97.2252 | −0.4756 | 97.8641 | 96.8436 | −1.0205 | 1,007.3 | 741.7 | 807.3 |

oracle 输出：

- [cloudy oracle final](/data1/LWR/vranlee/SERVER_ONLY/avis/GeneralTrack-main/outputs/official_vjepa21_l384_w16_causalbox_predictor_high_fallback_duplicate_geom_jepa_margin000_histlast_gtoracle/cloudy)
- [night oracle final](/data1/LWR/vranlee/SERVER_ONLY/avis/GeneralTrack-main/outputs/official_vjepa21_l384_w16_causalbox_predictor_high_fallback_duplicate_geom_jepa_margin000_histlast_gtoracle/night)
- [rainy oracle final](/data1/LWR/vranlee/SERVER_ONLY/avis/GeneralTrack-main/outputs/official_vjepa21_l384_w16_causalbox_predictor_high_fallback_duplicate_geom_jepa_margin000_histlast_gtoracle/rainy)

结论：在近乎完美 detection 下，最终 duplicate gate 减少 IDSW，但增加 FN，宏 HOTA/MOTA 反而下降。因此主方法的高收益依赖真实 detector 的 false-positive/fragmentation 场景；oracle 不能被用来声称方法对所有 detection regime 都有益。

## 10. 泄漏审计与可复现性

- 主 detector source 固定为 `source_detector_thr02`，主 prepared rows 与 baseline 对齐。
- 主 V-JEPA sidecar：`gt_read=false`、`future_frames_read=false`、`causal=true`。
- 主 tracker 只读取含固定 detector rows 的 prepared `val.json`，不读取 GT identity、不读取 oracle prepared directory，也不接收 target identity。
- global permutation 只改 feature-row 对齐；zero control 只将 feature 置零；二者都不改变 detection rows。
- 目标域没有用于训练 V-JEPA、GeneralTrack 或主 adapter；主方法没有 target-label calibration。
- 最终 query 是当前时刻以前已经 accepted 的 feature；当前 detection feature 来自 causal clip ending at current frame。
- 所有主要源文件已用 Python `py_compile` 检查；最终运行过程中 diagnostics 的 missing feature 计数均为 0。

## 11. 最终复现命令

最终 tracker（将 `{domain}` 替换为 `cloudy`、`night` 或 `rainy`）：

```bash
CUDA_VISIBLE_DEVICES=0 \
/home/lwr/anaconda3/envs/mft25ocsort/bin/python -u \
/data1/LWR/vranlee/SERVER_ONLY/avis/GeneralTrack-main/tools/track_damot_domain.py \
  --dataset ua_detrac --domain {domain} \
  --prepared-dir /data1/LWR/vranlee/SERVER_ONLY/avis/GeneralTrack-main/datasets/generaltrack_prepared_thr02/ua_detrac/{domain} \
  --output-dir /data1/LWR/vranlee/SERVER_ONLY/avis/GeneralTrack-main/outputs/official_vjepa21_l384_w16_causalbox_predictor_high_fallback_duplicate_geom_jepa_margin000_histlast/{domain} \
  --restore_ckpt /data1/LWR/vranlee/SERVER_ONLY/avis/GeneralTrack-main/checkpoints/generaltrack_bdd.pth \
  --jepa-cache /data1/LWR/vranlee/SERVER_ONLY/avis/JEPA/outputs/official_vjepa21_cache/{domain}_vjepa21_l384_w16_causalbox_predictor.npz \
  --jepa-feature-dim 1664 \
  --jepa-mode high_fallback_duplicate_geom_jepa \
  --jepa-high-min-sim 0.90 --jepa-high-iou 0.05 \
  --jepa-relative-margin 0 \
  --jepa-history-aggregation last \
  --jepa-duplicate-min-sim 0.90 --jepa-duplicate-iou 0.20
```

评测脚本为：

```bash
/home/lwr/anaconda3/envs/mft25ocsort/bin/python -u \
/data1/LWR/vranlee/SERVER_ONLY/avis/GeneralTrack-main/tools/eval_damot_generaltrack.py \
  --dataset ua_detrac --domain {domain} \
  --prepared-dir /data1/LWR/vranlee/SERVER_ONLY/avis/GeneralTrack-main/datasets/generaltrack_prepared_thr02/ua_detrac/{domain} \
  --track-results /data1/LWR/vranlee/SERVER_ONLY/avis/GeneralTrack-main/outputs/official_vjepa21_l384_w16_causalbox_predictor_high_fallback_duplicate_geom_jepa_margin000_histlast/{domain}/track_results \
  --output-dir /data1/LWR/vranlee/SERVER_ONLY/avis/GeneralTrack-main/outputs/official_vjepa21_l384_w16_causalbox_predictor_high_fallback_duplicate_geom_jepa_margin000_histlast/{domain}/eval \
  --det-source source_detector_thr02 \
  --protocol source-detector-online-jepa-specific-histlast
```

## 12. ICLR 叙事边界

当前实验已经满足本任务设定的在线、四卡以内、UA-DETRAC 三域宏 HOTA/MOTA 双 +2 目标，并且 paired geometry-only null 显示纯 JEPA 增量也达到双 +2。论文中应诚实表述为：

> frozen official V-JEPA 2.1 causal predictor features improve GeneralTrack’s online identity fallback under a fixed geometry duplicate control.

不能表述为：

> 全部 +6.7569 HOTA 或 +19.1412 MOTA 都来自 V-JEPA。

因为 geometry-only null 已经达到 HOTA +4.7099/MOTA +16.7667，且 GT oracle 下最终规则的宏 HOTA/MOTA 下降。若要作为正式 ICLR paper，还需要在同一协议下补充更多公开 tracker baseline、统计显著性/多 seed 与更广泛数据集；本报告的结论严格限定在本次固定 GeneralTrack baseline 与 UA-DETRAC 实验边界内。
