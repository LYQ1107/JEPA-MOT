# MOT baseline → JEPA 实验协议（独立实验）

## 目标

本实验把 JEPA 作为 MOT 的 per-track association/filter predictor，而不是改动 detector。先固定一个成熟 tracking-by-detection baseline，再在完全相同的检测流、图像和 TrackEval 配置上加入 JEPA。主要指标为 HOTA、IDF1、AssA、DetA、MOTA、IDSW，并同时保存 FP/FN、Frag、FPS。

## Baseline 候选与选择

已只读检查本机可运行实现：

| 候选 | 本地实现/权重 | 判断 |
|---|---|---|
| ByteTrack | `masa/HybridSORT/trackers/byte_tracker/byte_tracker.py`；MOT17/DanceTrack YOLOX 权重存在 | 可作为简单强 baseline，二阶段高/低分检测关联 |
| OC-SORT | `masa/HybridSORT/trackers/ocsort_tracker/ocsort.py`；MOT17 public wrapper 存在 | 运动/观测一致性成熟，接口可直接接固定 public detections |
| HybridSORT/Deep-OC-SORT | `masa/HybridSORT/trackers/hybrid_sort_tracker/*`；FastReID SBS-S50 权重存在 | 依赖完整 YOLOX/FastReID 推理，需额外 detector 配置，smoke 成本较高 |
| TrackTrack | 本地有完整 tracker 和 feature pickle | 已有 pickle 的序列/格式不覆盖本实验的多序列严格 split，暂不作为主入口 |

主 baseline 选择 **OC-SORT（public-detection mode）**：它是成熟的 online MOT 关联器，包含 Kalman motion、观测一致性和 re-association；接口 `update_public(dets, cates, scores)` 能保证 JEPA 与 baseline 共用完全相同的 detector rows。ByteTrack 将作为交叉 sanity control，不修改生产代码。

## 数据与检测流

- MOT17 根目录：`/data1/LWR/vranlee/SERVER_ONLY/avis/masa/data/MOT17`，使用每个 `train/<seq>/det/det.txt` 的公开 FRCNN 检测；不使用 GT box 作为 detector。
- DanceTrack 根目录：`/data1/LWR/vranlee/DATASETS/JDE/dancetrack`；优先使用已审计的 D-CTRL cache `/data3/testdata/vranlee/.MOTSynth.partial/LocateMOT_L1A/detections_ctrl`，cache 中无 track ID。
- 每个方法记录 detector 文件/cache digest、输入行数、frame range、GT digest 和图像/feature digest；方法之间禁止改变检测阈值或删行。

## Split 与阶段

### Smoke

- MOT17：train `MOT17-02-FRCNN`，eval `MOT17-04-FRCNN`，前 20/30 帧。
- DanceTrack：train `dancetrack0001`，eval `dancetrack0004`，前 20/30 帧。
- 只验证导入、轨迹格式、候选置换不变性、无 GT 泄漏和 TrackEval staging。

### 正式 sanity

- MOT17：train `MOT17-02/04/09-FRCNN`，eval `MOT17-05/11-FRCNN`。
- DanceTrack：train `dancetrack0001/0002/0006`，eval `dancetrack0004/0007/0014`。
- 首轮每序列前 120 帧，至少 seed 17 和 seed 23；通过 gate 后再扩展到完整可用验证序列。
- train/eval 视频完全不重叠；JEPA 训练只能读取 train GT，eval GT 只在最终 TrackEval 中使用。

## JEPA 放置与因果协议

对 track 在时刻 `t` 的最近 `K=4` 个 accepted detector observations，输入 appearance、box center/scale、velocity、age/Δt、score、missing/visibility mask。冻结 clean teacher 从 clean current crop 与同帧 hard negatives 产生 association filter/score target；student 从同一历史和 corruption current observation 预测该 target。预测结果 `omega_t` 只用于 `t+1` 的 Hungarian cost，不能读取同帧候选或同帧 assignment。visibility/uncertainty 只控制 memory update。保留 raw OC-SORT、history-only、no-teacher、random-frozen controls。

优先加载官方 GOT-JEPA/ToMP tracking predictor checkpoint；若 checkpoint 暂不可加载，使用明确标记为 provisional 的 train-fit teacher，冻结并记录 SHA，不能按 eval window 用 GT 拟合。

## 公平性与停止规则

所有方法输出相同 MOTChallenge 十列格式。TrackEval 使用同一份 staging GT/seqinfo，输出 HOTA 全 alpha 均值及 alpha=.5、AssA、DetA、IDF1、MOTA、IDSW、FP、FN、Frag、FPS。不得根据 eval GT 调阈值、挑选 seed 或删失败序列。

正反馈 gate：JEPA 相对 OC-SORT baseline 在 HOTA、IDF1、AssA、MOTA 至少主要指标不降，DetA 不因改变 detector 而变化，IDSW 不增加；并在两个 seed、多个 eval 序列方向一致。若失败，只允许一轮有理论依据的修复；连续失败则停止线上集成并保留完整 negative result。

本文件只记录协议和只读选择；每轮运行另写 `reflection.json` 与 `next_plan.md`，包含命令、环境、脚本 SHA、输入 digest、指标和失败根因。
