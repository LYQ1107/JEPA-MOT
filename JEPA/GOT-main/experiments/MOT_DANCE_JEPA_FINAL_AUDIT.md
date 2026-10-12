# MOT17 / DanceTrack × JEPA 总审计

日期：2026-08-31。Luna 负责执行实验，主 agent 负责协议审计、归因和下一步
规划。本文件只总结已有产物，不改动 LocateMOT、MASA、生产 tracker 或数据。

## 结论

当前的 history-only temporal JEPA 不能作为 MOT 的默认线上模块：DanceTrack 的
一个 seed 有提升，但第二个 seed 几乎不再提升；MOT17 两个 seed 都下降。更严格
的 lagged association-filter 诊断也没有通过质量门槛。因此当前 release gate 为
`tracker integration = NO-GO`，不能用挑 seed、调阈值或把 proxy 当 HOTA 来宣称
提升。

## 数据与协议

| 数据 | train | eval | 检测输入 | 范围 |
|---|---|---|---|---|
| MOT17 | MOT17-02/04/09-FRCNN | MOT17-05/11-FRCNN | public FRCNN `det/det.txt` | 每序列前 120 帧，eval detector recall@IoU=.5 宏平均约 .429 |
| DanceTrack | 0001/0002/0006 | 0004/0007/0014 | 固定 D-CTRL per-frame cache（无 ID） | 每序列前 120 帧，detector recall@IoU=.5 宏平均约 .974 |

训练/评估序列不重叠；所有方法共享同一 detector/feature digest；匹配使用每帧
一对一 Hungarian（IoU≥.30）。正式 temporal tracker 分支的 predictor 每条
track 每帧只调用一次，输入仅为 `H_<t`，候选只进入同一帧的 Hungarian cost，
因此没有当前候选进入 predictor 的泄漏；更严格的“`t` 产生的 filter 只用于
`t+1`”约束是在后续 filter probes 中单独执行的。线上 tracker 不读 eval GT。
TrackEval 暂存 GT 只保留 120 帧并把 `seqLength` 设为 120。

下表是 vendored 官方 TrackEval 的 `HOTA_official_mean`（19 个 alpha 的均值）、
IDF1 和 IDSW；不是完整 MOT17/DanceTrack leaderboard。DanceTrack 使用
MOTChallenge2DBox adapter 的截断 sanity 评估，不能当作官方 DanceTrack 分数。

## 官方结果（固定检测、三序列/两序列截断 sanity）

格式：`HOTA_mean / IDF1 / IDSW`。

| 数据 / seed | raw | history EMA | random frozen | JEPA temporal | no-teacher |
|---|---:|---:|---:|---:|---:|
| DanceTrack / 17 | .7043 / .7673 / 38 | .7147 / .7859 / 40 | .7263 / .7878 / 32 | **.7331 / .8027 / 29** | .7184 / .7878 / 33 |
| DanceTrack / 23 | .7043 / .7673 / 38 | .7147 / .7859 / 40 | .6931 / .7473 / 65 | **.7102 / .7711 / 36** | .7102 / .7711 / 36 |
| MOT17 / 17 | .4855 / .5251 / 17 | .4853 / .5251 / 20 | .4723 / .5023 / 23 | **.4666 / .5147 / 23** | .4634 / .5189 / 17 |
| MOT17 / 23 | .4855 / .5251 / 17 | .4853 / .5251 / 20 | .5042 / .5685 / 24 | **.4666 / .5147 / 23** | .4821 / .5189 / 22 |

相对 raw，Dance seed 17 的 JEPA 是 `+2.88 pp HOTA / +3.53 pp IDF1 / -9 IDSW`，
但 seed 23 只有 `+0.59 pp / +0.37 pp / -2`，并且与 no-teacher 完全相同；MOT17
seed 17/23 均为 `-1.89 pp HOTA / -1.03 pp IDF1 / +6 IDSW`。增益集中在
DanceTrack-0004，不能作为跨数据集稳定增益。

## 逐轮反思

1. 初始 80-D crop-vector JEPA、current-pair probe 和 sparse 分支均不能证明
   JEPA 的因果提升；其中 current-pair 会把当前候选用于生成同帧 prototype，已
   按泄漏风险排除线上集成。
2. 严格 temporal 分支把输入改为最近 4 个 detector history token（外观、box、
   速度、age/delta-t、score、missing），并加入 quality/uncertainty gate。它在
   Dance 的 seed 17 有信号，但在 MOT17 的低 recall/长 gap 场景失效；post-gap
   诊断显示低 history coverage 时 assignment 改变更多、准确率更差。要特别说明：
   这一分支是 JEPA-inspired 的 history→clean-target predictor，并没有真正加载
   论文中的预训练、冻结 teacher，因此不能称为 GOT-JEPA 的完整复现。
3. 为排除“teacher 不冻结”或“corruption 太强”的解释，做了 train-only global
   frozen filter fallback（无 eval items）：teacher held-out `R@1=.9079`、
   margin `+.1753`，数值条件稳定；但 student 在 clean view 只有 `R@1=.0724`、
   margin `-.0588`，stress view `R@1=.0789`、`-.0671`，而 raw/history 为
   `1.0`。因此瓶颈是目标空间与学生输入/任务不匹配，不是简单的冻结问题，
   gate 明确 `NO-GO`。该 probe 仍用 oracle identity history，只是诊断。
4. FastReID smoke 只作为 in-domain embedding upper bound，不能替代未泄漏的
   MOT benchmark 或 JEPA 因果实验。

## JEPA 应放在 MOT 的哪里

不建议先把 JEPA 加到 detector backbone，也不建议直接预测一个“干净 crop
embedding”。MOT 特有的目标是每条 track 在候选集合中的判别与生命周期管理，
因此下一版应预测 per-track association/filter model：

```text
H_<t = {last K accepted detector embeddings,
        box/center/scale, velocity, age/delta-t, score, missing/visibility}

teacher: T(H_<t, clean current crop, same-frame hard negatives)
         -> omega_T^t (association cls/filter + bbox/motion + quality)
student: S(H_<t, corrupted current observation, state)
         -> omega_S^t ≈ stopgrad(omega_T^t)

use omega_S^t only on frame t+1 candidates (proposed lagged adapter);
Hungarian cost = appearance/filter score + motion/IoU + uncertainty;
low-quality/occluded observation freezes appearance memory.
```

`omega` 优先应是与 ToMP `ClsDec/RegDec` 兼容的 filter，或先做 8–12D 的 pair
scorer（appearance cosine、IoU/motion、center distance、score、gap、missing、
visibility），而非 80-D 向量回归。训练损失可用 filter regression/cosine、同帧
hard-negative ranking、`t→t+1` future consistency 和 uncertainty calibration；
VICReg covariance 只作为防塌缩正则，不能代替 MOT ranking 目标。

这对应 [GOT-JEPA](https://arxiv.org/html/2602.14771v5) 的核心：相同历史下，
clean teacher 产生 tracking model，corrupted student 预测同一 model，再由可见性/
遮挡机制控制更新；在 MOT 中需要把 tracking model 改成“候选集合上的 per-track
association model”，并显式建模多目标竞争、漏检、遮挡和重现。

## 是否需要训练

需要。推荐流程是：

1. teacher 使用预训练 ToMP/GOT-JEPA predictor/filter 初始化并完全冻结；若没有
   checkpoint，才在 train-fit 上训练一个 state-conditioned teacher，随后固定
   参数和 hash。
2. student 只在 MOT17/DanceTrack train 序列离线训练，加入 mask、blur、copy-paste
   distractor、短时 dropout、低分/false-positive 等 MOT 风格扰动；部署输入保持
   clean detector feature，corruption 只在训练/stress 评估。
3. 先在 train-heldout frame/identity 上做质量 gate，再接线上 tracker；初期不做
   online retraining/EMA，避免把错误关联写回 memory。

## 下一阶段计划与放行条件

1. 在取得授权后下载并校验官方 `GOT_JEPA_378.tar`（Drive metadata：
   1,706,753,563 B，约 1.59 GB），在隔离的 CUDA 11.7/PyTorch 2.0 环境做
   `torch.load`、constructor、state-dict shape 和单目标 20-frame smoke。当前
   本机没有该文件、ToMPL 前置权重或 `cotracker2.pth`，本轮没有下载。
2. 实现严格 lagged per-track adapter：accepted-detector-only state、
   `ClsDec/RegDec` 或低维 pair-filter、显式 visibility/occlusion/reappearance，
   保留 raw/history/no-teacher/random controls。
3. 训练前冻结 split、扰动、损失、seed 和停止规则；在 train-heldout 要求
   teacher `R@1≥.80` 且 margin>0，student 同时超过 history/no-teacher/random
   的 margin（预注册最小差值 .005）。
4. 只有 gate 通过才在两数据集上扩展到完整序列、至少 2 个 seed，并用官方
   TrackEval 报告 HOTA/IDF1/AssA/DetA/MOTA/IDSW 及 gap/occlusion 分桶；否则保留
   `NO-GO`，不把局部正例写成论文结论。

## 主要产物

- [实验说明](/data1/LWR/vranlee/SERVER_ONLY/avis/JEPA/GOT-main/experiments/README_MOT_JEPA.md)
- [MOT17 TrackEval 审计](/data1/LWR/vranlee/SERVER_ONLY/avis/JEPA/GOT-main/experiments/TRACKEVAL_MOT17_AUDIT.md)
- [DanceTrack TrackEval 审计](/data1/LWR/vranlee/SERVER_ONLY/avis/JEPA/GOT-main/experiments/TRACKEVAL_DANCE_AUDIT.md)
- [当前候选 JEPA 审计](/data1/LWR/vranlee/SERVER_ONLY/avis/JEPA/GOT-main/experiments/CURRENT_PAIR_JEPA_AUDIT.md)
- [filter/teacher 设计与 checkpoint 审计](/data1/LWR/vranlee/SERVER_ONLY/avis/JEPA/GOT-main/experiments/FILTER_JEPA_DESIGN_AUDIT.md)
- [DanceTrack seed 17 官方 CSV](/data1/LWR/vranlee/SERVER_ONLY/avis/JEPA/GOT-main/experiments/results/real_dance_temporal_v3/trackeval_official/official_metrics.csv)
- [DanceTrack seed 23 官方 CSV](/data1/LWR/vranlee/SERVER_ONLY/avis/JEPA/GOT-main/experiments/results/real_dance_temporal_seed23/trackeval_official/official_metrics.csv)
- [MOT17 seed 17 官方 CSV](/data1/LWR/vranlee/SERVER_ONLY/avis/JEPA/GOT-main/experiments/results/real_mot_temporal_v3/trackeval_official/official_metrics.csv)
- [MOT17 seed 23 官方 CSV](/data1/LWR/vranlee/SERVER_ONLY/avis/JEPA/GOT-main/experiments/results/real_mot_temporal_seed23/trackeval_official/official_metrics.csv)
- [train-only frozen-filter fallback](/data1/LWR/vranlee/SERVER_ONLY/avis/JEPA/GOT-main/experiments/results/mot_global_filter_jepa_trainonly_mot20_v2/results.json)
