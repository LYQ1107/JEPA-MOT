# FastReID current-pair probe audit

本文件审计 `current_pair_fastreid_probe.py` 的 FastReID 版本 probe。它只回答
“固定强外观 encoder 后，当前-pair JEPA 原型是否值得继续拆分”，不构成 MOT
benchmark、无泄漏结果或 tracker 归因。

## 产物与 provenance

审计对象：

```text
experiments/results/current_pair_fastreid_probe_mot20_fix2/results.json
```

命令：

```bash
/home/lwr/anaconda3/envs/AVI/bin/python experiments/current_pair_fastreid_probe.py \
  --dataset mot17 --max-frames 20 --epochs 2 --max-samples 2000 \
  --device cuda --out-dir experiments/results/current_pair_fastreid_probe_mot20_fix2
```

该 JSON 是旧脚本生成的，内部 `script_sha256` 为
`aa767f80c68ef6df45dea220b0da503af6548ca2207f9cc0fa9d046d8110c593`，且没有
`base_script_sha256` 或后来加入的 `quality_gate` 字段。审计时工作区脚本已变为
`bc7c88bcc1a2035333ca42be7f0ef6e74169de05942a0856681e81b50e464ad0`，base
`current_pair_jepa_probe.py` 已变为
`b23b8450c2fd7fb33fa94d0d316aeb23e2191afd6ad31b1619438e05a71eff11`。因此不应
声称当前源码可 bitwise 重现 fix2；任何新实验必须使用新目录、记录两个源码
SHA 和完整命令。

固定输入核对通过：MOT17 train/eval 序列为 02/04/09 与 05/11，样本数 772/260，
per-frame Hungarian IoU>=0.30，duplicate assignment=0；detector digest 为
`5358145d…445caf`/`2029ccb6…c024c`。FastReID 配置为
`/data1/LWR/vranlee/M4FTMoveOut4Doing/HybridSORT/fast_reid/configs/MOT17/sbs_S50.yml`
（SHA `5410bfe2…168d`），权重为
`/data1/LWR/vranlee/SERVER_ONLY/avis/masa/HybridSORT/pretrained/mot17_sbs_S50.pth`
（SHA `eb2e83af…d4fa`），输出 2048-D，3076 crops、200 次 batched inference。

## 关键结果（MOT17，20 帧）

| branch | Recall@1 | target cosine | hard-negative margin |
|---|---:|---:|---:|
| current_jepa | 0.1731 | 0.00942 | 0.00051 |
| current_only | 0.1423 | 0.00503 | 0.00070 |
| full_mask/history-only | 0.2385 | 0.00853 | -0.00099 |
| no_teacher (direct clean-target) | **0.4615** | **0.05154** | **0.01711** |
| both_clean | 0.2038 | 0.01010 | 0.00136 |
| random_frozen | 0.1577 | -0.00227 | 0.00343 |

Paired bootstrap（同一 260 records）显示 current_jepa 相对 no_teacher 的
Recall@1 差 `-0.2885`，95% CI `[-0.3462,-0.2346]`；margin 差 `-0.01660`，
CI `[-0.01860,-0.01469]`。相对 full_mask 的 Recall@1 差 `-0.0654`，CI
`[-0.1077,-0.0231]`。这不是 JEPA 思想的否证：当前 teacher 只是从零训练的
clean-crop MLP，而不是论文中冻结的预训练 tracking filter/model；它提供的
pseudo target 质量很差（见下节），所以 no_teacher 胜出是预期的目标/容量诊断，
不能直接解释为“JEPA 无效”。

## in-domain 权重警告

本地 BoT-SORT/HybridSORT 文档明确说 MOT17 SBS-S50 是 MOT17 专用 ReID，训练
patch 生成脚本 `fast_reid/datasets/generate_mot_patches.py` 对每个 FRCNN 序列
按时间切分：前一半写入 `bounding_box_train`，后一半写入 test。脚本只保留
`visibility > 0.4` 的行。对当前 MOT17 数据重算后，七个 FRCNN 序列前半段共有
487 个身份；checkpoint 的 `heads.weight` 正好为 `(487, 2048)`，与该训练身份
集合吻合。MOT17-05/11 的前 120 帧（序列长度约 837/900，前半段约 418/450）
因此既属于训练时间段，也属于训练身份集合。

所以该 encoder 的同 ID/异 ID 可分性是有用的**工程上限诊断**，但不能用于
正式无泄漏 A/B 或宣称 JEPA 提升。官方 BoT-SORT README 也把 validation 定义
为每条 MOT17 train sequence 的第二半；若使用 `mot17_half_sbs_S50.pth`，必须
确证其训练半段、只评估对应 held-out 半段，并避免把前 120 帧当 validation。
本工作区没有该 half checkpoint，不能假定它存在或自动替代 full weight。

DanceTrack 权重同样是数据集专用；HybridSORT 文档说明其 ReID 与 DanceTrack+
CUHKSYSU 联合训练。`dancetrack_sbs_S50.pth`（SHA
`debc78e7…e57a70`，约 12,353 个 classifier classes）若用于 DanceTrack val，
也应视为 in-domain upper bound；另一个 `dance_sbs_S50.pth`（约 373 classes）
来源/覆盖范围不明确，不能混用而不记录 provenance。

## teacher quality gate 审计

当前脚本后来加入的 gate 使用：

```python
teacher_validation = metric_teacher(teacher, eval_rec)
```

这会读取 eval GT clean crop/identity，故不是 held-out train gate；如果用它来
决定是否进入 tracker，就构成 eval-label tuning。fix2 artifact 甚至没有记录
该 gate。正确做法是从训练序列内部预先划出 identity/frame-disjoint held-out
subset，只在该 subset 上评估 frozen teacher 的 clean-target cosine、hard-
negative margin 和 Recall@1；阈值、seed 和失败规则在运行前固定，并把 eval
结果留到最终报告，不能参与放行。

fix2 中 teacher 的 MOT17 held-out-like 质量已经很弱：eval target cosine
`0.04173`、margin `0.01486`、Recall@1 `0.4346`；train target cosine
`0.16597`、margin `0.0924`、Recall@1 `0.9870`。这同时提示 train/eval gap 和
目标未校准问题，当前版本应判定 `teacher gate = FAIL`。

## 上限诊断与正式因果实验的边界

**可以保留的上限诊断**：固定上述 FastReID encoder/crop/normalization，报告
同 ID/异 ID cosine、teacher quality、candidate-count-adjusted chance，以及
各 branch 的 paired bootstrap；明确 GT identity history、数据集专用权重和
当前候选输入。

**正式因果实验必须改成**：

1. encoder、detector、crop、特征空间和阈值全部冻结且各分支完全相同；优先用
   ImageNet/跨数据集 encoder，或只在 train half 训练并在严格 held-out split
   评估。
2. teacher 是预训练/稳定的 association filter 或 prototype（可用 clean
   current + history + hard negatives 拟合），训练后 stop-gradient/freeze；
   不把从零 clean-embedding MLP 的输出当作论文同等 target。
3. student 输入 corrupted current，训练 corruption；线上默认输入 clean
   detector feature。`predict()` 不得隐式污染部署输入。
4. 用 oracle history 只作表示 upper bound；正式 tracker 需从 track lifecycle
   产生 history，并把 `omega_t` 严格延迟到 `t+1` 匹配，加入 leave-current-out
   /candidate-swap 检查，杜绝当前候选 autoencoding。
5. 对 raw/history-only/no-teacher/random controls 做同 detector digest 的
   paired comparison，再进入官方 TrackEval；没有一致的 HOTA/IDF1/IDSW 和
   post-gap 改善时不放行。

## Release decision

```text
FastReID fix2：仅 upper-bound/engineering diagnostic，不能当 benchmark。
teacher held-out gate：FAIL（且旧 artifact 未记录 gate；当前 gate 误用 eval）。
JEPA current-pair -> lagged tracker：NO-GO。
```

下一步应先修 teacher target、train-only held-out gate、部署 corruption 语义和
lagged causal probe；修补后的结果必须另存目录并重新审计 SHA/digest。相关的
handcrafted current-pair 审计见 `experiments/CURRENT_PAIR_JEPA_AUDIT.md`。
