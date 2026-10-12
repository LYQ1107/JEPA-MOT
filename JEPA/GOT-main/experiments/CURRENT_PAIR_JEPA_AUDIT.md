# Current-pair JEPA probe: audit and release gate

本文件记录 `current_pair_jepa_probe.py` 的当前-pair smoke 结果、协议审计和
下一步最小修补方案。它是给审计/规划使用的 reflection，不是 MOT benchmark
结果，也不授权把 probe 直接接入 online tracker。

## 被审计的产物

修复后的短 smoke 产物是：

```text
experiments/results/current_pair_probe_mot20_fix2/results.json
experiments/results/current_pair_probe_mot20_fix2/validation.csv
```

命令为：

```bash
/home/lwr/anaconda3/envs/AVI/bin/python experiments/current_pair_jepa_probe.py \
  --dataset mot17 --max-frames 20 --epochs 3 --max-samples 4000 \
  --out-dir experiments/results/current_pair_probe_mot20_fix2
```

该 JSON 内记录的脚本 SHA-256 是
`8e0ea05c56f6556174d0852c89b3e06a7d4c2d0b78d1aad354b24d3402f3900c`。审计时当前
工作区脚本 SHA-256 已变为
`eb09e034b59d5c4c038c722397cb4a774eaba214ef1bcf72fce390cbbb4e56b0`，因此不能
声称用当前源码 bitwise 复现 fix2；后续 120-frame 结果必须保留其实际脚本
快照/SHA，且不得覆盖 fix2 目录。

fix2 的关键数据完整性检查通过：训练/评估序列分别为 MOT17-02/04/09 与
MOT17-05/11；样本数 772/260；per-frame Hungarian IoU>=0.30；duplicate
assignment 为 0；GT history 明确只使用当前帧之前的帧。训练和评估 detector、
feature、clean-target digest 已写入 JSON（分别为
`5358145d…445caf`/`2029ccb6…c024c`、`45a8361b…d4b2d`/
`743c149b…9602`、`89a73a6a…e9ec9`/`8e0cd63a…38a12`）。

## 结果（MOT17，20 帧；仅 representation probe）

| branch | target cosine | hard-negative cosine | margin | Recall@1 |
|---|---:|---:|---:|---:|
| current_jepa | 0.10999 | 0.10619 | 0.00380 | 0.1615 |
| current_only | 0.16664 | 0.16461 | 0.00203 | 0.1538 |
| full_mask/history-only | 0.09492 | 0.09348 | 0.00144 | 0.1462 |
| no_teacher (direct target regression) | 0.32164 | 0.32099 | 0.00065 | 0.1385 |
| both_clean (diagnostic) | 0.12908 | 0.12519 | 0.00390 | 0.1423 |
| random_frozen | 0.08972 | 0.09178 | -0.00207 | 0.1577 |

每个样本有 3--9 个候选，按候选数计算的 chance Recall@1 平均为 0.1509。
因此 current_jepa 的 0.1615 只是略高于 chance。260 个样本的 paired bootstrap
显示 current_jepa 相对 current_only 的 Recall@1 差为 0.0077，95% CI
`[-0.0346, 0.0500]`；相对 random 的差为 0.0039，95% CI
`[-0.0615, 0.0654]`。DanceTrack 20-frame smoke 也没有一致的 Recall@1
优势（current_jepa 0.2727，no_teacher 0.2867，random 0.2937）。

## 目标和 teacher 质量审计

代码中的目标现在是：

```text
teacher: T(H_<t, GT-box clean crop, g_t) -> omega_hat_t
student: S(H_<t, corrupted detector crop, g_t) -> stopgrad(omega_hat_t)
```

teacher 先用同一训练样本直接回归 clean GT crop，再复制参数并标记 frozen；
它不是 GOT-JEPA 论文中的预训练 tracking predictor/filter。独立重算冻结
teacher 的输出与 clean target 的质量如下：

```text
MOT17 train: cos=0.554, margin=-0.0094, Recall@1=0.0738
MOT17 eval : cos=0.494, margin=-0.0061, Recall@1=0.1577
Dance train: cos=0.217, margin= 0.0022, Recall@1=0.1267
Dance eval : cos=0.196, margin= 0.0126, Recall@1=0.2587
```

这说明 3 epochs 的 teacher 尚未形成可靠的 pseudo tracking model；student
cosine 不能被解释成对高质量 teacher 的 JEPA invariance。`both_clean` 也不能
作为可信 ceiling，因为其 teacher 本身很弱。任何 online 集成前必须先通过
held-out teacher quality gate（建议记录 clean-target cosine、hard-negative
margin、Recall@1，并预注册最低阈值），否则应判定为 probe failure。

另一个语义问题是 `predict()` 当前在非 `full_mask/history_only/both_clean`
分支对候选再次调用 `corrupt_feature()`。该 corruption 是训练增强，不应默认
出现在部署路径；若 tracker 直接调用此函数，会主动污染线上 detector feature。
部署接口应默认使用未污染候选，corruption 只在训练或显式 stress-eval 开关
下启用。

## 不能作因果结论的原因

1. **GT history**：probe 用 GT identity 组装 `H_<t>`；真实 tracker 不知道
   当前候选属于哪个 `gid`，所以这是 representation upper-bound，不是
   association result。
2. **当前候选泄漏/autoencoding**：student 输入了当前正候选（虽有 mask/noise），
   其 latent 随后在同一帧和所有候选比较。必须用 leave-current-out 或把
   `omega_t` 延迟到 `t+1` 匹配，不能用同一候选同时生成 prototype 和评估该
   候选。
3. **candidate-order audit 的含义有限**：列表本身没有进入 predictor，故
   permutation 后 max diff=0 是代码结构不变量，不是模型学到的 set invariance。
4. **control 不同构**：`current_only` 仍看到当前 box geometry；`full_mask`
   仍看到当前 geometry；`no_teacher` 的 clean-target 与 JEPA 的 teacher
   latent 目标不相同。应把这些 branch 当诊断 controls，而不是同一任务上的
   leaderboard。

## 最小修补方案（仍只做 probe，不接 tracker）

按以下顺序改，所有新结果另存新目录：

1. 固定源码快照和数据 digest；输出中增加 `source_sha256`、命令、环境、seed、
   corruption 参数和 teacher/student 参数 hash。
2. 将 corruption 从 `predict()` 默认路径移除，增加显式 `corrupt_train`/
   `stress_eval` 开关；训练仍保留 mask+Gaussian，部署语义保持 clean detector
   feature。
3. teacher 训练完成后，在**训练序列内部的 held-out frame/identity split**
   上计算 teacher→clean target cosine、hard-negative margin、Recall@1；没有
   达到预注册 quality gate 时只输出失败审计，不训练/不推广 student。
4. validation 同时报告 `student→teacher`、`student→clean-target` 和
   `teacher→clean-target`，并保留一个 identity/nearest-neighbour 直接基线。
   不使用 eval GT 调参。
5. 新增 causal diagnostic：同一 `omega_t` 只用于下一帧 `t+1` 的候选排序，
   并增加 leave-current-out/candidate-swap 检查；仍保留 GT-history 版本但明确
   标为 oracle upper bound。

## Release gate

截至 fix2：

```text
120-frame representation probe: CONDITIONAL GO（只能在上述修补后作诊断，
不宣称提升；需新 SHA/新目录/teacher quality gate）。
online tracker integration: NO-GO。
```

只有在 120-frame、多 seed、两条序列上 teacher quality 通过，且 causal
leave-current-out/lagged probe 相对 `raw`、`history-only`、`no_teacher` 和
`random` controls 方向一致后，才可申请下一阶段 tracker smoke；tracker 阶段
还必须单独做官方 TrackEval 和 post-gap/IDSW 审计。
