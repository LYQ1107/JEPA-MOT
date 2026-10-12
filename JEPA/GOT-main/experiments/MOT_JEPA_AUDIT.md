# MOT-JEPA preliminary audit

本文件是对 `mot_jepa_synthetic.py` 的可复现性与归因审计；数值结果是合成
机制 smoke test，**不是** MOTChallenge/BDD/VisDrone benchmark。

## 复现实验

基础 preliminary（保留为四方法结果，未改写）：

```bash
python -u experiments/mot_jepa_synthetic.py \
  --train-sequences 12 --eval-sequences 12 --frames 72 --objects 6 \
  --embedding-dim 12 --history-len 4 --scenarios mild,hard \
  --methods baseline,jepa_inv,jepa_cov,no_teacher --seeds 0,1,2 \
  --output experiments/results/mot_jepa_preliminary.json
```

gate 归因 control（同一配置、另存文件，不覆盖上面的结果）：

```bash
python -u experiments/mot_jepa_synthetic.py \
  --train-sequences 12 --eval-sequences 12 --frames 72 --objects 6 \
  --embedding-dim 12 --history-len 4 --scenarios mild,hard \
  --methods baseline,gated_ema,jepa_inv,jepa_cov,no_teacher --seeds 0,1,2 \
  --output experiments/results/mot_jepa_gated_control.json
```

AVI 兼容性已执行四方法 preliminary 命令（把 `python` 换成
`/home/lwr/anaconda3/envs/AVI/bin/python`），产物为
`mot_jepa_preliminary_avi.json/.csv`；Python 3.8.0/NumPy 1.23.4 的指标输出与
系统 Python 3.12.2/NumPy 1.26.4 一致。**尚未单独保留 AVI gate-control
artifact**，如需该环境的 gate control 需复跑第二条命令。脚本只 import NumPy，
不 import torch、DINO、timm 或 scipy。

## 产物校验

以下是本轮文件 SHA-256（便于审计时确认没有静默改写）：

```text
mot_jepa_synthetic.py                 9fbaadb4466ce9342fa56c6a416c4d9dc2a6ef310b8bfc21425fb233296069d5
README_MOT_JEPA.md                    7f0f49195ec3cb91b5db21a87047b5d4f047f2a28a0d2d4599e0e46ea41e1613
mot_jepa_preliminary.json             1347e4202972748c22cfb00fad3c3266605c405bc9fd8b2c2d7c7c6b3d5ff56e
mot_jepa_preliminary.csv              47cb31d164a467e2ba97cb40a7517238305f1533c292dad3f9fccc8233aac2c8
mot_jepa_preliminary_pre_gated.json   1347e4202972748c22cfb00fad3c3266605c405bc9fd8b2c2d7c7c6b3d5ff56e
mot_jepa_preliminary_pre_gated.csv    47cb31d164a467e2ba97cb40a7517238305f1533c292dad3f9fccc8233aac2c8
mot_jepa_gated_control.json           c405c1026939631dd52df19639e82049fd6bee4581e9fe9ef2935ca0a4e9629e
mot_jepa_gated_control.csv            72ef1ad82ad444a028c6911c75741bf42e43e0e1242dfa589b8b807f8b8c0aa6
mot_jepa_preliminary_avi.json        85aee9651de5e0b32d20b9612a0415c14ceb9d50ef2af08618331b9a41173701
mot_jepa_preliminary_avi.csv         47cb31d164a467e2ba97cb40a7517238305f1533c292dad3f9fccc8233aac2c8
```

结构检查已通过：

* 每个 scenario/seed 的五个方法（preliminary 中为四个）共享同一
  `detections_sha256_16`；因此 baseline/JEPA 读取同一检测输入。
* 训练 seed 为 `100000 + seed*1000 + k`，评估 seed 为
  `200000 + seed*1000 + k`；训练/评估序列不重叠。
* `jepa_inv` 和 `jepa_cov` 的 target 是 synthetic clean branch，拟合时不把
  clean target 放入 tracker context；`no_teacher` 将 target 行随机打乱。
* `_global_idtp` 使用实际匹配的 row/column，不假定 row 连续；指标函数有手工
  assignment case 检查。

## 结果与归因

四方法 preliminary 的均值（3 seed；每 seed 内再平均 12 条 eval 序列）：

```text
scenario  method      IDF1(proxy)  IDSW   identity-consistency  occ-consistency
mild      baseline       0.8813    11.42          0.8700             0.7692
mild      jepa_inv       0.8946    11.31          0.8898             0.7741
mild      jepa_cov       0.8948    11.31          0.8899             0.7751
mild      no_teacher     0.8822    11.06          0.8722             0.8018
hard      baseline       0.7819    10.47          0.7557             0.6905
hard      jepa_inv       0.7647    10.86          0.7399             0.6892
hard      jepa_cov       0.7680    10.83          0.7434             0.6938
hard      no_teacher     0.7643    10.11          0.7383             0.6794
```

关键混杂拆分（gate control）：

```text
scenario  method      IDF1(proxy)  IDSW   与 gated_ema 的 IDF1 差
mild      gated_ema      0.9324    11.72          0.0000
mild      jepa_inv       0.8946    11.31         -0.0378
mild      jepa_cov       0.8948    11.31         -0.0376
hard      gated_ema      0.7884    10.31          0.0000
hard      jepa_inv       0.7647    10.86         -0.0237
hard      jepa_cov       0.7680    10.83         -0.0204
```

因此不能把 preliminary 中相对 raw `baseline` 的 mild 增益归因于 JEPA：
`gated_ema`（没有 predictor，只有相同低分观测冻结策略）更好，说明主要信号
来自 visibility gate，当前线性 predictor 反而低于 gate-only control。`jepa_cov`
与 `jepa_inv` 几乎相同，toy 设置尚未显示 covariance loss 的独立贡献。

## 下一步（真实 MOT 前的最小拆分）

1. 在真实 detector stream 上固定同一 detections，比较 `raw EMA`、`gated EMA`、
   `JEPA predictor + gated EMA` 三组；先报告官方 CLEAR MOT/IDF1/HOTA。
2. 用真实高置信 crop 作为 stop-gradient teacher，训练 student 处理 mask、
   copy-paste distractor、blur 和短时缺失；不要把 GT visibility 作为推理输入。
3. 单独做 `invariance`、`invariance+variance`、`invariance+covariance` 消融，
   并按 occlusion length/回归帧数分层报告 IDSW。
4. predictor 稳定后再接入 ToMP 的 `JEPA_predictor_cls/breg` filter；当前结果
   不足以决定重训 DINO/CoTracker 或宣称真实 MOT 提升。
