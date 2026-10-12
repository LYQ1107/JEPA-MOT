# DanceTrack / TrackEval 官方指标审计

本报告只审计已有 `real_dance_sanity_v2` 输出，不修改
`experiments/mot_dance_jepa_real.py`、LocateMOT、MASA 或 TrackEval reference。
审计脚本是 [`audit_trackeval_real.py`](./audit_trackeval_real.py)。所有暂存的
GT、tracker 文件和指标均写在输入结果目录下的 `trackeval_official_v2/`。

## 数据与格式核验

核验日期：2026-08-31。使用的真实根目录为：

```text
DanceTrack: /data1/LWR/vranlee/DATASETS/JDE/dancetrack
TrackEval:  /data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT/references/TrackEval-official
```

DanceTrack `annotations/val.json` 是 COCO-video 格式：25 个 video、25,508
张 image、225,148 个 annotation；annotation 含 `category_id=1`、`track_id` 和
`bbox=[x,y,w,h]`。对应的本地视频目录同时提供 MOTChallenge 风格
`val/<sequence>/gt/gt.txt`，例如 `dancetrack0004` 的行是
`frame,id,x,y,w,h,mark,class,visibility`，与 COCO 解析结果一致。原始
`seqinfo.ini` 为 `dancetrack0004: 1920x1080, 20 FPS, 1203 frames`（其余
验证序列长度也由各自 ini 读取）。

v2 tracker 文件是标准十列 CSV（`frame,id,x,y,w,h,conf,x,y,z`），不是检测
文件。例如：

```text
1,1,624.850,379.120,222.180,630.580,0.92520,-1,-1,-1
```

输入检测协议在 v2 中是 LocateMOT D-CTRL 的逐帧
`frame,x,y,w,h,score` 缓存（无 track ID），不是已有 tracker 输出。v2 结果
记录的 detector cache 根目录和权重 hash 保持不变。每个方法共享同一序列的
输出帧集合；检测 digest 在 v2 JSON 中按序列记录。

## 截断与 TrackEval 配置

v2 命令使用 `--max-frames 120`。TrackEval 会按 `SEQ_INFO` 的长度读取每一帧，
若直接给它完整的 DanceTrack GT，会把第 121 帧以后的行判定为 invalid
timestep。因此审计脚本在 `trackeval_official_v2/staging/` 创建只读源的
逐行副本：

```text
staging/gt/<seq>/gt/gt.txt                 # 只保留 frame <= 120
staging/gt/<seq>/seqinfo.ini               # seqLength=120
staging/trackers/<method>/data/<seq>.txt   # 只保留 frame <= 120
```

暂存目录中的 `SEQ_INFO` 是三个验证视频各 120 帧；脚本同时记录每个源文件、
暂存文件的 SHA-256、源文件实际最大帧和保留行数。GT 的 `mark/class` 原样保留，
tracker 文件不足十列时才补齐 `-1` 世界坐标列。没有使用 GT 来改变 tracker
输出或阈值。

TrackEval 调用等价于：

```python
Evaluator({"USE_PARALLEL": False, "BREAK_ON_ERROR": True, ...})
MotChallenge2DBox({
    "GT_FOLDER": ".../staging/gt",
    "TRACKERS_FOLDER": ".../staging/trackers",
    "TRACKERS_TO_EVAL": methods,
    "BENCHMARK": "MOT17", "CLASSES_TO_EVAL": ["pedestrian"],
    "SEQ_INFO": {seq: 120 for seq in sequences},
    "SKIP_SPLIT_FOL": True, "TRACKER_SUB_FOLDER": "data",
})
metrics = [HOTA, CLEAR, Identity]
```

`BENCHMARK=MOT17` 这里只是让 TrackEval 走 MOTChallenge person-box 的标准
预处理路径；DanceTrack 评估序列全部是 `class=1/person`。这不是 DanceTrack
官方 leaderboard（视频被截断，也没有宣称完整 val 结果）。当前 vendored
TrackEval 仍引用 `np.float/np.int`，脚本只在进程内安装兼容别名，未编辑
reference checkout。

可复跑命令：

```bash
cd /data1/LWR/vranlee/SERVER_ONLY/avis/JEPA/GOT-main
python -u experiments/audit_trackeval_real.py \
  --result-dir experiments/results/real_dance_sanity_v2 \
  --dataset dance --max-frames 120 \
  --out-dir experiments/results/real_dance_sanity_v2/trackeval_official_v2 \
  2>&1 | tee experiments/results/real_dance_sanity_v2/trackeval_official_v2_run.log
```

脚本 hash（本次成功运行）：

```text
audit_trackeval_real.py
d7e8040663eef5c7a9c05e7464124aa1e5d82bbdec12bb64c33641e5e7f58da8
```

## 官方结果（截断 smoke）

完整逐序列结果在
`real_dance_sanity_v2/trackeval_official_v2/official_metrics.csv`，机器可读
审计记录在 `audit.json`。以下是 TrackEval 的 `COMBINED_SEQ`，HOTA/AssA/DetA
取官方 alpha=0.50，IDF1/MOTA/IDSW 取官方 Identity/CLEAR 字段；不是脚本中
自定义的 `*_proxy`：

| method | HOTA@0.50 | AssA@0.50 | DetA@0.50 | IDF1 | MOTA | IDSW |
|---|---:|---:|---:|---:|---:|---:|
| raw | 0.76047 | 0.66305 | 0.87221 | 0.76734 | 0.90023 | 38 |
| gated_ema | 0.75123 | 0.63034 | 0.89531 | 0.74093 | 0.90023 | 38 |
| history_ema | 0.78521 | 0.67846 | 0.90877 | 0.78594 | 0.90098 | 40 |
| random_frozen | 0.77380 | 0.67429 | 0.88799 | 0.77553 | 0.91190 | 27 |
| jepa_hist | 0.76822 | 0.65967 | 0.89464 | 0.76065 | 0.91039 | 31 |
| jepa_hist_assoc | 0.76703 | 0.65762 | 0.89464 | 0.76399 | 0.91002 | 32 |
| no_teacher | 0.76703 | 0.65762 | 0.89464 | 0.76399 | 0.91002 | 32 |

因此在这个首轮三视频、120 帧、固定检测的审计中，JEPA predictor 没有胜过
`history_ema`，也没有显示出相对于 `jepa_hist` 的 hard-negative
`jepa_hist_assoc` 增益；`no_teacher` 与后者相同。`random_frozen` 的结果提醒
需要保留容量/随机控制，不能把所有身份改善归因于 teacher。该结论只适用于
当前 frozen crop descriptor、生命周期和短前缀，不外推为完整 DanceTrack
结论。

## 已记录的阻塞与后续官方验证方案

第一次审计运行中 evaluator 已成功完成，但结果提取函数误把
`get_name` 当成类方法；错误和修复过程保留在
`trackeval_official_run.log`，最终 `trackeval_official_v2/audit.json` 的
`status=complete` 且 `failures=[]`。这不是数据格式失败。

扩展到完整 val 时应：

1. 使用所有 25 个 val 视频的完整 `seqinfo.ini` 长度，不再做 120 帧截断；
2. 为每个方法提供每个序列的十列 MOT 文件，并在运行前检查每帧 ID 唯一；
3. 固定 `BENCHMARK`/`DO_PREPROC`/`CLASSES_TO_EVAL`，只在 calibration/train
   视频上选择阈值，禁止用 val 结果调参；
4. 保存 TrackEval 的 `HOTA`, `CLEAR`, `Identity` summary/detailed 文件、
   source/detector digest 和命令环境；FPS 由 tracker 侧单独测量，不能从
   TrackEval 推断。

MOT17 也可复用同一脚本：将 `--dataset mot17`、`--mot-root` 和相应结果目录
传入，GT 路径改为 `train/<seq>/gt/gt.txt`，其余 `SEQ_INFO`/暂存规则不变。
