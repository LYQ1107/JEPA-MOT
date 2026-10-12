# MOT17 TrackEval 格式交叉核验（`real_mot_sanity_v5`）

这是对 DanceTrack 审计脚本的额外格式/指标交叉核验，不是新的 tracker 实验，
也不修改 `mot_dance_jepa_real.py` 或 MOT17 原始数据。输入为已有
`experiments/results/real_mot_sanity_v5`：MOT17-04-FRCNN 验证前 30 帧，
MOT17 FRCNN `det/det.txt`，7 个方法的十列输出。

复跑命令：

```bash
python -u experiments/audit_trackeval_real.py \
  --result-dir experiments/results/real_mot_sanity_v5 \
  --dataset mot17 --max-frames 30 \
  --out-dir experiments/results/real_mot_sanity_v5/trackeval_official \
  2>&1 | tee experiments/results/real_mot_sanity_v5/trackeval_official_run.log
```

官方 reference 为
`/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT/references/TrackEval-official`；
脚本 hash 与 DanceTrack 审计相同（`d7e8040663eef5c7a9c05e7464124aa1e5d82bbdec12bb64c33641e5e7f58da8`）。
TrackEval 运行使用进程内 NumPy 旧别名兼容，不改 reference。

## 结果（COMBINED_SEQ，官方 alpha=0.50）

| method | HOTA | AssA | DetA | IDF1 | MOTA | IDSW |
|---|---:|---:|---:|---:|---:|---:|
| raw | 0.67596 | 0.87998 | 0.51925 | 0.65150 | 0.51651 | 3 |
| gated_ema | 0.67596 | 0.87998 | 0.51925 | 0.65150 | 0.51651 | 3 |
| history_ema | 0.67596 | 0.87998 | 0.51925 | 0.65150 | 0.51651 | 3 |
| random_frozen | 0.69247 | 0.92348 | 0.51925 | 0.68252 | 0.51730 | 2 |
| jepa_hist | 0.69247 | 0.92348 | 0.51925 | 0.68252 | 0.51730 | 2 |
| jepa_hist_assoc | 0.69247 | 0.92348 | 0.51925 | 0.68252 | 0.51730 | 2 |
| no_teacher | 0.69247 | 0.92348 | 0.51925 | 0.68252 | 0.51730 | 2 |

数值与输入结果中的 proxy 趋势一致，但这里是 TrackEval 的真实 HOTA/CLEAR/
Identity 实现。前缀只有 30 帧，且只覆盖一个视频；不能作为 MOT17 leaderboard
结果。完整机器可读结果位于
`experiments/results/real_mot_sanity_v5/trackeval_official/`。
