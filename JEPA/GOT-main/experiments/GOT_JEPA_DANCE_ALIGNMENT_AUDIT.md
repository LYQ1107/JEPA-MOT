# GOT-JEPA frozen-feature DanceTrack audit

日期：2026-08-31。该实验是独立的 bounded 120 帧验证，不修改旧的 crop80
脚本、LocateMOT、MASA 或生产 tracker。

## 固定协议

- 训练序列：`dancetrack0001,0002,0006`；评估序列：`0004,0007,0014`。
- 输入检测：LocateMOT D-CTRL 逐帧缓存，行格式严格为
  `frame,x,y,w,h,score`，无 track ID；`oracle_boxes=false`。
- 官方 checkpoint：`/data3/testdata/vranlee/GOT_JEPA_checkpoints/GOT_JEPA_378.tar`，
  SHA-256 `5800f0fdc748708431743c5520e960bb6e45f4a786dacc0dbbc9107c8ae51a98`。
- 每个检测/训练 clean-GT crop 都经过同一个冻结 DINOv2 + ToMP head，输出
  `[256,27,27]`，空间平均并 L2 归一化为 256-D；缓存的 sidecar 同时绑定
  checkpoint 和输入框 hash。
- `TemporalPredictor` 使用最近 4 个 detector-history token（外观、可观测
  box/速度/age/score/missing mask），训练时加入 mask/noise/identity-mix，
  并保留 `no_teacher`、`random_frozen` 控制。当前候选只进入 Hungarian cost。
- 评估 tracker 函数不接收 GT；所有 track 文件写完并 hash 后才解析 val GT 做
  诊断，随后用 vendored TrackEval 在 `seqLength=120` staging 上评估。

## 官方 TrackEval 结果

数值依次为 `HOTA_mean / HOTA@0.50 / AssA@0.50 / DetA@0.50 / IDF1 / MOTA /
IDSW`，来自 `/data2/usr_for_deadline/mot_dance_jepa_got_full120_s17/trackeval_official/official_metrics.csv`。

| sequence | raw | jepa_temporal | no_teacher | random_frozen |
|---|---:|---:|---:|---:|
| 0004 | .785728/.859378/.861618/.857143/.877949/.865385/4 | .679790/.750051/.642944/.875000/.756923/.861111/6 | .682583/.752908/.647853/.875000/.756923/.858974/7 | .691136/.760107/.660301/.875000/.763077/.858974/7 |
| 0007 | .888070/.968632/.973597/.963693/.981511/.946875/2 | .888070/.968632/.973597/.963693/.981511/.946875/2 | .888070/.968632/.973597/.963693/.981511/.946875/2 | .889605/.969163/.974664/.963693/.981511/.963542/0 |
| 0014 | .556858/.584832/.425547/.803738/.609805/.885993/27 | .559142/.584921/.430514/.794707/.616979/.886808/26 | .559142/.584921/.430514/.794707/.616979/.886808/26 | .540585/.603934/.407519/.895015/.596253/.880293/28 |
| **COMBINED** | **.727609/.783502/.708034/.867014/.789288/.904367/33** | **.708561/.762432/.671468/.865718/.770690/.903991/34** | **.709049/.762940/.672363/.865718/.770690/.903614/35** | **.705667/.775518/.657380/.914886/.762135/.906627/35** |

因此该严格的“官方 256-D pooled representation + unconditional temporal
prediction”在此前缀是 **NO-GO**：相对 raw，JEPA 的 combined HOTA mean
`-0.01905`、HOTA@0.50 `-0.02107`、AssA `-0.03657`、DetA `-0.00130`、IDF1
`-0.01860`、MOTA `-0.00038`、IDSW `+1`。这不是 checkpoint/cache/GT 泄漏
问题；`got_protocol_audit.json` 与 TrackEval audit 均通过，训练样本
`x=[2841,1060]` 且 `history_future_violations=0`。

## 反思与下一步

官方 pooled appearance 已使 raw association 很强，当前 predictor 的无条件
替换会扰动容易帧。下一候选应预注册为 train-only ambiguity gate：仅在 native
top-1/top-2 cost margin 低于训练后缀确定的阈值时融合 JEPA，其他帧完全保留 raw；
必要时再做 train-only whitening/temperature calibration。不能把该 NO-GO 数值
与旧 crop80 provisional 正反馈结果混报，也不能用评估 GT 反向调阈值。

主要机器产物：

- `/data2/usr_for_deadline/mot_dance_jepa_got_120/smoke.json`
- `/data2/usr_for_deadline/mot_dance_jepa_got_full120_s17/results.json`
- `/data2/usr_for_deadline/mot_dance_jepa_got_full120_s17/got_protocol_audit.json`
- `/data2/usr_for_deadline/mot_dance_jepa_got_full120_s17/provenance.json`
- `/data2/usr_for_deadline/mot_dance_jepa_got_full120_s17/trackeval_official/official_metrics.csv`
