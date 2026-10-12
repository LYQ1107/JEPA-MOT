# 当前结果

研究状态：ACTIVE，Phase A 工程验证。尚无正式新方法结论。

UA 与 BDD 的 GeneralTrack、官方 V-JEPA encoder+predictor、缓存、在线跟踪与 TrackEval 均通过真实 GPU smoke。每次仅 6 个观测，不能据此选择正式方法。三份历史权重 SHA256 与 manifest 一致；当前 V100 实际 autocast=FP16，新增依赖置于隔离目录。

同容量历史网络有 1,072,640 个参数。seed 17 的判别、预测、预测+ranking 三个目标各训练 200 步。输入为两个 sunny source_train 视频各 8 个连续原生帧，共 165 个源 GT 对象，训练与内部验证视频分离。源验证 rank-1 分别为 89.13%、100.00%、97.83%；预测误差分别为 0.75545、0.02035、0.19795。

未经训练的 last-observation rank-1 为 97.83%，误差 0.01533。这些小样本结果不能支持 JEPA 优于冻结历史，也不能支持 MOT 或持续学习收益。梯度正常；从 step 100 恢复到 200 的参数与连续训练逐位相同，已完成任务重跑会直接跳过。

UA 原生四域共 138,252 帧，旧输入只包含抽样帧。BDD 四域 train/val 视频集合全部重合。旧 UA sunny 与 BDD clear detector 都用 all 文件训练，和新源域评测集重合。新划分仍为提案，干净的源模型尚未建立。

历史 HOTA（不是本次重训）：UA GeneralTrack、geometry-null、V-JEPA 为 33.5983、38.3082、40.3552；BDD 为 35.1767、35.0900、35.3925。

原始工程记录见 ENGINEERING_EVIDENCE.json 和 EXPERIMENT_REGISTRY.csv。本地 checkpoint 与日志保存在 runs/JEPA_CMOT，数据从只读 NAS 读取。
