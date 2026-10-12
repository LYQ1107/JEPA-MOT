# 当前结果

状态：ACTIVE，工程验证通过，正式源训练正在运行。没有本次完整视频或持续学习结果，尚不能支持 H1/H2/H3。

## 有效工程证据

UA 与 BDD 的实际图像、GeneralTrack、官方 encoder+predictor、缓存、在线跟踪与 TrackEval 均已在 V100 GPU 跑通。旧路径 smoke 每次 6 个观测，仅验证接口。

重建后的新检测输入使用 6 个连续原生 UA 帧。新跟踪器的 native、frozen-last、已训练 predictor 路线均通过；查询历史检查没有当前帧混入。额外插入一个空检测帧后，禁用 JEPA 的输出与逐帧调用原始 native tracker 一致，时钟正确推进。

修正跨视频 RGB buffer 后，源训练 v2 使用两个 sunny source_train 视频各 8 帧，共 165 个源 GT 对象。三个目标各有 1,072,640 参数，seed 17、200 步：判别 rank-1 86.96%，预测 95.65%，预测+ranking 91.30%；误差分别 0.75411、0.01921、0.18261。这是工程小样本，不作为研究方法选择依据。v1 缓存和相应指标已作废，保留原始记录。

延迟在线学习器通过因果、恢复和独立视频状态检查：同帧不更新，下一帧确认后更新，恢复后参数一致；清空视频 pending/history 时保留 optimizer/replay。合成评测检查证明 ignore 区域不会产生额外 FP、未列入 XML 的帧不被当作空 GT、BDD 类别错误不会匹配。

## 正式运行与输入边界

UA 100 视频、140,131 图片帧，其中 XML 列出 138,252 帧；BDD 111 视频、22,074 帧。所有 native 图像参与跟踪，缺少 XML 的帧单独标记。VIDEO_SPLITS.json 已冻结；source、adaptation、development、sealed 按视频互斥，sealed 尚未用于任何方法评测。

两个 source-only Faster R-CNN 从 COCO 公共权重初始化，各训练 3,000 步。UA 16 源视频/2,048 训练帧；BDD 25 源视频/3,174 帧。训练使用固定每视频 128 帧预算，评测保留完整视频。旧 all-source detector 不再用作本次正式输入。

官方 encoder 最后层为 1024 维，官方 predictor 为 1664 维。正式对照会分别训练匹配维度/读数的历史网络；同一读数族的判别/预测/预测+ranking 容量一致。正在生成源特征，并检查 dense current-image ROI 与 object-crop 的成本差异。

GeneralTrack 权重与作者公开文件 SHA256 完全一致。它是所有方法共享的 BDD 预训练关联先验，不能声称完整系统只有当前 source-weather GT 预训练。该限制见 FROZEN_PROTOCOL.json。

原始工程记录与失效原因见 ENGINEERING_EVIDENCE.json、FAILURE_ANALYSIS.md、EXPERIMENT_REGISTRY.csv。权重、缓存和日志保存在本地 runs/JEPA_CMOT；数据从只读 NAS 读取。
