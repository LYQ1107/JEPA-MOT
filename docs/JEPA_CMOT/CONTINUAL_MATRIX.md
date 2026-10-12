# 连续域学习矩阵

状态：ACTIVE。BDD forward seed17 的六组训练已完成 source→overcast→rainy→night 和44个适应视频，四个参数快照保存；stage×domain 冻结评测正在执行，完整矩阵尚未生成。UA forward 继续 cloudy；同教师 H2 五组在两个数据集持续运行。未评测 sealed。

路径：runs/JEPA_CMOT/continual_v1/{dataset}_forward_seed17；同教师比较为continual_h2_v1。所有 snapshot 评测关闭更新，不反馈到训练；物理 track/pending 在视频边界重置，参数、optimizer、replay/recent和RNG持续。

顺序：UA sunny→cloudy→rainy→night；BDD clear→overcast→rainy→night。反向顺序仍待执行，保持相同视频、更新预算和评测集合。

每个 stage 评测所有四个 development 域，报告 HOTA/AssA/IDF1/DetA/MOTA/IDSW。最终平均对四域等权，不按视频数量调权。Average Forgetting 使用旧域在学会后到最终阶段之前的最佳值减最终值，主表截断为非负，同时报告未截断值和相对学完当时的BWT；FWT使用该方法实际stage0未适应目标域分数为参考。continual_analysis.py强制完整矩阵后才生成最终汇总。

冻结预测器参考来自实际完整stage0评测，同一静态权重、校准和测试输入在每stage重复该结果；不借用旧历史抽帧实验值。配对统计从各视频 raw TrackEval 计数重新聚合。

BDD六组stage1已完成19视频/四域回测，详细记录BDD_CONTINUAL_STAGE1_PARTIAL_RESULTS.json；stage2/3仍待MOT回测。source内部完整23,623对上的24快照descriptor诊断已独立完成，见BDD_COMPLETE_SOURCE_VALIDATION_SNAPSHOT_AUDIT.json，不能填充缺失MOT矩阵。新增reverse17、UA23/42及pure-prediction在线任务已登记队列，状态仍QUEUED而非结果。
