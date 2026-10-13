# 连续域学习矩阵

状态：ACTIVE。BDD forward seed17 的六组训练已完成 source→overcast→rainy→night 和44个适应视频，四个参数快照保存；第一波六组完整stage×domain矩阵已生成。UA主线六组已完成45个适应视频并保存stage0–3边界快照，stage0–2冻结回测和指标已核验；通用 RKD-D 已完成 stage0–2 冻结回测，主线 stage3 正在外部节点续跑，RKD stage3尚未开始，固定教师及其它对照另有完整矩阵。未评测 sealed。

路径：runs/JEPA_CMOT/continual_v1/{dataset}_forward_seed17；同教师比较为continual_h2_v1。所有 snapshot 评测关闭更新，不反馈到训练；物理 track/pending 在视频边界重置，参数、optimizer、replay/recent和RNG持续。

顺序：UA sunny→cloudy→rainy→night；BDD clear→overcast→rainy→night。反向顺序仍待执行，保持相同视频、更新预算和评测集合。

每个 stage 评测所有四个 development 域，报告 HOTA/AssA/IDF1/DetA/MOTA/IDSW。最终平均对四域等权，不按视频数量调权。Average Forgetting 使用旧域在学会后到最终阶段之前的最佳值减最终值，主表截断为非负，同时报告未截断值和相对学完当时的BWT；FWT使用该方法实际stage0未适应目标域分数为参考。continual_analysis.py强制完整矩阵后才生成最终汇总。

冻结预测器参考来自实际完整stage0评测，同一静态权重、校准和测试输入在每stage重复该结果；不借用旧历史抽帧实验值。配对统计从各视频 raw TrackEval 计数重新聚合。

BDD六组stage1已完成19视频/四域回测，详细记录BDD_CONTINUAL_STAGE1_PARTIAL_RESULTS.json；stage2/3仍待MOT回测。UA主线stage0–2已由实际四域指标重建并通过source parity/optimizer核验，见UA_MAIN_COMPLETED_STAGES_1_2_ANALYSIS.json；stage3正在回测，不能填充缺失MOT矩阵。UA通用RKD-D stage0–2已通过native complete marker、source parity和optimizer核验，见UA_RKD_COMPLETED_STAGES_1_2_ANALYSIS.json；stage3仍缺失，不能填充最终遗忘指标。source内部完整23,623对上的24快照descriptor诊断已独立完成，见BDD_COMPLETE_SOURCE_VALIDATION_SNAPSHOT_AUDIT.json，不能填充缺失MOT矩阵。新增reverse17、UA23/42及pure-prediction在线任务已登记队列，状态仍QUEUED而非结果。

BDD first-wave stage0/1/2已完成，H2与C2 stage0/1已完成；明确存于部分阶段报告，不填未完成stage3。固定教师预算已用真实checkpoint的optimizer step/frames/confirmed/seen核对。当前H2 prediction priority不优于普通appearance-hard，C2 column相对row无实质增量；完整final矩阵仍需实际回测。

H2 stage2同预算回测已完成，见BDD_FIXED_TEACHER_STAGE2_RESULTS.json。prediction-need priority并未超过ordinary appearance-hard priority。新增frozen mean/EMA源校准与完整部署进行中；冻结prototype属于物理历史基线，并非参数持续学习，不把其每帧prototype更新计为optimizer更新。

第一波BDD完整六组final矩阵见BDD_FIRST_WAVE_FULL_CONTINUAL_RESULTS.json；原始6指标与完整4×4表全部包含。强静态geometry HOTA39.9022高于所有六组final（最佳feature KD39.4747），没有达到JEPA优于强同协议对照的目标。完整性仅限第一波seed17 forward，不推广到UA、多seed、reverse或最终sealed。

BDD H2完整五组4×4矩阵已完成，所有组1096步/8811帧/41563确认，详见BDD_FIXED_TEACHER_FULL_CONTINUAL_RESULTS.json。prediction-priority final39.4684，高于普通hard39.4031的区间跨0，且阶段趋势不一致；不支持稳定独立优势。完整性仍限seed17 forward development。

BDD C2完整4组4×4矩阵已完成，全部1094步/8811帧/41176确认，详见BDD_COMPETITION_FULL_CONTINUAL_RESULTS.json。Column最终39.32630，只比row39.30327多.02302、forgetting .22007 vs .25077；未形成实质量级优势，RKD-D与跨seed/顺序仍需完成。
