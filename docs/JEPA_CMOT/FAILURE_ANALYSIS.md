# 失败证据和诊断

## 工程 smoke v1：缺失 timm

Observation：UA 原始 GeneralTrack 跟踪和 TrackEval 成功，官方 V-JEPA 构造失败 `ModuleNotFoundError: timm`。

处理：在项目的 `runtime/site-packages` 安装 timm 0.9.16，保留 v1 日志。V100 使用 FP16，缓存记录实际精度。

结果：UA v2、BDD v1 的官方 predictor 缓存、在线跟踪和评测闭环均通过。此结果仅证明工程可执行。

## 历史 temporal adapter

旧 source temporal adapter 的 deploy_alpha=0，source held-out 直接预测更差。该证据不属于本次新训练；不加载为有效 warm start。

下一假设：无约束替换历史外观可能损害原有身份区分；共享零初始化残差预测器有利于分离训练目标效果与随机初始化扰动。需 prediction / ranking / combined 同容量对照验证。

## 数据协议问题

UA：MVI_39501 XML/图片 540 帧，历史 prepared 80 个抽样帧。不能将该输入当作完整原生视频。

BDD：train/val 文件数量一致，必须审计视频集合；源 detector 配置使用 clear_all，存在与新 source 测试划分重合的风险。需要根据训练视频清单重建冲突的源模型/缓存。

下一步：冻结完整视频划分之前先完成覆盖与训练 provenance 审计；不通过删除困难视频或更改测试分母解决。

## 源视频缓存重置修复

源缓存生成器只清空 frame_buffer，未清空 rgb_buffer；官方 predictor 裁剪读数因此可能携带前一独立视频像素。添加 rgb_buffer.clear()，作废 v1 小型源训练比较并重建 v2。目标缓存原本已正确清空两个缓冲。

## dense 读数的 source 排序与校准风险

多间隔 source-validation 的 frozen-last rank-1 为 55.98%，teacher 当前 GT latent 上限为 93.15%；11,831 样例中 teacher 上限有 1,538 个候选最高分并列，说明 dense ROI 分辨率不能区分所有身份。所有新比较使用最高分并列的分数信用，避免正样本总放在第 0 列导致虚高 rank-1。最初 source-history v1 的第一模型在这一规则冻结前已导入代码；保留原训练/中断记录，新比较重启至 v2。

初始 CE-only source rank-1 约 76.38% 而 cosine prediction error 约 0.61，存在“排序改善但绝对相似度尺度变差”的风险。旧 residual-only 正增益融合可能偏向高 raw cosine 的预测目标，需在正式模型比较中加入相同 source-supervised 校准与 row-relative scorer 控制，不能将尺度差异误归因于 JEPA。

源历史 v2 已冻结统一 tie 规则，三个正式目标均完成 2,000 步。共同 affinity 校准只拟合 source 内部验证上的两个单调参数，不改变候选排序；raw-cosine 路线保留用于判断改进是否来自尺度变化。校准结果及每个模型的 checkpoint/cache SHA256 见 SOURCE_UA_CALIBRATION.json。

## 六类到原生八类的语义接口

恢复工具把 COCO 六类直接映射为 0..5，但 GeneralTrack 原始 sig 将 0/1 视为 person/rider，2..5 为车辆，6/7 为两轮车。修正映射为 [0,2,3,4,6,7] 并把输出转换回六类。另发现 legacy JEPA positive residual 能让原生类别 mask=0 的边重新达到匹配阈值；新 scorer 对跨原生类别组的相似度置零。旧 full-video v1 原始记录保留并标记失效，v2 使用相同检测行和缓存重跑全部控制。这些是共同接口修正，不能归因于 JEPA。
