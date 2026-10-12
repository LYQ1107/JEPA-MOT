# 文献与审稿问题审计

状态：首轮审计，创新性尚未获得实验支持。审计日期：2026-10-13（Asia/Shanghai）。

|工作/一手资料|已核实的重叠|本任务需要证明的区别|
|---|---|---|
|[GOT-JEPA，TCSVT 2026 / arXiv v5](https://arxiv.org/abs/2602.14771)|从历史及观测预测跟踪模型，teacher/student 和遮挡处理|MOT 多候选竞争、无标签跨视频/跨域模型学习和旧关联关系保持；不能声称首次预测跟踪模型|
|[FACT，原始预印本 2024](https://arxiv.org/abs/2409.07904)；[期刊 DOI](https://doi.org/10.1016/j.knosys.2026.115935)|在线学习外观并使用过去跟踪信息、两阶段关联|共享数据/预算下预测目标相对判别学习的增量，以及域序列遗忘；期刊网页访问受限，不把未读的修订细节当作事实|
|[TCEI，CVPR 2026 作者预印本](https://arxiv.org/html/2603.21629v1)|短期记忆和跨测试视频经验校准，利用可靠与不可靠案例|候选预测关系的保持及独立可靠性证据必须超出一般经验记忆；需 confidence-only、replay、普通蒸馏对照|
|[V-JEPA 2.1](https://arxiv.org/abs/2603.14482)|视觉 latent 的掩码预测、稠密/多层监督|官方 predictor 不直接输出目标身份或下一帧坐标；预训练强特征和本次 trainable identity predictor 的收益分开|
|[MPM，CVPR 2025 原文](https://openaccess.thecvf.com/content/CVPR2025/papers/Sun_Unsupervised_Continual_Domain_Shift_Learning_with_Multi-Prototype_Modeling_CVPR_2025_paper.pdf)|无标签连续域变化、多原型和域特定表示|这里的困难是身份历史污染、一对一候选竞争和关联关系遗忘；不能把 prototype 或域保持本身当作首次创新|

DOI、CVF 部分页面存在访问限制，使用作者 arXiv 和 CVF 已索引的原文内容补充；未核实内容保留为待核实，不据此下新颖性结论。

## Why JEPA?

同一个历史网络使用三个目标：ranking-only、prediction-only、prediction+ranking。目标空间固定，输入、初始化、样本、更新预算和可训练参数一致。分别报告预测损失和排序项的增量，并加入 last/EMA、no-history、feature projection、shuffle 控制。

## Why continual?

必须比较冻结、顺序微调、普通 replay、特征/输出蒸馏、候选关系保持。单域正增益不能证明连续学习。memory 内容选择和预测驱动的更新需求只有在比简单置信度与 hard replay 更有效时才保留。

## Why MOT-specific?

历史预测不能读取当前候选；评分发生在候选出现之后。必须分析同帧相似车辆、消失重现、错误匹配污染、候选集改变和分配约束。跨无关视频只保留参数和许可的 replay，不能继承物理 track ID。

## 尚缺证据

已有工程训练，但尚无正式完整视频、连续域矩阵或三种子结果，因此不能声称创新成立或已达到 CVPR 论文标准。后续继续检索 2025–2026 的预测关联、在线 MOT、predictive replay，并审计公开实现和训练协议。

## 追加核实

[GeneralTrack 作者仓库](https://github.com/qinzheng2000/GeneralTrack) 的 checkpoint 与本地 SHA256 逐位相同。[作者论文](https://arxiv.org/html/2406.00429v1) 说明其关联表示先以 KITTI flow 预训练，并在 BDD 上训练/评测；不把共享 BDD 监督先验描述为本次单天气 source-only 预训练。新 source-only detector 和 history learner 的源视频列表单独记录。

当前 reliability gating、reservoir replay、候选 KL 保持仍属于待验证组件，不能仅靠组合命名称为新方法。需与 CE+同样门控、普通 replay 和蒸馏使用相同预算进行对照。
