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

已有正式 source 训练和两个完整 source 域开发视频的受控结果，但尚无完整连续域矩阵或三种子结果，因此不能声称创新成立或已达到 CVPR 论文标准。后续继续检索 2025–2026 的预测关联、在线 MOT、predictive replay，并审计公开实现和训练协议。

## 追加核实

[GeneralTrack 作者仓库](https://github.com/qinzheng2000/GeneralTrack) 的 checkpoint 与本地 SHA256 逐位相同。[作者论文](https://arxiv.org/html/2406.00429v1) 说明其关联表示先以 KITTI flow 预训练，并在 BDD 上训练/评测；不把共享 BDD 监督先验描述为本次单天气 source-only 预训练。新 source-only detector 和 history learner 的源视频列表单独记录。

当前 reliability gating、reservoir replay、候选 KL 保持仍属于待验证组件，不能仅靠组合命名称为新方法。需与 CE+同样门控、普通 replay 和蒸馏使用相同预算进行对照。

[MOTIP，CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Gao_Multiple_Object_Tracking_as_ID_Prediction_CVPR_2025_paper.html) 已将轨迹关联建模为基于轨迹上下文的 ID 解码。不能声称首次预测身份。本任务需要验证的是候选出现前的 latent forecasting 及其无标签连续适应价值。另需区分 [TrackTrack，CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/papers/Shim_Focusing_on_Tracks_for_Online_Multi-Object_Tracking_CVPR_2025_paper.pdf) 的关联/初始化改进与参数持续学习；当前共同 geometry-null 规则本身不是新贡献。

## 2026-10-13 新近原文核实：在线预测与保持并非空白

[JEPA-TTT（2026-09-30 arXiv v1）](https://arxiv.org/html/2610.00722v1) 已研究 frozen encoder、trainable latent predictor、跨 episode 保留参数/optimizer/replay 的自监督 test-time training。其对象是 action-conditioned planning，使用逐时间 offset 的 growing dense replay；原文 limitations 明确没有测试多次 dynamics shift、回到旧 dynamics 或遗忘。不能声称首次跨独立 episode 持续适应 JEPA。这里可检验的差别是 MOT 无标签身份关联中的伪对误差、一对一竞争以及多天气旧域回测；这是研究定位，尚未被结果支持。

[AdaJEPA（2026-06-30）](https://arxiv.org/abs/2606.32026) 同样从实际 next-state transition 做闭环预测适应。只将 observed future latent 作为自监督信号不足以构成新颖性。

[TCA，CVPR 2025 原文](https://openaccess.thecvf.com/content/CVPR2025/papers/Ni_Maintaining_Consistent_Inter-Class_Topology_in_Continual_Test-Time_Adaptation_CVPR_2025_paper.pdf) 已通过 class centroid 拓扑、一致性与类内紧致保持持续适应的关系结构。不能将泛化的 relation/topology retention 或 row-output KL 本身称为首次创新。若继续构建 MOT-specific 保持，需要显式区分同帧动态身份的 row/column 候选竞争与固定类别 centroid，提供同样 replay 和蒸馏预算的独立证据。

以上只依赖作者 arXiv 和 CVF 原文；未用第三方总结作技术证据。当前 row KL 仍按普通输出蒸馏对照解释。

## Candidate-column保持的待验证区别

新增C2显式记录同一已观测frame中，被多个过去轨迹query评分的同一个detector candidate，保持其跨query归一化的column竞争分布；普通row输出蒸馏在同memory和budget下作为对照。区别在动态物理track与检测一对一竞争，既不同于固定class-centroid拓扑，也不能仅靠row KL命名。teacher仍为旧forecast query，训练没有目标GT。首个完整适应视频证明loss路径/记忆/更新量运行，尚未证明独立收益或足够论文新颖性。需完整矩阵、对普通KD的增量、额外计算成本与错误归因。

## 关系保持的额外边界

[RKD，CVPR 2019 原文](https://openaccess.thecvf.com/content_CVPR_2019/html/Park_Relational_Knowledge_Distillation_CVPR_2019_paper.html) 已用样例间距离和角度蒸馏结构关系；[Comparative Knowledge Distillation，WACV 2025 原文](https://openaccess.thecvf.com/content/WACV2025/papers/Xu_Comparative_Knowledge_Distillation_WACV_2025_paper.pdf) 同样研究样例组关系与教师推理预算。因此不能以增加一个column归一化或跨query KL本身宣称新的蒸馏范式。C2的可检验定位限于：候选出现前的identity forecast，在动态同帧一对一竞争中保留旧关系，同时用无GT教师跨天气持续更新。完整旧域保持、相对普通KD/通用关系KD的增量和额外开销仍是必要证据；当前只有实际路径验证。

C2额外成本已经用实际after_assignment分支与真实保存样例做受控测量：column相对row p50从7.576升到10.731ms，50步有266列真实激活。它在完整BDD开发矩阵相对row仅+.023 HOTA，因此当前数据不支持“新增关系值得该计算开销”的结论；不能把nonzero loss或可实现性当论文新颖性。RKD全域效果仍待真实回测。
