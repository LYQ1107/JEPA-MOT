# 最终科研报告（进行中的证据草稿，尚非最终交付）

状态：ACTIVE。尚未冻结最终模型或超参数，sealed评测没有执行。多seed/order、UA完整持续矩阵与部分读数/结构对照仍未完成；不能给出项目最终成功或完成结论。

截至当前，已经真正训练并评测了trajectory-level latent predictor、同容量判别/联合目标、关系响应目标、需求优先与关系保持，以及普通projection/Transformer/pair/mean/EMA等控制。旧的“只完成工程smoke”占位说明已经过时。

模型预测的是当前目标的identity latent或候选响应状态，输入只有严格过去的轨迹描述符、box motion与quality metadata。当前候选在历史查询形成之后才用于评分；source可用GT，target仅用独立native证据与延迟确认构造伪对。候选column保持和一般KD/RKD都不能仅靠形式或nonzero loss称新颖性。

BDD seed17 forward第一波、固定教师需求对照、candidate-competition保持均已完成4stage×4domain真实development矩阵。最强静态geometry为39.9022 HOTA，第一波最高feature KD final39.4747，H2最高prediction-priority39.4684，C2 column39.3263。当前数据不支持这些方法稳定优于强简单对照；stage趋势、AF和弱FT对照上的正差不能替代最终强baseline比较。

BDD完整frozen Transformer纯预测39.9719比同架构CE39.4402高，但相对geometry只有+.0698且视频CI跨0；这尚非跨seed、持续学习或独立测试证据。普通mean/EMA未可靠超过geometry。对象crop与连续source clips的完整结果继续进入[当前结果](CURRENT_RESULTS.md)，所有负域保留。

实际GPU图像链路已测12个完整视频case：联合online UA1.833 FPS、BDD1.603 FPS，未达到实时。保存伪样本上的实际update call显示column新增成本；该计时排除图像/伪对生成，不能冒充live FPS。GPU/server共享与峰值显存统计范围见[实时效率报告](FULL_NATIVE_LIVE_EFFICIENCY_RESULTS.json)。

目前缺乏证据证明相机运动是根因；观测bbox速度与feature gap不是camera-motion或真实遮挡。Memory/局部GT诊断不能外推为全部训练错误率，也不能反馈到门控。现有证据不足以支撑已验证的CVPR持续学习主张。

最终报告仍须逐一回答任务书17问，补齐UA/多seed/逆序/RKD/pure-online/官方读数结果与冻结后的held-out，并明确最终checkpoint、hash、命令、所有主表和局限。完整待办与已完成范围见[完成证据表](EVIDENCE_STATUS.md)，实际指标、代码版本、路径和失败链见[当前结果](CURRENT_RESULTS.md)、[实验索引](EXPERIMENT_REGISTRY.csv)、[连续矩阵](CONTINUAL_MATRIX.md)及[新颖性审计](NOVELTY_AND_REVIEWER_AUDIT.md)。
