# 下一步

完成修正 native 类别接口后的 B0/B1、frozen-history、CE、预测、预测+ranking 和关系目标的完整开发视频对照。检测与特征队列继续准备 UA/BDD 的所有 adaptation/development 视频，sealed 不参与本阶段。

核查第一条实际在线训练的更新次数、可靠伪对、loss 与参数变化；跨域恢复必须同时载入 optimizer/replay/recent/counters，独立评测只加载权重并关闭更新。

随后建立完整顺序学习矩阵，比较无 replay、同容量 replay、普通 feature distillation 和候选关系保持。保持目标 GT 不进入训练、参数和 replay 跨视频持续、物理轨迹状态重置。

source 排序和一个完整开发视频仍不能支持最终结论；继续 no-history/projection/shuffle、两个域顺序、多种子和效率验证。
