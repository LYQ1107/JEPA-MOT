# 下一步

1. 审计旧 detector 与 GeneralTrack 的训练视频；重建存在已知重合的 source-only detector。
2. 构建 UA 全原生帧输入和 BDD 视频互斥划分。推理输入排除 target GT，评测标签分开保存；冻结类别、ignore 规则与检测流。
3. 帧开始时冻结历史查询，接入新预测器。所有控制使用相同生命周期，空检测帧更新时钟；跨视频保留 learner/optimizer/replay，重置轨迹 ID 和历史。
4. 在完整 development 视频比较预测、判别和冻结历史，诊断纠错与反改错；推进关系预测和保持机制的第二、第三轮结构性探索。
5. 完成两个数据集顺序适应、反向域顺序、旧域回测、多种子与效率验证。

工程 smoke 不用于方法选择；last-observation 在小型源验证达到 100% rank-1，需以更有信息量的预测间隔和困难关联验证 H1。
