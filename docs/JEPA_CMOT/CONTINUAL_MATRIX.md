# 连续域学习矩阵

状态：尚未执行正式连续适应；无新算法矩阵。

计划顺序：UA `sunny → cloudy → rainy → night`；BDD `clear → overcast → rainy → night`。第二 BDD 顺序 `clear → night → rainy → overcast` 使用相同视频和更新预算。

每个 stage 保存独立 snapshot，并以冻结 snapshot 评测所有域；评测不更新模型。所有独立视频重置轨迹。

将报告 HOTA、AssA、IDF1、DetA、MOTA、IDSW、final seen-domain average。Average Forgetting 使用各已学域历史最佳值减当前值；同时报告相对该域刚学完阶段的 BWT。FWT 的参考模型和未见域阶段固定在配置中。

没有填入历史单域值，因为它们不能作为本次 stage×domain 结果。
