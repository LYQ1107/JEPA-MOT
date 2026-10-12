# 下一步

等待两个源 detector 及 source-only frozen V-JEPA 特征完成；登记 checkpoint 哈希和实际耗时。统一生成完整 development/adaptation 视频检测流，sealed 推理与评测暂不运行。

首轮使用每个读数族内部同容量的三个目标，并固定 prediction gaps 1/2/4/8。在完整 development 跟踪比较 last、判别、预测、预测+ranking，分析纠错、反改错、候选竞争和重现间隔。随后比较预测对象、可靠性门控和关系保持机制，推进两个数据集完整顺序域适应。

已有训练/延迟更新/类感知评测都可执行，但尚无算法结论。坚持目标 GT 不进入训练；已冻结方法和超参数后才运行 sealed。
