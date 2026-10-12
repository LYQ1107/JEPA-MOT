# 下一步

完成正在运行的全 development frozen 对照与 forward 连续矩阵；优先汇总已经完整的域，严格标注 source/target、seed和评测覆盖。主表始终使用相同 detector/native frame/caches，逐域报告负结果。

UA clips三目标已经完成源训练与校准，正在部署全部开发视频；BDD clips继续生成。目标object-crop的 source校准和完整开发视频特征/匹配对照已启动。历史1/2/4/8使用相近容量、相同图像与2,000×64预算源训练，然后做匹配长度的冻结推理。

收集 H2 固定教师五组的相同实际更新预算结果，以及 H3 replay/feature/row-relation distillation 的独立增量。row-KL仍按普通输出蒸馏解释；完成这一波结果后，决定是否需要MOT动态候选竞争的结构性保持。

继续反向域顺序和seed23/42的真正在线跟踪验证，保持目标GT不参与训练/伪对门控。开发选择冻结后才执行sealed。

live_stream已经跑通真实图像/两环境detector IPC/encoder/query/关联/更新；完整适应视频计时等待GPU5的source clips队列释放。独立benchmark checkpoint不能用于研究训练；共享GPU数字明确标记，最终再核实实际fps、p50/p95和显存。

C2还需与通用RKD-D距离保持比较，相同教师/负例/九字段memory/实际更新预算；实际完整视频验证完成，两个数据集的完整序列已排队。不要把候选column的形式或非零loss当成新颖性/防遗忘证据。

BDD连续clips的四目标完整开发集部署已补排；UA历史长度回测迁移只在当前完整视频终态之后，GPU5先完成现有真实链路benchmark。
