# UA五种保持方法的实际边界预算与样本对齐

状态：三个完整适应域边界核验通过，最终4×4 MOT回测仍未齐备。固定教师/源checkpoint/源校准/学习配置一致，仅retention机制不同；没有通过GT调整任何采样或门控。

| 边界 | native帧 | 实际更新/optimizer step | seen | memory | recent | 对照数 |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 27440 | 3395 | 229359 | 128 | 28 | 5 |
| 2 | 42140 | 5232 | 337223 | 128 | 32 | 5 |
| 3 | 65430 | 8101 | 442278 | 128 | 32 | 5 |

对齐范围：每条九字段样本中的history features/geometry/mask/candidates/validity/physical candidate IDs/query IDs（字段0/1/2/3/4/7/8），reservoir与recent顺序、replay RNG、frames/updates/seen、确认供给统计及真实optimizer step。字段5/6为各learner自身保存的旧候选响应/旧query，随参数变化，不要求相同。

这不是逐batch训练trace，也不能由边界相同断言两个独立run每帧/每批必然相同。机制相关统计（如competition columns）不同应保留，不能强行要求一致。没有运行新增多种子任务。

最终均覆盖65,430 native帧、442,278 seen、8,101实际更新，memory128。普通RKD-D与column同供给/同预算的完整指标和配对统计待四阶段冻结回测完成；nonzero loss或样本对齐本身不是增益或新颖性证据。

[完整计数、五组各边界checkpoint SHA、protocol/代码hash及复现命令](UA_GRAPH_RKD_ACTUAL_BOUNDARY_PARITY.json)。
