# 下一步

收集正在运行的全部 development frozen 控制和 forward seed17 六组持续学习结果，逐域核实完整视频覆盖、实际更新数和参数/optimizer/replay 跨视频保留。完成矩阵后，用相同视频重采样汇总 HOTA、AssA、IDF1 差值及不确定性。评测继续保持独立、关闭更新，sealed 不参与选择。

信息控制已提示 shuffle 和分数尺度也能改变结果；收集常数 affinity 对照，检查 candidate ordering、融合导致的纠错/反改错与伪对质量，不能把固定 backbone 或几何收益归因于 JEPA。

新 source clips 使用同样图片预算，修正 UA uniform source 的时间间隔与逐帧在线 history 不一致；跑同容量 CE / 预测 / 联合，做相同 source-only 校准，比较完整开发域。保留之前 uniform 结果。

随后补齐 prediction-need 与 confidence/hard-example 的相同实际更新预算对照、候选关系保持与普通蒸馏的区别、反向域顺序和 seed23/42。只有在方法与超参数冻结后执行 sealed，最终报告真实端到端效率与负结果。
