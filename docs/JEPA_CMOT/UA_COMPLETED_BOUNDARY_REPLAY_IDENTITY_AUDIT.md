# UA完成适应边界的离线伪对偏差诊断

这是seed17固定教师C2的三个已完成域边界，最终4×4 MOT回测仍在运行。GT只用于本次离线诊断，未回流训练、priority或门控；sealed未用。

| stage | native帧 | 实际更新 | memory一致 | memory历史未匹配 | memory当前未匹配 | recent一致/历史未匹配/当前未匹配 |
|---|---:|---:|---:|---:|---:|---|
| 1 | 27440 | 3395 | 102 | 23 | 3 | 26/1/1 |
| 2 | 42140 | 5232 | 107 | 20 | 1 | 32/0/0 |
| 3 | 65430 | 8101 | 110 | 18 | 0 | 21/9/2 |

三阶段memory均为128。四种保持方法在相应边界的物理样本字段、frames/updates/seen逐项一致；old response/query随learner变化，不在物理样本相等断言中。当前样本中未出现confirmed identity mismatch或mixed history，不能据此声称全部伪对无偏：这是终态reservoir/recent的476个案例，不是442,278个确认对或按训练频次加权的抽样；一致只针对已匹配的history槽，无法匹配者单列，不作为正确或错误。

还原累计1,899条history观测，无clipped gaps。最大history feature误差5.96e−8、box误差5.53e−5像素；源物理frame/candidate keys可还原实际features。身份映射使用group-compatible nearest IoU≥.5、已匹配history主身份purity≥.75，因此属于有限离线代理。

Final memory中110一致、18历史未匹配；recent中21一致、9历史未匹配、2当前未匹配。这里没有确认错误组可以检验prediction need是否识别错误；不能把各组need均值直接当作选择机制因果证据。已完成的BDD审计存在错误案例，两数据集观察不能彼此替代。

[摘要、四组checkpoint SHA与命令](UA_COMPLETED_BOUNDARY_REPLAY_IDENTITY_AUDIT.json)记录完整范围；逐案例原文件保留在本地runs路径，SHA在摘要中，可复算而不删除失败/未匹配项。完整MOT成绩、AF与关系保持独立贡献待四阶段回测完成。
