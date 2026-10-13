# 完成证据与仍需执行的工作

状态：ACTIVE。用户于2026-10-13要求限制多种子耗时；当前优先seed17，批量seed23/42已撤下，详见SINGLE_SEED_PRIORITY_USER_UPDATE.json。完整三seed矩阵不再作为当前自动执行门槛；未完成的重复验证须如实列为结论局限。此表用于接续研究，不是完成验收。GitHub 研究分支已同步；只有相应完整 split/stage 的实际结果支持对应范围。

| 要求 | 当前证据 | 状态/范围 |
|---|---|---|
| 环境、官方权重、GPU训练/原生跟踪/评测闭环 | ENGINEERING_EVIDENCE.json、checkpoint SHA、各 source metrics、FULL_NATIVE_LIVE_EFFICIENCY_RESULTS | 两数据集已实际执行 |
| 冻结完整视频划分与独立物理状态 | VIDEO_SPLITS、FROZEN_PROTOCOL、native_protocol_v3 manifest、逐视频 complete/causal stats | 已落实；sealed未用 |
| 相同source预算的CE/prediction/joint | source_history_v1/v2与multiseed的实际metrics；SOURCE_MULTI_SEED_COMPLETE_RESULTS | 两数据集源训练完成；不能等价于三seed MOT完成 |
| 普通projection/MLP/Transformer/pair/mean/EMA强对照 | BDD_FULL_STRONG_HEAD_DEVELOPMENT_RESULTS、BDD_FULL_FROZEN_PROTOTYPE_RESULTS | 两数据集全部development完成；UA Transformer pure与EMA差的CI含0，独立优势未成立 |
| 预测目标与源时间支持诊断 | source_clips、source relation、crop source、BDD/UA clips与两数据集crop完整结果 | 已有真实结构探索和负结果；未所有读数/数据集部署完成 |
| 历史1/2/4/8与间隔/运动分组 | BDD_FULL_HISTORY_LENGTH_DEVELOPMENT_RESULTS、BDD_COMPLETE_LOCAL_EVENT_GAP_MOTION_STRATA | 两数据集历史完整；UA更长历史未得稳定增益；观测gap/速度不是实际occlusion或camera-motion证明 |
| Encoder与official predictor正式配对读数 | OFFICIAL_PAIRED_READOUT_REGISTRATION、UA_FULL_OFFICIAL_PAIRED_READOUT_RESULTS.json、完整202-frame接口验证、paired_readout_v1源BDDcache | BDD正式源cache/六head/物理候选池及两读数完整development完成；masked predictor pure优于同读数CE/last，但未可靠超geometry；UA六head、物理源样本池和17视频两读数完整development均已核验 |
| 无标签跨域θ/optimizer/replay/RNG连续与旧域回测 | BDD_FIRST_WAVE_FULL_CONTINUAL_RESULTS、BDD_FIXED_TEACHER_FULL_CONTINUAL_RESULTS、BDD_COMPETITION_FULL_CONTINUAL_RESULTS、UA_C2_FULL_CONTINUAL_ANALYSIS.json | BDD seed17 forward三个完整family；UA C2 四组0–3已完整，主线/H2/RKD其它阶段仍未完成 |
| 相同actual update预算 | H2 final1096步/41563确认；C2/RKD final1094步/41176确认及边界样本相同；真实optimizer step核验 | 同family已验证；不同family/自教师样本不能假定一致 |
| A/B/C机制独立贡献 | latent/response source目标、H2需求vsappearance-hard、C2 feature/row/column完整矩阵、UA_C2_FULL_CONTINUAL_ANALYSIS.json | UA C2 final HOTA 62.61165–62.83441、AF .30282–.59870；有实质探索，但未显示稳定超过强简单方法的证据 |
| 普通关系蒸馏RKD控制 | GENERIC_RELATIONAL_KD_ACTUAL_VERIFICATION、BDD/UA_GRAPH_RKD_ACTUAL_BOUNDARY_PARITY | BDD full4×4完成，三边界五组样本/1094更新对齐；RKD-D/column增量无可靠证据；UA三边界五组物理样本/RNG/8101更新对齐通过，UA完整回测仍待完成 |
| 多seed最终MOT均值/标准差 | source seeds17/23/42已完成；BDD23/42 forward已按用户要求暂停，后续多seed reverse撤下 | 未完成；Transformer三目标seed23/42队列已按用户要求撤下，不能用源seed或video CI替代 |
| 反向域顺序与pure-prediction在线贡献 | REMAINING_SEQUENCE_REGISTRATION、原始/补齐configs、实际queued_sequences handles | BDD pure seed17 forward完整；未可靠超自身frozen或geometry；seed17反向/UA仍运行；批量多seed延后 |
| BDD seed17 reverse完整矩阵 | BDD_REVERSE_SEED17_FULL_CONTINUAL_ANALYSIS.json；stage0–3四域native complete markers、实际snapshot/optimizer核对 | 四阶段已验证；固定seed开发结果，不替代UA逆序、跨seed或sealed证据 |
| UA seed17 H2/C2已完成阶段 | UA_H2_COMPLETED_STAGES_ANALYSIS.json、UA_C2_FULL_CONTINUAL_ANALYSIS.json、UA_C2_COMPLETED_STAGES_1_2_3_ANALYSIS.json；native complete markers、实际snapshot/optimizer核对 | C2四阶段及AF/BWT/FWT已验证；H2仍为stage1/2，主线/RKD后续阶段未完成 |
| UA seed17主线已完成阶段 | UA_MAIN_COMPLETED_STAGE1_ANALYSIS.json；stage1四域native complete marker、实际snapshot/optimizer核对 | stage1已验证；stage2/3及完整最终矩阵未完成 |
| UA seed17通用RKD已完成阶段 | UA_RKD_COMPLETED_STAGES_0_1_ANALYSIS.json；stage0/1四域native complete marker、source checkpoint parity与optimizer step核对 | stage0冻结源与stage1已验证；stage1为27440 frames/3395 updates/229359 confirmed-seen、HOTA等域平均62.77113（相对source +0.07434）；stage2–3及最终遗忘仍未完成 |
| UA official paired readout | UA_FULL_OFFICIAL_PAIRED_READOUT_RESULTS.json；encoder/predictor各17视频、25,460 native frames、相同prepared inputs与source audit | 两种official读数的完整frozen development已验证；masked predictor predictive HOTA 63.22523、predictive+ranking 63.31112，仍不是持续学习、sealed或跨seed证据 |
| 配对视频统计、全域负结果 | BDD full frozen、各partial/final报告、TrackEval raw counts bootstrap2000 | 已完成对应scope；fixed-seed开发、无selection correction |
| 实际图像到轨迹FPS/latency/显存/更新成本 | FULL_NATIVE_LIVE_EFFICIENCY_RESULTS、RETENTION_ACTUAL_UPDATE_COSTS | 12个整视频live案例+真实update-call成本；非全场景/新增FP32读数live速度 |
| Pseudo确认偏差与改变匹配事件 | BDD/UA_COMPLETED_BOUNDARY_REPLAY_IDENTITY_AUDIT、BDD_FULL_RANK_REWARD_RESULTS、event strata | 离线局部/终态memory诊断；非所有训练对或传播的counterfactual因果结论 |
| Future/targetGT严格限制 | 查询history严格past、source-only校准、target stream labels空、实际GT_read=false；官方masked readout扰动验证 | 已落实已运行路径；最终仍应审计所有新配置/代码版本 |
| 文献/真正新颖性 | NOVELTY_AND_REVIEWER_AUDIT | 原文审计已完成；未因机制命名宣称首次/论文成立 |
| 方法冻结后的独立held-out测试 | sealed split仍未消费 | 待开发选择与超参数冻结后执行 |
| 最终中文17问、主表、完整矩阵、最优模型/命令/SHA/索引 | CURRENT_RESULTS、完整JSON/SVG、EXPERIMENT_REGISTRY、FINAL_RESEARCH_REPORT草稿、REPRODUCTION与PRIMARY_SOURCE_CHECKPOINT_INDEX | 已整理17问证据草稿、实际主表/逐域/BDD八组矩阵；方法冻结、完整测试与最终附录仍未完成，不应标记goal complete |

不能由已完成的BDD三个family缩小成功定义为“所有要求完成”。必要的UA/seed/order/pure/RKD/readout/held-out工作仍在追踪；现阶段负结果也不等于在这些工作完成前结束研究。
