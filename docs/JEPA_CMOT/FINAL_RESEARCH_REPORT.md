# 最终科研报告（进行中的证据草稿，尚非最终交付）

状态：ACTIVE。用户于2026-10-13要求减少多种子耗时，当前集中seed17，批量seed23/42任务已撤下；完整三seed不再是当前自动执行门槛，仍不得声称已验证跨seed稳健性。本稿回答任务书17问，表格只采用已完成范围的实际结果。尚未冻结最终方法与超参数；sealed测试未执行，UA主线/RKD后续持续矩阵、UA逆序与多seed尚未齐备。BDD seed17 reverse 与 UA C2/H2 已完成四阶段矩阵，但不能据本稿声明全项目完成或CVPR主张成立。

所有分数以百分点表示；开发集按域分别聚合TrackEval后取四域等权平均。视频bootstrap是固定seed开发探索的不确定性，未做模型选择校正，不能替代跨seed均值/标准差或独立测试。以下不同训练目标和不同frozen读数的点估计不自动构成等信息、等算力或等伪对供给的因果对照。

## 1. 采用了哪个最终模型？

尚未选择最终模型。当前主框架为冻结GeneralTrack与V-JEPA、训练轻量轨迹历史预测器；MLP、Transformer和无可训练预测器的EMA均保留为候选/对照。UA Transformer pure与EMA差的区间含0；BDD官方masked predictor pure也未可靠超过geometry。因此不能将最大开发点估计直接封为最终方法。最终checkpoint路径、SHA和方法冻结声明待seed17完整顺序/独立评测证据及必要的小范围重复验证后补齐。

## 2. JEPA在MOT中究竟预测什么？

Candidate A预测当前目标的固定identity latent，q=Pθ(H(<t))，以stop-gradient当前正例latent作目标，并拆开纯cosine prediction和prediction+ranking。Candidate B探索候选关联响应目标与同帧候选竞争关系，当前候选仅在历史query形成后参与评分。Candidate C用独立native匹配证据和延迟确认构造在线伪对，比较confidence、prediction-need、appearance-hard优先及feature/row/column/RKD保持；这些机制的有效性分别验证，不用一个结果支持三个假设。

## 3. 为什么采用这个预测对象？

固定身份表征允许拆开预测目标、历史输入、匹配规则与在线更新，避免student/teacher同时漂移；source正例来自同GT轨迹，负例来自同帧候选。响应/竞争目标旨在检验MOT特有一对一分配和候选相似性是否比单纯feature regression有价值。官方V-JEPA latent并非直接预测坐标或天然具有身份监督，其读数优劣须用真实MOT验证。

## 4. 为什么不能简单使用MLP、普通Transformer或历史平均替代？

当前不能排除替代。MLP/Transformer是预测器架构，JEPA是学习目标，二者不是互斥方法。主对照采用相同descriptor输入、约66万参数、source 2,000×64预算；pair分类器、last/mean4/EMA、projection、shuffle/random均真实执行。UA EMA63.04514已接近Transformer pure63.16389，差+.11875、CI[-.27443,.28213]；MLP pure略低于EMA。BDD强geometry39.90215也未被纯预测稳定超过。相同容量不意味着所有计算与输入信息完全一致。

## 5. JEPA训练目标带来多少独立增益？

同架构/读数的目标差：BDD dense MLP pure比CE +.43789 HOTA CI[.0661,.7329]；Transformer pure比CE +.53167 CI[.10729,.99374]。UA Transformer pure比CE +.68317 CI[.08575,1.29499]。但相对强非JEPA方法的独立优势尚未成立。正式BDD masked predictor pure比同读数CE +.53837 CI[.20395,.82959]，比geometry仅+.20203 CI[-.20709,.59719]。不同读数有当前像素信息、维度、frozen参数与算力差别，不能把source cosine error的跨空间大小解释为MOT收益。全部比较和限制见各完整结果JSON。

## 6. UA-DETRAC和BDD100K分别取得什么结果？

下表为完整seed17 frozen development，UA17视频/25,460 native帧、BDD19视频/3,844帧；不是最终held-out。同一dataset内共享detector/native输入/current dense features与几何规则。所有完整方法及负结果保留在[UA强对照](UA_FULL_STRONG_HEAD_DEVELOPMENT_RESULTS.json)和[BDD强对照](BDD_FULL_STRONG_HEAD_DEVELOPMENT_RESULTS.json)。

| 数据集 | 方法 | HOTA | AssA | IDF1 |
|---|---|---:|---:|---:|
| UA-DETRAC | MLP/native | 62.16421 | 65.24941 | 74.29056 |
| UA-DETRAC | MLP/geometry_null | 62.27898 | 65.06942 | 74.55576 |
| UA-DETRAC | MLP/frozen_last | 62.68248 | 66.01533 | 75.27816 |
| UA-DETRAC | MLP/discriminative | 61.98899 | 64.61923 | 74.24776 |
| UA-DETRAC | MLP/predictive | 62.99334 | 66.64353 | 75.78808 |
| UA-DETRAC | MLP/predictive_ranking | 62.69680 | 66.06268 | 75.29288 |
| UA-DETRAC | Transformer/discriminative | 62.48072 | 65.53855 | 74.97376 |
| UA-DETRAC | Transformer/predictive | 63.16389 | 66.99867 | 75.99836 |
| UA-DETRAC | Transformer/predictive_ranking | 62.62472 | 65.86084 | 75.14756 |
| UA-DETRAC | Pair/pair_MLP | 62.37338 | 65.24778 | 74.77341 |
| UA-DETRAC | Prototype/frozen_mean4 | 62.50597 | 65.61115 | 75.05222 |
| UA-DETRAC | Prototype/frozen_ema | 63.04514 | 66.71684 | 75.93486 |
| BDD100K | MLP/native | 39.26758 | 45.45072 | 45.53272 |
| BDD100K | MLP/geometry_null | 39.90215 | 46.06395 | 46.84283 |
| BDD100K | MLP/frozen_last | 39.38565 | 45.26869 | 46.26338 |
| BDD100K | MLP/discriminative | 39.39121 | 45.32986 | 46.03992 |
| BDD100K | MLP/predictive | 39.82910 | 46.25516 | 46.79766 |
| BDD100K | MLP/predictive_ranking | 39.48615 | 45.48594 | 46.37294 |
| BDD100K | Transformer/discriminative | 39.44024 | 45.45352 | 46.25770 |
| BDD100K | Transformer/predictive | 39.97191 | 46.54873 | 47.03864 |
| BDD100K | Transformer/predictive_ranking | 39.47068 | 45.49437 | 46.22081 |
| BDD100K | Pair/pair_MLP | 39.24595 | 45.08773 | 45.76168 |
| BDD100K | Prototype/frozen_mean4 | 39.59214 | 45.74879 | 46.52141 |
| BDD100K | Prototype/frozen_ema | 39.65004 | 45.84466 | 46.36949 |

BDD官方paired frozen读数另有完整19视频结果：encoder pure39.84562、masked predictor pure40.10419。UA官方paired也已完成17视频/25,460 native frames：encoder pure62.66640、masked predictor pure63.22523，predictive+ranking分别为62.71962与63.31112。它们具有不同视觉信息/额外frozen计算，单列于[BDD官方读数结果](BDD_FULL_OFFICIAL_PAIRED_READOUT_RESULTS.json)和[UA官方读数结果](UA_FULL_OFFICIAL_PAIRED_READOUT_RESULTS.json)，不混进上述相同dense读数表；这些是固定seed frozen development，不是持续学习、sealed或跨seed证据。

## 7. 两个数据集各域HOTA、AssA、IDF1是多少？

下表列出相同dense输入的geometry、MLP pure、Transformer pure与EMA；完整其它方法、DetA/MOTA/IDSW也在上述JSON，不能只据某个正域下结论。

| 数据集 | 方法 | 域 | HOTA | AssA | IDF1 |
|---|---|---|---:|---:|---:|
| UA-DETRAC | MLP/geometry_null | sunny | 56.90711 | 59.72874 | 70.30979 |
| UA-DETRAC | MLP/geometry_null | cloudy | 71.87623 | 77.38509 | 86.58212 |
| UA-DETRAC | MLP/geometry_null | rainy | 49.76460 | 51.05292 | 59.49750 |
| UA-DETRAC | MLP/geometry_null | night | 70.56798 | 72.11093 | 81.83363 |
| UA-DETRAC | MLP/predictive | sunny | 56.91312 | 60.00269 | 70.78138 |
| UA-DETRAC | MLP/predictive | cloudy | 71.93844 | 77.54718 | 86.69608 |
| UA-DETRAC | MLP/predictive | rainy | 50.71657 | 53.14903 | 60.79334 |
| UA-DETRAC | MLP/predictive | night | 72.40524 | 75.87522 | 84.88154 |
| UA-DETRAC | Transformer/predictive | sunny | 57.12950 | 60.46758 | 71.17210 |
| UA-DETRAC | Transformer/predictive | cloudy | 71.86853 | 77.39581 | 86.51438 |
| UA-DETRAC | Transformer/predictive | rainy | 51.18903 | 54.10587 | 61.26608 |
| UA-DETRAC | Transformer/predictive | night | 72.46851 | 76.02544 | 85.04089 |
| UA-DETRAC | Prototype/frozen_ema | sunny | 57.16267 | 60.47789 | 71.33086 |
| UA-DETRAC | Prototype/frozen_ema | cloudy | 71.82696 | 77.27322 | 86.44926 |
| UA-DETRAC | Prototype/frozen_ema | rainy | 50.78509 | 53.25202 | 61.06546 |
| UA-DETRAC | Prototype/frozen_ema | night | 72.40585 | 75.86424 | 84.89385 |
| BDD100K | MLP/geometry_null | clear | 37.92694 | 41.37994 | 45.25825 |
| BDD100K | MLP/geometry_null | overcast | 41.87501 | 49.47224 | 47.12995 |
| BDD100K | MLP/geometry_null | rainy | 46.05768 | 53.21909 | 54.69715 |
| BDD100K | MLP/geometry_null | night | 33.74898 | 40.18452 | 40.28598 |
| BDD100K | MLP/predictive | clear | 38.09031 | 41.85022 | 45.54988 |
| BDD100K | MLP/predictive | overcast | 40.66813 | 47.81159 | 45.48887 |
| BDD100K | MLP/predictive | rainy | 46.24250 | 53.98301 | 55.18734 |
| BDD100K | MLP/predictive | night | 34.31547 | 41.37582 | 40.96453 |
| BDD100K | Transformer/predictive | clear | 37.96038 | 41.60348 | 45.58240 |
| BDD100K | Transformer/predictive | overcast | 40.87690 | 47.97238 | 45.75376 |
| BDD100K | Transformer/predictive | rainy | 46.58124 | 54.79621 | 55.52871 |
| BDD100K | Transformer/predictive | night | 34.46912 | 41.82283 | 41.28970 |
| BDD100K | Prototype/frozen_ema | clear | 37.40336 | 40.51449 | 44.80549 |
| BDD100K | Prototype/frozen_ema | overcast | 40.97034 | 48.23750 | 45.21048 |
| BDD100K | Prototype/frozen_ema | rainy | 46.40985 | 54.37025 | 55.12991 |
| BDD100K | Prototype/frozen_ema | night | 33.81662 | 40.25639 | 40.33209 |

## 8. 完整顺序学习后平均遗忘降低多少？

BDD seed17 clear→overcast→rainy→night完成44适应视频/8,811 native帧及每阶段19视频冻结回测。以下八组源初始化/自教师伪对并非全相同，AF比较应结合初始性能、最终性能和新域学习，AF低不等于适应成功。AF采用已学习旧域历史最佳阶段到final的非负降幅；signed AF、BWT与FWT另报，不能混用。UA C2 四组也已完成四阶段回测，但UA主线/H2/RKD其它family仍未齐，不能给出两数据集共同的遗忘结论。

UA C2 的最终四域等权 HOTA/AF/BWT/FWT 为：none 62.61165/.59870/-.56967/.16868，feature 62.83441/.30282/-.26502/.16249，relation 62.65220/.33442/-.33442/.08989，competition 62.65687/.33051/-.33051/.09083。UA H2 的五组对应结果为：prediction-priority 62.95704/.09769/-.02180/.07876，appearance-hard 62.85191/.08327/-.08327/.24031，CE-confidence 62.92082/0/.69487/-.02483，CE-prediction-priority 62.73757/.00401/.54839/-.02021，confidence-FIFO 62.61165/.59870/-.56967/.16868。完整六指标与实际预算见 [UA C2 完整分析](UA_C2_FULL_CONTINUAL_ANALYSIS.json) 和 [UA H2 完整分析](UA_H2_FULL_CONTINUAL_ANALYSIS.json)；这是固定 seed17 development 证据，不能替代其它 UA family、sealed 或跨 seed。

| BDD方法 | final HOTA | final AssA | final IDF1 | HOTA AF |
|---|---:|---:|---:|---:|
| discriminative_replay0_none | 38.94819 | 44.73853 | 45.70474 | 0.27766 |
| predictive_ranking_replay0_none | 39.35783 | 45.29865 | 46.11059 | 0.20483 |
| discriminative_replay128_none | 39.27449 | 45.23057 | 46.06761 | 0.00000 |
| predictive_ranking_replay128_none | 39.19219 | 44.84257 | 45.70684 | 0.18017 |
| predictive_ranking_replay128_features | 39.47467 | 45.39169 | 46.07376 | 0.05550 |
| predictive_ranking_replay128_relations | 39.39764 | 45.37483 | 46.09618 | 0.28946 |
| predictive_replay0_none | 39.73459 | 45.99166 | 46.68544 | 0.09938 |
| predictive_replay128_none | 39.69799 | 45.85356 | 46.54923 | 0.22356 |

Pure FT相对CE FT的HOTA AF减少.17828，但源初始化优势与不同自教师伪对供给混入比较；最终虽比CE FT +.78640 CI[.50379,1.07441]，相对自身frozen pure -.09451 CI[-.32327,.13889]。不能称已获得可靠新域学习增益。

## 9. Replay与预测关系保持各自贡献多少？

BDD pure Replay比pure FT -.03661 HOTA CI[-.35638,.31336]，AF从.09938升至.22356；Replay收益未可靠成立，实际采样/更新数不同。固定教师C2中同family五组均为1,094实际更新、41,176确认，三边界memory/RNG/optimizer对齐，但未冒充逐batch trace。column比Replay +.02537 CI[-.20013,.29735]，比普通RKD-D +.022815 CI[-.20609,.29542]；未证明MOT column关系保持有独立优势。H2 need比appearance-hard final+.06535 CI[-.08315,.23180]，stage1反而-.29818。见[完整关系保持矩阵](BDD_FULL_GRAPH_RKD_RETENTION_RESULTS.json)与[固定教师矩阵](BDD_FIXED_TEACHER_FULL_CONTINUAL_RESULTS.json)。

## 10. 哪些场景效果不好，原因是什么？

BDD dense规则的1,732个关联改变中，离线一步诊断纠错354、反改错300；gap2–4的51/85说明中间漏检间隔并无净正收益。该计数是在各方法自身状态上的局部counterfactual，不是独立传播后的整段IDSW因果变化。缩小reward规则后BDD增益几乎消失。UA更长history未提供稳定收益：pure K4比K1 -.19099 CI[-.57326,-.01093]；近容量匹配同时改变hidden width，不能归因为只有时间长度变化。对象crop和连续source clips改变输入/时间支持后的结果仍不稳定。诊断支持固定latent、历史噪声、匹配约束和在线伪对供给的瓶颈假设，尚未证明唯一根因。

## 11. 是否存在伪标签确认偏差？

存在实际错误案例。BDD C2 stage1 reservoir128中119一致、4混合、5无法匹配；stage2有114一致、6混合、1不匹配及其它无法完整核对项。这是终态memory样本的离线GT诊断，不能外推全部训练对的错误率。高prediction need可包含错误历史，不能把error大当作可靠学习信号。GT诊断不反馈训练、门控或优先规则。完整归类与范围见[伪对审计](BDD_COMPLETED_BOUNDARY_REPLAY_IDENTITY_AUDIT.json)。

## 12. 方法是否严格在线？

已运行路径先以严格过去history冻结query，再读当前候选评分；在线延迟确认使用保存的旧query，不重算未来history。独立视频reset physical IDs/history/pending，θ/optimizer/replay/recent/RNG跨视频连续；domain stage评测隔离冻结快照，不更新训练模型。Target训练stream标签为空，GT仅用于独立离线评分/诊断；source监督合法。官方masked predictor在patch projection前将当前两帧像素归零并mask tokens，FP32确定性扰动验证当前目标像素不影响该读数。但detector/GeneralTrack仍观察当前帧，不能声称整个系统看不到当前像素。GeneralTrack author BDD权重具有广泛BDD监督先验，不能声称整个系统仅在weather source受监督；BDD六类协议也不是官方八类benchmark。最终新配置的全范围因果审计仍待补齐。

## 13. 计算成本、显存、FPS和延迟是多少？

下面是真实NAS RGB→detector IPC→FP16 dense encoder→匹配/实际更新的联合prediction+ranking online整视频案例，不是pure目标或新FP32官方读数的速度。启动/载入不计，warmup10帧后统计，使用共享服务器单次执行；torch reservation peaks的进程峰值相加未必同时发生，排除driver contexts与其它任务。

| 数据集 | native帧 | 实际更新 | FPS | 延迟p50 ms | 延迟p95 ms | torch reserved峰值和 GiB |
|---|---:|---:|---:|---:|---:|---:|
| bdd100k | 202 | 25 | 1.6030 | 617.139 | 748.651 | 4.018 |
| ua_detrac | 1750 | 218 | 1.8333 | 528.369 | 671.482 | 3.986 |

未达到实时。MLP662400/Transformer662016/pair662229参数；不同history容量通过宽度匹配，官方predictor1664-D head662657参数且有额外frozen计算。实际保存batch4+4、5次warmup/50次更新中none/feature/row/column/RKD p50分别6.703/7.010/7.576/10.731/8.625 ms；该update-call计时排除图像、检测、伪对生成与reservoir，不能冒充整链路FPS。全部12整视频case、逐组件latency和显存范围见[效率结果](FULL_NATIVE_LIVE_EFFICIENCY_RESULTS.json)、[真实更新成本](RETENTION_ACTUAL_UPDATE_COSTS.json)。

## 14. 相机运动是否被证明确实影响JEPA？

未证明。观测bbox速度、feature间隔和漏检不等于相机运动或真实遮挡。原时间窗以当前box锚定历史可能时空错位的假设仍需独立相机运动估计与受控对照；该次要问题不作为主研究必须成功的前提，也不在报告中伪称根因。

## 15. 与GOT-JEPA、FACT、TCEI等工作的真正区别？

已有GOT-JEPA涉及tracking中的JEPA预测/遮挡机制，FACT涉及在线特征适应与memory，TCEI涉及瞬时/跨视频经验校准；因此不能主张首次JEPA tracking、首次经验保持或仅因命名不同而新颖。本文欲验证的区别是无标签多域顺序下轨迹历史→identity latent/候选竞争关系的持续学习与旧域回测，且对Replay/普通KD/RKD和强历史聚合逐项归因。目前独立效果未获充分支持，该设计差别不是已证实贡献。JEPA-TTT等也已有persistent predictor更新机制。原文链接、交叠和不确定的书目信息见[文献审计](NOVELTY_AND_REVIEWER_AUDIT.md)。

## 16. 当前是否足以支撑持续学习核心的CVPR论文？

不足。已有真实多轮结构探索、完整BDD forward/reverse、UA C2/H2完整回测与两数据集frozen负结果，不是仅工程smoke；但未证明稳定超过强非JEPA的MOT/遗忘收益，UA主线/RKD后续持续矩阵、UA逆序和冻结后的独立测试未齐，三seed稳健性受用户资源约束尚未证明。仅凭弱CE对照的提升、较小cosine error、nonzero relation loss或低AF不能成立主张。最终研究可能得到证据不足的诚实结论，不能预先宣判整个方向失败，也不能伪造支持。

## 17. 还缺哪些关键证据？

补齐实际运行中的UA主线/RKD后续持续矩阵及UA seed17 forward/reverse完整矩阵，分析全域最终性能与AF/BWT/FWT；UA C2/H2完整结果已记录于[UA C2完整分析](UA_C2_FULL_CONTINUAL_ANALYSIS.json)和[UA H2完整分析](UA_H2_FULL_CONTINUAL_ANALYSIS.json)，UA通用RKD的stage0/1已记录于[RKD已完成阶段分析](UA_RKD_COMPLETED_STAGES_0_1_ANALYSIS.json)，UA official paired frozen development已记录于[UA官方读数结果](UA_FULL_OFFICIAL_PAIRED_READOUT_RESULTS.json)，BDD seed17 reverse完整结果已记录于[反向持续分析](BDD_REVERSE_SEED17_FULL_CONTINUAL_ANALYSIS.json)。按用户要求限制重复种子工作，仅对少量明确候选考虑必要复验，三seed均值/标准差缺失应列为局限；开发阶段冻结方法/超参数后执行sealed一次性独立评测。随后汇总全范围因果审计、错误归因、效率适用范围、最终checkpoint与SHA、关键训练/评测命令、所有主表/核心消融/诊断图和完整实验索引。若新增读数成为最终方法，现有FP16 dense速度不能替代其效率。没有把运行中的任务或partial标记当成完成证据；当前状态仍ACTIVE。

## 已完成BDD八组顺序矩阵：HOTA

Stage0为source冻结初始化；stage1/2/3为分别适应overcast/rainy/night后的隔离冻结回测。表格直接来自实际完整报告，未挑选最佳stage。各单元格对应完整domain开发split，不是抽帧样本。全部六指标、预算和checkpoint信息见[纯预测与第一波完整报告](BDD_FULL_PURE_PREDICTIVE_CONTINUAL_RESULTS.json)。

| 方法 | stage | clear | overcast | rainy | night |
|---|---:|---:|---:|---:|---:|
| discriminative_replay0_none | 0 | 37.53764 | 40.42713 | 45.61854 | 33.98155 |
| discriminative_replay0_none | 1 | 37.12066 | 40.47212 | 45.22770 | 33.36157 |
| discriminative_replay0_none | 2 | 36.58624 | 40.26443 | 45.05829 | 33.13791 |
| discriminative_replay0_none | 3 | 36.70466 | 40.52798 | 45.11460 | 33.44552 |
| predictive_ranking_replay0_none | 0 | 37.33256 | 41.22863 | 45.87245 | 33.51094 |
| predictive_ranking_replay0_none | 1 | 37.58711 | 41.05605 | 45.79642 | 33.22700 |
| predictive_ranking_replay0_none | 2 | 36.96535 | 40.87882 | 45.69184 | 33.16307 |
| predictive_ranking_replay0_none | 3 | 37.32855 | 40.70012 | 45.92665 | 33.47601 |
| discriminative_replay128_none | 0 | 37.53764 | 40.42713 | 45.61854 | 33.98155 |
| discriminative_replay128_none | 1 | 36.93389 | 40.51947 | 45.25055 | 33.50984 |
| discriminative_replay128_none | 2 | 37.10349 | 40.32739 | 44.81825 | 33.49753 |
| discriminative_replay128_none | 3 | 37.64994 | 40.98434 | 45.00927 | 33.45442 |
| predictive_ranking_replay128_none | 0 | 37.33256 | 41.22863 | 45.87245 | 33.51094 |
| predictive_ranking_replay128_none | 1 | 37.34866 | 41.07064 | 46.40749 | 33.40137 |
| predictive_ranking_replay128_none | 2 | 37.14116 | 40.71960 | 45.48521 | 33.19066 |
| predictive_ranking_replay128_none | 3 | 37.08732 | 40.79147 | 45.66859 | 33.22137 |
| predictive_ranking_replay128_features | 0 | 37.33256 | 41.22863 | 45.87245 | 33.51094 |
| predictive_ranking_replay128_features | 1 | 37.07361 | 40.62167 | 45.58611 | 33.20955 |
| predictive_ranking_replay128_features | 2 | 37.32083 | 40.90868 | 46.05632 | 33.56684 |
| predictive_ranking_replay128_features | 3 | 37.29008 | 41.01385 | 45.93232 | 33.66243 |
| predictive_ranking_replay128_relations | 0 | 37.33256 | 41.22863 | 45.87245 | 33.51094 |
| predictive_ranking_replay128_relations | 1 | 36.95335 | 40.74010 | 45.53524 | 33.57788 |
| predictive_ranking_replay128_relations | 2 | 37.86814 | 40.91112 | 45.81200 | 33.39278 |
| predictive_ranking_replay128_relations | 3 | 37.20449 | 40.70638 | 46.18173 | 33.49797 |
| predictive_replay0_none | 0 | 38.09031 | 40.66813 | 46.24250 | 34.31547 |
| predictive_replay0_none | 1 | 38.03504 | 40.95046 | 46.03029 | 33.94798 |
| predictive_replay0_none | 2 | 38.09159 | 40.67591 | 45.87593 | 33.96962 |
| predictive_replay0_none | 3 | 38.11132 | 40.65232 | 46.06476 | 34.10997 |
| predictive_replay128_none | 0 | 38.09031 | 40.66813 | 46.24250 | 34.31547 |
| predictive_replay128_none | 1 | 38.23754 | 41.14415 | 46.09964 | 33.83891 |
| predictive_replay128_none | 2 | 37.76928 | 40.62740 | 46.09111 | 33.86889 |
| predictive_replay128_none | 3 | 38.05258 | 41.16282 | 45.60539 | 33.97116 |

## BDD seed17 reverse完整四阶段结果：最终平均与遗忘

逆序为clear→night→rainy→overcast，四域等权平均，结果来自19个完整development视频和实际stage0–3快照；AF为非负截断，BWT/FWT保留有符号值。完整HOTA/AssA/IDF1逐域矩阵和定义见[BDD reverse full continual analysis](BDD_REVERSE_SEED17_FULL_CONTINUAL_ANALYSIS.json)。

| 方法 | final HOTA | HOTA AF | HOTA BWT | HOTA FWT |
|---|---:|---:|---:|---:|
| discriminative_replay0_none | 38.92826 | 0.57363 | -0.47127 | -0.07952 |
| predictive_ranking_replay0_none | 39.32286 | 0.07810 | -0.00357 | -0.16620 |
| discriminative_replay128_none | 39.04392 | 0.31759 | -0.25703 | -0.14313 |
| predictive_ranking_replay128_none | 39.21328 | 0.17325 | -0.08774 | -0.12030 |
| predictive_ranking_replay128_features | 38.94321 | 0.43060 | -0.35052 | -0.03942 |
| predictive_ranking_replay128_relations | 38.96842 | 0.29433 | -0.23746 | -0.11847 |
| predictive_replay0_none | 39.63099 | 0.02702 | +0.01053 | -0.07663 |
| predictive_replay128_none | 39.70039 | 0.08116 | -0.04689 | -0.03039 |

## 复现与全部证据入口（最终附录待补齐）

GitHub分支为`research/predictive-jepa-cmot-20261013`，仓库[LYQ1107/JEPA-MOT](https://github.com/LYQ1107/JEPA-MOT/tree/research/predictive-jepa-cmot-20261013)。本稿不是方法冻结声明；最终commit与最优checkpoint尚未指定，不能以当前HEAD冒充最终交付SHA。

[实验全索引](EXPERIMENT_REGISTRY.csv)、[完整持续矩阵索引](CONTINUAL_MATRIX.md)、[完成证据与缺口](EVIDENCE_STATUS.md)、[实际结果与失败链](CURRENT_RESULTS.md)、[下一步](NEXT_ACTION.md)、[完整任务书](MASTER_GOAL.md)。各完整JSON保存实际protocol、输入/权重hash和分析invocation；源模型metrics保存训练配置、预算与checkpoint SHA。最终训练与评测命令将以冻结配置逐条列出，不能把仅分析命令当作训练复现命令。

核心诊断图： [UA强对照](figures/ua_complete_strong_heads_hota.svg)、[BDD强对照](figures/bdd_complete_strong_heads_hota.svg)、[UA历史](figures/ua_complete_history_length_hota.svg)、[BDD官方读数](figures/bdd_complete_official_paired_readouts.svg)、[BDD纯预测持续矩阵](figures/bdd_complete_pure_objective_continual.svg)。其它crop/history/relation/H2/C2证据与负结果见实验索引及对应完整报告，最终附录不得删去。
