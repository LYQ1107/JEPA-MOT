# 失败证据和诊断

## 工程 smoke v1：缺失 timm

Observation：UA 原始 GeneralTrack 跟踪和 TrackEval 成功，官方 V-JEPA 构造失败 `ModuleNotFoundError: timm`。

处理：在项目的 `runtime/site-packages` 安装 timm 0.9.16，保留 v1 日志。V100 使用 FP16，缓存记录实际精度。

结果：UA v2、BDD v1 的官方 predictor 缓存、在线跟踪和评测闭环均通过。此结果仅证明工程可执行。

## 历史 temporal adapter

旧 source temporal adapter 的 deploy_alpha=0，source held-out 直接预测更差。该证据不属于本次新训练；不加载为有效 warm start。

下一假设：无约束替换历史外观可能损害原有身份区分；共享零初始化残差预测器有利于分离训练目标效果与随机初始化扰动。需 prediction / ranking / combined 同容量对照验证。

## 数据协议问题

UA：MVI_39501 XML/图片 540 帧，历史 prepared 80 个抽样帧。不能将该输入当作完整原生视频。

BDD：train/val 文件数量一致，必须审计视频集合；源 detector 配置使用 clear_all，存在与新 source 测试划分重合的风险。需要根据训练视频清单重建冲突的源模型/缓存。

下一步：冻结完整视频划分之前先完成覆盖与训练 provenance 审计；不通过删除困难视频或更改测试分母解决。

## 源视频缓存重置修复

源缓存生成器只清空 frame_buffer，未清空 rgb_buffer；官方 predictor 裁剪读数因此可能携带前一独立视频像素。添加 rgb_buffer.clear()，作废 v1 小型源训练比较并重建 v2。目标缓存原本已正确清空两个缓冲。

## dense 读数的 source 排序与校准风险

多间隔 source-validation 的 frozen-last rank-1 为 55.98%，teacher 当前 GT latent 上限为 93.15%；11,831 样例中 teacher 上限有 1,538 个候选最高分并列，说明 dense ROI 分辨率不能区分所有身份。所有新比较使用最高分并列的分数信用，避免正样本总放在第 0 列导致虚高 rank-1。最初 source-history v1 的第一模型在这一规则冻结前已导入代码；保留原训练/中断记录，新比较重启至 v2。

初始 CE-only source rank-1 约 76.38% 而 cosine prediction error 约 0.61，存在“排序改善但绝对相似度尺度变差”的风险。旧 residual-only 正增益融合可能偏向高 raw cosine 的预测目标，需在正式模型比较中加入相同 source-supervised 校准与 row-relative scorer 控制，不能将尺度差异误归因于 JEPA。

源历史 v2 已冻结统一 tie 规则，三个正式目标均完成 2,000 步。共同 affinity 校准只拟合 source 内部验证上的两个单调参数，不改变候选排序；raw-cosine 路线保留用于判断改进是否来自尺度变化。校准结果及每个模型的 checkpoint/cache SHA256 见 SOURCE_UA_CALIBRATION.json。

## 六类到原生八类的语义接口

恢复工具把 COCO 六类直接映射为 0..5，但 GeneralTrack 原始 sig 将 0/1 视为 person/rider，2..5 为车辆，6/7 为两轮车。修正映射为 [0,2,3,4,6,7] 并把输出转换回六类。另发现 legacy JEPA positive residual 能让原生类别 mask=0 的边重新达到匹配阈值；新 scorer 对跨原生类别组的相似度置零。旧 full-video v1 原始记录保留并标记失效，v2 使用相同检测行和缓存重跑全部控制。这些是共同接口修正，不能归因于 JEPA。

## 第一条完整视频：预测排序与 MOT 分数并不一致

BDD source rank-1 中预测弱于 CE，但一个完整 clear 开发视频的 HOTA 51.99 高于 CE 50.99；frozen-last 51.71 仍接近预测。Observation：source ranking 不能代替因果在线分配指标。Hypothesis：绝对分数校准、geometry/matcher 与当前候选竞争、frame gap 分布影响最终输出。Discriminating experiment：在同一 native8 和 geometry 条件下完整跑各域，并记录纠错/反改错；结合 crop/dense 的读数能力上限。没有基于单视频判定支持或 NO-GO。

## 离线连续性事件诊断

首个 UA 视频相对 geometry-null，预测有 65 个连续性修复和 73 个反改错；prediction+ranking 69 / 72，CE 65 / 71，frozen-last 48 / 39。BDD 同类计数预测 21 / 21、联合 21 / 19、CE 27 / 20、last 14 / 11。定义为同一 GT 上次 IoU05 匹配到的 track ID 是否延续，仅比较共同可评测事件，不能直接等同官方 IDSW、AssA 或净 HOTA。说明需要具体诊断 harmful corrections，而非只按 source ranking 选模型。该离线 GT 诊断没有反馈到在线参数或伪标签门控。

## 打乱身份特征仍有单视频收益

Observation：BDD clear 开发视频 frozen-shuffle HOTA 51.91，接近预测 51.99；constant_low 51.23、constant_half 50.57、constant_high 43.87，而 geometry-null 51.08。Hypothesis：部分分数变化来自 residual fusion 对 native affinity 的衰减/提升，而不是正确身份信息。Discriminating experiment：全域信息控制与常数控制，配合共同可评测关联事件诊断；不能用该单视频推断 shuffle 整体更优或否定 JEPA。

## source 预测的时间支持与逐帧推理不同

Observation：UA uniform source 验证样例 native delta-t 最小为 6、中位 19，没有 delta-t=1 样例；BDD 12.42% 为 delta-t=1。Hypothesis：history learner 在短时间步上外推，source 排序与在线关联变化的一部分可能由时间支持失配造成。Discriminating experiment：相同每视频图片预算下的连续 clips，保留 native frame 编号，clip 间重置 history，三种同容量目标和相同 source-only 校准。这个数据协议修正本身不是 JEPA 创新。

## 单视频 +1.00 不代表 source 全域 +1.00

Observation：首个 BDD 视频预测比 CE 高1.00；全部5个 clear 开发视频汇总差值0.553、配对区间跨0，相对最强geometry仅0.163。Hypothesis：收益存在视频/事件依赖，当前证据不足以排除几何与分数尺度。Discriminating experiment：保留全部 target 开发域、三 seeds 与实际持续回测，分析事件而不选择有利视频。

## 连续源时间支持改善不等于 JEPA 改善

Observation：clips 内 CE88.54%、预测88.62%、联合89.80%，两种普通/预测模型都能较好排序。源验证图片与uniform不同，不能直接比较绝对分数。Hypothesis：源间隔失配可能影响部署，但其修正也可能被普通历史学习解释。Discriminating experiment：相同固定 native 开发输入上比较新三个目标，并加入训练过的不同历史长度/相近容量模型。

## 缓存数值与真实链路存在微小差异

Observation：实际 detector 数值相同、短序列轨迹一致；FP16 latent最大差约1.4e-3，3次更新后参数最大差2.8e-4。不能把位级 checkpoint 可复现扩展到两种 GPU 执行环境的数学近似。Discriminating experiment：完成长视频真实链路，保留精度和硬件竞争说明；正式受控主表仍共用同一固定 feature cache。没有用 reference cache 替代 live inference。

## 四域平均的随机特征与预测接近

Observation：predictive39.829、geometry39.902、random39.871；预测超过CE的分层区间为正，但不优于简单几何。Hypothesis：部分收益可能来自residual score scale而非身份排序。Discriminating experiment：rank_reward匹配CE/frozen/random，不整体衰减native，不产生原本不可行的边；记录实际一步counterfactual。首个完整视频目前无收益，全域保留。

## 更新间隔相同仍可能更新数不同

Observation：第一波FT1073/1074、replay1096、KD1097，真实state与optimizer step相同但不同policy间更新量略异。Discriminating experiment：H2/H3共享冻结teacher并逐帧断言更新数相等，负例与记忆字段一致；避免将伪对供给造成的更新量解释为预测或保持机制。

## 时间支持修正没有自动产生MOT增益

Observation：clips源验证的pure prediction排序接近CE，但同一三视频sunny native开发集合，clips预测HOTA56.292，uniform预测56.913，geometry56.907。Hypothesis：短间隔源排序、detector历史噪声与latent目标的可判别性可能不是同一问题；不能仅由源loss选方法。Discriminating experiment：继续匹配容量的真实历史长度部署、crop读数、事件mixed-history审计和多域持续回测，保留整个失败路径。未基于这三个source视频改变目标域split或在线GT门控。

## 保守竞争奖励没有建立JEPA优势

Observation：全BDD开发集新预测规则39.9134，geometry39.9022，同规则CE39.9124；预测只改变23次一步决策、局部修复和反改错各1。原残差虽改变1732次，修复与损害均存在。Hypothesis：目前latent预测的有效身份信息和检测噪声不足以稳定改变正确的native排序；过于保守的介入也会使机制近似geometry。Discriminating experiment：既有crop/history长度的完整域检验、纯预测的在线适应、固定教师同预算与多顺序遗忘矩阵；不能仅放宽rank阈值来得到更多改动，不能从这些事件数量宣布因果泛化结论。

## 伪关联错误不是已经证实的唯一原因

Observation：固定教师completed stage1/2的reservoir128中，身份一致119/114，确认错配0/1，混合历史4/6，另有GT无法匹配；真实global候选键与历史feature/box重构已验证。Hypothesis：除了伪对噪声，还可能有latent预测/排序受更新影响或固定source校准在新θ上失配。Discriminating experiment：在不改变θ的前提下，以完全相同source内部验证视频、候选、gaps复评所有域边界snapshots，后续与真实source AssA/HOTA对照。该终态reservoir诊断不等于全训练错误率，不据此删除训练样例或调目标GT门控。

## “预测更准”与“身份更可分”出现分离

Observation：完整原source-validation上联合replay最终error下降.02470，但rank1下降.00859、hard-negative cosine margin从.02083降至.01472；全局query covariance trace并未趋于零。Hypothesis：latent匹配的改善没有保护身份区分，可能受域变化、候选来源和source-calibration与更新θ的交互影响；全局collapse目前缺乏证据。Discriminating experiment：已有固定教师matched负例、candidate-column和通用RKD的完整矩阵，纯预测在线对照与对象crop/history长度；补齐encoder/official-predictor读数。Architectural Change：C2保护同帧跨轨迹候选竞争、对全部对照统一native-group负例，不改变检测/生命周期；结果仍待完整MOT回测，不能由源验证表宣告该机制有效。

## 预测需求优先暂未优于普通hard-example

Observation：H2相同328步和15756确认对，prediction priority四域HOTA低于appearance-hard .2982，固定seed配对CI全负；C2增加column相对row仅.00045，无实质量级。Hypothesis：高forecast discrepancy可能不是优质学习需求，保持已有弱关系也不足以增加身份信息。Discriminating experiment：完成旧域/新域矩阵、RKD、相同数据不同顺序、pure预测和多seed，而不提高replay预算或用GT筛对。Next decision：继续已有完整对照，当前不保留“预测priority更优”的结论。

## 最终第一波矩阵没有证明JEPA持续价值

Observation：完整first-wave final所有六组低于强geometry；联合FT优于CE FT .4096，但同replay联合低于CE .0823且遗忘更高（.1802 vs0）。Feature KD比geometry -.4275的固定seed视频CI全负。Hypothesis：普通latent贴近与source分数变化不足以建立更强身份关系，保护弱函数只能恢复部分稳定性；不能由FT弱对照产生的正号推导整体优势。Discriminating experiment：等待已运行的fixed-teacher、RKD、pure预测、多seed/order及crop/official/mean-EMA完整同协议结果。Next decision：保持预注册输入和预算，不通过扩大memory或目标GT阈值调参寻求成功。

## 相等实际预算之后仍缺乏需求优先的可靠增量

Observation：H2全流程五组1096步/41563确认，final预测priority比appearance-hard仅+.0653、比CE+priority +.2065，CI均跨0；阶段1/2的相对hard差均负。Hypothesis：有限单seed的阶段波动与原模型差异可以解释小宏正号，不足以说明JEPA discrepancy稳定识别有价值更新。Next decision：保留全部矩阵和既有多seed/order控制，不缩小任务到最终一个有利阶段；Mean/EMA和Transformer强普通基线继续，不能称新增结构本身是JEPA创新。

## 同帧候选column保持的最终增量很小

Observation：C2完整同预算final row+column39.32630、row39.30327、replay39.30093，column比row仅.02302；AF降低.03070，但仍低于geometry。Hypothesis：共享候选竞争的旧forecast query本身缺乏足够身份区分，保持其关系不自动产生更多有效身份信息。Next decision：等待通用RKD、SourceMask readout与强普通baseline的完整证据，不因有nonzero loss、较低AF或CI边界为0就声明创新/稳定优势。

## 局部事件目前不支持向reappearance收缩

Observation：原残差gap2-4的修复51/损害85，gap5-8为8/11；gap1为293/203。Hypothesis：较长观测间隔下的历史/读数错位可能更难，也可能涉及Hungarian全局互动；当前不是摄像机或遮挡根因的证据。Next decision：保留分组诊断，不据此向long-gap分支添加阈值搜索或宣称遮挡恢复创新；source分位定义的观测速度不能代替真实camera-motion/occlusion实验。

## 更强head没有建立相对简单几何的优势

Observation：Transformer pure预测39.9719高于自身CE .5317，但比geometry仅+.0698且CI跨0；普通pair源UA rank更高，却BDD部署39.2459低于geometry且CI全负。Hypothesis：源candidate ranking、loss尺度与实际native affinity不是同一任务；更多交互/attention并未自动增加可靠身份信息。Next decision：保留这些同容量强控制的负结果，继续已有pure online/多seed/order/official读数，不将Transformer结构变化本身归为JEPA价值，待UA全域判断域依赖性。

## object-centered target和源时间支持修正的部署边界

Observation：BDD crop pure与dense pure只差.00159、CI跨0；UA完整clips pure相对CE +.33416但相对last +.04196、CI跨0。Hypothesis：源身份排序和更清晰object target并非native图像关联收益的充分条件，普通history与score calibration可解释一部分差。Next decision：保留全域失败与source/readout预算区别，等已运行强geometry/EMA/Transformer/pair及持续矩阵；不改变数据划分或额外调lambda筛到有利分数。

## BDD完整通用RKD-D反事实机制对照完成

真实通用RKD-D已完成44适应视频/8811帧及全部4×4冻结回测（每阶段19视频/3844帧）。所有五组在相同固定教师、同native-group负例、相同source checkpoint/校准和三个域边界实际更新/确认/ordered buffers/RNG核验下比较；final均1094 optimizer steps、41176 confirmed/seen、memory128。每批训练样本未逐步记录，边界一致性不扩大为跨run逐batch完全一致。

Final HOTA/AF：Replay39.300926/.255566、feature39.235753/.349849、row39.303274/.250772、column39.326297/.220075、RKD-D39.303482/.258916。Column相对RKD-D仅+.022815 CI[-.206092,.295419]，RKD-D相对row+.000208 CI[-.291793,.251347]，相对Replay+.002556 CI[-.017555,.020145]。Column的AF比RKD-D低.038841只是seed17点估计，不能代替可重复的防遗忘证据。

实际静态geometry39.90215仍更强；五组相对geometry的HOTA区间均在0以下，RKD-D差-.598671 CI[-1.184840,-.130581]，column差-.575856 CI[-1.117550,-.146660]。候选竞争命名、非零loss和较小AF不足以建立MOT-specific创新；通用RKD-D没有产生可靠独立增量。这与实际update-call中column p50较高的成本一起，削弱当前H3主张。没有根据这些target评测GT调整门控/超参数。

完整六指标5×4×4、AF/signedAF/BWT/FWT、十一项final raw-count配对比较、actual protocol/checkpoint SHA/预算及复现命令见BDD_FULL_GRAPH_RKD_RETENTION_RESULTS.json，图figures/bdd_complete_graph_rkd_retention.svg。当前仅BDD seed17 forward，UA RKD及多seed/order尚未完整；sealed仍未用，不能因此宣布全项目完成。

## 磁盘恢复与UA完整强普通对照

磁盘满使BDD多seed夜间第40视频、pure forward stage0、UA主矩阵夜间第35视频、H2 stage0、C2最后第45视频和UA RKD首视频因Errno28退出；没有将部分文件或失败进程当作完成。最后完整视频的12/6/4 learner全状态实际torch.load核验optimizer step，按原source/calibration/target videos/order/seed预算接续。原保留的UA dependency PID保持T直到真正完整矩阵，BDD死依赖重绑到实际恢复队列，已排除重复启动。新增compact-native仅在完成native视频之后转换更早中间状态，保留source/domain-boundary/latest-two full states，减少阶段内磁盘峰值，不修改学习数学或GT规则。DISK_FULL_EXPERIMENT_RECOVERY.json记录真实存活的新controller/queue。

UA MLP/Transformer/pair-MLP/mean4/EMA全部17开发视频/25460 native帧完成，四域等权HOTA：geometry62.27898、last62.68248、MLP CE61.98899/pure62.99334/joint62.69680、Transformer CE62.48072/pure63.16389/joint62.62472、pair62.37338、mean4 62.50597、lifetime EMA63.04514。EMA不含trainable predictor，beta .25真实持续历史，仅两参数source-only score calibration；与其它组相同detector/native/current dense features/geometry。

Transformer pure比同Transformer CE +.68317 CI[.08575,1.29499]，比geometry +.88491 CI[.20845,1.32044]；但比EMA只有+.11875 CI[-.27443,.28213]。MLP pure比EMA -.05180 CI[-.22667,.05772]，比pair +.61997 CI[.39052,.94665]，比last +.31086 CI[.11770,.83744]。EMA本身比geometry +.76616 CI[.16143,1.37539]；因此不能用弱CE/geometry的正差宣称预测损失已超越强简单历史，Transformer best点估计仍无selection correction/cross-seed证据。源时间支持/crop/readout探索不从最佳视频/阶段选结果。

全部方法逐域六指标、十三项全native raw-count配对比较见UA_FULL_STRONG_HEAD_DEVELOPMENT_RESULTS.json，图figures/ua_complete_strong_heads_hota.svg。结果是seed17 frozen development，仍需UA持续矩阵、多seed/orders与official UA读数；sealed未用。

## UA完整历史1/2/4/8消融与独立轨迹复现

12组均完成17开发视频/25460 native帧；四域等权HOTA按history1/2/4/8分别：CE62.57138/62.42879/62.34729/62.31475，pure62.87244/62.74658/62.68145/62.69079，joint62.59632/62.61130/62.66136/62.61837。相同连续source clips、2,000×64/gaps1,2,4,8/同detector/native/current dense features/source-only校准，近似容量K1/2/4/8=660864/660685/662400/661821；hidden width随history调整，不能把差异解释成仅增加time tokens而其它结构完全相同。

pure history4比history1 -.19099 CI[-.57326,-.01093]，history8比history1 -.18165 CI[-.50792,.08258]，history8比geometry +.41181 CI[-.14166,.79067]；CE history4比history1 -.22408 CI[-.44736,-.07791]，joint history8比history1 +.02205 CI[-.32898,.36664]。本匹配预算协议未提供更长历史稳定收益，不从K1最佳点选择删掉其它长度，也不等价于证明一切历史结构无效。

history4全部三目标×17视频轨迹与独立完成的UA source-clips部署bytes逐条相同；Source head SHA未变，GT只用于离线指标，sealed未用。完整12组逐域六指标、九项配对raw-count bootstrap、51条轨迹SHA见UA_FULL_HISTORY_LENGTH_DEVELOPMENT_RESULTS.json，图figures/ua_complete_history_length_hota.svg。磁盘满失败视频已从合法完整节点恢复，不以16/17替代完整scope。尚未完成的持续矩阵/多seed/orders/official UA读数继续。

## BDD纯预测FT/Replay完整持续矩阵完成

seed17 pure两组完成全部44适应视频/8811 native帧及完整4×4冻结回测（每阶段19视频/3844帧）；初始stage0逐六指标与实际frozen pure对齐，checkpoint SHA匹配。Pure FT final HOTA39.73459/AF.09938/AssA45.99166/IDF1 46.68544；pure Replay39.69799/AF.22356/AssA45.85356/IDF1 46.54923。实际FT1073步、41703 confirmed/seen、memory0；Replay1096步、41562 confirmed/seen、memory128。与CE对应组的最终更新数恰好相同，但self-teacher轨迹/伪对供给不同，不声称每帧或每batch相同。

pure FT相对CE FT +.78640 HOTA CI[.50379,1.07441]，相对joint FT +.37676 CI[.04084,.72665]；pure Replay相对CE Replay +.42350 CI[-.06098,.95699]，相对joint Replay +.50580 CI[.10204,1.05359]。这类目标对比混合了源初始化优势与在线适应/保持差异，不能全部当作target learning gain。

相对实际frozen pure39.82910，FT -.09451 CI[-.32327,.13889]，Replay -.13111 CI[-.46871,.22496]；相对geometry39.90215，FT -.16756 CI[-.52401,.18123]，Replay -.20416 CI[-.53792,.12487]。Replay相对pure FT -.03661 CI[-.35638,.31336]，AF反而更高（单seed/实际采样与更新数不同）。当前支持pure比弱CE/joint的性能保留更好，但未建立online adaptation或Replay可靠额外收益；AF低也不等于学会新域或超过静态强方法。

完整八组4×4逐域六指标、AF/signedAF/BWT/FWT、九项配对raw-count区间、实际边界optimizer/确认数/checkpoint SHA/原protocol与复现命令见BDD_FULL_PURE_PREDICTIVE_CONTINUAL_RESULTS.json，图figures/bdd_complete_pure_objective_continual.svg。源预算/同容量head/同source-only校准流程、detector/native/geometry/nominal更新预算保持，targetGT只用于offline metrics，sealed未用。反向顺序、三seed及UA counterpart仍在运行，不将本scope声明为最终全项目完成。
