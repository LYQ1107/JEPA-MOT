# 下一步

2026-10-13 的实际核查显示当前容器 `/dev/nvidia*` 不存在、CUDA 不可用。BDD reverse 四阶段矩阵、UA C2/H2 四阶段完整矩阵、UA主线stage0–2、UA RKD stage0–2、UA official paired encoder/predictor 已通过完整 native marker、快照/optimizer或source parity核验；UA主线stage3与RKD stage3正在外部节点续跑，不重启或新增多种子任务。

完成正在运行的全 development frozen 对照与 forward 连续矩阵；优先汇总已经完整的域，严格标注 source/target、seed和评测覆盖。主表始终使用相同 detector/native frame/caches，逐域报告负结果。

UA clips三目标已经完成源训练与校准，正在部署全部开发视频；BDD clips继续生成。目标object-crop的 source校准和完整开发视频特征/匹配对照已启动。历史1/2/4/8使用相近容量、相同图像与2,000×64预算源训练，然后做匹配长度的冻结推理。

收集 H2 固定教师五组的相同实际更新预算结果，以及 H3 replay/feature/row-relation distillation 的独立增量。row-KL仍按普通输出蒸馏解释；完成这一波结果后，决定是否需要MOT动态候选竞争的结构性保持。

继续反向域顺序和seed23/42的真正在线跟踪验证，保持目标GT不参与训练/伪对门控。开发选择冻结后才执行sealed。

live_stream已经跑通真实图像/两环境detector IPC/encoder/query/关联/更新；完整适应视频计时等待GPU5的source clips队列释放。独立benchmark checkpoint不能用于研究训练；共享GPU数字明确标记，最终再核实实际fps、p50/p95和显存。

C2还需与通用RKD-D距离保持比较，相同教师/负例/九字段memory/实际更新预算；实际完整视频验证完成，两个数据集的完整序列已排队。不要把候选column的形式或非零loss当成新颖性/防遗忘证据。

BDD连续clips的四目标完整开发集部署已补排；UA历史长度回测迁移只在当前完整视频终态之后，GPU5先完成现有真实链路benchmark。

纯预测目标的在线贡献不能由prediction+ranking替代。已补排seed17纯预测FT/replay的两数据集forward与reverse；reverse17的CE/联合六组同时保留。UA source seeds23/42已真实训练和source-only校准完成，forward12组及reverse17/23/42共18组已补排（各附pure17两组），相同detector/caches/splits。队列等待现有完整矩阵/开发组终态之后占用GPU；排队状态不能作为已运行的科学结果。BDD原seed23/42两顺序队列继续；seed17纯预测forward已提前在GPU6实际启动（无Replay/Replay128），保持原注册输出。剩余reverse17队列同时等待多seed两顺序与pure forward完整终态，避免重复启动。GPU0队列等待六个已有完整实验。

完整source-validation诊断已经覆盖24快照，需与后续source AssA/HOTA逐方法对应；避免混合CE与预测模型不同loss尺度。官方1664-D predictor先前小规模工程cache不能代替正式读数对照；补充需要相同合法时序输入、明确encoder与masked predictor的信息差别、匹配head参数/源预算和全视频部署，作为仍未完成的要求保留。

官方配对读数完整pipeline已实际排队，等待PID20894/20960退出后用GPU2，先source paired caches、三目标/校准，再两数据集全部development；不是复用旧1664工程缓存。相同4帧观察窗口与source clips图片预算，mask导致当前像素信息有意不同，新增frozen predictor计算另计。源4帧GPU检查完成，202帧无标签整视频接口检查继续。

普通mean4与真实lifetime EMA（beta .25）已补齐source-only matched两参数校准并启动全部native development；这比源GT聚合诊断更直接地检查简单历史是否解释身份收益。完整后与同inputs实际geometry/last/predictive/CE对照，以source stage0按video配对聚合，不能从单video挑最强。

同容量Transformer三目标源训练+source校准+完整两数据集development已启动；history-only输入不接触当前候选，SourcePairs预算/源负例与原MLP一致。源CE已完成，原始源cache没有paired readout的额外ranking字段时保留既有采样。后续仍需明确普通pair-MLP scalar scorer的独立对照，不能把dot-product history CE等价描述为所有pair scorer都已经测过。

同容量普通pair-MLP源模型/校准已完成并部署全部development；历史状态先冻结，candidate只进入final scalar scorer。与MLP/Transformer同源物理候选及2,000×64预算，不用其较高UA源rank直接决定目标域方法。完整后联合普通mean/EMA/Transformer判断源latent预测是否仍有独立意义。

## 纯预测在线对照提前启动

GPU6启动时已用约7.3 GiB/32 GiB；仅调整注册实验的调度顺序，未更改checkpoint、校准、目标视频、域顺序或超参数。第一个overcast适应视频已经实际运行到100/202 native帧，两组共同输入、目标GT训练=false、sealed未使用。输出为`runs/JEPA_CMOT/continual_completion_v1/bdd100k_forward_pure_predictive_seed17`；完整44适应视频及全部4×4冻结回测完成后再报告HOTA/AF和实际更新预算。源checkpoint仍是pure predictive，不用联合目标代替其独立贡献。启动/依赖记录见PURE_PREDICTIVE_FORWARD_LAUNCH.json；当前不构成科学结果。

## BDD通用RKD-D与四种关系保持的真实训练边界对齐

RKD-D已完成全部44适应视频/8811 native帧。对实际保存的stage1/2/3完整checkpoint逐tensor核对：五组（Replay、feature、row、row+column、RKD-D）的累计optimizer steps均为328/570/1094，confirmed/seen为15667/26973/41176，video resets为13/23/44；源checkpoint/校准/固定教师SHA一致，除retention之外OnlineConfig完全相同。

三个边界的reservoir与remaining recent具有相同顺序，字段0/1/2/3/4/7/8（历史latent、geometry、mask、候选、valid与物理候选/轨迹key）逐tensor完全一致；replay RNG状态一致，pending为空，全部optimizer step等于记录updates。旧teacher response和query字段5/6由各learner不同θ产生，不要求相等。column-specific competition_columns是额外机制计数，不等同伪对供给差异。

这些是实际边界样本/预算证据，没有逐步记录跨独立run的每个batch，不能把边界一致扩大为每帧或每批完全一致的证明。RKD-D完整4×4 frozen evaluation仍在运行，不提前宣称HOTA/AF或候选竞争创新。可复现审计模块`jepa_cmot.audit_matched_replay_buffers`与命令、checkpoint SHA、各阶段计数详见BDD_GRAPH_RKD_ACTUAL_BOUNDARY_PARITY.json；不读取target GT，不修改θ。

## 官方配对读数的真实源样本池与训练完成

BDD paired w4 source来自25个clear source_train视频的原3174帧/38466 GT实例，实际features1024-D encoder与1664-D masked predictor均完成三目标2,000×64训练及source-only校准。head参数662400/662657（差0.039%），保持history4/gaps1,2,4,8。实际source NPZ的box/frame/video/track/segment/image/offset全部相同，predictor存储的negative-ranking proxy与encoder feature逐元素相同；直接构造原SourcePairs后，83146 fit与20175 validation样本的完整history/positive/ordered negative物理row索引全部一致，并记录各pool SHA。这补充了此前仅合成样例的负例池检查。

真实源held-out rank1：encoder CE .555494 /pure .497362 /joint .547662；masked predictor CE .473748 /pure .476078 /joint .481818。pure source prediction error为encoder .116265、predictor .039299；1664-D predictor较低的cosine error不能直接解释为更强身份关联，不同读数的latent空间/信息不同。源rank不用于选择目标域视频或调整门控。

两者分享合法4帧window和ROI，但encoder能看到当前像素，official predictor当前两帧像素在patch projection前归零且tokens masked；其额外22973056个frozen predictor参数与计算另计。这里匹配物理样本、源图像/可训练预算和近似head容量，不声称可用信息、frozen参数或算力相同。完整开发MOT部署仍在运行，UA对应pipeline尚未完成，不把源训练/池一致性当成HOTA结果。详见BDD_OFFICIAL_PAIRED_SOURCE_POOL_AUDIT.json，含六个真实checkpoint SHA、source-cache SHA、原采样定义下重建的seed17 index SHA和复现命令；重建sampler不声称保存了每步训练trace。

## UA完整object-centered target与强geometry对照

对象crop（context1.5）六组完成全部17开发视频/25460 native帧，four-domain平均HOTA：native62.16421、geometry62.27898、last62.57372、CE62.31090、pure62.88103、joint62.70728。pure相对geometry +.60205 CI[-.26407,1.05961]，相对last +.30731 CI[-.00141,.45103]，相对CE +.57013 CI[-.41858,1.06302]，相对joint +.17375 CI[-.39423,.44930]。正点估计尚未获得跨视频/跨seed稳定证据，不从多个读数中best选择宣称成功。

pure全域HOTA为sunny56.77390/cloudy72.03579/rainy50.36298/night72.35145；逐域六指标和HOTA/AssA/IDF1配对raw-count bootstrap见UA_FULL_OBJECT_CROP_DEVELOPMENT_RESULTS.json。图figures/ua_complete_object_crop_hota.svg。actual source checkpoint hashes unchanged；sealed未使用。

与此前连续source clips的17个prepared detection/native clock逐条相同。复用实际完整crop geometry作同输入基线后，clips pure相对geometry +.40247 CI[-.23687,.79159]；crop pure相对clips pure +.19958 CI[-.14183,.35637]。cross-readout同时改变source时间支持及source/target读数，不解释为单因素因果增量；两者都未显著超过strong geometry。crop native/geometry与目前已完成的原dense各12视频轨迹字节完全相同，不能将该12视频验证称作原dense17视频重新运行完成。crop提高对象分辨率并增加frozen encoder调用，不声称frozen算力相等。所有结果是fixed-seed开发/frozen阶段，不替代持续或held-out证据。

## UA固定教师五组完整适应预算

五组完成45个目标适应视频/65430 native帧，域顺序sunny→cloudy→rainy→night；实际final checkpoint均为8101 optimizer updates、442278 delayed-confirmed/seen、memory128、45 video resets、pending为空。三个完整域边界均核对frames与原manifest、optimizer step与learner count、五组确认/供给数量一致。当前已经进入stage0冻结回测，仍未完成4×4指标与AF/BWT/FWT。priority选择fresh样本不同，不将数量一致解释为所有batch完全相同，更不能解释为预测需求优先级有效。checkpoint路径/SHA/三边界计数/原protocol见UA_FIXED_TEACHER_COMPLETED_ADAPTATION_BUDGET.json；targetGT训练=false，sealed未用。

## BDD完整通用RKD-D反事实机制对照完成

真实通用RKD-D已完成44适应视频/8811帧及全部4×4冻结回测（每阶段19视频/3844帧）。所有五组在相同固定教师、同native-group负例、相同source checkpoint/校准和三个域边界实际更新/确认/ordered buffers/RNG核验下比较；final均1094 optimizer steps、41176 confirmed/seen、memory128。每批训练样本未逐步记录，边界一致性不扩大为跨run逐batch完全一致。

Final HOTA/AF：Replay39.300926/.255566、feature39.235753/.349849、row39.303274/.250772、column39.326297/.220075、RKD-D39.303482/.258916。Column相对RKD-D仅+.022815 CI[-.206092,.295419]，RKD-D相对row+.000208 CI[-.291793,.251347]，相对Replay+.002556 CI[-.017555,.020145]。Column的AF比RKD-D低.038841只是seed17点估计，不能代替可重复的防遗忘证据。

实际静态geometry39.90215仍更强；五组相对geometry的HOTA区间均在0以下，RKD-D差-.598671 CI[-1.184840,-.130581]，column差-.575856 CI[-1.117550,-.146660]。候选竞争命名、非零loss和较小AF不足以建立MOT-specific创新；通用RKD-D没有产生可靠独立增量。这与实际update-call中column p50较高的成本一起，削弱当前H3主张。没有根据这些target评测GT调整门控/超参数。

完整六指标5×4×4、AF/signedAF/BWT/FWT、十一项final raw-count配对比较、actual protocol/checkpoint SHA/预算及复现命令见BDD_FULL_GRAPH_RKD_RETENTION_RESULTS.json，图figures/bdd_complete_graph_rkd_retention.svg。当前仅BDD seed17 forward，UA RKD及多seed/order尚未完整；sealed仍未用，不能因此宣布全项目完成。

## BDD官方masked predictor与encoder配对读数完整部署

两种official w4读数均完成全部19开发视频/3844 native帧。平均HOTA encoder last39.51336/CE39.40392/pure39.84562/joint39.59494；masked predictor last39.53400/CE39.56582/pure40.10419/joint39.46577。三目标每读数2,000×64 source-only训练，83146 fit/20175 validation物理history/positive/ordered negatives完全一致，head662400/662657参数；source真实checkpoint SHA与完整部署protocol一致，θ未更新。

encoder pure比同读数CE +.44170 CI[.19336,.75556]，但比geometry -.05653 CI[-.37277,.24199]。masked predictor pure比同读数CE +.53837 CI[.20395,.82959]，比last +.57019 CI[.21397,.93152]，比geometry +.20203 CI[-.20709,.59719]。联合目标比pure -.63841 CI[-1.09235,-.22567]。predictor pure比encoder pure +.25857 CI[-.11895,.64641]，不能从多个探索候选中的最大点估计或未经选择校正的开发区间宣称稳定独立优势。

当前首次有正式全source/head/全native development的1664-D predictor读数证据，替代先前工程cache占位，但不是持续学习或三seed证据。mask predictor不能看到当前两帧像素（patch projection前归零、tokens masked），extra frozen22973056参数与FP32计算另计；encoder1024-D能看到当前像素。相同合法4帧window/ROI、source物理预算和近似可训练容量，不等价于信息量/frozen算力相同。更低source cosine error也不能跨不同latent空间解释为更强identity target。

完整八组逐域六指标、十二项配对视频raw-count区间、source pool/checkpoint/实际prepared annotation SHA和复现命令见BDD_FULL_OFFICIAL_PAIRED_READOUT_RESULTS.json，图figures/bdd_complete_official_paired_readouts.svg。全部原prepared detections、native frames和GT评分覆盖逐条匹配，targetGT没有进入训练/gates，sealed未用。UA official pipeline继续，不以BDD seed17 frozen结果替代全项目完成。

UA通用RKD-D的seed17 forward已完成stage2冻结回测：stage1/2分别为27440/42140累计adaptation frames、3395/5232 updates，stage2四域等权HOTA 63.10675；完整阶段报告见UA_RKD_COMPLETED_STAGES_1_2_ANALYSIS.json。stage3已在外部节点续跑，最终AF/BWT/FWT仍待完整矩阵，不把启动记录或阶段点估计当作最终结果。早期启动和依赖关系仍见UA_GENERIC_RKD_EARLY_LAUNCH.json。

## data1磁盘清理

磁盘曾出现ENOSPC，按已完成native状态保留source/domain-boundary及每阶段最近两个完整视频的续跑状态，388个更早中间状态仅保留逐tensor一致的模型参数；清理2313个已完成frozen评测的重复权重（与canonical source θ逐tensor核对），以及6个非主要detector快照（保留500/1500/3000，final SHA不变）。本次释放13.39 GiB，核对时可用16.18 GiB；NAS原数据、主要checkpoint、轨迹/指标/报告保留。完成的run_controls可直接复用完整标记；实际清理后重入验证通过，轨迹与完成标记SHA未改变。未提交的analysis源码因失败写入变空，已从原执行patch恢复，SHA与既有结果记录一致。详细清单/保留策略见DATA1_STORAGE_CLEANUP.json和本地runs manifest。空间不足期间的任务终态与续跑节点还需单独核验，不以旧PID声称仍在运行。

## 磁盘恢复与UA完整强普通对照

磁盘满使BDD多seed夜间第40视频、pure forward stage0、UA主矩阵夜间第35视频、H2 stage0、C2最后第45视频和UA RKD首视频因Errno28退出；没有将部分文件或失败进程当作完成。最后完整视频的12/6/4 learner全状态实际torch.load核验optimizer step，按原source/calibration/target videos/order/seed预算接续。原保留的UA dependency PID保持T直到真正完整矩阵，BDD死依赖重绑到实际恢复队列，已排除重复启动。新增compact-native仅在完成native视频之后转换更早中间状态，保留source/domain-boundary/latest-two full states，减少阶段内磁盘峰值，不修改学习数学或GT规则。DISK_FULL_EXPERIMENT_RECOVERY.json记录真实存活的新controller/queue。

UA MLP/Transformer/pair-MLP/mean4/EMA全部17开发视频/25460 native帧完成，四域等权HOTA：geometry62.27898、last62.68248、MLP CE61.98899/pure62.99334/joint62.69680、Transformer CE62.48072/pure63.16389/joint62.62472、pair62.37338、mean4 62.50597、lifetime EMA63.04514。EMA不含trainable predictor，beta .25真实持续历史，仅两参数source-only score calibration；与其它组相同detector/native/current dense features/geometry。

Transformer pure比同Transformer CE +.68317 CI[.08575,1.29499]，比geometry +.88491 CI[.20845,1.32044]；但比EMA只有+.11875 CI[-.27443,.28213]。MLP pure比EMA -.05180 CI[-.22667,.05772]，比pair +.61997 CI[.39052,.94665]，比last +.31086 CI[.11770,.83744]。EMA本身比geometry +.76616 CI[.16143,1.37539]；因此不能用弱CE/geometry的正差宣称预测损失已超越强简单历史，Transformer best点估计仍无selection correction/cross-seed证据。源时间支持/crop/readout探索不从最佳视频/阶段选结果。

全部方法逐域六指标、十三项全native raw-count配对比较见UA_FULL_STRONG_HEAD_DEVELOPMENT_RESULTS.json，图figures/ua_complete_strong_heads_hota.svg。结果是seed17 frozen development，仍需UA持续矩阵、多seed/orders与official UA读数；sealed未用。

补查恢复UA历史1/2/4/8：磁盘满导致第17个开发视频MVI_40792退出，前16全native标记有效；GPU4实际controller26247从该未完成视频接续，完整之后自动执行跨长度bootstrap与history4全部轨迹bytes对照。见UA_HISTORY_DISK_RECOVERY.json；尚未以16/17当作完整消融结果。

## UA完整历史1/2/4/8消融与独立轨迹复现

12组均完成17开发视频/25460 native帧；四域等权HOTA按history1/2/4/8分别：CE62.57138/62.42879/62.34729/62.31475，pure62.87244/62.74658/62.68145/62.69079，joint62.59632/62.61130/62.66136/62.61837。相同连续source clips、2,000×64/gaps1,2,4,8/同detector/native/current dense features/source-only校准，近似容量K1/2/4/8=660864/660685/662400/661821；hidden width随history调整，不能把差异解释成仅增加time tokens而其它结构完全相同。

pure history4比history1 -.19099 CI[-.57326,-.01093]，history8比history1 -.18165 CI[-.50792,.08258]，history8比geometry +.41181 CI[-.14166,.79067]；CE history4比history1 -.22408 CI[-.44736,-.07791]，joint history8比history1 +.02205 CI[-.32898,.36664]。本匹配预算协议未提供更长历史稳定收益，不从K1最佳点选择删掉其它长度，也不等价于证明一切历史结构无效。

history4全部三目标×17视频轨迹与独立完成的UA source-clips部署bytes逐条相同；Source head SHA未变，GT只用于离线指标，sealed未用。完整12组逐域六指标、九项配对raw-count bootstrap、51条轨迹SHA见UA_FULL_HISTORY_LENGTH_DEVELOPMENT_RESULTS.json，图figures/ua_complete_history_length_hota.svg。磁盘满失败视频已从合法完整节点恢复，不以16/17替代完整scope。尚未完成的持续矩阵/多seed/orders/official UA读数继续。

## BDD纯预测FT/Replay完整持续矩阵完成

seed17 pure两组完成全部44适应视频/8811 native帧及完整4×4冻结回测（每阶段19视频/3844帧）；初始stage0逐六指标与实际frozen pure对齐，checkpoint SHA匹配。Pure FT final HOTA39.73459/AF.09938/AssA45.99166/IDF1 46.68544；pure Replay39.69799/AF.22356/AssA45.85356/IDF1 46.54923。实际FT1073步、41703 confirmed/seen、memory0；Replay1096步、41562 confirmed/seen、memory128。与CE对应组的最终更新数恰好相同，但self-teacher轨迹/伪对供给不同，不声称每帧或每batch相同。

pure FT相对CE FT +.78640 HOTA CI[.50379,1.07441]，相对joint FT +.37676 CI[.04084,.72665]；pure Replay相对CE Replay +.42350 CI[-.06098,.95699]，相对joint Replay +.50580 CI[.10204,1.05359]。这类目标对比混合了源初始化优势与在线适应/保持差异，不能全部当作target learning gain。

相对实际frozen pure39.82910，FT -.09451 CI[-.32327,.13889]，Replay -.13111 CI[-.46871,.22496]；相对geometry39.90215，FT -.16756 CI[-.52401,.18123]，Replay -.20416 CI[-.53792,.12487]。Replay相对pure FT -.03661 CI[-.35638,.31336]，AF反而更高（单seed/实际采样与更新数不同）。当前支持pure比弱CE/joint的性能保留更好，但未建立online adaptation或Replay可靠额外收益；AF低也不等于学会新域或超过静态强方法。

完整八组4×4逐域六指标、AF/signedAF/BWT/FWT、九项配对raw-count区间、实际边界optimizer/确认数/checkpoint SHA/原protocol与复现命令见BDD_FULL_PURE_PREDICTIVE_CONTINUAL_RESULTS.json，图figures/bdd_complete_pure_objective_continual.svg。源预算/同容量head/同source-only校准流程、detector/native/geometry/nominal更新预算保持，targetGT只用于offline metrics，sealed未用。反向顺序、三seed及UA counterpart仍在运行，不将本scope声明为最终全项目完成。

纠正两个尚未执行的reverse17/UA reverse-all队列开关：此前false会误跑forward，现true并重载无child的等待队列；预期BDD clear→night→rainy→overcast、UA sunny→night→rainy→cloudy。未产生任何错误标签的reverse实验，也未修改已完成forward数据/学习预算。见REVERSE_QUEUE_FLAG_CORRECTION.json；多seed BDD原队列的reverse=True保持正确。
