# 下一步

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
