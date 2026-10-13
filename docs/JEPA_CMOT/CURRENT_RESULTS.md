# 当前结果

状态：ACTIVE。两个数据集的正式 source 模型和首个完整开发视频对照已完成，完整多域开发对照与六组持续学习正在运行。没有最终持续学习矩阵、多种子或 sealed 结果，尚不能支持 H1/H2/H3。

## 有效工程证据

UA 与 BDD 的实际图像、GeneralTrack、官方 encoder+predictor、缓存、在线跟踪与 TrackEval 均已在 V100 GPU 跑通。旧路径 smoke 每次 6 个观测，仅验证接口。

重建后的新检测输入使用 6 个连续原生 UA 帧。新跟踪器的 native、frozen-last、已训练 predictor 路线均通过；查询历史检查没有当前帧混入。额外插入一个空检测帧后，禁用 JEPA 的输出与逐帧调用原始 native tracker 一致，时钟正确推进。

修正跨视频 RGB buffer 后，源训练 v2 使用两个 sunny source_train 视频各 8 帧，共 165 个源 GT 对象。三个目标各有 1,072,640 参数，seed 17、200 步：判别 rank-1 86.96%，预测 95.65%，预测+ranking 91.30%；误差分别 0.75411、0.01921、0.18261。这是工程小样本，不作为研究方法选择依据。v1 缓存和相应指标已作废，保留原始记录。

延迟在线学习器通过因果、恢复和独立视频状态检查：同帧不更新，下一帧确认后更新，恢复后参数一致；清空视频 pending/history 时保留 optimizer/replay。合成评测检查证明 ignore 区域不会产生额外 FP、未列入 XML 的帧不被当作空 GT、BDD 类别错误不会匹配。

## 正式运行与输入边界

UA 100 视频、140,131 图片帧，其中 XML 列出 138,252 帧；BDD 111 视频、22,074 帧。所有 native 图像参与跟踪，缺少 XML 的帧单独标记。VIDEO_SPLITS.json 已冻结；source、adaptation、development、sealed 按视频互斥，sealed 尚未用于任何方法评测。

两个 source-only Faster R-CNN 从 COCO 公共权重初始化，各训练 3,000 步。UA 16 源视频/2,048 训练帧；BDD 25 源视频/3,174 帧。训练使用固定每视频 128 帧预算，评测保留完整视频。旧 all-source detector 不再用作本次正式输入。

官方 encoder 最后层为 1024 维，官方 predictor 为 1664 维。正式对照会分别训练匹配维度/读数的历史网络；同一读数族的判别/预测/预测+ranking 容量一致。正在生成源特征，并检查 dense current-image ROI 与 object-crop 的成本差异。

GeneralTrack 权重与作者公开文件 SHA256 完全一致。它是所有方法共享的 BDD 预训练关联先验，不能声称完整系统只有当前 source-weather GT 预训练。该限制见 FROZEN_PROTOCOL.json。

原始工程记录与失效原因见 ENGINEERING_EVIDENCE.json、FAILURE_ANALYSIS.md、EXPERIMENT_REGISTRY.csv。权重、缓存和日志保存在本地 runs/JEPA_CMOT；数据从只读 NAS 读取。

修正缓存后的 frozen-last 源验证 rank-1 为 97.83%，高于三种训练目标；这进一步说明工程小样本不能提供 H1 支持。UA detector 的 3,000 步已完成并登记 SHA256，完整 1,505 帧的 MVI_39311 与 800 帧的 MVI_20064 检测正在生成。

两个正式 detector 均完成 3,000 步，SHA256 已登记于 SOURCE_DETECTOR_UA_RESULT.json 和 SOURCE_DETECTOR_BDD_RESULT.json。正式 source history 队列等待各自完成的 source-only cache，然后按固定 2,000 步、seed 17、gaps 1/2/4/8 执行三个同容量目标。

## 首轮正式 source identity 训练

UA dense-current 1024-D，三个模型各 662,400 参数，seed 17 / 2,000 步 / batch 64 / gaps 1、2、4、8。13 个视频拟合、3 个内部 source 视频验证，共 43,921 / 11,831 样例。计入候选最高分并列分数后，判别 / 预测 / 预测+ranking 的 rank-1 为 72.17% / 66.29% / 70.85%；prediction error 为 0.61463 / 0.06641 / 0.20999。frozen-last / mean / recent-weighted 为 55.98% / 49.41% / 52.55%。这只支持历史学习优于这些冻结摘要，不支持 JEPA 优于判别目标。

当前 GT latent 的 source 读数上限为 93.15%，1,538 个样例因 dense 网格出现最高分并列。记录见 SOURCE_UA_BASELINE_DIAGNOSTICS.json、SOURCE_HISTORY_UA_V2_RESULTS.json。完整视频验证与相同 source-only 二参数 affinity 校准正在执行，不以这些 source 排序分数代替 HOTA。

## 首个完整 native 视频

全 1,505 帧 MVI_39311（sunny development）已完成实际图像读取、统一新 detector 输入、native GeneralTrack 和离线评测。B0 HOTA 49.49、AssA 52.27、IDF1 62.11、IDSW 43。这是一个完整开发视频的基线，尚无同视频 JEPA 对照，也不是最终全域证据。geometry-null 对照和匹配 source/readout 的完整特征缓存正在运行。

第二类关系目标的 source 训练已完成：relation-KL rank-1 74.74%，relation-KL+ranking 74.43%，均为同样 662,400 参数和 2,000 步。其 source 排序高于 CE 的 72.17%，但尚无完整 MOT 对照；该损失与普通 soft distillation 重合，不能据此声称 JEPA 创新成立。

BDD dense-current 的同容量 source rank-1：CE 52.49%、预测 46.95%、预测+ranking 51.53%；source 样例 97,827 / 23,623，20 视频拟合、5 视频内部验证。尚未形成 JEPA 优于判别的 source 证据。

MVI_39311 geometry-null HOTA 49.73、AssA 52.29、IDF1 62.53、IDSW 34。该相同盒子规则已接入所有学习目标的匹配控制，避免把几何处理收益算作 JEPA。

## 完整视频接口修正

上述 first_full_video_v1 完整视频分数已标记 SUPERSEDED_NATIVE_CATEGORY_MAPPING，不能作为正式基线。六类原始检测输出依次为 person/car/truck/bus/motorcycle/bicycle，原始 GeneralTrack 的 BDD8 编号为 0/2/3/4/6/7；直接使用 0..5 会使类别组错误。新 v2 在 matcher 输入边界统一转换，输出映射回六类评测，JEPA 与 geometry duplicate 都限制在原生类别组内。检测行、图像、特征缓存与 source 模型不变；所有控制重跑。

## 类别接口修正后的 v2

统一 native8 映射后，完整 MVI_39311 native / geometry-null HOTA 为 48.58 / 48.83，AssA 50.24 / 50.27，IDF1 60.11 / 60.52。完整 BDD clear 开发视频 b202cae2-672e61c5 共 203 帧，native HOTA 50.70、AssA 53.96、IDF1 59.57。还没有同视频最终学习目标比较，均不是全域研究结论。

object-crop source 读数已完成：同容量 CE / 预测 / 预测+ranking rank-1 为 81.54% / 84.02% / 83.70%。这一趋势与 dense-current 不同，需和 crop 的 frozen-history 对照及全视频指标区分，不能把读数变化算作 JEPA 收益。

已开始第一条实际无标签在线学习：BDD overcast adaptation 的完整 202 帧，预测+ranking / 无 replay / 延迟确认。只有 images、detections 和因果缓存进入训练；该步骤尚未完成域顺序和遗忘回测。

第一条真实无标签在线训练已完成：BDD overcast adaptation / b22f385b-5d7e5202 全 202 帧，25 次更新、904 个延迟确认伪对、458 个可靠性拒绝，6 个参数 tensor 发生变化，replay=0。原始指标和首末 loss 见 FIRST_ACTUAL_ONLINE_TRAINING.json；这证明在线更新实际运行，不证明多域遗忘改善。

## 第一个完整 BDD 受控比较

同一 203 帧开发视频、native8 类别接口、相同 geometry 和 source-only 二参数校准：geometry-null HOTA 51.08，frozen-last 51.71，CE 50.99，预测 51.99，预测+ranking 51.81。预测比 CE 高 1.00，但仅比更强的 frozen-last 高 0.28。不能把一个 source 域开发视频的结果当作目标达成。共享执行的 geometry-null 和 CE 轨迹与独立运行逐字节相同。

BDD 的 63 个 adaptation/development 视频已全部生成统一 detector stream，特征队列继续生成无 GT 缓存。UA 的完整 detector 与特征队列仍运行，使用每视频互斥锁并保持进行中的 extractor。六个相同更新间隔和 batch 的顺序学习对照已在两个数据集启动等待/消费流水线；评测将在全部训练后从保存的快照独立运行，不反馈到未来训练。

## 首个完整 UA 受控比较

MVI_39311 全 1,505 帧同预算、同校准、同 geometry：HOTA geometry-null 48.83、frozen-last 48.22、CE 47.67、预测 48.67、预测+ranking 48.55。预测比 CE 高约 0.99，但低于更强 geometry-null。geometry-null 与 CE 的共享计算输出再次与独立运行逐字节一致。该结果不支持 JEPA 达到目标；还需全域、顺序学习和多种子。

相近容量单观测 projection 已训练：660,864 参数，历史模型 662,400 参数；UA source rank-1 72.22%，与历史 CE 72.17% 接近。projection 只读最后一次 appearance 与运动 metadata；此结果提示多次 appearance 历史本身未必提供明显增量。它尚待完整 MOT 与 continual 对照。

## 信息控制、时间尺度诊断与完整开发集扩展

首个 BDD clear 视频的相近容量 projection HOTA 49.74，预测模型仅保留最后观测为 51.65；冻结特征 shuffle 为 51.91，random 为 51.35。真正的历史预测 51.99 与 shuffle 的差距很小，单视频不足以证明正确身份信息是收益原因。见 FIRST_BDD_DIAGNOSTIC_CONTROL_RESULTS.json。

同一 eligibility 的常数 affinity（不使用身份信息）1e-6 / 0.5 / 1.0 的 HOTA 为 51.23 / 50.57 / 43.87。它们与 geometry-null 51.08 的差异说明融合分数尺度会改变完整分配；极高常数会明显伤害结果。不能将该现象归因于 JEPA。见 FIRST_BDD_CONSTANT_AFFINITY_RESULTS.json。

source box-crop 的 frozen-last：UA rank-1 73.81%，BDD 62.46%。BDD 同容量 CE / 预测 / 联合为 66.62% / 63.79% / 68.81%，因此 crop 也没有两数据集一致的纯预测优势。UA crop 的 source 当前 GT latent 排序上限 100%，dense-current 为 93.15%，说明 ROI 网格别名会限制读数。

uniform source 的实际预测帧间隔：UA 最小 6、中位 19、p90 75，delta-t=1 的比例为 0；BDD 中位 3、p90 13，delta-t=1 占 12.42%。这些统计是 source 内部验证样例的观测间隔，不是跟踪评测抽帧。以相同 source 视频及相同每视频 128 图片预算准备 8×16 连续 clips，源历史在 clip 边界重置；保持检测器、encoder、参数量、2,000×64 训练预算和现有开发集不变，比较 CE / 预测 / 联合。新 source clips 训练已排队，不能预先断言收益。

完整 frozen 开发集队列已启动：BDD 10 个控制、UA 12 个控制（含两种 relation objective），覆盖全部 source 和三个目标域的 development 视频；适应和 sealed 不进入这些 frozen 对照。逐视频完成后汇总 TrackEval，新增 paired video bootstrap 会从原始计数重新聚合，以相同视频重采样而非平均每视频 HOTA。

六组持续学习的 BDD forward seed17 已完成 overcast、rainy 两阶段并进入 night；UA cloudy 已持续跨多个完整视频更新。未完成全域矩阵，不报告遗忘结论。

## H2 实际更新预算与教师流控制

冻结 frozen-last 教师在 BDD overcast adaptation 的完整 202 帧上提供同一因果轨迹与 native 独立可靠性证据。P+ranking 的 confidence-FIFO / prediction-priority / frozen-appearance-hard，以及 CE 的 confidence-FIFO / prediction-priority，共五个学习器都执行 25 次更新、确认 907 个伪对、replay 128，history/candidates 的 replay 内容逐 tensor 一致，每个学习器 6 个参数 tensor 改变；两个 prioritization 的最终参数与 FIFO 不同。每帧断言实际更新预算一致。该验证没有评测 GT，也不证明 H2 有收益。

该控制保持预测查询在当前 detection/feature 行选择前冻结，下一帧独立确认后才训练。优先级仅重新排列同一 fresh buffer，不硬拒绝低预测误差样例，从而避免 gate 降低训练次数的混淆。完整序列会在学习结束后，用各学习器快照单独部署评测；教师的 adaptation 轨迹不能冒充已学习模型的跟踪输出。见 MATCHED_TEACHER_H2_FIRST_ACTUAL_RESULTS.json 和 MATCHED_PRIORITY_PROTOCOL_VERIFICATION.json。

source uniform-dense 的 seed23/42 训练与校准已经启动，BDD seed23 完成；多种子 MOT 矩阵仍未完成。存储按域边界保留完整 model/optimizer/replay 状态，中间视频保留参数、loss 和更新记录；仅在已完成域且 domain snapshot 保留后压缩重复状态，失败记录和原始权重保留。

BDD uniform-dense 的 seed23、42 已完成同预算三个目标和校准：source rank-1 约为 CE 51.91% / 51.89%，预测 47.43% / 47.29%，联合 51.81% / 51.59%。与 seed17 相同，pure prediction 在该 source 排序检查中低于 CE；这仍不等于目标域 MOT 或持续学习结论。UA 多种子正在运行，完整记录见 SOURCE_MULTI_SEED_PARTIAL_RESULTS.json。

BDD 全部 63 个 adaptation/development native 特征视频已准备完毕；完整开发对照和多域学习继续。H2 五组同教师完整序列已在两个数据集启动。当前 BDD forward 第一波已完成两个域、37 个适应视频，UA 9 个；这只是运行进展，未形成终态矩阵。BDD 已完成域的中间 state 保留参数后释放约 2.20 GB 重复 optimizer/replay，source 和域边界完整状态均保留。

## 首个完整 source 域开发集合与统计区间

BDD clear development 的 5 个原生完整视频、1,013 帧：native / geometry-null / frozen-last / CE / 预测 / 联合 HOTA 分别为 37.87 / 37.93 / 37.58 / 37.54 / 38.09 / 37.33；projection 36.95、预测 last-only 37.51、shuffle 37.05、random 37.67。所有视频均来自冻结 split，未选有利片段。

预测相对 geometry-null 差值 +0.163，2,000 次视频级配对 bootstrap 95% 区间 [-0.827, +0.764]；相对 CE 为 +0.553，区间 [-0.166, +1.063]。相对 frozen-last 为 +0.514，区间 [+0.231, +0.775]。这是 source 域、单 seed 的结果，不能证明跨域 H1 或整体目标达成。见 BDD_CLEAR_FULL_DEVELOPMENT_RESULTS.json 与 figures/bdd_clear_development_paired_hota.svg。

BDD 第一波六组 forward 无标签训练已完成全部 overcast/rainy/night、44 个适应视频，保存了 4 个 stage 快照；完整冻结回测仍在运行。H2 同教师五组也已完成 overcast 并进入 rainy。没有填造终态矩阵或遗忘值。

## 连续 clips 的首个源比较

UA 同 source 视频、每视频 128 图片预算、8×16 clips：42507 拟合 / 10111 验证样例，三个模型均 662,400 参数、seed17、2,000×64。CE / 预测 / 联合 rank-1 为 88.54% / 88.62% / 89.80%；误差为 0.48431 / 0.02398 / 0.11792。该验证帧/样例与 uniform 不同，不能把它们的绝对分数变化写成方法增益；可以比较同一 clips 内的三个目标。原生预测间隔最小1、中位2、p90 8，delta-t=1 占31.08%。新的 source-only 校准和全 development 对照已启动；BDD clips 继续生成。

历史长度1/2/4/8 的相近容量控制已登记并开始源训练，参数分别660864/660685/662400/661821；不同目标在同一长度内结构相同。K1 不含观测速度，K4/K8 能包含过去运动；冻结推理将用对应训练长度，不能把只在推理时遮掉历史当作完整 no-history 训练控制。

## 真实在线计算链路

新的 live_stream 从 label-free 图片路径运行 source detector、官方当前图像 encoder、query、原生关联与真实在线更新。MMDetection 与 GeneralTrack 使用已有不同 Python/torch 环境，IPC、图片读取、decode/resize 和更新均计入墙钟；不以特征缓存喂入推理。

BDD 24/203 帧的工程检查：detector 与已有 source-only 检测逐数值一致，缓存与真实推理输出轨迹逐字节相同，实际执行3次更新，query 因果违例0。FP16特征最大数值差约0.00144，更新后的参数最大差约0.000284，因此不声称 latent/checkpoint 位级等同。

单次 decode 的 RGB IPC 检查保持检测与轨迹结果：共享 GPU 条件下 warmup 后约1.74 FPS，相比重复读取约1.45 FPS。这只是24帧工程测量，不能作为最终效率数字。完整适应视频的 geometry / frozen-feature / CE /预测联合 / online 计时已排队；benchmark 学习器独立，禁止恢复进正式研究训练。记录 p50/p95、实际更新耗时及两个进程的 torch memory，明确硬件竞争。

Native 可靠性计算从每个 pair 重复 sort 改为每行/列一次 sort；空矩阵、单行/列和 ties 检查证明所有 evidence 字段逐值相同。完成视频的 pending 不会跨独立视频确认，保存完整域状态时移除该无效 pending，保留参数、optimizer、replay/recent、RNG和计数。

## BDD 四域完整 frozen 开发结果

全部19视频、3844原生帧已完成。每域固定权重1/4：geometry-null 39.902，frozen-last 39.386，CE39.391，预测39.829，联合39.486，单观测projection38.989，预测last-only39.437，shuffle39.120，random39.871。source checkpoint hashes 未变化，sealed未使用。

预测相对CE +0.438，按域分层的视频配对bootstrap区间[+0.066,+0.733]；相对geometry -0.073，区间[-0.417,+0.215]；相对random -0.042，区间[-0.347,+0.313]。source clear和三个target域都完整纳入，不能只强调CE对比而忽略更强的简单基线。这是seed17 frozen结果，不能当作持续学习结果或跨seed结论。详见BDD_FULL_FROZEN_DEVELOPMENT_RESULTS.json与figures/bdd_full_development_hota.svg。

## 一步分配反事实与受限介入

记录真实native原始cost、融合cost、语义分数、过去raw框、当前选项、branch、更新状态；GT只在独立audit_counterfactual中匹配诊断。原残差预测在首个203帧视频改变91个一步分配：23个局部身份修复、20个反改错，另有混合历史/未评分/两边都正确等情况。这不是独立baseline整条轨迹，也不是官方IDSW。开启日志后原轨迹逐字节不变。

新rank_reward只在同native类别组、两个原始score-fused可行候选、native margin≤0.1、语义margin≥0.02、短gap和排序分歧时给予小正奖励。其余cost保持不变，避免随机低分导致整体衰减。首个视频预测/CE/frozen-last的HOTA约51.04/51.03/51.09，random与geometry 51.0847逐值相同；预测仅改变4个一步分配，没有该视频的正收益。全19视频对照继续，不据此调阈值或挑视频。

## 实际训练预算与进一步H3控制

BDD第一波44适应视频、8811原生帧已核查完整model/optimizer/replay状态。最终FT的CE/联合更新1073/1074，replay为1096/1096，feature/row-KL为1097/1097。相同interval不等于相同实际更新次数；这是self-teacher伪对供给差异，主表明确记录，不能声称全六组exact budget。

新增固定教师四组H3：相同native-group负例、记忆容量128、相同9个tensor字段、batch8/interval8/lr1e-4。replay、feature角距离、普通row-KL、row-KL+同帧candidate-column竞争保持。候选全局frame/detection键和物理track键只用于同帧关系，不把独立视频ID链接起来；重复replay副本按物理键去重，描述符相同的不同轨迹仍参与竞争。

在完整202帧BDD overcast验证中，四组均25更新、900确认，replay的histories/candidates/physical keys一致；新机制累计处理264个竞争column。只有旧预测query响应作为stop-gradient teacher，没有目标GT。Column KL额外计算成本另计；这是MOT-specific保持假设，尚未证明优于普通KD或遗忘更低。完整多域训练将继续。

## 完整实时效率与多seed/order推进

在独立的202帧适应视频、同一GPU5、startup权重载入不计、warmup10帧后：geometry2.607 FPS、frozen-last1.763、CE frozen1.660、联合frozen1.693、CE online1.711、联合online1.603。在线两组实际25更新，update-call p50约9.70/12.20ms；该调用包含伪对/batch/replay和optimizer。整个wall time包含图片读取、detector IPC、encoder和关联；torch peak memory按进程列出。所测GPU起止未发现其他job，但整个server仍共享，不把单视频数值泛化为全部场景或实时30FPS。

source初始化的benchmark权重独立，禁止恢复到研究训练；P_*路径实际为prediction+ranking，未称pure prediction。完整表见BDD_FULL_LIVE_EFFICIENCY_RESULTS.json，UA全视频计时继续。

seed23/42的BDD六组CL已排队，共享的仅为frozen image-pair计算，学习器/优化器/记忆与物理轨迹分别独立；随后反顺序使用相同44视频和development集合。检测器始终seed17共同输入，source heads与online sampler种子不同。

## UA 完整 sunny 开发集合与源时间支持诊断

三个完整sunny开发视频的uniform-dense HOTA：native56.722、geometry56.907、frozen-last56.444、CE56.354、预测56.913、联合56.724、projection57.007、预测last-only57.123、random57.025。预测相对geometry仅约0.006，不能以该域source结果宣称预测优势。

相同开发图片与检测输入，连续clips源模型：frozen-last56.430、CE56.299、预测56.292、联合56.792。source clips排序改善未转化为该完整source域上的pure prediction提升；该负结果保留，目标域仍需完成。见UA_SUNNY_FULL_DEVELOPMENT_TIME_SUPPORT_RESULTS.json。Crop读数已有两个完整source视频，不作为完整三视频汇总。

1/2/8历史的18个source目标全部训练完毕，matching capacity原生部署正在进行。UA源排序不同长度接近；BDD一观测CE约56.75%，8历史CE55.66%，pure预测约52.51%/52.99%。不能仅靠更多历史或source排序证明机制，完整方法见HISTORY_LENGTH_SOURCE_COMPLETE_RESULTS.json。

UA完整cloudy适应视频MVI_40152的geometry实时链路已完成1750帧：warmup后3.016 FPS，起止GPU未发现其他job，其他live方法继续。该测量是孤立效率benchmark，不改变研究checkpoint，也不是持续学习分数。

已导出15个实际启动的控制配置到configs/JEPA_CMOT，替换的只有机器特定repo前缀；原始相对路径指向本地权重/校准。当前BDD仍仅stage0完整回测，后续阶段运行中，不提前填造forgetting。完成域的intermediate checkpoint压缩累计释放约4.10GB，域边界完整state及所有中间weights保留。

## 复用实际相同的冻结stage0结果

C2的四组初始θ、source-only校准、native输入和geometry完全相同于已完成的BDD source prediction+ranking frozen对照。reuse_frozen_evaluation逐项核对checkpoint SHA、校准JSON、19视频图片/检测/特征路径和帧数，再链接真实已生成的轨迹与disabled-learner快照；76个轨迹文件SHA均验证一致。这里只复用相同静态source评测，不复用任何adapted stage。

run_controls在全部video均完成时无需建立新的shared matcher；已修正其None-resume分支并用真实76个output完成恢复验证。C2后续stage评测仍需实际运行，forwards=0仅标明本次恢复没有重新计算，不作为新FPS。记录BDD_COMPETITION_STAGE0_ACTUAL_REUSE_VERIFICATION.json。

## 通用关系KD与资源调度

针对RKD等已知关系蒸馏，新增normalized pairwise distance + smooth-L1的RKD-D式对照（teacher为各记忆样例的旧query），不使用当前候选列。它与replay、candidate-column组共享source checkpoint、native-group负例、九字段memory128、batch8/interval8/lr1e-4和冻结teacher。完整202帧实测三组均25更新、900确认，memory的history/candidates/physical keys一致。原candidate-column项在25步内mean约0.000336、max0.004129，13步大于1e-7，不以最后一步接近0误判整条loss路径无效。只证明运行与预算，不能证明防遗忘或新颖性。详见GENERIC_RELATIONAL_KD_ACTUAL_VERIFICATION.json。

两数据集的通用RKD完整顺序对照排队在C2之后执行；沿用相同视频和独立冻结快照评测。C2的extra compute与普通row/feature/RKD对照必须最终报告。

BDD clips source模型/校准已经准备好，完整开发集部署排队在rank_reward后运行。GPU0多组回测拥挤，UA历史长度组计划等GPU5实时benchmark完成后，先停止控制器派发、等待活跃完整视频终态，再迁移；已完成输出和模型不改写，不能因观察超时重启正在训练的模型。

## BDD 完整开发集的关联规则检验

19个开发视频、3,844个native帧、source head seed17，冻结相同输入和source-only校准。四域等权HOTA：原预测残差39.8291、新预测rank reward39.9134、相同规则CE39.9124、frozen-last39.9204、random和geometry均39.9022。预测相对geometry +0.01122，2,000次分层配对视频bootstrap 95%区间[-0.02173,0.04409]；预测相对同规则CE约+0.00098。这是固定seed的开发集不确定性，未做选择校正，不能替代多seed或sealed结论。

原残差日志1732次局部分配变化，354个局部身份修复、300个反改错；新预测规则23次变化，仅1个修复、1个反改错。其它记录含混合历史、未评分、双方正确/错误，不能相加解释为官方IDSW。日志开启后的原预测19条轨迹与此前结果逐字节一致；random新规则19条轨迹与geometry逐字节一致。事件文件按变化惰性写入，完整且开启audit的零事件视频可以没有jsonl。

当前证据支持“规则显著限制了语义介入”，不支持“JEPA有实际独立收益”。不据此放宽目标GT门控、改split或用新阈值筛选有利视频。完整结果BDD_FULL_RANK_REWARD_RESULTS.json，生成命令为`./run_python.sh -m jepa_cmot.rank_reward_analysis --root runs/JEPA_CMOT/development_rank_reward_v1/bdd100k_seed17 --reference runs/JEPA_CMOT/development_all_v1/bdd100k_seed17 --manifest runs/JEPA_CMOT/native_protocol_v3/manifest.json --out docs/JEPA_CMOT/BDD_FULL_RANK_REWARD_RESULTS.json`。

纯预测FT/replay、两数据集reverse17以及UA23/42的完整序列已注册源权重和配置并排队。当前新增队列不读取开发指标，也不会基于该负结果改变预算；等待既有完整实验释放GPU后启动。未完成任务不计作科研证据。

## BDD 第一目标域的完整回测（持续矩阵仍不完整）

第一波六组stage1完成全部19开发视频回测，stage0逐指标等于此前实际source frozen结果。适应overcast后HOTA变化（clear / overcast）：CE FT -0.417 / +0.045，联合FT +0.255 / -0.173；CE replay -0.604 / +0.092，联合replay +0.016 / -0.158；feature KD -0.259 / -0.607，row KD -0.379 / -0.489。联合replay四域均值相对本身source仅+0.071，其它组均值下降。

联合模型在这一阶段的旧source HOTA较稳定，但新域分数下降，普通KD也没有自动改善，不能据此宣称stability-plasticity优势。self-teacher实际预算差异仍须报告，固定教师H2/H3进一步检验。stage2/3回测未完成，不填最终forgetting/BWT/FWT；完整指标见BDD_CONTINUAL_STAGE1_PARTIAL_RESULTS.json。

## 固定教师伪样本的离线身份诊断

在C2已完成overcast/rainy域边界，按九字段memory的全局frame/detection key还原正例和历史框；正例descriptor逐值一致，608/619个可还原历史观测的feature最大差5.96e-8、bbox最大误差6.87e-5像素。四组的frames/updates/seen和memory/recent的7个输入/物理字段完全相同；query/teacher分布允许因模型不同而不同。目标GT只在独立诊断使用。

stage1 memory128：119身份一致、4混合历史、5历史无GT匹配；remaining fresh25全部一致。stage2 memory128：114一致、6混合、1确认身份错配、6历史未匹配、1当前检测未匹配；remaining fresh28全部一致。这是reservoir和残余fresh buffer的终态，不能外推为全部15,667/后续确认对的错误率，也不能代表按梯度重复次数加权的训练分布。

stage2那1个确认错配的prediction need=0.1333，而一致样本均值约0.1206；不能由单个错误证明error gate有效。高预测误差也可能反映困难身份/检测噪声。当前有限记忆中大多可核对的正例身份正确，持续学习负迁移不能直接全归咎于伪对错配；完整source-validation snapshot检查继续，以区分表示/排序退化和MOT融合问题。聚合证据BDD_COMPLETED_BOUNDARY_REPLAY_IDENTITY_AUDIT.json。

## 完整源验证快照：预测loss下降但身份排序下降

对六组×四stage的24个实际快照，使用原source内部5个heldout视频的全部23,623对、相同gaps1/2/4/8和source_GT候选，CPU FP32无优化复评。按训练fingerprint核验输入几何1280×720，stage0 source rank1与原GPU验证最大差小于1e-8、预测error差小于3e-8。不是新的目标GT监督。

从stage0到stage3：CE replay rank1 .524940→.507086，prediction error .671088→.593682；联合replay rank1 .515276→.506683，error .232598→.207897；联合FT .515276→.503150、error .232598→.185719。联合feature KD最终rank1 .506580，row KD .504503。latent接近真目标并不自动保持hard-negative区分：联合replay的positive-minus-hard-negative mean cosine .020826→.014724。

该联合replay全源query covariance trace .542556→.567305，distinct-pair mean cosine .457421→.432671，未显示全局完全collapse；不能用这一项排除条件性/身份级退化。四stage的source描述符诊断已完整，stage2/3的MOT回测仍在运行，不能把该表称为完整MOT遗忘证据。不同目标的loss尺度不同，不跨模型池化报告loss与AssA相关性；等真实完整矩阵后再做逐方法探索性分析。见BDD_COMPLETE_SOURCE_VALIDATION_SNAPSHOT_AUDIT.json。

已将53个本次完成source模型的冗余中间optimizer/RNG状态缩减，保存所有901份中间模型weight、全部final.pt及首/中/末完整训练state；逐文件验证θ tensor相同并记录前后SHA，释放约4.797GB。历史备份与NAS不改。详见SOURCE_CHECKPOINT_COMPACTION_SUMMARY.json。

## 官方encoder/predictor配对读数：正式任务已排队

补齐任务书encoder-only与官方predictor feature对照。两者相同4帧因果窗口、相同现有连续source clips/128图像预算、相同ROI/detector rows；encoder使用当前tubelet，predictor全final tubelet masked，仅由t-3/t-2预测1664-D map。两种读数信息因mask不同，这正是该补充对照要记录的差别，不能声称相同当前像素信息或总计算量。source identity head参数662400/662657，hidden128/79，相差0.04%；三目标各2,000×64、seed17、相同source-only校准。两数据集全development都在pipeline中，尚无正式收益结果。

初始化缺失历史用normalized-zero，不复制当前帧到过去context；source窗口按video/segment重置，不跨clip洞。FP16扰动检查失败，相同input也波动（一次repeat raw token max差0.05469），故不能由该差直接断言信息泄漏。FP32+deterministic的4帧/64对象实际GPU验证：同input和masked-target变化均0，改变过去像素的max token差13.1614；两读数1024/1664维，有限、L2 norm约1，offsets相同。工程smoke不计source正式训练。

代码paired_readout_cache/pipeline独立于现有tracker；全实验等待UA两feature producer完成后使用GPU2。完整202帧无标签适应视频的读数/跟踪接口验证继续，仅作为工程闭环，不用于选阈值。详见OFFICIAL_PAIRED_READOUT_REGISTRATION.json。

## 两数据集12个完整图像链路效率案例完成

新增UA完整cloudy MVI_40152（1,750 native帧）六方法已完成，GPU5各case起止未观察other-job，source初始化、warmup10在原轨迹内，权重启动时间不计。geometry3.016 FPS，frozen-last1.901，CE frozen1.804，联合frozen1.849，CE online1.846，联合online1.833；online两组均218次真实更新，update-call p50 9.919/9.778ms，p95 21.468/20.922ms，包含伪对、fresh/replay、梯度和optimizer。联合online整链路p50/p95 528.369/671.482ms，torch两进程reservation峰值之和约3.986GiB（不含driver context，峰值未必同时）。

BDD六组202帧结果保持：联合online1.603 FPS、25次更新、end-to-end p50/p95 617.139/748.651ms。12案例均来自真实NAS图片+private detector IPC+current encoder+frozen matcher，cached_input_used=false、GT_read=false、history current violations0，benchmark θ从未接续到研究训练。原目录P_*实际是prediction+ranking；不是pure预测。

同一个dataset共享stream和detector SHA已核验，但每方法只有一次执行、整个server/CPU/NAS仍共享，只检查GPU起止；CE在线偶然更快不作为加速结论。主full-video launch没有传parity reference，默认reference error=0不能理解为bitwise cache parity。这个表尚未测C2/RKD或者新FP32 w4读数的额外成本。完整可复核报告FULL_NATIVE_LIVE_EFFICIENCY_RESULTS.json。

官方配对读数完整202帧BDD适应视频接口验证已完成：encoder/predictor各2824条detection rows，维度1024/1664；FP32重复/目标扰动差0，过去扰动max token差11.0013。两读数均实际跑完native geometry+frozen-last，202帧、updates0、GT_read=false、history violations0；两套geometry轨迹逐字节一致。校准和source训练未在该工程验证中进行，不报告HOTA或把它当作全development收益。详见OFFICIAL_PAIRED_READOUT_NATIVE_TRACKING_VERIFICATION.json。

配对读数的源负例进一步固定为共同encoder的last-feature cosine挖掘。Predictor源cache额外携带对齐的encoder ranking features，仅用于选取同一物理候选，不输入predictor历史网络；两读数的正例/历史/gaps与hard-negative行完全相同。针对不同1024/1664特征、每帧17个可用负例的sampler检查，18个样例candidate index列表一致。旧cache没有该字段时SourcePairs保持原行为；旧checkpoint与source训练记录不改写。

## 固定教师第一阶段与第二目标域回测

BDD H2第一阶段5组均2630 native适应帧、328个实际optimizer步、15756确认对、memory128，原source stage0逐指标等于独立frozen开发评测。相对source四域HOTA变化：联合confidence -0.2736，prediction priority -0.2025，普通appearance-hard priority +0.0956；CE confidence -0.4458、CE prediction priority -0.2680。预测priority相对confidence +0.0711 CI[-0.1139,0.3159]，相对普通appearance-hard -0.2982 CI[-0.5045,-0.1435]，没有JEPA需求信号的独立正收益。固定seed开发探索，不是选择校正或跨seed结论。详见BDD_FIXED_TEACHER_STAGE1_RESULTS.json。

BDD C2第一阶段4组均2630帧、328步、15667确认对、memory128；比H2少89对来自共同native-group负例过滤，不声称跨family的样本完全相同。四组相对source HOTA变化replay -0.1091、feature -0.1628、row -0.1207、row+column -0.1203。column相对row仅+0.0004508，相对replay -0.01115 CI[-0.04232,0.00937]，当前无实质增量。详见BDD_COMPETITION_STAGE1_RESULTS.json，RKD与后续回测继续。

第一波六组stage2（适应rainy后）全部19视频回测也已完成：相对本身source均值CE FT -0.6295、联合FT -0.3114、CE replay -0.4545、联合replay -0.3520、feature KD -0.0230、row KD +0.0099。row KD新rainy域仍-0.06045，old overcast -0.31751，不能用宏均值正号称适应成功。实际预算记录见BDD_CONTINUAL_STAGE2_PARTIAL_RESULTS.json；final night快照回测未完成，不计算最终forgetting。

## 完整BDD历史长度对照

同source clips、相近容量、2,000×64，全部19开发视频/3844 native帧：H1/2/4/8 CE HOTA39.388/39.517/39.635/39.394，pure预测39.527/39.820/39.718/39.644，联合39.747/39.656/39.746/39.989。联合H8相对H1 +0.2422 CI[0.0951,0.3889]，但相对强geometry39.902仅+0.0866 CI[-0.3717,0.5471]；不从选出的best长度宣称JEPA优势。H4三目标19条轨迹与独立clips部署逐字节一致。详见BDD_FULL_HISTORY_LENGTH_DEVELOPMENT_RESULTS.json。

GPU7/8/9原外部任务终态后，已对UA main/H2/C2安排完整native child边界迁移；H2已经在GPU8接续，main/C2仍等待当前child自然结束。旧控制器停止派发但保留到实际完整experiment终态，既有dependent队列不会因PID变更提前启动；θ/optimizer/replay/recent/RNG仍从完整视频checkpoint继续，未停止训练child、未改参数/数据。GPU5的历史长度迁移也继续等待它的当前完整视频。记录UA_NATIVE_BOUNDARY_GPU_MIGRATION.json为该时刻快照，不把待迁移当作已完成。

## 进一步补齐普通mean/EMA强对照

此前SourcePairs的mean/recent-weighted仅是GT轨迹源诊断，实际native frozen控制器使用last；不能把源诊断当作完整MOT EMA基线。新增独立PrototypeHistoryTracker：mean4用最多四个严格过去描述符的单位均值；EMA使用完整physical-track历史、beta .25，每次更新后归一化，视频重置。当前观测只改变下一帧prototype，单元检查验证本帧已冻结query逐值不变。核心CausalHistoryTracker/online/anchor控制器不修改，新run_controls仅按显式query_aggregation选择子类，默认last行为保持。

两种prototype都无trainable identity head，source-only两参数BCE校准200步，使用原匹配源验证video/positive/candidate/gaps：BDD23623对、UA11831对；不是用目标GT校准。与geometry、last同一source detector/dense-current cache/geometry/lifecycle/lambda .15，完整两数据集development已启动。当前只有校准与运行记录，不提前报告收益，见FROZEN_PROTOTYPE_CONTROL_REGISTRATION.json。

## H2第二阶段：需求信号未形成独立优势

BDD H2 stage2全部19视频/四域已实际回测，五组均4560适应帧、570真实optimizer步、27193确认/seen样例、memory128。相对source均值联合confidence -0.06315、prediction priority +0.04177、appearance-hard +0.11335；CE confidence -0.51344、CE prediction priority -0.33464。prediction priority相对appearance-hard -0.07158 CI[-0.21970,0.07691]；其新rainy域相对source -0.34295，appearance-hard -0.02717。不能由old clear上+0.66329或小宏正号宣称新域改善；final night仍待完整回测。见BDD_FIXED_TEACHER_STAGE2_RESULTS.json。

新增prototype部署已经完成两个完整BDD开发视频。原geometry/last与此前独立全开发集对应轨迹逐字节相同，验证新增子类选择不改变既有默认对照。mean/EMA全19视频仍在运行，不对这两个视频选择方法或改参数。

## BDD第一波完整持续矩阵完成（仍非项目最终结论）

六组完成source→overcast→rainy→night全部44适应视频/8811 native帧，所有4stage×4domain×6指标均已真实回测，final开发覆盖19视频/3844帧。最终等域HOTA / AvgForgetting：CE FT38.9482/.2777，联合FT39.3578/.2048，CE replay39.2745/.0000，联合replay39.1922/.1802，feature KD39.4747/.0555，row KD39.3976/.2895。AssA/IDF1/DetA/MOTA/IDSW、BWT/FWT及逐域负结果完整保存在BDD_FIRST_WAVE_FULL_CONTINUAL_RESULTS.json。

遗忘使用已学会域在final前的historical best减final，主表max(0,·)，另保留signed值；CE replay的0不意味着所有域超过其source模型，也不表示适应成功。最终几何静态基线39.9022，高于这六个学习方法。联合FT比CE FT +.4096 CI[.0359,.7468]，但联合replay比同replay CE -.0823 CI[-.7217,.4466]；feature KD比CE replay +.2002 CI[-.3344,.6734]，比强geometry -.4275 CI[-.9591,-.0294]。联合replay比geometry -.7100 CI[-1.3526,-.2166]。这是固定seed的开发阶段证据，不能由弱FT对照上正差称JEPA持续学习优于强简单方法。

实际final self-teacher步数CE/联合FT1073/1074、两replay1096/1096、两KD1097/1097；样本供给仍可不同，不等价于固定teacher完全同样本。后续H2/C2/RKD、多seed/order、pure预测、crop/official读数与UA仍须完成，sealed未用。图bdd_first_wave_complete_continual_hota.svg；final paired report BDD_FIRST_WAVE_FINAL_PAIRED_COMPARISONS.json。完成矩阵后原BDDseed23/42队列已经满足首个依赖，继续实际完整数据流。

完整4stage源验证error与source开发AssA/HOTA的逐方法配对也已记录：BDD_SOURCE_PREDICTION_ERROR_ASSOCIATION_EXPLORATION.json。每方法只有4个相关的时序点，源descriptor验证与source-MOT开发来自不同视频集合；仅描述Pearson系数，不给p值/因果结论、不跨CE与联合模型loss尺度池化。

## 同容量Transformer历史关联/预测对照

新增两层history-only Transformer（hidden128、4 heads、FF512、learned temporal positions、dropout0），从严格过去feature/motion/quality预测状态，然后与当前候选评分。source三目标CE/pred/联合分别同init/sample seed17、同原cache/SourcePairs/history4/gaps1/2/4/8、同2,000×64预算。参数662016 vs MLP662400，零residual初始化同last，训练/推理padding值扰动都不改变valid query，梯度有限。旧history/projection结构与state_dict字段不改，只新增显式architecture分支。

两数据集实际CE源训练完成：BDD源rank1 .513198、error .693113；UA .714606/.646686（原MLP CE .524940/.671088、.721734/.614629）。三目标源训练均已完成：BDD pure .479112/.101517、联合 .509405/.252874；UA pure .675866/.062879、联合 .713662/.206808。不用source排序代替MOT结论。每数据集三源模型/source-only校准后部署全部development；新源中间state按已有规则压缩保留weights与主要state。详细MATCHED_TRANSFORMER_CONTROL_REGISTRATION.json。

## BDD H2完整固定教师矩阵完成

五组全部44适应视频/8811帧和4×4真实开发矩阵完成，均1096实际optimizer步、41563确认/seen、memory128，预算相等已从完整checkpoint核验。final等域HOTA / AvgForgetting：联合confidence39.24177/.26008、prediction priority39.46844/.23199、appearance-hard39.40309/.43569、CE confidence39.07277/.23244、CE prediction priority39.26195/.12968。

最终prediction priority相对appearance-hard +.06535 CI[-.08315,.23180]，相对相同priority CE +.20649 CI[-.32410,.67217]；联合confidence相对CE confidence +.16901 CI[-.26274,.53218]。这些固定seed开发CI跨0；最强H2 final仍低于静态geometry39.9022。prediction priority在stage1/2低于ordinary appearance-hard，不能仅用final的小正号称学习需求机制稳定有效；其forgetting也不是所有对照最小。完整6指标/4stage表、BWT/FWT、实际预算和不确定性见BDD_FIXED_TEACHER_FULL_CONTINUAL_RESULTS.json及BDD_FIXED_TEACHER_FINAL_BUDGET_AND_COMPARISONS.json。

## 普通MLP pair scorer同容量控制

独立PairAssociationModel先编码严格过去history/metadata为122-D状态，再以candidate projection与concat MLP输出scalar logit；当前候选仅进入评分器、不进入历史encoder。这是普通association classifier，不称latent JEPA，prediction error明确None。参数662229 vs history MLP662400，相差0.026%；源原cache/fit-val videos/SourcePairs candidates/gaps/seed17/2,000×64预算相同，raw logits的source-only两参数BCE校准200步。

两数据集真实源训练完成：UA source rank1 .764714、BDD .504800（MLP CE .721734/.524940）；源排名不能替代MOT。历史状态对改变当前candidate保持不变，score随candidate变化；padding/有限梯度检查通过。FrozenPairReader仅推理不更新，PairHistoryTracker独立覆盖评分，默认CausalHistoryTracker路径不变。全部native development已排队启动，细节MATCHED_PAIR_MLP_CONTROL_REGISTRATION.json。

## BDD普通prototype完整开发对照完成

19视频、3844 native帧、source calibration matched：geometry39.90215、last39.38565、mean4 39.59214、lifetime EMA39.65004。EMA比last+.26440 CI[-.21251,.64579]，比mean4+.05790 CI[-.23867,.35040]，比geometry-.25211 CI[-.81295,.21041]；都为fixed-seed开发视频uncertainty，不外推跨seed。新增运行的geometry/last全部19轨迹与之前独立完整frozen结果逐字节一致；普通聚合目前没有可靠越过强geometry。全域指标和SHA见BDD_FULL_FROZEN_PROTOTYPE_RESULTS.json，UA完整对照继续。

普通pair scorer已实际完成三个native BDD开发视频，GT_read=false、updates0、current history violations0，历史state122维/候选feature1024维的独立评分接口已跑通。源UA rank较高不能作为开发/持续收益证据；全部开发deployment继续。

## BDD C2完整同预算保持矩阵完成

四组全部44适应视频/8811 native帧、四stage×四domain真实回测完成；实际更新均1094、确认/seen均41176、memory128。final HOTA / AvgForgetting：replay39.30093/.25557、feature KD39.23575/.34985、row KD39.30327/.25077、row+column39.32630/.22007。column比row +.02302 CI[0,.05350]，比replay +.02537 CI[-.20013,.29735]，比feature +.09054 CI[-.12480,.35599]。量级很小、固定seed开发探索、没有方法选择校正，不能称MOT-specific relation机制获得可靠优势；所有组仍低于强静态geometry39.90215。

完整6指标/4×4/AF/BWT/FWT与actual budget见BDD_COMPETITION_FULL_CONTINUAL_RESULTS.json，逐视频raw-count bootstrap见BDD_COMPETITION_FINAL_BUDGET_AND_COMPARISONS.json。与H2不同的native-group负例过滤改变确认对供给，机制增量只能在同C2 family内解释。通用RKD-D序列已实际开始，当前已完成8个完整适应视频，仍待后续域及全部回测；不以待完成结果宣称创新。

## 完整开发关联事件的间隔/观测运动分组

BDD完整19开发视频recorded one-step changes重新离线匹配GT，分组阈值仅由source_GT bbox native速度的25/75分位定义（image-diagonal/frame .00139888/.01337447）。原残差在gap1的identity corrected/harmful=293/203，gap2-4=51/85，gap5-8=8/11，gap>8=2/1。保守rank_reward只出现1次修复（gap2-4）与1次反改错（gap1）。

间隔指最后一次已提交feature的frame gap，不是真实遮挡标注；bbox speed混合camera/object motion/关联错误。统计仅条件于改变的局部分配，不是全部机会率或独立baseline后续轨迹；长gap也可能因其它row改动而间接改变Hungarian结果，不能把2/1当作长遮挡forecast证据。没有基于这些target GT分组改变门控。见BDD_COMPLETE_LOCAL_EVENT_GAP_MOTION_STRATA.json。

## BDD强普通head完整开发对照完成

Transformer三目标与ordinary pair MLP均完成19开发视频/3844 native帧。相同原source cache/SourcePairs/2,000×64、近似capacity（MLP662400、Transformer662016、pair662229）、同source-only score calibration、same detector/native clock/geometry。final frozen平均HOTA：Transformer CE39.44024、pure预测39.97191、联合39.47068；pair MLP39.24595。源UA pair rank .764714未自动转化为BDD部署优势；UA全开发仍继续。

Transformer纯预测比同Transformer CE +.53167 CI[.10729,.99374]，但比强geometry39.90215仅+.06976 CI[-.37109,.49923]，比MLP预测39.82910 +.14281 CI[-.09065,.44687]。pair MLP比MLP CE -.14527 CI[-.67304,.28914]、比geometry -.65621 CI[-1.19582,-.24943]。MLP pure预测比pair +.58316 CI[.17442,1.01228]，比EMA +.17906 CI[-.23841,.69538]。不能以弱对照的正差、多个候选中的best点估计或未选择校正的开发区间声称稳定JEPA优势。

这些是frozen阶段，不能替代持续学习结果；图bdd_complete_strong_heads_hota.svg、所有域六指标/actual protocol/raw-count paired bootstrap见BDD_FULL_STRONG_HEAD_DEVELOPMENT_RESULTS.json。paired official w4全源BDDcache已真实生成，后续训练/development继续；RKD已完成15个真实适应视频，多seed forward已9视频，仍不提前汇报final。
