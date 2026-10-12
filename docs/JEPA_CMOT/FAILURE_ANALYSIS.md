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
