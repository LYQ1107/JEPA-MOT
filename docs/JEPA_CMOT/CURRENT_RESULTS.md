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
