# 当前状态：ACTIVE / 完整原生视频持续适应与诊断

研究分支：`research/predictive-jepa-cmot-20261013`。历史结果与新训练分开；尚未形成新方法结论。

# JEPA-MOT 长周期自主科研任务书

## FINAL GOAL

**Predictive Identity JEPA for Continual Multi-Object Tracking**

目标：在 UA-DETRAC 和 BDD100K 两个数据集上，研究、实现并验证一种以 JEPA 潜在预测为核心的无标签多域持续学习 MOT 方法。

最终希望证明：

1. JEPA 式潜在预测学习相比普通判别式关联学习，能够提供额外且可重复的身份关联收益。
2. 这种预测关系能够成为连续域适应中的有效学习信号，在学习新域的同时减少旧域身份关联知识的遗忘。
3. 优势来自可解释、可消融的预测机制，而不是仅仅增加参数量、加入 Replay、改变跟踪阈值或者使用更强的预训练特征。
4. 方法满足严格的因果在线 MOT 协议，不能依赖未来帧、目标域 GT 或离线轨迹修正。
5. 最终在完整视频和完整域顺序上取得可信的 HOTA、AssA、IDF1、Average Forgetting 结果，并提供参数、显存、推理延迟和实际在线更新成本。

**这是一项持续研究任务，不是执行三个预设实验后给出 GO/NO-GO 的短任务。**

你必须按照实验结果主动诊断瓶颈、形成新的可证伪假设、修改模型、重新训练、继续验证。

但科研结论必须真实。禁止伪造增益、挑选有利测试视频、修改评测协议、泄漏 GT 或将无效实验宣称成功。

---

# 0. 总体执行原则

你是本项目的自主算法研究工程师，兼任严格的 CVPR 审稿人。

在整个执行过程中，必须同时思考：

- Researcher：如何提出真正有价值的 JEPA-MOT 算法？
- Engineer：怎样可靠实现、训练和评测？
- Reviewer：当前提升能否被简单 MLP、Transformer、Replay、EMA 或几何规则解释？
- Scientist：当前实验能否支持相应的科研结论？

不允许只负责运行既有脚本。你需要根据证据修改模型结构、训练目标和持续学习策略。

遇到负结果，首先执行诊断和下一轮改进，而不是立即结束项目。

**禁止以下行为：**

- 仅尝试几组 lambda、similarity threshold、IoU threshold 后宣布方向失败。
- 把普通特征融合或可学习门控直接称作 JEPA 创新。
- 只比较原始 GeneralTrack，不比较同几何规则、同容量的强对照。
- 使用评测 GT 指导目标域在线训练、模型选择或阈值调参。
- 只报告最好的随机种子、最佳视频或最佳阶段。
- 将旧实验结果复制为本次重新训练结果。
- 为了取得提升改变 detector rows、GT 协议、tracker 生命周期或测试集划分。
- 通过删除困难视频、排除负结果或调整测试集获得所谓成功。
- 未经许可覆盖历史 checkpoint、NAS 原始数据或其他项目目录。
- 无限制增加复杂度而不做必要的受控消融。

研究中必须把“技术上运行成功”和“科学假设获得支持”分开。

---

# 1. 项目环境与代码恢复

GitHub：

https://github.com/LYQ1107/JEPA-MOT

当前已知本地工作区：

`/data1/liuyeqiang/JEPA_MOT_resume_20261012`

NAS：

`/data1/liuyeqiang/nas/A100`

主要代码：

- `GeneralTrack-main/`
- `JEPA/`
- `DA-MOT/darth-main/`
- `reference_reports/`
- `results/`
- `workspace_manifest.json`
- `recovery_state.json`
- `archive_paths.py`
- `run_python.sh`

开始前必须读取仓库根目录和相关子目录的 `AGENTS.md`，遵守已有执行约束。

### 1.1 首先检查

1. 核实 GitHub 远端、当前分支、HEAD、工作树状态。
2. 创建独立研究分支：`research/predictive-jepa-cmot-20261013`。若同名分支已经存在，则检查并接续，禁止覆盖别人的工作。
3. 检查 Conda、CUDA、PyTorch、GPU、可用显存和磁盘。
4. 验证官方 V-JEPA 2.1 checkpoint。
5. 检查 GeneralTrack checkpoint 和训练入口。
6. 验证 NAS 数据集与历史特征缓存能否访问。
7. 核对 `workspace_manifest.json` 中的 SHA256。
8. 检查可执行的在线跟踪器入口。
9. 检查原始评测脚本及 TrackEval。
10. 检查 GitHub 推送权限。

优先复用已有环境，不要随意升级共享 Conda 包。

GPU 使用应根据实际空闲情况选择，不要擅自占用已有任务使用的显卡，不要批量结束其他用户进程。

如果现有环境存在缺失，应优先在隔离环境中修复。

### 1.2 历史研究基线

必须阅读：

- `reference_reports/ALL_DATASETS_VJEPA21_final_audit.md`
- `reference_reports/UA-DETRAC_VJEPA21_Online_TBD_ICLR_audit.md`
- `reference_reports/JEPA_independent_tuning_audit.md`
- `DA-MOT/darth-main/experiments/jepa/DA_MOT_JEPA_PROTOCOL.md`
- `JEPA/GOT-main/experiments/MOT_JEPA_FINAL_REPORT.md`

历史观察：

UA-DETRAC：

GeneralTrack HOTA 33.5983；Geometry-null 38.3082；V-JEPA21 40.3552。

BDD100K：

GeneralTrack HOTA 35.1767；Geometry-null 35.0900；V-JEPA21 35.3925。

这些数字只是历史参考，不是本次新方法的结果。

需要特别注意：

- 当前 GeneralTrack 中 V-JEPA 主要用于身份关联 fallback。
- 官方 V-JEPA predictor 输出视觉潜在 token，并不直接预测下一帧目标坐标。
- 原有时间窗裁剪使用当前 detection box 作为历史裁剪锚点，可能引起目标运动或相机运动下的时空错位。
- 旧 temporal adapter 的 `deploy_alpha=0`，不能未经验证就认定它提供有效的预测初始化。
- DA-MOT 中的 256-D QDTrack TemporalJEPAAdapter 和当前官方 V-JEPA 2.1 的 1664-D 特征并非同一套完整模型。

需要明确区分历史模型、当前恢复状态和此次新训练。

首先跑真实 GPU smoke，验证模型前向、缓存读取、完整在线跟踪及评测链路。

此前 40 组 MOTSynth 指标复算成功，只证明历史预测可以复评，不代表新模型已经训练成功。

---

# 2. 审稿人的核心质疑

在设计方法之前，必须完成文献与创新性审计。

至少核对：

1. GOT-JEPA，2026  
   https://doi.org/10.1109/TCSVT.2026.3675005

2. FACT，2026  
   https://doi.org/10.1016/j.knosys.2026.115935

3. TCEI，CVPR 2026  
   https://openaccess.thecvf.com/content/CVPR2026/html/Guo_Dual-level_Adaptation_for_Multi-Object_Tracking_Building_Test-Time_Calibration_from_Experience_CVPR_2026_paper.html

4. V-JEPA 2.1，2026  
   https://arxiv.org/abs/2603.14482

5. Unsupervised Continual Domain Shift Learning with Multi-Prototype Modeling，CVPR 2025  
   https://openaccess.thecvf.com/content/CVPR2025/html/Sun_Unsupervised_Continual_Domain_Shift_Learning_with_Multi-Prototype_Modeling_CVPR_2025_paper.html

搜索是否还有 2025—2026 年更接近的在线 MOT、predictive representation、continual domain adaptation、predictive replay、association-model prediction 方法。

不能假设这些工作与我们没有重叠。

必须重点回答三个问题。

### Q1：Why JEPA?

为什么使用 JEPA 式预测学习，而不是：

- 普通历史特征平均；
- EMA prototype；
- Temporal MLP；
- Temporal Transformer；
- 同容量 GRU；
- 普通关联分类器；
- BYOL-style 自蒸馏；
- 传统监督/伪监督的 metric learning？

你必须通过相同特征输入、相同参数量或参数预算、相同训练数据和训练预算进行受控比较。

JEPA 的核心不应只是某个模型名称，而应该体现在清晰的预测目标与学习机制上。

### Q2：Why Continual Learning?

为什么这种潜在预测学习特别有利于连续域变化？

必须验证：

- 无标签新域数据能否提供可靠的预测监督；
- 是否能够降低参数更新造成的旧域身份关联退化；
- 是否优于普通 Replay 和知识蒸馏；
- 是否具备更好的 stability-plasticity trade-off；
- 是否能够对连续出现的未知域进行适应，而不是针对已知域单独训练模型。

禁止把单域 HOTA 提升等同于持续学习成功。

### Q3：Why MOT-specific?

如果同样的方法可以用于普通图像分类或单目标跟踪，那么 MOT 特有的研究机制是什么？

优先考虑：

- 一个轨迹对应多个候选检测的竞争；
- 同帧相似车辆形成的 hard negatives；
- 在线关联错误会污染后续身份历史；
- 遮挡和漏检导致的轨迹历史不完整；
- 不同域中身份关联关系的保留；
- 多目标一对一分配约束；
- 在线伪标签错误会跨时间累积；
- 跨独立视频的模型参数持续学习，但轨迹 ID 不能跨无关视频直接继承。

研究目标应该是预测与保留 **identity association relations**，不是泛泛预测视频特征。

文献审计需要形成 `NOVELTY_AND_REVIEWER_AUDIT.md`，记录重叠点、差异点和所需实验。

---

# 3. 核心研究假设

本项目不预先假设 JEPA 天然优于其他方法。

提出三个可证伪假设：

**H1：Predictive Identity Representation**

从轨迹历史预测当前身份潜在表征，相比直接判别式学习，能够改善跨时间、遮挡和域变化下的身份关联。

**H2：Predictive Learning for Domain Adaptation**

轨迹预测误差结合独立的关联可靠性证据，能够在无标签目标域构造比普通匹配置信度更有价值的在线学习信号。

**H3：Predictive Relation Retention**

保持旧域的轨迹-候选预测关系，相比普通特征 Replay 或参数蒸馏，更有利于降低持续域适应中的关联知识遗忘。

三个假设分别验证，不得用一项实验结果同时宣称三个都成立。

相机运动可能影响预测质量，但目前只是次要诊断假设，不作为核心研究成立的必要前提。

---

# 4. 主算法设计：Track-Level Predictive JEPA

主框架：

**Frozen GeneralTrack + Frozen V-JEPA 2.1 + Trainable Track-Level Predictor + Continual Adaptation**

初始阶段不微调完整 V-JEPA 模型。

保留冻结 V-JEPA encoder/predictor 的现有特征抽取能力，优先复用合法、因果、detector-row 对齐的现有缓存。

新增轻量轨迹预测器，第一版优先使用紧凑 MLP/GRU，后续根据证据决定是否需要更复杂的注意力结构。

## 4.1 轨迹级预测

对每个目标 i：

输入为当前时刻之前的 K 个历史观测，包括：

- 冻结视觉特征；
- bounding box 几何；
- 相对位移和速度；
- 帧间间隔；
- 检测置信度；
- 轨迹 age；
- 缺失帧数量；
- 历史观测有效 mask。

定义：

`q_i(t) = P_theta(H_i(<t))`

其中 `q_i(t)` 是依据历史预测的当前身份潜在表征。

当前检测特征为：

`z_j(t) = stopgrad(T(F(x_j(t))))`

其中：

- F 是冻结 V-JEPA 特征抽取器；
- T 是固定或独立控制的目标投影；
- 当前检测不得提前进入历史预测器 P_theta；
- 当前检测只能在预测完成后用于计算匹配损失或身份相似度。

预测器输出可以包括：

- identity latent；
- prediction uncertainty；
- 可选的轨迹可观测质量。

但这些辅助输出必须有独立训练信号，不能只添加网络头而没有有效监督。

首轮优先固定 target representation space，避免 Student/Teacher 同时漂移造成无法解释的改进或退化。

## 4.2 训练目标

第一版比较两类真正不同的训练目标。

A. 普通判别式关联学习：

直接学习历史轨迹与候选检测的匹配概率，训练 binary/listwise association classifier。

B. JEPA 式潜在预测：

根据历史预测当前目标的潜在表征，以 stop-gradient target 表征作为预测目标。

可以使用：

`L_pred = 1 - cosine(q_i(t), z_i(t))`

另加身份区分约束：

`L_rank = -log exp(sim(q_i,z_pos)/tau) / sum_j exp(sim(q_i,z_j)/tau)`

最终：

`L = lambda_pred * L_pred + lambda_rank * L_rank`

需要注意：

- L_pred 是潜在预测目标；
- L_rank 是 MOT 任务专用的身份区分目标；
- 两者贡献必须拆开报告；
- 加入 ranking 后不能把全部收益直接归因于 JEPA；
- 所有学习方法应尽可能保持相同预测器结构和训练预算；
- 防止 collapsed representation；
- 训练与验证中的正负候选来源必须清楚。

首先判断潜在预测目标是否提供额外价值，再优化其结构。

---

# 5. 不要只实现一个 JEPA 方案

这次必须真正探索至少三种不同的 JEPA-MOT 结合方式，而不是只对同一特征融合器修改阈值。

## Candidate A：Predictive Latent Association

使用历史轨迹预测当前 identity latent，利用预测 latent 与检测 latent 做关联。

优势：

- 简单；
- 参数轻；
- 容易复用现有 V-JEPA 缓存；
- 便于严格验证是否优于普通历史特征。

先跑这条。

## Candidate B：Predictive Association-Relation JEPA

从单纯预测目标特征进一步转向预测轨迹的候选关联响应。

输入仍然是当前候选出现前的轨迹历史。

Predictor 预测轨迹关联状态或用于候选评分的低维参数，而不是直接输出二分类结果。

随后将候选检测映射到这个预测的关联状态，形成同一轨迹下的候选排序。

重点让模型学习：

- 不同身份之间的区分；
- hard negatives；
- 候选集合改变时的稳定排序；
- 遮挡或长间隔后的身份恢复。

需要与同参数量的 MLP pair scorer、Transformer association head 做公平比较。

不能把它描述为首次预测跟踪模型；GOT-JEPA 已经有相邻思想。

我们要探索的是其针对 **MOT 多候选竞争和持续域适应** 的独立价值。

## Candidate C：Predictive Consistency-Guided Continual JEPA

如果 A/B 在基础关联上显示有用信号，再研究在线持续学习。

关键不是简单增加 Replay，而是利用预测关系判断：

1. 新域哪些观测值得学习？
2. 哪些高误差来自域变化？
3. 哪些高误差来自遮挡、匹配错误或不可靠历史？
4. 哪些旧域关联关系需要保护？
5. 更新是否正在破坏旧模型明确的身份排序？

候选机制：

- frozen teacher 提供目标特征；
- student 从过去历史进行预测；
- 只有通过独立一致性检查的伪关联才能成为在线正样本；
- 预测误差提供学习需求信号；
- 匹配一致性、候选 margin、运动约束和轨迹持久性提供可靠性证据；
- 使用小容量 replay 保存旧域具有代表性的轨迹历史与候选关系；
- 旧模型对候选的关联分布作为关系保持目标；
- 如果当前梯度会明显破坏旧域关系，可以探索约束更新方向或保守更新。

明确区分“需要学习”和“可以安全学习”。

预测误差大不自动等于新域信息，也可能是错误身份关联。

建议使用两种信号：

`adaptation_need = prediction_discrepancy`

`adaptation_reliability = independent_temporal_association_evidence`

只有同时满足适应需求与可靠性时，才允许较大的在线更新。

这样 JEPA 预测不仅用于当前匹配，还参与选择更新样本和保护旧域预测关系。

但这些机制必须与简单置信度门控、hard-example replay 和普通知识蒸馏比较。

若独立价值不成立，删除无效模块，不因命名需要而保留。

---

# 6. 数据集与严格协议

第一轮只使用：

1. UA-DETRAC
2. BDD100K

暂不开展 MOTSynth、VKITTI2 的完整训练。

## 6.1 UA-DETRAC

计划研究域：

`sunny -> cloudy -> rainy -> night`

先核对数据可用性、源模型、源标签和每个域的完整视频数量。

如果现有源域模型与特征缓存来自不同训练源，必须建立新的公平源模型和一致的 detector 输入，不允许直接把不可比结果并列。

## 6.2 BDD100K

计划研究域：

`clear -> overcast -> rainy -> night`

初步使用三个目标阶段。

后续验证另一种域顺序，例如：

`clear -> night -> rainy -> overcast`

不同顺序需要保持训练视频、评测集合、更新预算一致。

## 6.3 数据使用规则

对每个数据集建立明确的：

- source training split；
- target adaptation stream；
- development split；
- final held-out evaluation split。

按完整视频/序列划分，不能按帧随机打散相邻内容。

源域可用身份 GT 训练初始预测器。

无标签目标域只能使用图像、检测、因果历史和在线生成的伪标签。

GT 仅供离线评测和独立诊断使用，不得成为目标域模型输入、梯度来源、在线阈值优化依据。

开发集可用于模型选择，但不得将其指标当作最终独立测试证据。

最终 held-out test 在冻结方法和超参数后运行；禁止反复使用 test 指导下一轮设计。

每个评测阶段应从该阶段保存的模型快照独立运行，评测结束必须恢复快照，不能让测试序列改变后续训练模型。

每段独立视频重置跟踪 ID、轨迹状态和短期身份历史；持续学习的 Predictor 参数、许可的历史 replay 和必要的优化状态可以跨视频及域保留。

必须定义清楚在线推理、在线适应与离线评测的时间顺序。

如果已有 checkpoint 或缓存与新拆分重合，需要做 provenance audit，并重建有冲突的部分。

---

# 7. 实验 Wave 1：建立强基线并判断预测学习是否有价值

先完成以下方法：

**B0：GeneralTrack original**

原始 GeneralTrack。

**B1：GeneralTrack + Geometry-only**

与 JEPA 版本保持相同生命周期和几何规则，但无有效 JEPA 特征。

**B2：GeneralTrack + Frozen V-JEPA 2.1**

复现现有冻结 JEPA 关联方法。

**B3：Frozen V-JEPA + History Last/EMA**

直接使用历史最后一次或 EMA 特征关联，不训练预测器。

**B4：Frozen V-JEPA + Temporal MLP Association**

普通判别式关联学习。

**B5：Frozen V-JEPA + Temporal JEPA Predictor**

同容量历史预测器，使用潜在预测损失。

**B6：B5 + MOT Ranking**

预测损失加身份判别约束。

补充：

- Encoder-only 与 official predictor feature；
- trainable prediction vs equal-capacity feature projection；
- feature shuffle/random control；
- no-history control；
- geometry-only control。

优先在 UA-DETRAC 与 BDD100K 的 development 视频上完成真实训练和评测。

先使用固定种子 17 定位算法趋势。

之后对有价值候选使用种子 17、23、42 重复训练。

不得用一个随机种子的最佳结果代表最终效果。

第一轮重点观察：

- HOTA；
- AssA；
- IDF1；
- IDSW；
- 新纠正的关联错误；
- 新引入的关联错误；
- 训练目标是否塌缩；
- 是否出现高预测相似但错误身份的情况；
- 预测损失下降与身份关联提升是否一致。

如果 JEPA 损失降低但 HOTA 不变，说明目前预测任务可能没有学习到 MOT 所需要的判别信息。

如果训练集改善但跨域不改善，研究预测目标是否过拟合源域。

如果分类器与 JEPA 表现一致，检查目标潜在空间、hard negatives、时间历史、遮挡和目标预测对象，而不立即放弃 JEPA。

---

# 8. 实验 Wave 2：针对失败原因主动改进

第一轮不达目标时，不得只记录 NO-GO。

必须进行错误归因，至少检查：

### 8.1 预测目标是否错误

冻结 V-JEPA 预测的 token 可能包含背景和类别信息，却不足以区分不同身份。

尝试：

- target latent normalization；
- identity-discriminative projection；
- object-centered target；
- 跨时间稳定目标；
- relation-level predicted target；
- 同帧 hard-negative-aware target。

所有目标都必须能从合法观测或源域标注生成。

### 8.2 历史构造是否错误

检查当前裁剪代码的时空对齐问题。

分别比较：

- 现有固定坐标 causal crop；
- 单帧冻结视觉特征 + 正确轨迹历史；
- 每帧依据因果轨迹位置进行的 track-aligned crop；
- 可选的相机运动补偿。

不能使用当前未来检测框反向修正此前的预测历史。

摄像机运动只是待检验因素，不得事先断言它是失败根因。

### 8.3 模型是否没有利用历史

检查历史长度 1、2、4、8 的变化。

检查遮挡、短期消失、长间隔重新出现时的预测表现。

如果 JEPA 在连续可见帧没有收益，但在重关联场景有效，允许把模型收缩到困难关联或 reappearance 分支。

这比强制对所有关联使用 JEPA 更符合已有证据。

### 8.4 预测无法改善真实关联

开展 event-level counterfactual audit。

针对每一个 JEPA 改变的匹配事件记录：

- 原始 GeneralTrack 决策；
- JEPA 决策；
- 是否纠正错误；
- 是否反而引入错误；
- 对后续轨迹的影响；
- 当前预测相似度和候选 margin；
- 是否处于遮挡、漏检或场景变化；
- 事件发生时模型是否已更新。

离线可以利用 GT 判断纠错与反改错，但这些标签绝不能反馈到目标域在线训练器。

重点降低 harmful corrections，而不仅是增加 JEPA 介入次数。

### 8.5 必须形成下一轮研究假设

每次主要失败都写明：

`Observation -> Root Cause Hypothesis -> Discriminating Experiment -> Architectural Change -> Result -> Next Decision`

不要在缺乏证据时反复改变多个变量。

---

# 9. 实验 Wave 3：真实多域持续学习

只有在真实在线训练流程能够运行后，才正式开展这一步。

但如果基础预测器尚未取得稳定单域提升，也允许继续测试持续学习假设，因为预测目标的价值可能主要体现于连续适应，而不是冻结模型的即时关联性能。

需比较：

- Frozen Predictor；
- Sequential Fine-Tuning；
- Temporal MLP Online Adaptation；
- Temporal MLP + Replay；
- JEPA Online Adaptation；
- JEPA + Standard Replay；
- JEPA + Standard Distillation；
- JEPA + Predictive Relation Retention；
- 完整候选方法。

所有方法在同一数据流下使用一致或明确报告的：

- online update steps；
- 可训练参数；
- 训练样本；
- replay memory 容量；
- 训练预算；
- 推理及在线更新预算。

不允许让完整模型多使用大量 replay 或 GPU 计算，然后直接归因于 JEPA。

## 9.1 持续学习流程

每个域依次处理：

Step 0：保存源模型，评测所有域的初始性能。

Step 1：使用目标域 1 的无标签视频在线更新。

Step 2：保存模型，评测全部域。

Step 3：在不重置 Predictor 参数的情况下继续学习目标域 2。

Step 4：保存模型，评测全部域。

Step 5：继续学习目标域 3。

Step 6：保存最终模型，评测全部域。

必须生成每个 stage × 每个 domain 的：

- HOTA；
- AssA；
- IDF1；
- DetA；
- MOTA；
- IDSW；
- Average Forgetting；
- Backward Transfer；
- Forward Transfer；
- final seen-domain average。

遗忘统计必须明确参照历史最佳阶段或者该域学习完成阶段。

对于负迁移、记忆损失、域顺序敏感性也要报告。

## 9.2 历史预测关系保留

探索 memory 保存：

- 旧域轨迹历史；
- 相应的历史候选集合；
- 旧模型的预测 latent；
- 旧模型对候选的关联分布；
- 预测不确定性和可靠性。

可以设计：

`L_retain = KL(p_old(candidate | history) || p_new(candidate | history))`

但需要与普通 logit distillation、feature distillation、random replay、hard-example replay 比较。

不要把 KL distillation 本身当作创新。

研究重点是预测关系的选择、保留与更新是否对身份关联持续学习有特殊价值。

---

# 10. 实验 Wave 4：自动研究迭代，不允许过早结束

这一阶段是本任务与普通短实验脚本的主要区别。

如果前几轮效果不理想，继续根据证据探索结构不同的方法。

探索至少覆盖：

**方向 A：改变预测对象**

feature prediction、relation prediction、candidate-response prediction。

**方向 B：改变预测所使用的信息**

单帧特征、历史轨迹、时间 mask、运动与质量信息。

**方向 C：改变学习策略**

supervised source prediction、target pseudo-prediction、reliability-gated adaptation。

**方向 D：改变知识保持方式**

feature replay、predictive function distillation、candidate-relative relation retention。

至少完成三个在科学假设上不同的方案，并对值得进一步验证的方案进行有针对性的二次改进。

不是必须对每一种方法做无意义的巨大网格搜索。

每轮优先选择最可能区分两个研究假设的实验。

当新实验失败时：

1. 检查数据和协议；
2. 检查梯度、loss、激活和匹配事件；
3. 检查预测目标是否有效；
4. 检查简单基线是否已经解释提升；
5. 根据证据修改结构；
6. 重新训练；
7. 回到完整在线 MOT；
8. 更新科研结论和下一轮计划。

不要因为第一次或第二次未超过 baseline，就直接将整个 Goal 标记为失败。

也不要为了永不结束而无限随机试验。

完成多轮结构性探索后，如果仍不存在可信收益，应保留完整负结果和最优候选，客观说明为什么 JEPA 不能优于更简单方法。

---

# 11. 量化科研目标

将以下指标作为研究努力方向，而不是必须伪造达到的结果。

### 核心目标

在两个数据集的顺序域学习协议下，相较最强的同预算非 JEPA 持续学习对照，希望获得：

- 有实际意义的最终平均 HOTA 或 AssA 提升；
- 优先争取至少 +1.0 个百分点的最终平均 HOTA 增益；
- 更低的平均遗忘；
- 更好的新旧域性能平衡；
- 多个域/多个阶段出现一致趋势；
- 至少三个种子支持结果；
- 视频级配对统计不确定性报告；
- 负面域结果完整保留。

+1.0 HOTA 是目标，不是强制结果过滤条件。

如果 HOTA 增量较小，但 AssA、IDSW 和遗忘改善稳定且有统计支持，可以保留为有效研究信号，但不得夸大成全面优势。

如果方法仅在 UA-DETRAC 有效而 BDD100K 退化，需要明确揭示域依赖性，并继续诊断。

不允许通过改变不同域的测试比例人为提高宏平均。

### 在线性能目标

需要实际测量：

- adapter 参数量；
- 模型推理附加延迟；
- online update latency；
- p50/p95；
- GPU peak memory；
- tracker-only FPS；
- V-JEPA 特征提取 FPS；
- 完整图像到轨迹输出的端到端 FPS。

预缓存特征的 tracker FPS 不得冒充完整在线系统 FPS。

如果官方 V-JEPA 2.1 太慢，可以研究缓存加速、轻量特征蒸馏或稀疏调用，但这属于独立工程优化，必须单独报告与原模型的比较。

---

# 12. 统计分析和归因

最终必须有完整的研究审计：

1. 不同随机种子平均值与标准差。
2. 按视频分组的 paired bootstrap 或等价不确定性分析。
3. 完整域结果，不只报告宏平均。
4. Prediction loss 与 AssA 的关系。
5. Corrected vs Harmful Association 事件数量。
6. JEPA Predictive Loss 的独立增量。
7. Replay 和 Retention 的独立增量。
8. 相同参数量、数据量与训练预算下的比较。
9. 不同历史长度、遮挡间隔和运动难度的分组结果。
10. 检查全部用于指标汇报的配置是否和实际运行配置一致。

需要形成审稿人可验证的结论：

- 哪些收益来自冻结预训练视觉表征？
- 哪些来自 JEPA 潜在预测目标？
- 哪些来自 MOT 关联损失？
- 哪些来自 Replay？
- 哪些来自关系保持机制？
- 哪些只是几何规则和生命周期差异？
- 当前是否存在普通 MLP 可以等价替代的方法？

如果不能证明 JEPA 的独立价值，报告中必须如实说明。

---

# 13. 科研过程管理与 GitHub 交付

所有代码、实验记录和研究报告应持续推送 GitHub。

工作分支：

`research/predictive-jepa-cmot-20261013`

禁止未经授权强推 main。

建议建立目录：

`docs/JEPA_CMOT/`

至少包含：

- `MASTER_GOAL.md`：最终 Goal 和状态；
- `RESEARCH_PLAN.md`：研究阶段及方案；
- `NOVELTY_AND_REVIEWER_AUDIT.md`：文献和审稿问题；
- `EXPERIMENT_REGISTRY.csv`：全部实验索引；
- `CURRENT_RESULTS.md`：当前最佳结果与完整对照；
- `FAILURE_ANALYSIS.md`：失败证据与诊断；
- `CONTINUAL_MATRIX.md`：域顺序及遗忘矩阵；
- `NEXT_ACTION.md`：下一轮执行计划；
- `FINAL_RESEARCH_REPORT.md`：最终科研报告。

新算法代码应放在清晰、独立的模块中。

尽量不要把所有新实验逻辑继续堆进既有的巨大 tracker 文件里。

原始 GeneralTrack 模型和历史 JEPA 结果应保留。

每个实验至少记录：

- experiment ID；
- Git commit；
- dataset/domain/split；
- random seed；
- config；
- checkpoint SHA；
- 特征缓存 hash；
- 模型结构及训练目标；
- train/update steps；
- 评测协议；
- HOTA/AssA/IDF1；
- forgetting；
- FPS/latency；
- 结果目录；
- 状态；
- 下一步判断。

GitHub 只推送必要的代码、配置、指标、图表和研究报告。

大型数据集、预训练权重、轨迹缓存、临时中间文件留在本地，并在文档中记录可复现路径和 SHA。

每完成一个实质性阶段，立即：

1. 保存实验结果；
2. 更新研究状态；
3. 提交 Git commit；
4. 推送远端；
5. 验证远端提交存在；
6. 继续下一轮研究。

如果 GitHub 暂时不可访问，不能丢弃实验结果，先保留本地提交和待推送记录，恢复后再同步。

不要仅在整个任务结束时才提交一次。

---

# 14. 长运行任务与恢复机制

这是可能包含多次训练和完整数据集评测的长任务。

需要实现可恢复的实验执行流程。

要求：

- 已完成的相同配置实验不重复训练；
- 中断后从合法 checkpoint 恢复；
- 未完成的评测自动补齐；
- 长实验写入独立日志；
- 不因一个实验失败终止所有后续研究；
- CUDA OOM 等问题应诊断后调整执行策略；
- 不要频繁轮询 GPU 或持续输出大量无意义日志；
- 不覆盖有效 checkpoint；
- 不删除历史失败证据；
- 不擅自杀掉其他项目的进程。

执行过程中应持续维护当前最佳候选、已测试假设、被否定的假设以及未解决问题。

若某轮训练无法运行，应进行环境修复，或切换到不依赖该故障资源的有效诊断任务。

真正的硬件、数据或权限阻塞必须具体报告，不能凭空声称任务完成。

---

# 15. 科研阶段验收与停止条件

不要使用简单的“跑完三个实验没有提升，因此 NO-GO”规则。

采用以下研究推进原则。

**Phase A：工程闭环**

数据、缓存、模型、训练、跟踪和评测全部可执行。

**Phase B：预测机制验证**

至少完成普通历史学习与 JEPA 潜在预测的严格受控对照，并对负结果进行结构性诊断。

**Phase C：持续学习闭环**

至少完成两个数据集的顺序域适应及旧域回测。

**Phase D：方法创新探索**

至少比较不同的潜在预测对象、在线学习信号和关系保持机制，并证明或否定其独立贡献。

**Phase E：研究结论确认**

对有价值的模型进行多种子、完整视频、多域顺序、效率与错误归因验证。

只有当研究证据足以支持明确结论、必要实验完成，或存在无法克服的实际资源约束且已完成充分探索时，才结束本次执行。

科研成功不能以完成预设命令数量判断。

也不能以能否凑到目标分数判断。

**最终需要的是一个真正经过研究验证的 JEPA-MOT 方法，而不是一个漂亮但没有科学归因的实验表格。**

如果效果达到目标，状态标记为：

`GOAL_SUPPORTED_BY_EVIDENCE`

如果系统完整但算法结论仍不充分：

`RESEARCH_COMPLETE_EVIDENCE_INSUFFICIENT`

如果受环境因素阻塞：

`BLOCKED_WITH_RECOVERY_PLAN`

不能因为一两轮结果不好就将整个方向宣布失败。

---

# 16. 最终交付标准

最终报告必须用中文清楚回答：

1. 采用了哪个最终模型？
2. JEPA 在 MOT 中究竟预测什么？
3. 为什么采用这个预测对象？
4. 为什么不能简单使用 MLP、普通 Transformer 或历史特征平均来替代？
5. JEPA 的训练目标到底带来多少独立增益？
6. 在 UA-DETRAC 和 BDD100K 分别取得什么结果？
7. 两个数据集各个域的 HOTA、AssA、IDF1 分别是多少？
8. 完整顺序学习后平均遗忘降低多少？
9. Replay 和预测关系保持各自贡献多少？
10. 哪些场景效果不好，原因是什么？
11. 是否存在伪标签确认偏差？
12. 方法是否严格在线？
13. 计算成本、显存、FPS 和延迟分别是多少？
14. 相机运动是否被证明确实影响 JEPA？
15. 与 GOT-JEPA、FACT、TCEI 等相关工作的真正区别是什么？
16. 当前证据是否足以支撑一篇以持续学习为核心的 CVPR 论文？
17. 如果尚不充分，还缺少哪些关键证据？

必须给出所有主表、核心消融表、完整多域顺序学习矩阵及必要的诊断图。

最终附上：

- GitHub 分支；
- 最终 commit SHA；
- 研究报告；
- 关键训练命令；
- 关键评测命令；
- 模型配置；
- 最优 checkpoint 的本地路径及 hash；
- 全部实验索引。

---

# 17. 立即开始执行

不要只写一份新计划然后停止。

从以下顺序立即开始：

**第一步：** 审计当前仓库、GPU、NAS、checkpoint 和历史结果，完成真实环境恢复与基础 smoke。

**第二步：** 确定两个数据集的源域训练、目标域无标签适应及最终评测划分，并冻结评测协议。

**第三步：** 基于现有 GeneralTrack + V-JEPA 代码实现统一的 Track-level Predictor 训练和推理接口。

**第四步：** 训练同容量 Temporal MLP 判别模型、JEPA Predictive Latent 模型、JEPA + MOT Ranking 模型。

**第五步：** 在 UA-DETRAC 与 BDD100K 上进行受控在线跟踪评测，分析预测机制是否具有独立收益。

**第六步：** 如果效果不好，不要停止。根据模型输出、关联纠错/反改错、预测目标、历史输入、梯度和域变化诊断原因，设计第二轮、第三轮具有明确假设的模型改进。

**第七步：** 建立可恢复的无标签顺序域适应训练器，完成完整持续学习对照、历史域保持和跨域迁移实验。

**第八步：** 继续根据实验结果优化预测对象、学习信号和抗遗忘机制，进行多种子完整验证，并持续提交 GitHub。

**你的最高研究目标是：证明一个 MOT-specific 的 JEPA 预测学习机制能够在连续无标签域变化中改善身份关联，并比强非 JEPA 对照具有更好的新域适应与旧域保持平衡。**

请直接执行代码修改、训练、诊断、完整评测和 GitHub 推送。

不要在完成几个实验后仅以 NO-GO 结束。

不要为了达成 GOAL 而牺牲实验真实性。

用充分、可复现的科研证据决定最终结论。