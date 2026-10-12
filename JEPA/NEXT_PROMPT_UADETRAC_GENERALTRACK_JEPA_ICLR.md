# UA-DETRAC：GeneralTrack + JEPA 单 Agent 实验执行提示词

你是本任务唯一的规划、实现、训练、评测、审计和反思 agent。不要创建、调用或委派任何子 agent、子线程或额外执行代理；所有代码修改、命令执行、实验设计、结果分析和下一轮计划都由你自己完成。

## 1. 总目标

以 UA-DETRAC 作为第一优先数据集，以 GeneralTrack 的原始结果作为唯一主要比较基线，探索能够显著提升多目标跟踪效果的 JEPA 接入方式。

核心目标不是只提升 IDF1，而是同时关注并优先提升：

- HOTA
- MOTA

同时记录 IDF1、DetA、AssA、LocA、IDSW/IDs、FP、FN、运行时间和显存占用。只有当相对 GeneralTrack 原始结果取得可重复、具有实际意义的整体提升时，才可以宣布阶段性成功；单个指标的小幅波动不能作为停止理由。

用户的研究目标是 ICLR，因此实验必须具备可复现性、严格的公平比较、明确的因果归因和完整的失败分析。当前先攻克 UA-DETRAC；只有在 UA-DETRAC 上得到可信结果后，才考虑扩展到其他数据集和域。

硬件约束：最多使用 4 张 GPU。允许大规模训练，但必须记录 GPU 数量、显存、训练时长、有效 batch size、随机种子和完整命令。

## 2. 工作目录和已有代码

先读取并遵守：

`/data1/LWR/vranlee/SERVER_ONLY/avis/JEPA/AGENTS.md`

主要代码位置需要自己核实，但当前已知的重点位置包括：

- GeneralTrack：`/data1/LWR/vranlee/SERVER_ONLY/avis/GeneralTrack-main`
- GeneralTrack 评测入口：`GeneralTrack-main/yolox/evaluators/mot_evaluator.py`
- GeneralTrack 跟踪器：`GeneralTrack-main/yolox/tracker/byte_tracker_bdd.py`
- GeneralTrack 关联代价：`GeneralTrack-main/yolox/tracker/matching_bdd.py`
- Point2InstanceRelation：`GeneralTrack-main/core/Point2InstanceRelation.py`
- JEPA/GOT 代码：`/data1/LWR/vranlee/SERVER_ONLY/avis/JEPA/GOT-main`

先检查当前 git 状态、已有未提交修改、已有 checkpoint、实验日志、UA-DETRAC 数据路径、检测结果缓存和历史结果。不要覆盖用户已有修改，不要删除失败实验，不要为了整理目录而复制大量数据或日志。

## 2A. 已核实的可复用 GeneralTrack 结果

已经检查过 `/data1/LWR/vranlee/SERVER_ONLY/avis/GeneralTrack-main`。它不只是源码，还包含一轮已经完成的运行结果；因此不需要为了得到第一版 UA-DETRAC 基线而从头重跑。

可直接复用的已有产物：

- 运行根目录：`/data1/LWR/vranlee/SERVER_ONLY/avis/GeneralTrack-main/outputs/generaltrack_thr02`
- 汇总表：`outputs/generaltrack_thr02/generaltrack_thr02_summary.csv`
- 运行日志：`outputs/generaltrack_thr02/logs/generaltrack_thr02_20260709_020841.log`
- 各域指标：`outputs/generaltrack_thr02/ua_detrac/{cloudy,night,rainy}/metrics.csv` 和 `metrics.json`
- 各视频轨迹：`outputs/generaltrack_thr02/ua_detrac/{cloudy,night,rainy}/track_results/`
- 关联 checkpoint：`/data1/LWR/vranlee/SERVER_ONLY/avis/GeneralTrack-main/checkpoints/generaltrack_bdd.pth`

该轮日志显示：使用 `/home/lwr/anaconda3/envs/ByteTrack/bin/python`、GPU `0 1`、`MAX_PARALLEL=2`，检测来源为 `source_detector_thr02`，准备数据目录为 `datasets/generaltrack_prepared_thr02`，运行时间为 2026-07-09 02:08:41 至 05:30:40。

目前已核实的 UA-DETRAC 内部基线如下：

| target domain | videos | frames | HOTA | MOTA | IDF1 | DetA | AssA | IDSW | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| night | 28 | 5015 | 34.5917 | 21.3725 | 46.0185 | 30.5086 | 39.7572 | 1697 | 8093 | 15657 |
| cloudy | 30 | 5060 | 35.2668 | 15.8500 | 48.6058 | 31.8860 | 39.5842 | 3820 | 18768 | 20416 |
| rainy | 19 | 4999 | 30.9364 | 19.0118 | 39.1974 | 25.0934 | 39.2198 | 2080 | 10072 | 37842 |

### 复用规则

1. 将这些轨迹和指标作为已有的 **内部 GeneralTrack `source_detector_thr02` 基线**，优先复用，不重复进行完整 baseline 推理。
2. 在正式 JEPA 实验前，只做轻量审计：检查 detector 文件、prepared metadata、输入检查表、评测脚本和关键配置是否仍然存在；检查指标 CSV/JSON 是否一致；确认轨迹视频数和帧数没有缺失。
3. 必须做一次 `lambda=0` 或完全关闭 JEPA 的 no-op 对照，至少在 UA-DETRAC 三个 target domain 上确认输出指标和轨迹与已有基线一致；若不一致，先找出原因再进入 JEPA 实验。
4. 所有 JEPA 变体固定使用同一份 `source_detector_thr02` 检测结果、同一 prepared input、同一评测器和同一类别/坐标协议。不要因为 JEPA 结果不理想而换 detector。
5. `thr02` 是检测来源/准备数据变体的名称，不能未经核实就称为论文中未经限定的“GeneralTrack 原始结果”。报告中同时写清楚：代码基线是 GeneralTrack，当前可复用的数值基线是 `GeneralTrack + source_detector_thr02`。
6. 如果论文原始结果使用不同 detector、不同阈值或不同评测协议，另列为外部参考，不得和当前公平比较的主表混在一起。用户要求的提升默认以这轮已核实、可复现的 `source_detector_thr02` 数值为主比较对象，并在报告中解释这一点。

除非上述审计失败，否则不要重跑已经完成的 UA-DETRAC GeneralTrack 主干推理；把时间和四张卡优先用于 JEPA 特征、adapter 和关联策略实验。

## 3. 先完成基线审计，再写新代码

必须先确认 GeneralTrack 的真实执行链路，不能只依据论文名称或文件名推断：

1. 确认 GeneralTrack 是否使用 ByteTrack 框架、哪些分支仍是标准 ByteTrack、哪些关联分支被 Point2InstanceRelation 替换。
2. 确认 detector、检测结果缓存、检测阈值、track threshold、匹配阈值、生命周期管理、lost/removed 逻辑和评测脚本。
3. 确认 UA-DETRAC 的数据划分、帧顺序、图像颜色格式、图像尺寸、坐标格式和评测协议。
4. 找到 GeneralTrack 在 UA-DETRAC 上的原始结果或原始复现实验；若没有现成结果，使用 release 代码和固定检测结果重新跑出基线。
5. 记录基线的 commit、checkpoint、检测结果来源、配置文件、命令、随机性、运行时间和输出文件哈希。

基线表至少包括：HOTA、MOTA、IDF1、DetA、AssA、LocA、IDSW/IDs、FP、FN、GT 数量、帧数和运行时间。若历史结果与重新复现结果不一致，必须分别保留并解释差异，不能擅自选择更好的一组。

公平比较要求：GeneralTrack 和所有 JEPA 变体必须使用完全相同的检测结果流、数据划分、评测脚本、类别过滤和坐标处理。除非单独定义 detector ablation，否则不能同时修改 detector 来制造提升。

## 4. 严格数据泄漏审计

在任何正式结果之前，建立一份数据流审计：

- UA-DETRAC 测试集 GT 只能用于最终评测，不能用于阈值、门控、checkpoint、训练轮数、模型选择或失败样本挑选。
- JEPA temporal adapter 如果训练，必须使用明确声明的 source/train split；需要划分 validation 来选超参数，不能使用 test GT。
- 跟踪时每条轨迹的 JEPA 预测只能使用当前帧之前已经接受的观测，不能读未来帧、未来检测或当前帧 GT。
- 检测结果缓存、图像索引、轨迹历史和特征缓存必须一一对齐；记录 frame id、detection row id 和图像路径。
- 不能把当前候选检测先写入历史，再用该历史预测当前候选。
- 不得使用 oracle association、GT track id、GT visibility 或任何测试标注衍生字段。

如果发现任何实验使用了测试 GT 调参，立即标记该实验为 invalid，不得把它放入主结果表，只保留作为失败/泄漏证据。

## 5. GitHub 和新方法检索要求

需要自己检索 GitHub 和对应论文，优先寻找 2025、2026 年公开且与以下方向直接相关的方法：

- JEPA、V-JEPA、视频表征预测或时序 latent prediction
- 多目标跟踪中的遮挡恢复、re-identification、关联代价学习
- ByteTrack/GeneralTrack 类高低分检测关联
- 利用时序预测改进 tracklet association、lost-track reactivation 或 ambiguity resolution

如果 2025/2026 没有可复现的直接相关实现，再按相关性扩大到更早年份。不得捏造论文、GitHub 仓库、实验结果、代码接口或发布日期。

每个候选方法都要记录：论文标题、作者、年份、论文链接、GitHub 链接、commit 或 release、许可证、可运行入口、依赖、是否提供 checkpoint、和它对本任务的实际可迁移性。没有可验证代码的工作可以作为设计参考，但不能伪装成已复现方法，也不能直接作为主实验结果。

优先复用真实仓库的代码或明确借鉴其结构；如果由于依赖、数据或许可无法复现，要记录具体原因，并基于已有本地代码设计可审计的替代实验。

## 6. JEPA 接入策略

不要一开始替换 GeneralTrack 的 detector 或 Point2InstanceRelation。先保留 GeneralTrack 的检测器、生命周期和主要关联机制，把 JEPA 作为可插拔的时序关联信号。

按以下优先级设计实验：

### A. lost-track reactivation

优先检查 GeneralTrack 中 lost track 与剩余高分检测之间仍使用 IoU 的分支。将 JEPA 的历史表征预测作为短时重现的辅助关联信号，尤其针对短时间遮挡、车辆外观变化和运动不连续。

### B. 高分关联中的歧义门控

在已有 Point2InstanceRelation 关联代价之后、最终线性分配之前，仅对关联 margin 小、短时遮挡、候选冲突或质量不确定的样本启用 JEPA。容易匹配的样本保持原逻辑，避免 JEPA 污染已经正确的关联。

### C. 其他经过证据支持的接入方式

如果 A、B 没有提升，继续探索有明确动机的方案，例如：

- JEPA 作为候选 pair 的 residual association score
- JEPA temporal prototype 与 GeneralTrack relation score 的校准融合
- 只在 reappearance 场景使用 uncertainty-aware gate
- 在 source-only 数据上训练轻量 temporal adapter，再冻结 GeneralTrack 主体
- 真实新论文中验证过的其他时序预测或遮挡恢复结构

每次只改变一个主要科学因素，保证能够解释提升来自哪里。不能同时改 tracker lifecycle、detector、多个阈值和 JEPA 网络后再声称 JEPA 有效。

## 7. JEPA 实现约束

当前 JEPA/GOT 代码和 checkpoint 必须先审计后复用。特别检查：

- checkpoint 版本和输入输出维度
- BGR/RGB 转换
- 原图与 resize 后图像的坐标缩放
- crop、ROI、padding 和归一化方式
- 特征缓存是否与 UA-DETRAC 的 frame/detection row 精确对应
- 现有脚本是否硬编码 MOT17 的 `det.txt` 或其他数据集路径

优先使用离线特征缓存，避免 GeneralTrack 和 JEPA 两套环境同时加载大模型，并提高实验重复性。缓存必须带有元数据：图像根目录、检测文件哈希、checkpoint 哈希、crop 规则、颜色格式、坐标规则、特征维度和生成命令。

JEPA predictor 或 temporal adapter 必须是因果的：每条轨迹根据过去若干个已接受观测预测一个 candidate-independent temporal prototype；当前候选只能参与最终相似度计算，不能参与 prototype 生成。

候选 detection 的 JEPA 特征应按帧预先计算，每个 detection 只计算一次，不能在每个 track-detection pair 中重复提取。

建议先实现以下可解释的分数融合：

```text
S_base = 1 - C_GeneralTrack
S_jepa = (cosine(candidate_feature, temporal_prototype) + 1) / 2
S_final = S_base + lambda * gate * (S_jepa - S_base)
C_final = 1 - S_final
```

其中 `gate` 必须由过去信息和当前关联不确定性构成，例如短 gap、轨迹质量、预测 uncertainty、匹配 margin 或候选冲突；不得使用测试 GT。必须记录 gate 命中率、被 JEPA 改变的匹配数、被拒绝的匹配数和各类样本上的效果。

## 8. 必须完成的实验阶梯

至少按以下顺序执行：

1. **Exact GeneralTrack baseline**：UA-DETRAC 原始结果，固定 detector rows。
2. **Frozen JEPA control**：只加入冻结 JEPA 特征，不训练 temporal adapter，用于确认接入和缓存没有问题。
3. **Causal JEPA reactivation**：只在 lost-track reactivation 分支启用 causal temporal adapter。
4. **Ambiguity-gated fusion**：在高分关联中只对歧义样本启用 JEPA。
5. **Best variant tuning**：只用 source train/validation 选择少量预先声明的 `lambda`、gap、uncertainty 和 margin 参数。
6. **Ablation**：至少比较无 JEPA、无 gate、只 reactivation、只 high-score、完整方案，并报告关联改变率。
7. **Stability**：最佳方案至少使用多个随机种子复测；若方法无训练随机性，也要重复完整推理并固定环境。

每轮实验完成后都要检查：

- HOTA、MOTA 是否同时改善
- IDF1、AssA 是否改善或退化
- DetA 是否保持不变；若 detector 固定但 DetA 改变，必须解释
- IDSW、FP、FN 的变化来源
- JEPA 实际触发率和实际改变的 assignment 数量
- 提升是否集中在少数视频或少数阈值
- 是否存在 NaN、空轨迹、帧错位、坐标错误或异常 runtime

如果某一方案失败，不要据此结束任务。根据失败类型设计下一轮：

- gate 几乎不触发：检查场景判定和分数尺度；
- gate 大量触发但指标下降：收紧 gate 或校准融合；
- raw JEPA 有效但 adapter 无效：检查训练目标、时间窗口和特征泄漏；
- lost 分支有效但高分分支无效：保留局部接入，不强行全局融合；
- 所有方案无效：重新检索 2025/2026 方法，考虑更换 JEPA 表征、训练目标或接入点。

不能把一次失败实验包装成结论，也不能因为训练时间长就提前宣布成功。

## 9. 成功标准和停止规则

在基线复现完成后，先在实验日志和报告中写明可审计的成功标准，再进行主方案选择。用户尚未给出精确的百分点评估门槛，因此不得在看到结果后临时改变标准或把极小波动称为“大幅提升”。至少要求：

- 相对 GeneralTrack 原始 UA-DETRAC 结果，HOTA 和 MOTA 都有明确的绝对提升；
- 提升在重复运行中保持；
- 不是通过测试集调参获得；
- IDF1、AssA 或 IDSW 没有出现无法接受的反向退化；
- 能解释提升对应的场景和机制。

如果结果没有达到“大幅提升”，继续修改和实验，不要输出“最终完成”的措辞。若 UA-DETRAC 达到标准，再扩展到其他数据集验证泛化；在扩展之前不能声称方法具有跨域有效性。

## 10. ICLR 级别报告要求

最终整理为 Markdown 文件，建议路径：

`/data1/LWR/vranlee/SERVER_ONLY/avis/JEPA/reports/UA_DETRAC_GeneralTrack_JEPA_ICLR_audit.md`

报告至少包含：

1. 研究问题和成功标准；
2. GeneralTrack 精确基线和执行链路；
3. UA-DETRAC 数据划分、检测流和评测协议；
4. GitHub/论文检索表及可验证来源；
5. 每个 JEPA 变体的动机、代码位置和配置；
6. 训练资源、时间、GPU、checkpoint 和随机种子；
7. 所有实验的完整指标表，不只报告最好结果；
8. HOTA/MOTA/IDF1/DetA/AssA/LocA/IDSW/FP/FN 分析；
9. 数据泄漏审计和公平性审计；
10. 失败实验、失败原因和下一轮决策依据；
11. 最佳方案的消融、重复性和运行代价；
12. 当前结论、仍未解决的问题和扩展到其他数据集的计划；
13. 每个关键结果对应的日志、配置、checkpoint 和输出文件路径。

报告必须区分“已复现事实”“本地实验结果”“来自论文的结果”“推测/待验证假设”，不得把未验证内容写成事实。

## 11. 最终交付格式

完成当前阶段后，聊天回复只提供最终 Markdown 报告的可点击文件链接，不要把完整报告正文直接粘贴到聊天中。若实验尚未达到成功标准，不得伪装成完成；继续执行下一轮，或在报告中明确记录真实阻塞原因、已有证据和下一步实验计划。

开始执行时，先给出一个简短的里程碑计划，然后立即进行 GeneralTrack-UA-DETRAC 基线审计和复现；不要先大规模训练未经验证的 JEPA 方案。
