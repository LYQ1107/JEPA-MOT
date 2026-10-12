# Predictive Identity JEPA：研究计划

状态：ACTIVE，先完成 Phase A 和数据协议审计。主研究仅 UA-DETRAC / BDD100K。

## 假设与可以否定它的实验

|假设|受控比较|需要的证据|
|---|---|---|
|H1：潜在预测提供身份收益|同一个残差历史 MLP，ranking-only / prediction-only / prediction+ranking|完整开发视频的 HOTA、AssA、IDF1 和纠错/反改错；loss 下降不等于支持 H1|
|H2：预测误差提供有效无标签学习信号|相同伪对、更新步数、网络；confidence-only / discrepancy+independent reliability|新域收益与伪对可靠性，排除只增加更新预算|
|H3：保留候选预测关系缓解遗忘|同容量 memory：random replay / feature distillation / candidate-relative retention|所有 stage×domain 的矩阵、遗忘、BWT/FWT、三个种子|

## 执行顺序

1. Phase A：原始 GeneralTrack、官方 V-JEPA encoder/predictor、NAS 读图、在线跟踪、TrackEval 的真实 GPU smoke；失败日志保留。
2. 数据审计：检查完整帧覆盖、源 checkpoint 训练视频与新划分的重合，建立按完整视频的 source train / source dev / adaptation / target dev / sealed test。
3. Candidate A：共享历史预测器和固定目标空间，分别训练普通判别关联、潜在预测、潜在预测+排序。首先 seed 17；不复用旧 alpha=0 的训练结果。
4. 结构性诊断：目标空间的身份区分性、历史信息增量、遮挡恢复、候选竞争、纠错与反改错；由证据选择第二轮变更。
5. Candidate B：预测关联状态/候选响应，并与同预算 pair scorer 比较。当前候选不能进入历史预测接口。
6. Candidate C：严格延迟伪对学习、独立可靠性证据、相同 memory/update budget 下的关系保持。
7. 完整连续域评测；视频结束只重置轨迹，不重置模型/许可的 optimizer 和 replay。评测快照只读，不更新训练模型。
8. 有价值模型进行 seeds 17/23/42、第二个域顺序和效率/配对统计验证；证据足够后形成最终报告。

## 不变约束

- 所有学习对照固定同一特征输入、同一个 detector stream、GeneralTrack、几何规则、阈值和生命周期。
- 历史输入只含 `<t` 的观测；当前候选只用于预测后的评分/损失。
- target GT 不进入在线训练、伪对筛选、阈值选择或 replay。
- 测试集不用于改模型；开发指标和 sealed 指标分开。
- 输出和 checkpoint 在 `runs/JEPA_CMOT/`，NAS 和旧结果只读。

## 当前工程发现

V100 不支持 BF16 autocast；实际使用 FP16，并在缓存元数据中记录。隔离安装 timm，不升级共享 Conda。

历史 UA-DETRAC 的 MVI_39501 在原始 XML/图片中有 540 帧，旧 prepared 只有 80 个抽样帧；正式研究必须核对/重建完整帧输入。

BDD 的文件名 `train` / `val` 不能作为划分证据：初步数量完全一致，正在检查视频集合重合。已有源 detector checkpoint 的 config 使用 `clear_all`，不能未经重合审计用于新 source held-out 结论。

## 第二类预测对象的可执行定义

关系预测沿用完全相同的历史网络、参数和输入，在候选出现后计算其响应分布。teacher target 为停止梯度的正身份当前 latent 与同帧竞争候选的相似分布，student 为仅由 H<t 生成 query 的相似分布。KL 关系预测与 KL+ranking 已加入训练入口；当前候选仍不进入 P 的前向。该定义有与普通软标签蒸馏重合的风险，必须在同容量、同预算下比较，不能凭命名称为创新。

1024-D encoder 路线的三个模型各有 662,400 个参数；1664-D official-predictor 路线各有 1,072,640 个参数。主对照在同一特征族内进行，跨族差异需要容量与算力审计。
