# MOT 接入 GOT-JEPA：最小机制实验

`mot_jepa_synthetic.py` 是一个不依赖 PyTorch、torchvision、scipy、timm 或
GOT checkpoint 的小型实验。它的目的只是验证“JEPA 放在 MOT 的轨迹关联层”这
个想法能否端到端运行，**不是 MOTChallenge/BDD/VisDrone 的正式 benchmark**。
因此脚本可在系统 Python，也可在 `/home/lwr/anaconda3/envs/AVI` 的旧版
PyTorch 环境中直接运行；脚本本身只需要 NumPy。

## 与 GOT-JEPA 的对应关系

GOT-JEPA 的关键设计是：给定相同历史，teacher 从干净当前帧产生 pseudo
tracking model，student 从当前帧的遮挡/扰动版本预测同一个 model；OccuSolver
再用 object prior 细化 visibility。仓库中的对应调用链是：

```text
Head_JEPA_context (masked context)
       -> JEPA_predictor_cls / JEPA_predictor_breg
       -> classifier / bb-regressor
CoTracker visibility -> 是否写入 tracking memory
```

在 MOT 中建议把目标换成“每条轨迹的 association model/token”：

```text
track memory (过去 K 帧 embedding、位置、速度、mask)
       -> context encoder + predictor (student)
       -> 预测当前 track latent / association prototype
       -> 与 detector embeddings + motion 做 Hungarian 匹配
clean/high-confidence crop -> target encoder (teacher, 训练时 stop-gradient)
visibility/quality head -> 低置信度时冻结 appearance memory
```

注意：当前仓库的 `MultiObjectWrapper` 只是为每个 object 启动独立的 ToMP
SOT（`multiobj_mode='parallel'`），本身没有 detector-to-track 的候选集合、
生命周期管理或 Hungarian 关联。因此不能只打开这个 wrapper 就得到 MOT；实际
MOT 需要新增一个共享 encoder 的 tracking-by-detection loop，再把下面的 token
接入该 loop。

实际项目中可以先只改 association，不动 DINO/ToMP backbone：

1. 在 `pytracking/pytracking/evaluation/multi_object_wrapper.py` 或现有 MOT
   wrapper 中，为每条 track 保存 `K` 个 head feature、bbox delta 和 visibility
   mask。
2. 对每个 detector crop 生成 teacher token；把 crop mask、邻近目标 copy-paste、
   blur/dropout 作为 student 输入。
3. 用 smooth-L1 invariance loss 预测 teacher token，并逐步加入 VICReg variance/
   covariance loss；只在高 visibility 时更新 raw prototype。
4. 匹配代价使用
   `motion + cosine(det_embedding, predicted_track_token)`，并在 occlusion
   时提高 predicted token 权重、暂停污染性更新。
5. 最后再把预测 token 接到 ToMP 的分类/回归 filter（对应
   `JEPA_predictor_cls/breg`），避免一开始同时重训 backbone 和 association。

## 实验内容

脚本生成有 crossing、随机/几何 occlusion、低质量 contaminated detections 和
false positives 的合成序列。训练序列与评估序列使用不相交的 seed；每个 seed
下列五个方法读取同一份 detector list（JSON 中记录
`detections_sha256_16`）：

* `baseline`：raw appearance EMA + motion。
* `gated_ema`：不使用 predictor，但采用与 JEPA 分支相同的低分观测冻结门，
  用来拆分“visibility gate”本身与 JEPA predictor 的贡献。
* `jepa_inv`：context-to-clean-target 线性 predictor，只用 invariance/ridge
  目标。
* `jepa_cov`：相同容量，额外加入 off-diagonal covariance penalty（VICReg 风格）。
* `no_teacher`：相同 predictor 容量，但把 teacher target 行随机打乱，作为无效
  teacher control。

输出指标包括 global IDF1 proxy（**仅在 detector-observed GT 集合上，且不做
IoU/frame matching**）、IDSW、identity consistency（JSON 中同时保留旧别名
`association_accuracy`）、detector recall、occlusion/post-occlusion identity
consistency 和 `HOTA_proxy`。最后一项只是
`sqrt(det_recall * association_accuracy)` 的透明诊断，**不是官方 HOTA**。

## 运行

先做最小 smoke：

```bash
python experiments/mot_jepa_synthetic.py \
  --train-sequences 2 --eval-sequences 2 --frames 24 --objects 4 \
  --seeds 0 --cov-iterations 10 \
  --output experiments/results/mot_jepa_smoke.json
```

四方法 preliminary（保留的基础结果；3 个 seed、mild+hard，不按结果挑 seed）：

```bash
python experiments/mot_jepa_synthetic.py \
  --train-sequences 12 --eval-sequences 12 --frames 72 --objects 6 \
  --embedding-dim 12 --history-len 4 --scenarios mild,hard \
  --methods baseline,jepa_inv,jepa_cov,no_teacher --seeds 0,1,2 \
  --output experiments/results/mot_jepa_preliminary.json
```

同时生成同名 `.csv`。若要在 AVI 环境复核，只需把第一行换成：

```bash
/home/lwr/anaconda3/envs/AVI/bin/python experiments/mot_jepa_synthetic.py ...
```

归因时应使用包含 `gated_ema` 的五方法 control（结果已保存在
`experiments/results/mot_jepa_gated_control.json/.csv`）：

```bash
python experiments/mot_jepa_synthetic.py \
  --train-sequences 12 --eval-sequences 12 --frames 72 --objects 6 \
  --embedding-dim 12 --history-len 4 --scenarios mild,hard \
  --methods baseline,gated_ema,jepa_inv,jepa_cov,no_teacher --seeds 0,1,2 \
  --output experiments/results/mot_jepa_gated_control.json
```

## 如何解读

该实验只回答“预测型轨迹 token + visibility gate 是否值得接着做真实 MOT
实验”。`gated_ema` 尤其重要：若 JEPA 与它相近，增益主要来自 visibility gate，
不能归因于 predictor。这里的 toy student 只根据历史 context 预测 synthetic
clean latent，并没有完整复现论文中“clean current crop teacher / corrupted
current crop student”的神经网络结构；真实实验必须补上这一对分支。该实验不能
证明真实数据上的 IDF1/HOTA 提升：合成 embedding、检测误差、
遮挡分布和 tracker 超参数都与真实场景不同。若 `jepa_inv`/`jepa_cov` 在多个
seed、相同 detection digest 下稳定降低 IDSW 并提高遮挡恢复，再进入真实数据；
若只在某个条件有增益，应优先检查 predictor 的 target 定义、低质量 memory
mask 和 motion/appearance 权重，而不是直接扩大模型。
