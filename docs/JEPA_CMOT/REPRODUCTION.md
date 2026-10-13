# 关键复现命令与源模型索引

状态：进行中的复现入口，未指定最终方法/最优checkpoint。用户已要求限制多种子耗时，目前优先seed17；本文命令不会自动启动实验。

从 `/data1/liuyeqiang/JEPA_MOT_resume_20261012` 执行，使用 `./run_python.sh` 的私有运行环境。大权重、detector输入、特征和结果在本地，RGB数据仍读只读NAS `/data1/liuyeqiang/nas/A100`。命令中的 `GPU_ID` 替换为经核对可用的GPU编号，不改共享环境。

## 输入、版本与实际权重

冻结划分为 `docs/JEPA_CMOT/VIDEO_SPLITS.json`，native协议为 `runs/JEPA_CMOT/native_protocol_v3/manifest.json`；detector/feature输入在 `runs/JEPA_CMOT/formal_detection_streams_v1/DATASET`。每个split保留完整native视频，target训练流不含GT；离线评分独立读取标签。BDD评估六类，非官方八类benchmark；author GeneralTrack BDD权重具有广泛BDD监督先验。

[六个主源checkpoint索引](PRIMARY_SOURCE_CHECKPOINT_INDEX.json)记录dense-current seed17 MLP的CE/pure/joint配置、实际绝对路径、权重/缓存SHA、fit/validation视频、预算和训练代码SHA。权重和缓存hash均已现场重新计算。这些是源初始化，不能当作最终适应模型；各持续序列 `snapshots.json` 指向source/domain边界快照，最终选型后还须核验所选适应权重。

下面的新输出目录命令复现关键配置和预算，未声称重新训练已完成。精确历史复现还需匹配metrics记录的training_code_sha256、cache SHA和配置；记录的git_commit不自动证明当时工作树所有代码均已提交。当前版本结果与原训练应区分，fingerprint不一致时不能绕过恢复检查。

## Source训练与校准

三目标共用history4/hidden128、seed17、2,000×64、gaps1/2/4/8、AdamW lr1e-3/wd1e-4/clip1；当前候选不进入历史预测器。source验证按独立视频，校准仅两参数/200步。

BDD源clear、1280×720：

```bash
for method in discriminative predictive predictive_ranking; do
  CUDA_VISIBLE_DEVICES=GPU_ID ./run_python.sh -m jepa_cmot.train_source \
    --cache runs/JEPA_CMOT/source_identity_cache_v1/bdd100k/features_dense_current.npz \
    --out runs/JEPA_CMOT/reproduction_source/bdd100k/seed17/$method \
    --method "$method" --source-domain clear --seed 17 \
    --history 4 --hidden 128 --architecture history --steps 2000 --batch 64 \
    --prediction-gaps 1 2 4 8 --width 1280 --height 720
done
./run_python.sh -m jepa_cmot.calibrate_source \
  --cache runs/JEPA_CMOT/source_identity_cache_v1/bdd100k/features_dense_current.npz \
  --models runs/JEPA_CMOT/reproduction_source/bdd100k/seed17 \
  --out runs/JEPA_CMOT/reproduction_calibration/bdd100k/seed17 --dataset bdd100k
```

UA同预算，将dataset替换为ua_detrac、source-domain为sunny、width/height为960/540，使用独立输出目录。UA原正式head在 `source_history_v2/ua_dense_current/seed17`，不复用已判无效的旧源checkpoint。

## 完整冻结开发评测

```bash
CUDA_VISIBLE_DEVICES=GPU_ID ./run_python.sh -m jepa_cmot.development_controls \
  --dataset bdd100k --manifest runs/JEPA_CMOT/native_protocol_v3/manifest.json \
  --detections runs/JEPA_CMOT/formal_detection_streams_v1/bdd100k \
  --controls configs/JEPA_CMOT/bdd100k_frozen.json \
  --feature-producer-pid 0 --out runs/JEPA_CMOT/reproduction_development/bdd100k_seed17
```

新训练模型必须在新controls文件同时替换checkpoint与匹配的source calibration；上述原配置指向本次已有权重。UA使用对应UA配置/dataset，完整17视频/25,460 native帧；BDD19视频/3,844帧。其它强对照采用登记的Transformer/pair/prototype配置；official/crop读数须指定对应feature文件与源模型。

## 无标签顺序适应与独立回测

```bash
CUDA_VISIBLE_DEVICES=GPU_ID ./run_python.sh -m jepa_cmot.continual_controls \
  --dataset bdd100k --manifest runs/JEPA_CMOT/native_protocol_v3/manifest.json \
  --detections runs/JEPA_CMOT/formal_detection_streams_v1/bdd100k \
  --feature-producer-pid 0 --controls configs/JEPA_CMOT/bdd100k_continual.json \
  --compact-intermediate --compact-native \
  --out runs/JEPA_CMOT/reproduction_continual/bdd100k_forward_seed17
./run_python.sh -m jepa_cmot.continual_analysis \
  --complete runs/JEPA_CMOT/reproduction_continual/bdd100k_forward_seed17/complete.json \
  --out runs/JEPA_CMOT/reproduction_continual/bdd100k_forward_seed17/analysis.json
```

BDD正序clear→overcast→rainy→night、逆序clear→night→rainy→overcast；UA正序sunny→cloudy→rainy→night、逆序sunny→night→rainy→cloudy。逆序同时使用新输出目录与 `--reverse`。轨迹物理状态逐视频reset，θ/optimizer/replay/recent/RNG继续；各stage独立冻结回测，评测不反馈训练。compact只改变存储保留方式。完整marker对应所有native视频后才可产生最终AF；partial不能替代最终矩阵。

纯预测使用 `bdd100k_pure_predictive_continual.json` / `ua_detrac_pure_predictive_continual_seed17.json`，逆序seed17使用 `bdd100k_reverse_seed17_with_pure_predictive.json` / `ua_detrac_reverse_seed17_with_pure_predictive.json`。H2/C2/RKD用各登记的matched-teacher配置，并传对应 `source_calibration_v1/bdd_dense_current/seed17/frozen_last.json` 或 `ua_dense_current` 路径作 `--teacher-calibration`。供给对齐只在相应固定教师family内成立，不假定不同自教师算法伪对相同。

## 单域指标复算

```bash
./run_python.sh -m jepa_cmot.evaluate_stream \
  --manifest runs/JEPA_CMOT/native_protocol_v3/manifest.json \
  --tracks runs/JEPA_CMOT/continual_v1/bdd100k_forward_seed17/evaluation/stage3/discriminative_replay0_none \
  --dataset bdd100k --domain rainy --split development \
  --out runs/JEPA_CMOT/reproduction_metrics/bdd100k_CE_FT_stage3_rainy.json
```

这个独立评分入口读取GT。UA缺失XML帧记录为unknown/unscored，不改为空GT。六指标原始计数在完整结果JSON，汇总按域等权；bootstrap按域内配对视频重采样，不能当作跨seed区间。stage0可复用核验输入/权重/校准/hash一致的完整已有评测，不能据此推断后续stage性能，见 [实际复用记录](SINGLE_SEED_SOURCE_EVALUATION_REUSE.json)。

全部其它结构/读数与负结果入口：[实验索引](EXPERIMENT_REGISTRY.csv)、[当前结果](CURRENT_RESULTS.md)、[中文报告草稿](FINAL_RESEARCH_REPORT.md)。当前单种子优先与多种子暂停安排见 [用户范围调整](SINGLE_SEED_PRIORITY_USER_UPDATE.json)；旧多种子配置作为历史记录保留，不因此自动恢复任务。
