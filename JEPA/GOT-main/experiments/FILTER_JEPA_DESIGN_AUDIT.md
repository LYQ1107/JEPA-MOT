# Closed-form ridge-filter JEPA：设计审计清单与伪代码

本文是将 GOT-JEPA 的 teacher/student 思路接到 MOT association filter
时的**设计审计**，不是实验结果，也不授权把尚未通过清单的实现接入线上
tracker。目标是把 teacher 定义成一个可复现、数值稳定、严格因果的 tracking
filter，而不是把一个从头训练的 clean-embedding MLP 误称为 GOT-JEPA teacher。

## 0. 必须先固定的语义

对轨迹 (i) 和帧 (t)，令

* (H_{i,<t}) 是线上在帧 (t) 到来前已经保存的观测（检测框、外观特征、
  时间戳、运动状态和 missing mask）；它只能包含帧号小于 (t) 的信息。
* (c_t) 是帧 (t) 的一个候选检测，(g_t) 是可公开的上下文（例如相机/序列
  元数据；不能偷偷编码 GT identity）。
* (phi(H,c)) 是 association/filter 特征，(y) 是匹配标签或 bbox/motion
  回归目标。
* (omega_t) 是 teacher 生成的**tracking filter/model**（至少拆成
  (omega^{cls}_t,omega^{box}_t)；也可以是轨迹 prototype），不是 clean crop
  的原始 embedding，也不是 GT identity label。

论文式训练目标应写成：

```text
teacher (train only):  T(clean current c_t, H_<t, labels) -> stop-grad omega_t
student:               S(corrupt current c_t, H_<t, g_t) -> omega'_t
loss:                  D(omega'_t, stopgrad(omega_t)) + optional supervised terms
online at frame t:     stored omega_{t-1} scores detector candidates c_t
```

这里有一个很重要的时间约束：**由 (t) 的 current pair 得到的
(omega_t) 只能在 (t+1) 或更晚使用**。如果同一个 (c_t) 既用于拟合/生成
(omega_t)，又被 (omega_t) 排名，就算输入做了 mask/noise，也属于
current-target autoencoding，不能称为 causal online association。

训练 teacher 可以使用 clean GT crop、同帧 GT positives 和 hard negatives，
但只限于预先声明的 train split。teacher 的质量必须在 train-held-out 的
(t+1) 候选上评测，不能读取 eval labels 来做 gate、调阈值或选择模型。
student 在 eval 时可以读取 detector 的 (c_t)，但只用已保存的
(omega_{t-1}) 给 (t)（即下一时刻）候选排序；这仍要标记为 causal
student eval，不能把 teacher/oracle-current/history 结果叫作 online。

## 1. 数据与泄漏清单（全部必须 PASS）

### 1.1 Split、时间和 ID

- [ ] 在运行前固定 train/teacher-fit、train-heldout、eval 的**序列、帧区间和
  identity 区间**；记录 split 文件 hash。不得随机把同一 identity 的未来帧泄漏
  到 fit 或质量 gate。
- [ ] 每个样本保存 `frame_t`、`history_max_frame`，并断言
  `history_max_frame < frame_t`；空 history、断帧、重复 frame 要显式计数。
- [ ] GT 只用于离线样本配对/标签、train-only teacher 拟合和 train-heldout
  质量评估。GT `gid`、未来框、visibility、遮挡标签不得进入 `H_<t>`、候选
  geometry、quality target、online state 或阈值选择。
- [ ] detector-to-GT 匹配必须是一帧内一对一（例如预注册 IoU 门限的 Hungarian）；
  记录 duplicate assignment、unmatched、IoU 分布和每帧候选数。
- [ ] 同一视频的评估必须按时间顺序运行；禁止用整段 GT 预先组装每条轨迹的
  history。`gid` 只能在离线构造训练标签时出现。

### 1.2 Current crop/候选泄漏

- [ ] teacher 的 clean current crop/同帧正样本与 hard negatives 只出现在
  train-fit（以及独立 train-heldout 的评估输入）；不得用 eval GT crop 拟合
  teacher 或制作 pseudo target。
- [ ] student 的 corruption 只作用于当前观测或显式训练增强；target
  (omega_t) 不得由被 corruption 的内容生成，且须 `stop_gradient`。
- [ ] 生成 (omega_t) 的样本集合必须和被评分候选集合 leave-current-out，或
  明确采用 (t\rightarrow t+1) lag。特别检查正候选不会同时出现在 filter
  拟合矩阵和同帧 Recall@1 评测矩阵中。
- [ ] candidate list 置换、加入无关候选、删除当前正候选的审计结果要写入
  artifact。若 predictor 根本不读取候选集合，置换不变只说明代码结构，而不是
  学到的 set invariance。
- [ ] candidate bbox/score/feature 不得通过 cache、排序索引、文件名或 tensor
  shape 间接携带 GT identity。清理 crop 必须使用 detector 框；GT crop 只能
  明确标作 `oracle_clean_current`。

### 1.3 Online 状态

- [ ] online 初始化只用 detector；不能以 GT box/ID/visibility 初始化轨迹。
- [ ] 帧 (t) 的 assignment 完成前，不把 (c_t) 写回用于本帧评分的
  history/filter。接受的检测只能在评分后更新 (H_{i,<t+1})。
- [ ] teacher 网络/参数在 eval 和 online 均 frozen；线上不 fit ridge、不读
  GT、不更新 batch statistics。若有自适应更新，必须另写成 student memory
  ablation，并报告 update 时刻和接受条件。

## 2. Ridge filter teacher 的定义与数值审计

### 2.1 推荐的闭式目标

对每个 train sample（或一个预定义的轨迹窗口）构造

```text
X_cls = [phi(H_<t, c_j)]_j,       y_cls[j] in {+1, -1}
X_box = [psi(H_<t, c_j)]_j,       Y_box[j] = normalized box/motion target
```

分类/association filter 与 bbox/motion filter 分开拟合：

[
w_{cls} = (X_{cls}^T X_{cls}+\lambda_{cls}I)^{-1}X_{cls}^T y_{cls},
\]

[
W_{box} = (X_{box}^T X_{box}+\lambda_{box}I)^{-1}X_{box}^T Y_{box}.
]

当样本数 (n) 小于特征维度 (d) 时允许使用等价 dual 形式：

[
w = X^T(XX^T+\lambda I_n)^{-1}y.
]

teacher 输出应至少包含 `omega_cls`, `omega_box`（以及拟合的 feature
normalization/scaler 状态和 `fit_frame`）。若只输出一个 clean embedding，必须
标作 `embedding_teacher`，不能宣称 filter teacher。

### 2.2 实现必须通过的数值检查

- [ ] 全部 fit 在 `float64` 中进行；输入、标签、输出和 loss 每一步做
  `isfinite` 检查。最终线上推理可转换到 float32/16，但要记录转换点。
- [ ] 不得显式计算 `inv(A)`。使用 Cholesky/`solve`；若矩阵不正定或条件数
  超过预注册上限，回退到 SVD/pseudoinverse 并记录 `solver_fallback`。
- [ ] 先按 train-fit 统计量对每个 feature/target 做中心化和尺度标准化；
  零方差列要删除或固定 scale=1，并记录列 mask。train-heldout/eval 只能使用
  fit 的 scaler。
- [ ] λ（以及可能的 λ_cls/λ_box 网格）必须在 fit 前预注册，选择只能在
  train-heldout；禁止看 eval 指标后再改 λ。记录最终 λ、单位（是否随 n 归一化）
  和 grid。
- [ ] 记录 `rank(X)`, `min_eig(A)`, `max_eig(A)`, `cond(A)`, solver、残差
  `||Aw-X^Ty||`, dual/primal consistency residual 和 fallback 次数。建议
  `cond(A) <= 1e8`（项目可更严格，但阈值须事先固定）。
- [ ] 在 (d\gg n) 与 (n\gg d) 两种形状各做一次 primal/dual 数值等价性
  测试；相对误差超过预注册阈值（建议 (10^{-5}) in float64）则 fail。
- [ ] 当某个窗口没有正样本、只有一个类别、候选全被过滤或 history 为空时，
  不伪造标签/不除以零；返回 `invalid_teacher_window` 并从质量统计中单独计数。
- [ ] 评分前后做稳定的 L2/温度归一化和有限值裁剪；裁剪比例也要报告，不能
  用 NaN→0 掩盖病态 filter。
- [ ] bbox target 的坐标、尺度和速度按固定定义（例如 log-width/log-height、
  Δcenter/Δsize）标准化；不得把 eval GT 的全局均值/方差混入。

### 2.3 闭式 teacher 的伪代码（offline fit）

```python
def fit_ridge_filter(train_fit_windows, lam_cls, lam_box, scaler):
    # train_fit_windows only; every window has frame_t and history < t.
    cls_rows, cls_y, box_rows, box_y = [], [], [], []
    for win in train_fit_windows:
        assert win.history_max_frame < win.frame_t
        # GT is permitted here only because this is the declared train-fit split.
        clean_pos, hard_negs = win.clean_gt_positive, win.same_frame_hard_negatives
        for cand, label in [(clean_pos, +1), *[(c, -1) for c in hard_negs]]:
            cls_rows.append(phi(win.history, cand))
            cls_y.append(label)
            box_rows.append(psi(win.history, cand))
            box_y.append(normalize_box_target(win, cand, scaler.box))

    Xc, yc = scaler.cls.fit_transform(np.asarray(cls_rows, dtype=np.float64)), np.asarray(cls_y, dtype=np.float64)
    Xb, Yb = scaler.box_x.fit_transform(np.asarray(box_rows, dtype=np.float64)), np.asarray(box_y, dtype=np.float64)
    assert np.isfinite(Xc).all() and np.isfinite(yc).all()
    omega_cls, diag_cls = stable_ridge(Xc, yc, lam_cls)
    omega_box, diag_box = stable_ridge(Xb, Yb, lam_box)
    assert diag_cls.ok and diag_box.ok
    return FrozenFilter(omega_cls, omega_box, scaler, diag_cls, diag_box)

def stable_ridge(X, Y, lam):
    # Use solve/Cholesky/SVD, never np.linalg.inv.
    X, Y = X.astype(np.float64), np.asarray(Y, dtype=np.float64)
    d = X.shape[1]
    A = X.T @ X + float(lam) * np.eye(d, dtype=np.float64)
    b = X.T @ Y
    try:
        w = np.linalg.solve(A, b)
        solver = "solve"
    except np.linalg.LinAlgError:
        u, s, vh = np.linalg.svd(A, full_matrices=False)
        w = (vh.T * (1.0 / np.maximum(s, SVD_EPS))) @ (u.T @ b)
        solver = "svd_fallback"
    diag = ridge_diagnostics(X, Y, w, lam, solver)
    return w, diag
```

`FrozenFilter` 序列化时必须包含参数 hash、scaler hash、λ、fit split hash、
solver diagnostics 和源码 SHA；读取时重新检查 hash 和有限值。

## 3. Strict lag (t\rightarrow t+1) 协议

推荐的最小 causal 训练/评估流程如下。注意 `teacher_omega_t` 只作为下一帧
的 target/context；它不对生成它的 (c_t) 做同帧排名。

```python
# Offline train -------------------------------------------------------------
for t in train_fit_frames:
    H_t = observable_history(track_memory_before(t))       # frames < t only
    omega_t = teacher.fit_or_apply(clean_gt_c_t, H_t, train_labels=True)
    # corruption is student-only augmentation
    z_t = student(H_t, corrupt(detector_c_t), g_t)
    loss += distance(z_t, stopgrad(omega_t))
# Fit/freeze teacher first; do not use train-heldout or eval labels here.

# Train-heldout quality (must be t+1 candidate ranking) ---------------------
for t in train_heldout_frames:
    H_t = observable_history_before(t)
    omega_t = frozen_teacher_from_train_fit(clean_gt_c_t, H_t)  # diagnostic only
    # Do not score c_t with omega_t.  Evaluate on the next frame.
    c_next = detector_candidates(t + 1)
    pred_next = student(H_t, corrupt_or_clean(detector_c_t), g_t)
    rank_next = score(pred_next, c_next)
    measure(rank_next, gt_match_at_t_plus_1)  # labels only for heldout metric

# Online/eval ---------------------------------------------------------------
for t in frames_in_order:
    H_t = observable_history_before(t)
    c_t = detector_candidates(t)
    pred_t = stored_student_or_filter_from_previous_step  # omega_{t-1}
    assignment = rank_and_match(pred_t, c_t)
    update_memory_after_assignment(t, accepted_detector_boxes_only=True)
    # only now build/cache state to be used at t+1
```

There are two acceptable implementations, which must not be conflated:

1. **Lagged filter**: fit/apply `omega_t` from frame (t), store it, and use it on
   candidates at (t+1). This is the default causal protocol.
2. **Leave-current-out filter**: fit `omega_t` from history and candidates excluding
   the candidate being scored, then score that candidate. It is more expensive and
   must record the exclusion set for every score.

Any result using clean current GT to construct a filter and then ranking the same
frame is an `oracle-current` diagnostic. A result whose history is assembled with GT
IDs/boxes is an `oracle-history` diagnostic. Both can be useful upper bounds, but
neither is an online result; include these labels in filenames, tables and plots.

## 4. Student/teacher and control matrix

All branches must use the same detector boxes, encoder/crop preprocessing, candidate
sets, frame range, matching thresholds, seeds and post-processing. At minimum record:

| Branch | Current input | History | Target/filter | Meaning |
|---|---|---|---|---|
| `ridge_teacher_oracle` | clean GT/current | GT or oracle history (explicit) | closed-form ridge | ceiling/diagnostic, not online |
| `student_jepa_lagged` | corrupted detector (c_t) | detector memory | stop-grad `omega_t`, used at (t+1) | proposed causal JEPA |
| `raw_detector` | clean detector | none/raw feature | cosine/IoU baseline | association baseline |
| `history_only` | masked current | detector memory | no-current control | tests history signal |
| `current_only` | detector current | masked history | no-history control | tests current signal |
| `direct_no_teacher` | same student input | same | direct supervised target | distinguishes JEPA target effect |
| `shuffled_target` | same | same | other track/frame filter | leakage/signal negative control |
| `random_frozen` | same | same | random frozen filter | chance/reference |
| `clean_current_ceiling` | clean detector | same | same lag | corruption ceiling |

`current_only`/`history_only` must actually mask both geometry and appearance paths;
merely zeroing an embedding while leaving current bbox geometry is not a pure control.
`ridge_teacher_oracle` and any GT-history branch must never be included in the
“online improvement” column.

## 5. Teacher quality gate（train-heldout only）

Teacher quality is a release gate, not a post-hoc explanation. Compute it only on a
predeclared train-heldout split, with frame (t) teacher state evaluated on **(t+1)
candidates**. Do not read eval GT to decide whether to continue, choose λ, choose a
checkpoint, or set an association threshold.

Required fields per split and aggregate:

- clean-current/filter objective (teacher fit residual and heldout residual);
- teacher→clean target cosine (if an embedding diagnostic is retained);
- hard-negative score margin and Recall@1/Recall@K on (t+1);
- candidate-count-adjusted chance Recall and paired confidence interval;
- rank/condition number, min/max eigenvalue, primal-dual residual, fallback/invalid
  window counts;
- student→teacher distance/cosine and student→clean-target metrics;
- fraction of samples with positive margin and post-gap/reappearance bins;
- exact train-fit/train-heldout IDs, detector/feature/GT digests, source SHA, seed,
  λ/scaler hash, corruption policy and teacher parameter hash.

Pre-register numerical gates before looking at eval. A practical initial gate (to be
reviewed for dataset scale) is:

```text
all metrics finite;
heldout teacher margin > 0 and Recall@1 > chance by a declared delta;
student→teacher is not worse than direct_no_teacher by the declared tolerance;
cond(A) <= 1e8 (or documented SVD fallback), residual <= 1e-5 float64;
invalid/duplicate windows and candidate leakage count = 0.
```

If the gate fails, label the run `teacher_quality_fail` and stop before any tracker
integration. A positive eval score cannot override a failed train-heldout gate.

## 6. Required automated audit tests

- [ ] `assert_no_future_history`: mutate/delete all frames ≥ (t), output at (t)
  must be unchanged.
- [ ] `assert_no_gt_online`: run online with GT boxes/IDs removed; code must not
  import/read GT paths after dataset label construction.
- [ ] `assert_lag`: perturb (c_t); scores at (t) must not change, while permitted
  state at (t+1) may change only after assignment/update.
- [ ] `assert_leave_current_out`: remove the candidate used to make the filter;
  its score must be unavailable or recomputed from an exclusion set.
- [ ] `assert_candidate_permutation`: permuting candidates changes only row order,
  not per-candidate scores; adding an unrelated candidate does not alter existing
  scores unless a documented normalization requires it.
- [ ] `assert_primal_dual`: compare primal and dual ridge outputs in both matrix
  aspect ratios; fail above the predeclared relative tolerance.
- [ ] `assert_lambda_and_scaler_frozen`: eval cannot alter λ, scaler, teacher weights,
  batch norm or quality thresholds.
- [ ] `assert_corruption_scope`: train corruption is on student input only; clean
  detector features are the default deployment path; any stress corruption is an
  explicitly named evaluation mode.
- [ ] `assert_reproducibility`: rerun with identical SHA/seed/digests and compare
  teacher parameter hash, sample digest and metrics within tolerance.

Suggested leakage/lag probe pseudocode:

```python
base_t = run_step(t, history=H_before_t, candidates=C_t)
future_mut = run_step(t, history=H_before_t + mutate_frames_from(t), candidates=C_t)
assert max_abs(base_t.scores - future_mut.scores) == 0

perturbed_current = run_step(t, history=H_before_t, candidates=mutate(C_t))
assert scores_at_t_are_from_stored_omega_only(base_t, perturbed_current)
next_a = run_step(t + 1, history=state_after(base_t), candidates=C_next)
next_b = run_step(t + 1, history=state_after(perturbed_current), candidates=C_next)
assert difference_is_explained_by_accepted_assignment(next_a, next_b)
```

## 7. Artifact schema and decision rule

Every run must emit a machine-readable `audit.json` containing at least:

```json
{
  "protocol": "ridge_filter_jepa_lagged_v1",
  "source_sha256": "...",
  "trackeval_sha256_or_commit": "...",
  "train_fit_split": "...",
  "train_heldout_split": "...",
  "eval_split": "...",
  "detector_digest": "...",
  "feature_digest": "...",
  "gt_digest_for_offline_labels_only": "...",
  "lambda_cls": 0.0,
  "lambda_box": 0.0,
  "scaler_hash": "...",
  "teacher_parameter_hash": "...",
  "corruption_policy": "...",
  "lag_frames": 1,
  "leave_current_out": true,
  "gt_history": false,
  "oracle_semantics": null,
  "duplicate_assignment_count": 0,
  "quality_gate": {"split": "train_heldout", "status": "pass"}
}
```

Use actual λ values in place of the placeholders and include solver diagnostics and
per-sequence counts. If `gt_history=true` or `oracle_semantics` is non-null, the run
must be excluded from online tables and clearly named `oracle-history` or
`oracle-current`.

Decision rule for the next experiment:

1. First pass the data/leakage, lag, numerical and train-heldout teacher gates.
2. Then compare lagged JEPA against every control on at least two sequences, multiple
   seeds, and both normal/occlusion or post-gap bins.
3. Only a consistent causal gain with an independent TrackEval audit can justify a
   tracker smoke. The existing current-pair and FastReID probes do not meet this bar:
   they are representation/oracle diagnostics, and their `no_teacher` wins cannot
   falsify GOT-JEPA because their teacher is not a frozen pretrained tracking filter.

## 8. MOT17 20-frame ridge-filter smoke reflection（NO-GO）

当前已有两个同协议快照：旧目录
`experiments/results/mot_filter_jepa_probe_mot20/` 与补充 candidate-order audit
后的 `experiments/results/mot_filter_jepa_probe_mot20_v3/`。v3 使用的脚本 SHA-256
为 `28468a4a8d0fa32b3b640016651a0ba83ba35463cc86d8d4f712b6c75b52a7a1`，训练/评估
样本为 717/241，MOT17 train=02/04/09、eval=05/11，max 20 帧，lag=1，
per-frame Hungarian IoU≥0.30，duplicate=0。v3 的候选置换字段均通过（但该
predictor 不读取 next-candidate list，因此这是结构性不变量，不是 learned set
invariance）。

v3 的 eval representation 数值如下（Recall@1 / next-frame margin）：

```text
raw              0.9917 / +0.0897
history          0.9876 / +0.0760
ridge teacher*   0.9751 / +0.2480
current_jepa     0.1950 / -0.0625
current_only     0.1618 / -0.0557
full_mask        0.1701 / -0.0731
no_teacher       0.1743 / -0.0458
random_frozen    0.1784 / -0.0579
both_clean       0.1618 / -0.0594
```

`ridge teacher*` 对每个 eval frame (t) 用 eval GT clean crop、eval GT identity
history 和 GT-derived same-frame hard negatives 构造 `teacher_w_t`，再在
(t+1) 候选上测量；它是 `oracle-current + oracle-history` diagnostic，**不是
train-heldout quality gate，也不是 online teacher**。因此其高 Recall@1 不能
被用来证明可部署的 teacher；同理，所有 student 分支的 history 由 GT identity
回放组装，仍是 representation upper bound。

在相同 oracle replay 下，`current_jepa` 的 Recall@1（0.1950）和 margin
（−0.0625）显著低于 `raw`（0.9917/+0.0897）与 `history`
（0.9876/+0.0760），也低于该实验中的 `no_teacher`（0.1743/−0.0458）仅为
很小的绝对差；所有 student margin 为负，不能作为关联 filter。故当前结果的
决策为：

```text
ridge-filter JEPA 20f smoke: 诊断产物已保存；tracker integration = NO-GO。
```

这不是对 GOT-JEPA 思路的否定：这里的 teacher 是从头用极少样本、低维手工 crop
特征和单个固定 λ 的闭式 filter 目标，尚不是论文所说的 frozen pretrained
tracking predictor/filter。下一步只有在不覆盖上述目录的前提下，完成以下修复
才可重新评估：

1. 将 teacher fit 与质量 gate 拆为 `train_fit`/`train_heldout`；只在
   **train-heldout 的 (t+1) detector 候选**上测 teacher margin、Recall@1 和
   candidate-count-adjusted chance，绝不读取 eval 指标来选 λ、阈值或 checkpoint。
2. dual ridge 改为 float64 `solve`/Cholesky（必要时 SVD fallback），输出
   `rank/cond/eigenvalue/residual`、primal-dual 一致性和 invalid-window 计数；
   scaler 与 λ 只由 train-fit 固定并 hash。
3. 保留严格 (t\rightarrow t+1) lag；eval/student 在帧 (t) 只接收 detector
   (c_t)，用已存 `omega_{t-1}` 排下一时刻候选，assignment 后才更新 memory。
   现有 GT-history/clean-current 版本只标 oracle，不进入 online 表格。
4. 通过质量 gate 后再做至少两序列、多 seed、遮挡/post-gap 分桶，并用官方
   TrackEval 审计；若仍低于 raw/history/no-teacher controls，则停止集成。

### v5 artifact audit（补充判定）

`experiments/results/mot_filter_jepa_probe_mot20_v5/reflection.json` 保存了不改写
原始指标的逐项审计。v5 的 teacher gate 可以记为 PASS（train-heldout 的 (t+1)
teacher Recall@1=0.8816、mean margin=+0.1804；float64 ridge 的 condition
max=51.45、residual max=1.58e-14、invalid=0），但这只说明离线 filter 数值稳定，
不说明 student 可部署。student gate 必须记为 FAIL：heldout `current_jepa`
Recall@1=0.0789、margin=-0.0713，而 `raw/history` 都是 Recall@1=1.0、
margin 分别 +0.0785/+0.0739；外部 eval 诊断也只有 Recall@1=0.1909、margin=-0.0630，
远低于 `raw`=0.9917、`history`=0.9876。它仅略高于很弱的 `no_teacher`/`random`
并不足以形成 gate，尤其没有预注册的正 margin、chance-adjusted meaningful delta、
raw/history 要求或 paired CI。因此 v5 的代码字段 `proceed_lagged_tracker=true`
是**误报**，最终 integration decision 必须是 `NO-GO`，旧 `results.json` 不应被
覆盖。失败更可能反映 handcrafted 80-D crop filter 与 student 的 teacher-target
不匹配（以及 oracle GT-history replay），而不是对 GOT-JEPA 思路的反证。

### Train-only held-out wrapper（v4，严格不读 eval GT）

为避免把 eval oracle 混进 quality gate，新增了独立 wrapper
`experiments/mot_filter_jepa_probe_trainheldout.py`，运行产物为
`experiments/results/mot_filter_jepa_trainheldout_mot20_v1/results.json`。它将
`mot_eval` 明确设为空，并断言 loader 返回 0 个 eval item；artifact 中记录
`eval_gt_read=false`。因此下面的 gate 不依赖 MOT17-05/11 的 GT，也没有把外部
eval 指标用于选参。base probe SHA 为
`21acf2f0893228f931d91769ab24a48224a6e903de84adae569d9c7288bd974e`，wrapper SHA
为 `7701dc9e36a98fe76a966ef08eddfab51d5917ce50091122c5340bd0d8da6c8a`。

在 MOT17 train 02/04/09 的 20 帧 smoke 中，按每序列前 15 帧 fit、16--19 帧
heldout（717 总样本，565/152）得到：

```text
teacher (t+1 heldout): R@1=.8816, margin=+.1804, chance=.0695
ridge: float64, cond_max=51.45, residual_max=1.58e-14, invalid=0
current_jepa: R@1=.0789, margin=-.0713, Δchance=.0095
raw/history: R@1=1.0000/1.0000, margin=+.0785/+.0739
```

teacher 与数值 gate 通过，但 student gate 失败（margin 非正、相对 chance
增益远小于预注册 .10，且显著落后 raw/history），所以 wrapper 明确输出
`proceed_lagged_tracker=false`。该结果是当前最小的**不读 eval GT** heldout
证据；仍因 GT identity history 而属于 oracle representation diagnostic，不能
称 online tracker。还需注意：当前 `build_filter_records` 为每个 heldout 样本
即时用该样本的 clean GT/current 和 GT-derived negatives 生成 `teacher_w`；它没有
把 fit split 训练出的全局 frozen filter 应用于 heldout。因此这里的 teacher gate
应准确称为 **heldout oracle-filter quality**，不是 frozen-teacher 泛化验证；要
验证真正的 GOT-JEPA teacher，下一版必须先在 train-fit 固定 teacher/filter
参数（或明确 per-track filter 的可冻结状态），再在 heldout 的 (t+1) 候选测量。

## 10. v6 方向：真正 frozen global teacher（当前尚未通过）

v5 与 train-only wrapper 的 `teacher_w` 都是“每个窗口重新用 clean GT 拟合一把
ridge filter”。即使 heldout 不读 eval GT，这仍是 `oracle-current` target，不能
回答“一个在 train-fit 学到并冻结的 teacher 能否泛化到 heldout/eval”。因此 v6
应使用独立文件（建议 `experiments/mot_global_filter_jepa_probe.py`），不要覆写
旧 `mot_filter_jepa_probe*.py` 或已有结果目录。

### 10.1 Frozen global teacher 的最小定义

1. 在 **train-fit** 窗口收集低维 association rows（例如
   `phi(H_<t, c_t)` 的外观/几何/时间差），标签只在这里用 train GT 生成：正确
   匹配 `+1`，同帧 hard negatives `-1`。预注册 λ、scaler、采样和
   identity/sequence split。
2. 用所有 train-fit rows 一次拟合全局
   `w_T=(X.T@X + lambda*I)^(-1)@X.T@y`（或等价 dual），保存并 hash `w_T`、
   scaler、λ、train-fit digests；拟合后完全 frozen。不得在每个 heldout 样本
   用其 clean GT current 重新求 `w`。
3. 在 train-heldout/eval 的帧 `t`，只把可观测 detector history/current 送给
   student；将 frozen `w_T`（或由固定 teacher filter 对历史状态得到的
   `omega_t`）作为 stop-gradient target。由 `t` 得到的预测只在 `t+1` 候选上
   评分，严格 lag=1。若 filter 需要 track state，state 只能由帧 `<t` 的
   detector memory 更新。
4. heldout/eval 标签仅用于最后的 t+1 metric；不能反向影响 teacher、student
   超参、quality gate 或 checkpoint。若要 per-track adaptation，必须另定义只
   使用 accepted detector observations 的线上更新，并把 GT/per-track clean fit
   明确标作 oracle ablation。

### 10.2 v6 必须输出的对照与状态

至少输出 `global_teacher_frozen`, `student_jepa_global`, `raw`, `history`,
`direct_no_teacher`, `random_frozen` 和 `clean_current/oracle`（后者不进 online
表）。artifact 中要有 `teacher_fit_split`, `teacher_parameter_hash`,
`teacher_frozen=true`, `heldout_teacher_refit=false`, `eval_gt_read` 语义、
lag assertion、candidate set digest、paired CI 和 TrackEval 释放状态。

### 10.3 v6 当前审计状态

截至本审计，v6 尚未完成独立审计；Luna-MOT 正在尝试实现新脚本。此前 v5 的
teacher gate 即使数值 PASS，也仍是 `oracle-filter quality`。在 frozen global
teacher 实现并通过 train-heldout 质量 gate 之前，v6 teacher/student 与 tracker
integration 均记为 `NO-GO`。任何继续使用 heldout GT 现拟合 `teacher_w` 的结果必须
标注 `oracle-current/history`，不得用来声称 GOT-JEPA 的 frozen-teacher 泛化。

### Global-filter fallback v2（clean/stress 双视图）

为排除“student 指标只是被 synthetic corruption 污染”的歧义，新增了
`experiments/mot_global_filter_jepa_probe_fallback.py` 的 v2 运行，产物为
`experiments/results/mot_global_filter_jepa_trainonly_mot20_v2/results.json`，
逐项反思保存在同目录 `reflection.json`。该运行仍严格将 `mot_eval` 置空并记录
`eval_gt_read=false`、`eval_items_loaded=0`，所以没有使用 MOT17-05/11 的 GT。
teacher 是在 train-fit（565 rows）上拟合的一把全局 float64 ridge mapping，冻结
后在 train-heldout（152 rows）的 (t+1) 候选测量；输出 target variance=.0124、
centered rank=55，故不是常量 sanity target。全局 teacher 的 heldout R@1=.9079、
margin=+.1753，cond=26672、residual=2.44e-12。

同时报告两种 student 输入：

```text
                           deployment_clean       stress_corrupt
student_jepa_global       R@1=.0724 / m=-.0588    R@1=.0789 / m=-.0671
raw/history                R@1=1.0000 / +.0785     R@1=1.0000 / +.0785
no_teacher                 R@1=.0921               R@1=.0658
```

`deployment_clean` 将 heldout row 的 corruption 字段替换为**未污染 detector
candidate**，`stress_corrupt` 保留 mask+Gaussian 增强；两者都只在 (t+1) 候选
评分。student 在 clean/stress 下均 margin 为负且远低于 raw/history，因此
`proceed_lagged_tracker=false`，不因 clean view 而放宽 NO-GO。需要继续强调：
history 仍由 GT identity 回放组装，属于 oracle representation diagnostic，
不是线上 MOT。

下一步停止对手工 80-D/全局 ridge 的调参，转向真正的 pretrained 或
train-fit state-conditioned tracking predictor：优先检查本地 ToMP/HybridSORT/GOT
checkpoint 与配置的存在、大小、SHA 和训练覆盖；若 checkpoint 不在本机，先记录
缺失/下载需求，不在未获授权时下载约 1.7 GB 大文件。teacher 输出应与
ClsDec/RegDec 兼容（而非仅一个 crop-vector），然后沿用同一 frozen、lagged、
train-heldout/no-eval-GT gate。

## 11. GOT-JEPA/ToMP checkpoint 与运行依赖只读审计（2026-08-31 04:42 +08:00）

本节只做静态代码检查、文件存在性检查和公开 Google Drive/GitHub HTML 元数据
查询；**没有下载任何权重，也没有修改 tracker 代码**。因此下表中的远端 SHA
为空是预期行为，不能把网页元数据当作已验证的二进制内容。

### 11.1 本地文件与加载链

| 项目 | 本地状态 | 证据/尺寸 |
|---|---|---|
| `pytracking/pytracking/parameter/tomp/got_jepa_378.py` | 存在 | 2,262 B；SHA-256 `a2ab55321a401197eadeee5cd94a5734f87afecc2b57c291daeae09c7f681923` |
| `got_jepa_378_cla.py` / `got_jepa_378_gpt.py` | 存在 | 各 2,333 B；SHA-256 均为 `0c077b3b3c06041a6ab424d8d6e38c5556b14985c696a68e40a4656d57e391d6` |
| `pytracking/pytracking/features/net_wrappers.py` | 存在 | 2,437 B；SHA-256 `9f4c893ec7ac4dd586f56f5b05255e223792fc8f70d2f50588d24441eb2e9764` |
| `pytracking/pytracking/tracker/tomp/tomp_gotejepa.py` | 存在 | 48,519 B；SHA-256 `237e80f112a18c438c71535d4010abf1a72e015c2239682eca85ea006a152b48` |
| `GOT_JEPA_378.tar`（配置相对路径、`pytracking/pytracking/networks/`） | **缺失** | `pytracking/pytracking/networks/` 目录也不存在 |
| `GOT_JEPA_378.tar`（`cla/gpt` 硬编码路径） | **缺失** | `/data1/sfchen94/pytrackingcsf/pytracking/checkpoints/ltr/tomp/GOT_JEPA/GOT_JEPA_378.tar`；`/data1/sfchen94` 本身不存在 |
| `tomp50.pth.tar` / `tomp101.pth.tar` | **缺失** | GOT-main 本地树及已检查的 `/data1/LWR/vranlee`、`/home/lwr`、`/home/user` 均未找到 |
| 训练前置 `ToMPL.tar`、`got_jepa_pretrain.tar`、`got_jepa_finetune.tar` | **缺失** | 训练脚本以当前工作目录的相对文件名调用，仓库内无这些文件 |
| `cotracker2.pth` | **缺失** | 仓库内 `checkpoints/` 目录无该文件 |

静态加载链为：`got_jepa_378.py` 建立
`NetWithBackbone(net_path="GOT_JEPA_378.tar", use_gpu=True)` →
`pytracking.utils.loading.load_network` → `ltr.admin.loading.load_network` →
`torch.load(..., map_location="cpu")`，再依据 checkpoint 中的 `constructor` 重建
网络并加载 `checkpoint_dict['net']`；随后 wrapper 调用 `.cuda()`（若
`use_gpu=True`）和 `.eval()`。相对路径要先经过
`pytracking/evaluation/environment.py` 的 `env_settings().network_path`。
当前没有 `pytracking/pytracking/evaluation/local.py`，只有未自动导入的
`local_plain.py`；也没有 `pytracking/ltr/admin/local.py`。未配置这两个 local.py
时，环境模块会生成默认文件并抛出 `YOU HAVE NOT SETUP YOUR local.py`，所以即使
权重放入仓库也不能直接运行。

训练脚本的前置关系也不是一个文件：
`GOT_JEPA_378_pretrain.py` 读取 `ToMPL.tar`，finetune 读取
`got_jepa_pretrain.tar`，`GOT_JEPA_378_PT.py` 读取 `got_jepa_finetune.tar`；公开
文件名 `tompL_378_JEPAs2.pth.tar` 与这些相对名不相同，取得后需要经过明确的
路径/命名配置（不能默认为可直接加载）。

当前 shell 的只读运行时探针为 `/home/lwr/anaconda3/bin/python` 3.12.2，
`import torch` 失败（`ModuleNotFoundError`），因而本轮不能做 Python checkpoint
load smoke；这不等同于所有 conda 环境都没有 torch，但说明当前默认环境不可运行
GOT-JEPA。仓库 pinned README 要求 CUDA 11.7、PyTorch 2.0.0，并要求配置
`ltr/admin/local.py`、`pytracking/evaluation/local.py` 以及
`resnet.py`/`build_cotracker.py` 中的几何 backbone 路径。Python 3.12 与该旧版
PyTorch/CUDA 组合还存在兼容性风险，需在隔离环境中实际验证。

### 11.2 公开 Drive/GitHub 元数据（只查询页面，不下载）

仓库 README 的 GOT-JEPA 使用说明固定在
[GitHub commit `84e9324317e4afe62c06b2c51a97563f79730a2e`](https://github.com/chenshihfang/GOT/tree/84e9324317e4afe62c06b2c51a97563f79730a2e)，
并链接 [GOT-JEPA Drive 根目录](https://drive.google.com/drive/folders/1fM0MKmMDTL7bmONtmaeVTVuEuf0cVryc?usp=sharing)
和 [GOT-Pretrain 根目录](https://drive.google.com/drive/folders/1ixG9UVb8ikcrBEi_IvCc8_6mCcbQJ1xm?usp=sharing)。
2026-08-31 查询到的公开条目如下（字节数来自 Drive 页面内嵌 metadata，括号为
页面显示值）：

| 远端目录/文件 | Drive file ID | metadata 大小 | 页面显示 |
|---|---|---:|---:|
| `GOT-JEPA/Checkpoints/GOT_JEPA_378.tar` | `1P8e_2nKIaVKcxUkj59hgGVZMD54gKpdn` | 1,706,753,563 B | 1.59 GB |
| `GOT-JEPA/Checkpoints/cotracker2.pth` | `16rqb19G4Cph_FM6ylapSJDKO9ZdQ_sTn` | 204,396,415 B | 194.9 MB |
| `GOT_Pretrain/ToMPL_378_pre/tompL_378_JEPAs2.pth.tar` | `1d7B2nAuFzeoEmBwuX4Oa353FcQLQAip9` | 1,514,051,273 B | 1.41 GB |
| `GOT_Pretrain/ToMPL_252_pre/tompL_252_JEPAs2.pth.tar` | `1ngFRlextr0VnivuR-mNElkzdbINY0BC8` | 1,804,980,703 B | 1.68 GB |

这确认了用户所说约 1.7 GB 的 GOT-JEPA 文件确实在公开目录中，且下载按钮和
文件名可见；但本机没有对应副本，远端内容的 SHA、checkpoint key 和构造器尚未
验证。`GOT-JEPA` 根目录页面还显示 `Checkpoints` 子目录（ID
`17LYU6qoUE6HKIkaij7AClGmGKiGruGow`），`GOT_Pretrain` 根目录显示
`ToMPL_378_pre`（ID `1zzwf7s2PS5i-3JJpJXXVfGJxt8LoOWcN`）和
`ToMPL_252_pre`（ID `1GETDXhUvuoPjB3HLuRDeRNx8IG4ABa-7`）。这些是可复核的
公开元数据，不是授权下载或加载结果。

### 11.3 可行性判断与适配风险

* **下载可行性：条件性可行。** 公开文件存在、大小与预期一致；但下载约
  1.7 GB 需要用户/项目明确授权、足够磁盘空间和完整性校验。本轮遵守“只检查，
  不下载”的约束。
* **加载可行性：当前不可执行，取得权重后有条件可行。** LTR loader 的代码路径
  完整，但缺 checkpoint、PyTorch/CUDA 和两个 `local.py`，且 `cla/gpt` 的绝对
  路径是旧机器路径。下载后必须先在兼容环境中 `torch.load`/构造器 smoke，记录
  文件 SHA、顶层 keys（至少 `constructor`/`net`）和 state-dict shape；不能把
  FastReID、DINO 或其它 `.pth` 当作 ToMP/GOT-JEPA 替代物。
* **几何依赖另计。** `build_cotracker.py` 读取 `cotracker2.pth`；异常分支仍
  写死 `/yourpath/GOT/pytracking/ltr/cotracker2/co_tracker/checkpoints/cotracker2.pth`。
  README 要求同时修正 `resnet.py` 与该文件，故 GOT-JEPA tar 存在并不代表
  OccuSolver/CoTracker 可用。
* **MOT 适配不是换权重。** `got_jepa_378.py` 设置 `use_gt_box=True`，原始
  tracker 是单目标 ToMP：每个 MOT track 需要独立 history、filter/model state、
  lagged candidate scoring、association 后更新和 occlusion/reappearance 逻辑。
  GOT-JEPA 的 teacher 输出要接到 ToMP 的 `ClsDec/RegDec` 兼容接口，而不是当前
  实验中的 80-D crop/ridge 向量；还要避免把 MOT GT history/current 泄露进线上
  teacher。
* **训练成本与命名风险。** pinned README 的 pretrain/finetune 示例使用 8 GPU，
  PT 使用 4 GPU；公开 ToMPL 文件名、脚本相对名和 tracker 推理文件名不一致，需
  明确哪个是初始化模型、哪个是最终 GOT-JEPA 模型，并保留构造器兼容性记录。

### 11.4 当前实验决策与下一步

综合 v5 train-heldout、global-filter v2 clean/stress 和现有 TrackEval 稀疏审计：
学生分支在 heldout/clean/stress 的 margin 仍为负或明显低于 raw/history，现有
ridge/global probe 只是诊断且没有线上 per-track state；因此 **当前 MOT 的 JEPA
tracker integration = NO-GO**。本节的 checkpoint 元数据发现不会放宽这个 gate，
也没有改写任何旧结果目录。

建议的有序下一步：

1. 获得明确授权后，先下载 `GOT_JEPA_378.tar`（以及确需的 ToMPL/CoTracker 文件），
   下载前检查空间；下载后只做 SHA-256、大小、顶层 key/constructor/state-shape
   记录，保留原始文件名和 Drive ID。
2. 建立隔离的 CUDA 11.7/PyTorch 2.0（优先与项目兼容的 Python 版本）环境，补齐
   两个 `local.py` 和所有 backbone 路径；先做单目标 20-frame GOT-JEPA/ToMP
   inference smoke，再做 decoder 输出 shape 与显存检查。
3. 用冻结 pretrained predictor/filter 实现真正的 MOT per-track adapter：严格
   `t→t+1` lag、只用 accepted detector observations 更新 state、显式 visibility/
   occlusion/reappearance 和 association；train-fit 固定 teacher，train-heldout
   评估，eval GT 只在最后 TrackEval 使用。
4. 通过 teacher/student/numeric/leakage gate 后，才做至少两序列、多 seed、遮挡与
   post-gap 分桶，并由官方 TrackEval 审计；若仍不超过 raw/history/no-teacher
   controls，停止集成并保留 NO-GO。
