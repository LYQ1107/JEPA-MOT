# MOTSynth mapped source_detector_thr02 + V-JEPA 2.1 正确协议调参审计报告

生成时间：2026-09-09T07:34:30+08:00

## 协议修正

本报告只使用 `generaltrack_prepared_mapped/source_detector_thr02`。每个 MOTSynth 视频只保留 detector 实际覆盖的帧（通常 450 帧/视频）；snow 为 9 个视频、4050 帧。之前 1798 帧/视频的 dense preparation 结果不纳入本报告。
Baseline 复用已核实的 `outputs/generaltrack_mapped/source_detector_thr02/motsynth`，其五域宏平均 HOTA 为 16.7432。
V-JEPA 缓存直接从本报告的 mapped annotation 重提取；每个 image 的 offsets 已逐帧核对为与 detector 行数完全一致。缓存对 detector score > 0.35 的行计算官方 predictor 表示，低于该值的行保留零向量但不删除，从而不改变 GeneralTrack/ByteTrack 的检测输入协议。
候选只搜索 JEPA `high-min-sim` 与预测框 IoU；detector、GeneralTrack checkpoint、生命周期和所有非 JEPA 阈值固定。候选选择只看 tune videos，holdout 不参与选择。

## 候选与选择

| candidate | high-min-sim | predicted-box IoU | tune HOTA | tune MOTA | joint score |
|---|---:|---:|---:|---:|---:|
| s085_iou000 | 0.85 | 0.00 | 13.0175 | 12.3616 | 12.6896 |
| s085_iou005 | 0.85 | 0.05 | 16.2634 | 18.2290 | 17.2462 |
| s090_iou000 | 0.90 | 0.00 | 14.3331 | 15.2075 | 14.7703 |
| s090_iou005 | 0.90 | 0.05 | 16.5544 | 18.8895 | 17.7219 |
| s095_iou000 | 0.95 | 0.00 | 16.4881 | 18.6490 | 17.5686 |
| s095_iou005 **selected** | 0.95 | 0.05 | 16.6211 | 19.0503 | 17.8357 |

## Full mapped-domain 结果

宏平均为五个 MOTSynth 域等权平均；IDSW/FP/FN 为域平均计数。

| method | HOTA | DetA | AssA | MOTA | IDF1 | avg IDSW | avg FP | avg FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| GeneralTrack baseline | 16.7432 | 25.1723 | 11.4046 | 11.4434 | 18.4443 | 4338.8000 | 18174.4000 | 43356.8000 |
| Geometry-null | 19.4460 | 27.1936 | 14.1757 | 23.0634 | 22.5468 | 2970.2000 | 10054.8000 | 44271.0000 |
| Frozen V-JEPA21 | 19.3192 | 27.1220 | 14.0805 | 22.9255 | 22.3640 | 2943.4000 | 10171.6000 | 44288.0000 |
| Tuned V-JEPA21 | 19.4429 | 27.1983 | 14.2364 | 23.1473 | 22.4116 | 2917.4000 | 10043.0000 | 44269.8000 |

JEPA-specific 与 baseline 对比：
- tuned − geometry-null：HOTA -0.0031，MOTA +0.0839
- tuned − baseline：HOTA +2.6997，MOTA +11.7039
- frozen − baseline：HOTA +2.5760，MOTA +11.4821

## Untouched holdout 结果

宏平均为五个 MOTSynth 域等权平均；IDSW/FP/FN 为域平均计数。

| method | HOTA | DetA | AssA | MOTA | IDF1 | avg IDSW | avg FP | avg FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| GeneralTrack baseline | 17.1652 | 25.4332 | 11.8366 | 12.3260 | 19.1248 | 3477.0000 | 14656.6000 | 34761.6000 |
| Geometry-null | 20.0794 | 27.5980 | 14.8910 | 24.2122 | 23.4911 | 2353.8000 | 7976.2000 | 35479.2000 |
| Frozen V-JEPA21 | 19.9323 | 27.5234 | 14.7782 | 24.0750 | 23.2742 | 2325.0000 | 8062.8000 | 35503.6000 |
| Tuned V-JEPA21 | 20.1069 | 27.6146 | 14.9864 | 24.3440 | 23.4278 | 2302.6000 | 7935.2000 | 35491.6000 |

JEPA-specific 与 baseline 对比：
- tuned − geometry-null：HOTA +0.0275，MOTA +0.1317
- tuned − baseline：HOTA +2.9417，MOTA +12.0179
- frozen − baseline：HOTA +2.7670，MOTA +11.7489

## Full mapped-domain 逐域结果

| domain | method | HOTA | DetA | AssA | MOTA | IDF1 | IDSW | FP | FN |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| cloudy | Baseline | 13.5513 | 23.4089 | 8.0050 | 2.2028 | 14.1493 | 5287 | 21999 | 44458 |
| cloudy | Geometry-null | 15.5432 | 25.0300 | 9.8139 | 14.3430 | 17.1320 | 3904 | 13491 | 45443 |
| cloudy | Frozen V-JEPA21 | 15.5481 | 24.9440 | 9.8341 | 13.9667 | 17.2646 | 3886 | 13753 | 45475 |
| cloudy | Tuned V-JEPA21 | 15.5409 | 25.0110 | 9.8142 | 14.5120 | 17.0362 | 3839 | 13379 | 45496 |
| dark | Baseline | 17.8080 | 22.9411 | 13.9751 | 13.0283 | 20.8022 | 2969 | 13330 | 42413 |
| dark | Geometry-null | 19.9806 | 25.0204 | 16.1033 | 24.1605 | 24.4708 | 1984 | 6158 | 43055 |
| dark | Frozen V-JEPA21 | 19.5333 | 24.8923 | 15.4700 | 23.9383 | 23.9812 | 2006 | 6277 | 43064 |
| dark | Tuned V-JEPA21 | 20.0091 | 24.9984 | 16.1625 | 24.2064 | 24.4441 | 1922 | 6178 | 43066 |
| fog | Baseline | 16.1053 | 23.1640 | 11.4666 | 9.9355 | 17.1640 | 5178 | 21439 | 58829 |
| fog | Geometry-null | 18.9478 | 24.8246 | 14.7118 | 20.7680 | 21.6321 | 3368 | 11803 | 59998 |
| fog | Frozen V-JEPA21 | 19.3820 | 24.7605 | 15.6839 | 20.5097 | 21.4107 | 3462 | 12017 | 59935 |
| fog | Tuned V-JEPA21 | 19.6869 | 24.8530 | 16.1144 | 20.9113 | 21.8824 | 3326 | 11746 | 59961 |
| rain | Baseline | 19.0601 | 29.1963 | 12.5913 | 17.0503 | 20.7684 | 4426 | 18627 | 37049 |
| rain | Geometry-null | 22.6125 | 31.5839 | 16.4110 | 28.5263 | 25.5358 | 3150 | 10584 | 38053 |
| rain | Frozen V-JEPA21 | 22.3977 | 31.5970 | 16.0735 | 28.8451 | 25.5335 | 2932 | 10477 | 38147 |
| rain | Tuned V-JEPA21 | 21.8481 | 31.6118 | 15.2595 | 28.5208 | 24.8915 | 3059 | 10677 | 38055 |
| snow | Baseline | 17.1914 | 27.1512 | 10.9851 | 15.0000 | 19.3377 | 3834 | 15477 | 34035 |
| snow | Geometry-null | 20.1459 | 29.5092 | 13.8383 | 27.5191 | 23.9631 | 2445 | 8238 | 34806 |
| snow | Frozen V-JEPA21 | 19.7349 | 29.4163 | 13.3409 | 27.3678 | 23.6300 | 2431 | 8334 | 34819 |
| snow | Tuned V-JEPA21 | 20.1294 | 29.5172 | 13.8314 | 27.5860 | 23.8038 | 2441 | 8235 | 34771 |

## Holdout 逐域完整指标

| domain | method | HOTA | DetA | AssA | MOTA | IDF1 | IDSW | FP | FN |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| cloudy | Baseline | 13.6326 | 23.9957 | 7.8898 | 4.0717 | 13.8645 | 4246 | 18584 | 36753 |
| cloudy | Geometry-null | 15.7912 | 25.7913 | 9.8181 | 16.6538 | 16.9884 | 3078 | 11084 | 37606 |
| cloudy | Frozen V-JEPA21 | 15.8348 | 25.7130 | 9.8818 | 16.3511 | 17.2137 | 3061 | 11250 | 37645 |
| cloudy | Tuned V-JEPA21 | 15.8393 | 25.7749 | 9.8774 | 16.8953 | 16.9316 | 3008 | 10948 | 37662 |
| dark | Baseline | 18.5372 | 25.3452 | 13.7036 | 15.6011 | 22.4131 | 2303 | 9416 | 26220 |
| dark | Geometry-null | 21.2096 | 27.9251 | 16.2593 | 27.7852 | 27.1650 | 1584 | 4251 | 26627 |
| dark | Frozen V-JEPA21 | 20.6091 | 27.7236 | 15.4624 | 27.2602 | 26.3513 | 1591 | 4467 | 26640 |
| dark | Tuned V-JEPA21 | 21.3204 | 27.9352 | 16.4182 | 27.9787 | 27.3400 | 1506 | 4217 | 26652 |
| fog | Baseline | 16.4076 | 23.3981 | 11.8231 | 10.6268 | 17.5667 | 4554 | 18530 | 50581 |
| fog | Geometry-null | 19.3370 | 25.1502 | 15.1646 | 21.7619 | 22.1573 | 2948 | 10000 | 51539 |
| fog | Frozen V-JEPA21 | 19.8937 | 25.0878 | 16.4040 | 21.4561 | 22.0327 | 3037 | 10208 | 51494 |
| fog | Tuned V-JEPA21 | 20.1337 | 25.1850 | 16.7382 | 21.9269 | 22.4510 | 2921 | 9923 | 51507 |
| rain | Baseline | 19.4535 | 27.9686 | 13.7080 | 16.2266 | 21.4009 | 3499 | 15379 | 33503 |
| rain | Geometry-null | 23.1359 | 30.3279 | 17.9334 | 27.5369 | 26.1537 | 2426 | 8571 | 34312 |
| rain | Frozen V-JEPA21 | 22.9847 | 30.3370 | 17.6615 | 27.8824 | 26.3775 | 2242 | 8418 | 34433 |
| rain | Tuned V-JEPA21 | 22.4107 | 30.3400 | 16.7554 | 27.5209 | 25.7870 | 2348 | 8623 | 34348 |
| snow | Baseline | 17.7952 | 26.4584 | 12.0584 | 15.1040 | 20.3790 | 2783 | 11374 | 26751 |
| snow | Geometry-null | 20.9235 | 28.7954 | 15.2797 | 27.3233 | 24.9910 | 1733 | 5975 | 27312 |
| snow | Frozen V-JEPA21 | 20.3390 | 28.7557 | 14.4811 | 27.4250 | 24.3960 | 1694 | 5971 | 27306 |
| snow | Tuned V-JEPA21 | 20.8304 | 28.8379 | 15.1430 | 27.3980 | 24.6296 | 1730 | 5965 | 27289 |

## Holdout 逐域 HOTA/MOTA

| domain | baseline | geometry-null | frozen JEPA | tuned JEPA | tuned-null |
|---|---:|---:|---:|---:|---:|
| cloudy | 13.6326/4.0717 | 15.7912/16.6538 | 15.8348/16.3511 | 15.8393/16.8953 | +0.0481/+0.2415 |
| dark | 18.5372/15.6011 | 21.2096/27.7852 | 20.6091/27.2602 | 21.3204/27.9787 | +0.1108/+0.1935 |
| fog | 16.4076/10.6268 | 19.3370/21.7619 | 19.8937/21.4561 | 20.1337/21.9269 | +0.7967/+0.1650 |
| rain | 19.4535/16.2266 | 23.1359/27.5369 | 22.9847/27.8824 | 22.4107/27.5209 | -0.7252/-0.0160 |
| snow | 17.7952/15.1040 | 20.9235/27.3233 | 20.3390/27.4250 | 20.8304/27.3980 | -0.0931/+0.0747 |

## 路径与边界

- mapped prepared: `/data1/LWR/vranlee/SERVER_ONLY/avis/GeneralTrack-main/datasets/generaltrack_prepared_mapped/source_detector_thr02/motsynth`
- mapped V-JEPA cache: `/data1/LWR/vranlee/SERVER_ONLY/avis/JEPA/outputs/official_vjepa21_cache_mapped_source_detector_thr02_score0`
- tuning/holdout/full outputs: `/data1/LWR/vranlee/SERVER_ONLY/avis/GeneralTrack-main/outputs/jepa_motsynth_mapped_correct_tuning_vjepa21_score0`
- 本轮 MOTSynth 结果不与之前 dense 1798-frame preparation 的结果混合。
