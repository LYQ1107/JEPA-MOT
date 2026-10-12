# JEPA 独立调参实验审计报告

生成时间：2026-09-08T23:01:55+08:00

## 实验协议

冻结配置的 full-domain 结果已先封存；本实验只在 BDD100K、VKITTI2、MOTSynth 上调 JEPA 参数。每个域按视频确定性切分为约 20% tuning videos 和 80% holdout videos。
候选选择只使用 tuning videos 的 HOTA/MOTA 等权平均；最终 tuned 结果只在未参与选参的 holdout videos 上报告。

仅搜索 JEPA fallback 的 `high-min-sim` 和 `predicted-box IoU`：

| candidate | high-min-sim | predicted-box IoU |
|---|---:|---:|
| s085_iou000 | 0.85 | 0.00 |
| s085_iou005 | 0.85 | 0.05 |
| s090_iou000 | 0.90 | 0.00 |
| s090_iou005 | 0.90 | 0.05 |
| s095_iou000 | 0.95 | 0.00 |
| s095_iou005 | 0.95 | 0.05 |

Detector rows、GeneralTrack checkpoint、track/det threshold、native lifecycle、history=last、geometry duplicate IoU=0.20 和评测脚本均固定；没有调 detector 或 tracker 阈值。

## 每个数据集的 tuning 选择

| dataset | candidate | tuning HOTA | tuning MOTA | joint score | frozen candidate score |
|---|---|---:|---:|---:|---:|
| bdd100k | s085_iou005 | 33.1319 | 23.0832 | 28.1076 | 28.0728 |
| vkitti2 | s085_iou005 | 19.1207 | 13.6148 | 16.3677 | 16.3425 |
| motsynth | s090_iou005 | 1.0763 | -28.2182 | -13.5709 | -13.5709 |

## Holdout 宏平均

宏平均为同一数据集 holdout domains 等权平均；JEPA-specific 为 tuned - geometry-null。

| scope | method | HOTA | DetA | AssA | MOTA | IDF1 | avg IDSW | avg FP | avg FN |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| bdd100k | Baseline | 36.0070 | 27.0374 | 48.4949 | 28.0421 | 42.1044 | 477.0000 | 2191.0000 | 30659.5000 |
| bdd100k | Frozen V-JEPA21 | 36.2255 | 26.8423 | 49.4316 | 28.0859 | 42.2081 | 426.0000 | 2062.5000 | 30815.0000 |
| bdd100k | Geometry-null | 35.9261 | 26.8628 | 48.5851 | 28.0620 | 41.9436 | 441.7500 | 2076.5000 | 30793.2500 |
| bdd100k | Tuned V-JEPA21 | 36.1822 | 26.8374 | 49.3320 | 28.0706 | 42.1409 | 428.0000 | 2057.2500 | 30823.2500 |
| vkitti2 | Baseline | 23.0145 | 13.5247 | 39.7313 | 16.2041 | 26.0835 | 54.2000 | 248.2000 | 38943.4000 |
| vkitti2 | Frozen V-JEPA21 | 22.5794 | 13.1442 | 39.3518 | 15.7519 | 25.3083 | 46.4000 | 235.8000 | 39175.4000 |
| vkitti2 | Geometry-null | 22.5930 | 13.1610 | 39.3578 | 15.7669 | 25.3131 | 47.4000 | 236.6000 | 39166.6000 |
| vkitti2 | Tuned V-JEPA21 | 22.5664 | 13.1417 | 39.3132 | 15.7489 | 25.2544 | 46.8000 | 236.0000 | 39176.2000 |
| motsynth | Baseline | 0.3070 | 0.2578 | 0.4826 | -12.9871 | 0.0518 | 69.2000 | 30611.8000 | 239802.8000 |
| motsynth | Frozen V-JEPA21 | 0.3143 | 0.2474 | 0.5184 | -11.0019 | 0.0509 | 55.8000 | 25873.4000 | 239818.8000 |
| motsynth | Geometry-null | 0.3109 | 0.2483 | 0.5070 | -11.0967 | 0.0495 | 54.8000 | 26126.6000 | 239820.2000 |
| motsynth | Tuned V-JEPA21 | 0.3143 | 0.2474 | 0.5184 | -11.0019 | 0.0509 | 55.8000 | 25873.4000 | 239818.8000 |
| ALL | Baseline | 18.6168 | 12.6473 | 28.2178 | 9.1609 | 21.3638 | 180.3571 | 11647.4286 | 108312.0714 |
| ALL | Frozen V-JEPA21 | 18.5265 | 12.4519 | 28.3627 | 9.7210 | 21.1163 | 158.2143 | 9914.0000 | 108445.0714 |
| ALL | Geometry-null | 18.4446 | 12.4641 | 28.1189 | 9.6856 | 21.0420 | 162.7143 | 10008.7143 | 108436.2143 |
| ALL | Tuned V-JEPA21 | 18.5094 | 12.4497 | 28.3204 | 9.7155 | 21.0778 | 158.9286 | 9912.5714 | 108447.7143 |

## 调参后的 JEPA-specific 增益

| scope | tuned − geometry-null HOTA | tuned − geometry-null MOTA | tuned − frozen HOTA | tuned − frozen MOTA |
|---|---:|---:|---:|---:|
| bdd100k | +0.2560 | +0.0086 | -0.0434 | -0.0153 |
| vkitti2 | -0.0266 | -0.0179 | -0.0130 | -0.0030 |
| motsynth | +0.0033 | +0.0948 | +0.0000 | +0.0000 |
| ALL | +0.0648 | +0.0299 | -0.0170 | -0.0054 |

## Holdout 逐域结果

| dataset | domain | selected | frozen HOTA/MOTA | tuned HOTA/MOTA | tuned-null HOTA/MOTA |
|---|---|---|---:|---:|---:|
| bdd100k | dawn_dusk | s085_iou005 | 41.2988/34.2296 | 41.2412/34.1990 | +0.2293/-0.0306 |
| bdd100k | night | s085_iou005 | 24.5280/14.0947 | 24.5684/14.0835 | +0.4212/-0.0128 |
| bdd100k | overcast | s085_iou005 | 38.3310/30.6478 | 38.2768/30.6731 | +0.1067/+0.0655 |
| bdd100k | rainy | s085_iou005 | 40.7443/33.3713 | 40.6422/33.3266 | +0.2669/+0.0122 |
| vkitti2 | fog | s085_iou005 | 13.8503/7.6802 | 13.8332/7.6823 | -0.0375/-0.0171 |
| vkitti2 | morning | s085_iou005 | 29.0174/20.3630 | 28.9758/20.3395 | -0.0365/-0.0470 |
| vkitti2 | overcast | s085_iou005 | 27.3593/20.0512 | 27.3553/20.0555 | +0.0430/-0.0107 |
| vkitti2 | rain | s085_iou005 | 17.8704/12.2857 | 17.8639/12.2900 | -0.1189/-0.0085 |
| vkitti2 | sunset | s085_iou005 | 24.7998/18.3794 | 24.8038/18.3773 | +0.0168/-0.0064 |
| motsynth | cloudy | s090_iou005 | 0.3308/-10.7804 | 0.3308/-10.7804 | +0.0129/+0.1035 |
| motsynth | dark | s090_iou005 | 0.4044/-11.0919 | 0.4044/-11.0919 | +0.0034/+0.0824 |
| motsynth | fog | s090_iou005 | 0.2526/-8.7885 | 0.2526/-8.7885 | +0.0004/+0.1630 |
| motsynth | rain | s090_iou005 | 0.3926/-11.2518 | 0.3926/-11.2518 | +0.0019/+0.1485 |
| motsynth | snow | s090_iou005 | 0.1909/-13.0967 | 0.1909/-13.0967 | -0.0019/-0.0234 |

## 输出与解释边界

- tuning outputs: `/data1/LWR/vranlee/SERVER_ONLY/avis/GeneralTrack-main/outputs/jepa_independent_tuning_vjepa21`
- holdout tuned outputs: `/data1/LWR/vranlee/SERVER_ONLY/avis/GeneralTrack-main/outputs/jepa_independent_tuning_vjepa21/holdout_tuned`
- holdout controls: `/data1/LWR/vranlee/SERVER_ONLY/avis/GeneralTrack-main/outputs/jepa_independent_tuning_vjepa21/holdout_controls`
- 这是一组 target-domain video-split adaptation 结果，不应与冻结零样本主结果混写；最终 holdout 仍未参与候选选择。
- 如果 tuned 相对 frozen 改善，只能说明在该数据集内部视频划分上调 JEPA 阈值有效；不能把它表述成无需调参的跨域泛化。
