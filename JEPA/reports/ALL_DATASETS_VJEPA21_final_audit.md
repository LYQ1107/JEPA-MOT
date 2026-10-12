# V-JEPA 2.1 Online GeneralTrack 多数据集完成报告

生成时间：2026-09-08T02:43:08+08:00

## 实验结论

本报告汇总 UA-DETRAC、BDD100K、VKITTI2 和 MOTSynth 的固定 GeneralTrack/TBD 在线评测。
所有最终方法与 geometry-only null 使用相同 detector rows、checkpoint、tracker lifecycle、阈值和评测协议；null 只将 V-JEPA 特征置零，因此 `VJEPA21 - geometry_null` 是本实验中 JEPA-specific 的配对增益估计。

## 固定配置

- V-JEPA 2.1 官方 ViT-L/384 encoder + 官方 predictor，16 帧 causal box crop，context=1.5，最后层预测 token mean-pool，1664-D L2-normalized feature。
- 在线接入 GeneralTrack 原生 relation association；V-JEPA 只处理未匹配高置信轨迹/检测候选，并保留 geometry-only duplicate birth gate。
- detector rows、score threshold、track lifecycle、匹配阈值和 checkpoint 均未因数据集改变。
- 缓存只使用 detector 输入和过去帧，未读取 GT；GT 只用于最终评测。

## 逐域完整指标

数值均为 TrackEval evaluator 输出；HOTA、DetA、AssA、MOTA、IDF1 为百分点，IDSW/FP/FN 为计数。

| dataset | domain | method | HOTA | DetA | AssA | MOTA | IDF1 | IDSW | FP | FN |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ua_detrac | cloudy | GeneralTrack | 35.2668 | 31.8860 | 39.5842 | 15.8500 | 48.6058 | 3820 | 18768 | 20416 |
| ua_detrac | cloudy | Geometry-null | 42.0007 | 38.4536 | 46.4469 | 41.5603 | 59.7234 | 2410 | 6551 | 20904 |
| ua_detrac | cloudy | V-JEPA21 | 42.5496 | 38.6279 | 47.4135 | 42.6796 | 60.8585 | 2076 | 6219 | 20998 |
| ua_detrac | night | GeneralTrack | 34.5917 | 30.5086 | 39.7572 | 21.3725 | 46.0185 | 1697 | 8093 | 15657 |
| ua_detrac | night | Geometry-null | 39.2052 | 34.4646 | 45.0923 | 37.4058 | 53.5699 | 1138 | 3289 | 15831 |
| ua_detrac | night | V-JEPA21 | 42.4302 | 35.3953 | 51.4086 | 41.4967 | 58.4092 | 606 | 2498 | 15830 |
| ua_detrac | rainy | GeneralTrack | 30.9364 | 25.0934 | 39.2198 | 19.0118 | 39.1974 | 2080 | 10072 | 37842 |
| ua_detrac | rainy | Geometry-null | 33.7186 | 26.5970 | 43.9020 | 27.5684 | 43.7140 | 1526 | 4963 | 38223 |
| ua_detrac | rainy | V-JEPA21 | 36.0858 | 26.7104 | 49.8961 | 29.4816 | 47.1309 | 810 | 4461 | 38260 |
| bdd100k | dawn_dusk | GeneralTrack | 41.9095 | 33.0010 | 53.6498 | 35.9031 | 50.2578 | 593 | 2679 | 35081 |
| bdd100k | dawn_dusk | Geometry-null | 41.6631 | 32.6599 | 53.5681 | 35.6842 | 49.8180 | 552 | 2548 | 35384 |
| bdd100k | dawn_dusk | V-JEPA21 | 41.9961 | 32.6840 | 54.3673 | 35.7343 | 50.1078 | 537 | 2485 | 35432 |
| bdd100k | night | GeneralTrack | 24.0862 | 15.8786 | 36.9956 | 13.5413 | 26.2033 | 882 | 4394 | 64248 |
| bdd100k | night | Geometry-null | 24.0354 | 15.7794 | 37.0634 | 13.6284 | 25.9747 | 833 | 4231 | 64390 |
| bdd100k | night | V-JEPA21 | 24.5917 | 15.7928 | 38.7501 | 13.6508 | 26.5204 | 791 | 4267 | 64378 |
| bdd100k | overcast | GeneralTrack | 36.8581 | 26.1636 | 52.1590 | 28.0667 | 41.5787 | 474 | 1949 | 39909 |
| bdd100k | overcast | Geometry-null | 36.7770 | 26.0147 | 52.2176 | 28.1415 | 41.5026 | 427 | 1773 | 40088 |
| bdd100k | overcast | V-JEPA21 | 36.8034 | 25.9574 | 52.3941 | 28.1670 | 41.4580 | 413 | 1738 | 40122 |
| bdd100k | rainy | GeneralTrack | 37.8529 | 27.7759 | 52.0336 | 28.1874 | 42.9402 | 437 | 2296 | 26917 |
| bdd100k | rainy | Geometry-null | 37.8846 | 27.3453 | 52.8923 | 28.4998 | 43.2196 | 400 | 2021 | 27100 |
| bdd100k | rainy | V-JEPA21 | 38.1789 | 27.3188 | 53.7580 | 28.5264 | 43.4536 | 368 | 2024 | 27118 |
| vkitti2 | fog | GeneralTrack | 13.6313 | 6.3514 | 29.4900 | 7.7640 | 14.0647 | 25 | 309 | 49966 |
| vkitti2 | fog | Geometry-null | 13.5539 | 6.2586 | 29.5830 | 7.6503 | 13.4853 | 17 | 293 | 50052 |
| vkitti2 | fog | V-JEPA21 | 13.5360 | 6.2473 | 29.5590 | 7.6338 | 13.4262 | 19 | 293 | 50059 |
| vkitti2 | morning | GeneralTrack | 28.7201 | 17.5238 | 47.1219 | 20.3561 | 32.6416 | 81 | 296 | 43056 |
| vkitti2 | morning | Geometry-null | 28.1413 | 17.0441 | 46.5176 | 19.8005 | 31.8602 | 68 | 291 | 43377 |
| vkitti2 | morning | V-JEPA21 | 28.2235 | 17.0208 | 46.8503 | 19.7822 | 31.9892 | 62 | 291 | 43393 |
| vkitti2 | overcast | GeneralTrack | 27.4883 | 17.1181 | 44.2391 | 20.2388 | 32.0672 | 73 | 377 | 43047 |
| vkitti2 | overcast | Geometry-null | 26.8551 | 16.6072 | 43.5285 | 19.6281 | 31.0209 | 66 | 358 | 43406 |
| vkitti2 | overcast | V-JEPA21 | 26.8843 | 16.5958 | 43.6497 | 19.6263 | 31.1510 | 65 | 356 | 43410 |
| vkitti2 | rain | GeneralTrack | 17.7722 | 9.9127 | 32.0045 | 12.2547 | 21.5183 | 43 | 353 | 47455 |
| vkitti2 | rain | Geometry-null | 17.6723 | 9.7824 | 32.0661 | 12.1245 | 21.0521 | 35 | 328 | 47559 |
| vkitti2 | rain | V-JEPA21 | 17.5709 | 9.7673 | 31.7536 | 12.1099 | 20.8805 | 36 | 327 | 47567 |
| vkitti2 | sunset | GeneralTrack | 24.9158 | 15.6998 | 39.6473 | 18.6929 | 28.9778 | 93 | 318 | 43929 |
| vkitti2 | sunset | Geometry-null | 24.3367 | 15.1993 | 39.0738 | 18.0988 | 28.0973 | 89 | 294 | 44281 |
| vkitti2 | sunset | V-JEPA21 | 24.3890 | 15.1869 | 39.2716 | 18.0970 | 28.1570 | 85 | 295 | 44285 |
| motsynth | cloudy | GeneralTrack | 0.6916 | 0.6348 | 1.0475 | -17.0908 | 0.1204 | 258 | 50353 | 292601 |
| motsynth | cloudy | Geometry-null | 0.7087 | 0.6151 | 1.0863 | -14.4530 | 0.1154 | 223 | 42614 | 292643 |
| motsynth | cloudy | V-JEPA21 | 0.7280 | 0.6129 | 1.1326 | -14.2752 | 0.1245 | 214 | 42098 | 292647 |
| motsynth | dark | GeneralTrack | 0.6965 | 0.5977 | 0.8874 | -13.6183 | 0.2181 | 371 | 37379 | 268715 |
| motsynth | dark | Geometry-null | 0.7144 | 0.5703 | 0.9775 | -11.4336 | 0.2317 | 279 | 31459 | 268834 |
| motsynth | dark | V-JEPA21 | 0.7070 | 0.5773 | 0.9441 | -11.3854 | 0.2172 | 281 | 31333 | 268828 |
| motsynth | fog | GeneralTrack | 0.3396 | 0.2336 | 0.6533 | -15.0437 | 0.0819 | 127 | 57275 | 378900 |
| motsynth | fog | Geometry-null | 0.3515 | 0.2217 | 0.7303 | -12.5799 | 0.0753 | 101 | 47920 | 378937 |
| motsynth | fog | V-JEPA21 | 0.3580 | 0.2194 | 0.7860 | -12.4467 | 0.0773 | 98 | 47417 | 378938 |
| motsynth | rain | GeneralTrack | 0.6998 | 0.8406 | 0.6705 | -18.0581 | 0.2029 | 519 | 53109 | 288286 |
| motsynth | rain | Geometry-null | 0.6984 | 0.8134 | 0.6850 | -15.2433 | 0.2061 | 435 | 44945 | 288382 |
| motsynth | rain | V-JEPA21 | 0.7277 | 0.8064 | 0.7609 | -15.0545 | 0.2291 | 400 | 44409 | 288406 |
| motsynth | snow | GeneralTrack | 0.3231 | 0.1150 | 1.1545 | -16.9981 | 0.0320 | 40 | 42691 | 250712 |
| motsynth | snow | Geometry-null | 0.3305 | 0.1114 | 1.2148 | -14.6844 | 0.0327 | 34 | 36886 | 250720 |
| motsynth | snow | V-JEPA21 | 0.3246 | 0.1099 | 1.2137 | -14.6713 | 0.0313 | 32 | 36851 | 250724 |

## 逐域增益审计

| dataset | domain | ΔHOTA final-baseline | ΔMOTA final-baseline | JEPA ΔHOTA final-null | JEPA ΔMOTA final-null |
|---|---|---:|---:|---:|---:|
| ua_detrac | cloudy | +7.2828 | +26.8296 | +0.5489 | +1.1193 |
| ua_detrac | night | +7.8385 | +20.1242 | +3.2250 | +4.0909 |
| ua_detrac | rainy | +5.1494 | +10.4698 | +2.3672 | +1.9132 |
| bdd100k | dawn_dusk | +0.0866 | -0.1688 | +0.3330 | +0.0501 |
| bdd100k | night | +0.5055 | +0.1095 | +0.5563 | +0.0224 |
| bdd100k | overcast | -0.0547 | +0.1003 | +0.0264 | +0.0255 |
| bdd100k | rainy | +0.3260 | +0.3390 | +0.2943 | +0.0266 |
| vkitti2 | fog | -0.0953 | -0.1302 | -0.0179 | -0.0165 |
| vkitti2 | morning | -0.4966 | -0.5739 | +0.0822 | -0.0183 |
| vkitti2 | overcast | -0.6040 | -0.6125 | +0.0292 | -0.0018 |
| vkitti2 | rain | -0.2013 | -0.1448 | -0.1014 | -0.0146 |
| vkitti2 | sunset | -0.5268 | -0.5959 | +0.0523 | -0.0018 |
| motsynth | cloudy | +0.0364 | +2.8156 | +0.0193 | +0.1778 |
| motsynth | dark | +0.0105 | +2.2329 | -0.0074 | +0.0482 |
| motsynth | fog | +0.0184 | +2.5970 | +0.0065 | +0.1332 |
| motsynth | rain | +0.0279 | +3.0036 | +0.0293 | +0.1888 |
| motsynth | snow | +0.0015 | +2.3268 | -0.0059 | +0.0131 |

## 数据集宏平均

宏平均为同一数据集各域等权平均；`ALL` 为 17 个域等权平均，不按帧数或检测数加权。

| scope | method | HOTA | DetA | AssA | MOTA | IDF1 | avg IDSW | avg FP | avg FN |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ua_detrac | GeneralTrack | 33.5983 | 29.1627 | 39.5204 | 18.7448 | 44.6072 | 2532.3333 | 12311.0000 | 24638.3333 |
| ua_detrac | Geometry-null | 38.3082 | 33.1717 | 45.1471 | 35.5115 | 52.3358 | 1691.3333 | 4934.3333 | 24986.0000 |
| ua_detrac | V-JEPA21 | 40.3552 | 33.5779 | 49.5727 | 37.8860 | 55.4662 | 1164.0000 | 4392.6667 | 25029.3333 |
| bdd100k | GeneralTrack | 35.1767 | 25.7048 | 48.7095 | 26.4246 | 40.2450 | 596.5000 | 2829.5000 | 41538.7500 |
| bdd100k | Geometry-null | 35.0900 | 25.4498 | 48.9353 | 26.4885 | 40.1287 | 553.0000 | 2643.2500 | 41740.5000 |
| bdd100k | V-JEPA21 | 35.3925 | 25.4383 | 49.8174 | 26.5196 | 40.3849 | 527.2500 | 2628.5000 | 41762.5000 |
| vkitti2 | GeneralTrack | 22.5055 | 13.3212 | 38.5006 | 15.8613 | 25.8539 | 63.0000 | 330.6000 | 45490.6000 |
| vkitti2 | Geometry-null | 22.1119 | 12.9783 | 38.1538 | 15.4604 | 25.1032 | 55.0000 | 312.8000 | 45735.0000 |
| vkitti2 | V-JEPA21 | 22.1207 | 12.9636 | 38.2168 | 15.4498 | 25.1208 | 53.4000 | 312.4000 | 45742.8000 |
| motsynth | GeneralTrack | 0.5501 | 0.4843 | 0.8826 | -16.1618 | 0.1311 | 263.0000 | 48161.4000 | 295842.8000 |
| motsynth | Geometry-null | 0.5607 | 0.4664 | 0.9388 | -13.6788 | 0.1322 | 214.4000 | 40764.8000 | 295903.2000 |
| motsynth | V-JEPA21 | 0.5691 | 0.4652 | 0.9675 | -13.5666 | 0.1359 | 205.0000 | 40421.6000 | 295908.6000 |
| ALL | GeneralTrack | 20.9871 | 15.2550 | 30.0185 | 9.4371 | 24.9839 | 683.1176 | 17100.6471 | 114513.9412 |
| ALL | Geometry-null | 21.6851 | 15.7964 | 30.9791 | 13.0233 | 26.1000 | 507.8235 | 13574.3529 | 114712.4118 |
| ALL | V-JEPA21 | 22.1226 | 15.8606 | 31.9947 | 13.4796 | 26.7189 | 405.4706 | 13374.2353 | 114729.1176 |

## JEPA-specific 宏平均增益

| scope | JEPA ΔHOTA (V-JEPA21 - null) | JEPA ΔMOTA (V-JEPA21 - null) | total ΔHOTA vs baseline | total ΔMOTA vs baseline |
|---|---:|---:|---:|---:|
| ua_detrac | +2.0470 | +2.3745 | +6.7569 | +19.1412 |
| bdd100k | +0.3025 | +0.0311 | +0.2158 | +0.0950 |
| vkitti2 | +0.0089 | -0.0106 | -0.3848 | -0.4115 |
| motsynth | +0.0084 | +0.1122 | +0.0189 | +2.5952 |
| ALL | +0.4375 | +0.4562 | +1.1356 | +4.0425 |

## 结果文件

- V-JEPA21 final outputs: `/data1/LWR/vranlee/SERVER_ONLY/avis/GeneralTrack-main/outputs/official_vjepa21_other_datasets_l384_w16_causalbox_predictor_high_fallback_duplicate_geom_jepa_margin000_histlast`
- Geometry-null outputs: `/data1/LWR/vranlee/SERVER_ONLY/avis/GeneralTrack-main/outputs/official_vjepa21_other_datasets_l384_w16_causalbox_predictor_high_fallback_duplicate_geom_geometry_null_iou020`
- V-JEPA cache and sidecars: `/data1/LWR/vranlee/SERVER_ONLY/avis/JEPA/outputs/official_vjepa21_cache`
- UA-DETRAC prior audit: `/data1/LWR/vranlee/SERVER_ONLY/avis/JEPA/reports/UA-DETRAC_VJEPA21_Online_TBD_ICLR_audit.md`

## 审计边界

geometry-null 只能隔离本次 paired protocol 中的 JEPA 特征增益，不能证明所有总体增益都来自 JEPA；若作为 ICLR 正式论文，还应补充多 seed、更多公开 tracker baseline 和统计显著性。
