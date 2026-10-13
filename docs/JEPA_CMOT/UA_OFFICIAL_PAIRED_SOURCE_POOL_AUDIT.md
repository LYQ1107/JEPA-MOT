## UA官方配对读数：源样本池核验完成

六个seed17源head均完成2,000×64训练。实际encoder/predictor cache的boxes/frame/video/track/segment/image/offset逐项一致，predictor负例排序proxy与encoder features一致；重建SourcePairs的42,507 fit、10,111 validation样本中history/positive/ordered negatives逐条相同。实际六个checkpoint/cache SHA、source视频划分、gaps1/2/4/8和预算均匹配。详见UA_OFFICIAL_PAIRED_SOURCE_POOL_AUDIT.json；seed17采样索引SHA是从实际样本列表重建，未冒充保存的逐batch trace。

源held-out rank1 encoder CE .875762/pure .799245/joint .870141，masked predictor CE .777767/pure .806267/joint .791465。不同latent空间、当前像素信息与frozen参数/算力不同，不能跨读数把source rank/error当作独立MOT增益。未访问targetGT做训练；sealed未用。完整两读数17开发视频/25,460 native帧仍待pipeline完成，已注册自动完整分析（UA_OFFICIAL_PAIRED_ANALYSIS_REGISTRATION.json），不以源pool审计替代目标域性能或持续学习证据。
