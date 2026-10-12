# 失败证据和诊断

## 工程 smoke v1：缺失 timm

Observation：UA 原始 GeneralTrack 跟踪和 TrackEval 成功，官方 V-JEPA 构造失败 `ModuleNotFoundError: timm`。

处理：在项目的 `runtime/site-packages` 安装 timm 0.9.16，保留 v1 日志。V100 使用 FP16，缓存记录实际精度。

结果：UA v2、BDD v1 的官方 predictor 缓存、在线跟踪和评测闭环均通过。此结果仅证明工程可执行。

## 历史 temporal adapter

旧 source temporal adapter 的 deploy_alpha=0，source held-out 直接预测更差。该证据不属于本次新训练；不加载为有效 warm start。

下一假设：无约束替换历史外观可能损害原有身份区分；共享零初始化残差预测器有利于分离训练目标效果与随机初始化扰动。需 prediction / ranking / combined 同容量对照验证。

## 数据协议问题

UA：MVI_39501 XML/图片 540 帧，历史 prepared 80 个抽样帧。不能将该输入当作完整原生视频。

BDD：train/val 文件数量一致，必须审计视频集合；源 detector 配置使用 clear_all，存在与新 source 测试划分重合的风险。需要根据训练视频清单重建冲突的源模型/缓存。

下一步：冻结完整视频划分之前先完成覆盖与训练 provenance 审计；不通过删除困难视频或更改测试分母解决。
