JEPA + MOT 接续工作区

本地目录：/data1/liuyeqiang/JEPA_MOT_resume_20261012
NAS 挂载：/data1/liuyeqiang/nas/A100（只读）

JEPA/                  JEPA 实验代码及官方 V-JEPA 2.1 vendor 源码
JEPA/checkpoints/      官方 V-JEPA 2.1 权重及历史 source temporal adapter
GeneralTrack-main/     最新实际跟踪器代码、TrackEval 及 GeneralTrack 权重
DA-MOT/darth-main/     原 DA-MOT 主体及持续学习相关代码
reference_reports/     从 A100 恢复的最终实验报告
results/               本轮结果；最新 MOTSynth 的 40 组指标复算全部一致
runtime/site-packages/ 本项目独立补充依赖，不修改共享 Conda 环境
workspace_manifest.json 权重大小、SHA256 校验与 NAS 链接清单

数据集、prepared annotation 和已有特征缓存通过软链接读取 NAS。
新实验输出写入本地，原服务器归档不覆盖。
archive_paths.py 用于将旧 /data1/LWR、/data3/testdata 和 /data2/usr_for_deadline
路径及其绝对软链接解析为当前 NAS 路径。GeneralTrack 读图和 V-JEPA 缓存
构建入口已经接入这个解析器。

运行示例：
  ./run_python.sh GeneralTrack-main/tools/track_damot_domain.py --help
  ./run_python.sh JEPA/build_generaltrack_vjepa21_cache.py --help
  ./run_python.sh reproduce_latest_metrics.py

历史聊天：
  设计JEPA结合MOT实验：01a0537f-313a-7f33-af56-3de76a70ca91
  规划并审计 DA-MOT JEPA 集成：01a056f1-71a7-7470-a73c-0621251b0061
  最近实验完成：2026-09-09；最近技术问答：2026-09-15

最新冻结模型实验：
  GeneralTrack + 官方 V-JEPA 2.1，四个数据集。
  MOTSynth 必须使用 mapped source_detector_thr02，排除旧 dense 1798 帧协议。
  MOTSynth full HOTA：baseline 16.7432，geometry-null 19.4460，tuned JEPA 19.4429。
  因此该数据集相对 baseline 的总体增益不能直接归因于 JEPA。

本聊天用户选择的后续路线：轨迹级 JEPA 持续学习，复用官方 V-JEPA 2.1 表征。
新训练尚未启动；先完成本地代码和 checkpoint 整理。
历史 source temporal adapter 的 deploy_alpha=0，旧 source 评估表明直接使用
学习预测输出会下降。它保留作参考，不应未经验证作为有效 warm start。

此工作区完成文件恢复和评测复现；完整跟踪/训练仍需单独完成 GPU smoke。
