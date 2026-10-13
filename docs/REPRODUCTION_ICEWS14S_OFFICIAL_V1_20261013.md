# ICEWS14s 新基线复现：ICEWS14s-official-v1-20261013-200900

训练及最终 TEST 已完成，退出码 0。耗时 44.39 分钟。
代码提交：`bedb5b1c51d26c6cb32366b5d98beb3d9dd0a9f1`；协议：`official-dgar-v1`；GPU：0（RTX 4090）。
参数为作者默认值；没有显式 seed，没有方法模块，没有从旧实验恢复。

论文来源：[DGAR ACL 2025](https://aclanthology.org/2025.acl-long.537/)，表 1，五次运行平均；本次为一次运行。

| 指标 | 论文（%） | 本次（%） | 差值（百分点） |
|---|---:|---:|---:|
| Current MRR | 58.59 | 51.4229 | -7.1671 |
| Average MRR | 50.12 | 50.0060 | -0.1140 |
| Average Hits@1 | 39.36 | 39.1579 | -0.2021 |
| Average Hits@10 | 70.48 | 70.1332 | -0.3468 |

Pooled filtered entity MRR：0.495069；Hits@1/3/10：0.385511/0.558355/0.697551。
Pooled 不能替代论文 Average macro。

历史 Average 覆盖 363 个快照；Current 仅有 18 条事实，含逆向共 36 个 query。
Current 受少数 query 排名变化影响较大；判断整体复现优先看 Average 与历史 Hits。

另外保留最后阶段 diffusion 更新前的诊断 MRR：61.0585%。该诊断使用不同的模型状态、HCP 抽样和随机流，不替代最终 TEST Current。
结论：历史 Average 和 Hits 已接近论文；本次最终 Current 仍存在差距，不能宣称所有论文指标均已复现。

本次遵循作者的训练完成后回顾式评估历史传递方式；并非所有支路都采用严格在线历史。
单次结果不能证明某一个被恢复的公共改动单独造成了此前下降，也不用于判定新方法收益。

完整日志：`/data1/zihao/DGAR-code/.worktrees/original-baseline/logs/ICEWS14s-official-v1-20261013-200900.log`。
科学参数、source/data/cache SHA256 见 config.json；原代码快照保存在对应 logs/*.source.tar。

## 结果解读

历史 Average MRR 与论文相差 -0.1140 个百分点，Average Hits@1/10 分别相差
-0.2021/-0.3468 个百分点。就本次结果而言，历史平均性能已基本对齐论文，
此前较低的 Original 分数不应继续作为这个新基线的固定参考。

最终 Current MRR 相差 -7.1671 个百分点，因此没有复现全部论文指标。
Current 仅 36 个双向 query，本次单次运行与论文五次均值不同；这限制了对差距
的归因，不能直接断言差距全部来自随机性。公开代码同时含更新前诊断和最终
全流程测试，论文没有指出所报告 Current 对应哪个打印区段。本次固定使用
Final full-pipeline TEST 的 Current，并保留诊断数值，避免混用口径。

更新前诊断的 MRR 为 61.0585%，最终 Current 为 51.4229%。两者之间发生了
最后阶段 diffusion 更新、模型 best checkpoint 加载、引导模型/模式变化，
且 HCP 抽样范围和随机流不同，不能将两者之差归因于某一个单独模块。

本次没有多跑 seed、没有调整超参数以追论文分数，也没有改 FAHC/ECRI/QCDR。
以后新方法从该基线派生，关闭方法时保持协议一致；公平方法对照双方应显式
采用同一 seed、GPU、数据/cache 和科学参数。

## 完成及 checkpoint 审核

训练覆盖 364 个阶段，最终 TEST 覆盖 time_idx=1..364；进程退出码 0。
latest.pt 的 completed_stage=364、version=2；source/data/cache provenance 与
config.json 相同。model.pt（101 tensors）及 diffusion.pt（90 tensors）所有
浮点值有限，并与 latest 中的 best 权重逐元素一致；两个 Adam 状态也全部有限。
Adam 最后步数：predictor 3351..3358，diffusion 3113..3121，已保留末次更新状态。
训练及监督/汇总进程均已退出，两张 GPU 已释放。

## 可核对的实验记录

- `/data1/zihao/DGAR-code/.worktrees/original-baseline/experiment_runs/ICEWS14s-official-v1-20261013-200900/summary.json`
- `/data1/zihao/DGAR-code/.worktrees/original-baseline/experiment_runs/ICEWS14s-official-v1-20261013-200900/paper_comparison.csv`
- `/data1/zihao/DGAR-code/.worktrees/original-baseline/experiment_runs/ICEWS14s-official-v1-20261013-200900/checkpoint_audit.json`
- `/data1/zihao/DGAR-code/.worktrees/original-baseline/experiment_runs/ICEWS14s-official-v1-20261013-200900/execution_environment.json`
- `/data1/zihao/DGAR-code/.worktrees/original-baseline/experiment_runs/ICEWS14s-official-v1-20261013-200900/data_statistics.json`
- `/data1/zihao/DGAR-code/.worktrees/original-baseline/experiment_runs/ICEWS14s-official-v1-20261013-200900/config.json`

本提交只存档报告；训练日志、checkpoint、数据和完整实验目录均不进入 Git。

Source SHA256：`6eda8c0c9b3426cca861087a715309c8a7c5c1efa33d6d5ec5b1bdaf6c681538`

Data SHA256：`a81dea25dfbbd5b7420563e9291e356d8e14971cab73a409ed2efbcf514aa038`

History SHA256：`6185eadbb881aacd9cddf566fd1cfcae9b0adacaf3f81a5adc8db032fd6eb804`
