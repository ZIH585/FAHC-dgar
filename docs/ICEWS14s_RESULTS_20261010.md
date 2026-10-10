# ICEWS14s 三组 v3 实验结果分析

日期：2026-10-10。分析代码修订：`c61278d`。
只读取日志、CSV、数据和代码，没有运行训练、评估模型或使用 GPU。

## 1. 证据和完成状态

输入文件：

- `logs/ICEWS14s-dgar-seed1-v3.log`
- `logs/ICEWS14s-random-B4096-seed1-v3.log`
- `logs/ICEWS14s-fahc-B4096-eta0.1-seed1-v3.log`
- `logs/ICEWS14s-validation-summary.csv`

CSV 有 6 条尝试，其中 3 条 `failed`，不进入比较；3 条成功运行均包含
`Completed:`、完整最终 TEST 和 VALIDATION 区段。成功三组的验证完整性及
历史模式完整性标记均为 True，历史快照数为 363。日志没有 Traceback。
三个成功 Namespace 除 `gpu`、`hcp_selection`、`fahc_log` 外一致，均为
seed1、k35、B4096、eta0.1、KD0、lr0.001、每阶段轮次上限10/10、时间线遍历1次。

本工作区只有上述 stdout 和汇总 CSV，没有日志所引用的 `experiment_runs`
目录。因此能够交叉核对最终完整验证结果，但不能独立重算阶段 BWT、每阶段
回放量或核对三组源码/数据/缓存指纹。以下阶段统计以提供的 CSV 为依据。

## 2. 完整推理测试成绩

从 `Final full-pipeline TEST` 到 `Final full-pipeline VALIDATION` 之间提取，
不是读取日志最后一个 MRR。数字均转成百分数。

| 方法 | 旧快照宏平均 MRR | 最后快照 MRR | 按查询汇总 MRR | 旧快照 Hits@1 | 旧快照 Hits@10 |
| --- | ---: | ---: | ---: | ---: | ---: |
| original DGAR | 50.2162 | 53.9699 | 49.7292 | 39.7307 | 70.2630 |
| random B4096 | 50.1166 | 57.1572 | 49.7434 | 39.2073 | 70.7197 |
| FAHC B4096 eta0.1 | 49.6452 | 54.6475 | 49.2632 | 38.8021 | 70.1890 |

FAHC 相对 random：旧快照宏平均 -0.4714 个百分点，查询汇总 -0.4802
个百分点，最后快照 -2.5097 个百分点。当前 seed1 的测试结果没有显示优势。
测试仅用于描述本轮结果，下一配置的筛选仍依据验证集。

## 3. 与论文怎么比较

[论文 Table 1 和 §5.1–5.2](https://aclanthology.org/2025.acl-long.537.pdf)
对 ICE14 报告：Current MRR=58.59，Average MRR=50.12，Average Hits@1=39.36，
Average Hits@10=70.48；论文报告五次运行的平均结果。

因此，应将当前输出的旧快照宏平均与论文 Average 比，将最后快照与 Current
比，而不能把查询汇总 MRR 或 CSV 的验证分数与论文 Current 比。

- original 的 Average MRR 为 50.2162，比论文 50.12 高 0.0962 个百分点。
- random 的 Average MRR 为 50.1166，与论文相差 -0.0034 个百分点。
- original 的 Current 低 4.6201 个百分点；random 低 1.4328 个百分点。

所以“两个基线所有主要指标都远低于论文”不成立。历史平均数值已很接近，
差距主要在最后快照；但这不代表严格的数据与实现复现已完成。

当前 loader 把三个文件按时间片合并，再逐片按行分为80/10/10。最后一片
177 行，训练141、验证18、测试18；测试经逆关系扩展后只有36个查询。
Hits@1 改变一个命中就相差约2.78个百分点。单次 Current 应报告，但不足以
单独判断模型普遍退化，更不能挑最好的种子与论文五次均值比较。

还有一个数据统计差异：[论文 Appendix B.1](https://aclanthology.org/2025.acl-long.537.pdf)
写 ICE14 实体数6869；本地 stat 和全部原始三元组的实体去重计数均为7128，
全部 ID 均有出现。这不是简单的未使用实体 ID 问题。三个原文件的行数
74845/8514/7371 则与论文表中一致。差异可能来自数据处理版本或论文统计误写，
现在无法定因；需核对官方数据内容及最终快照，而不是直接把 stat 改成6869。

## 4. 验证集筛选结论

| 方法 | 完整推理旧快照 MRR | 完整推理最后快照 MRR | 完整推理查询汇总 MRR |
| --- | ---: | ---: | ---: |
| original | 50.2472 | 66.1097 | 49.4343 |
| random | 50.1924 | 71.3330 | 49.4386 |
| FAHC | 49.5972 | 68.5610 | 48.8243 |

FAHC 相对 random 的主验证指标 -0.5952 个百分点，最后快照 -2.7720 个百分点，
查询汇总 -0.6143 个百分点。它没有通过预先建议的“旧快照至少+0.5个百分点、
最后快照下降不超过1个百分点”的筛选门槛。

历史模式诊断也不支持相对 random 更好的保留：

| 方法 | 最终历史模式旧快照 MRR | BWT | 到达至最终正向截断遗忘 |
| --- | ---: | ---: | ---: |
| original | 40.0105 | +3.3672 | 2.9852 |
| random | 40.3210 | +5.2017 | 2.6424 |
| FAHC | 39.2810 | +3.8035 | 3.2080 |

FAHC 的 BWT 虽高于 original，但低于 random；截断遗忘则高于两者。
不能单挑“高于 original 的 BWT”来声称改善。所有 BWT 为正，表示这些
运行的平均最终历史模式分数高于各阶段到达时，但部分阶段仍可退化。
这些是历史模式验证指标，不能直接替代完整推理测试指标或峰值遗忘。

## 5. 预算和耗时

| 方法 | 实际回放事实数中位数 | 保留比例中位数 | 达到 B4096 比例 | GPU 参数 | wall min |
| --- | ---: | ---: | ---: | ---: | ---: |
| original | CSV 未汇总 | 不限额 | 不适用 | 1 | 70.23 |
| random | 1604 | 1.0 | 0 | 2 | 49.12 |
| FAHC | 1695 | 1.0 | 0 | 1 | 67.75 |

两组 controlled 路径均未达到4096上限。结合预算实现，当前阶段已没有
B128 那样的强裁剪。FAHC 实际回放中位数还比 random 高约5.67%，却没有
提高主验证指标。因此继续增加 B 目前没有依据。
同一上限下实际数量不同，这仍不是严格逐阶段等事实数量的选择消融。

FAHC 比旧 B128 运行的查询汇总测试 MRR 提高10.7936个百分点；但是预算、
eta、版本及运行环境均有变化，不能把恢复全部归因于其中一个变量。

GPU1 与 GPU2 的设备型号、负载和运行指纹未随这次文件一起提供，不能把
67.75/49.12 当作严格的策略开销比。成功日志还包含 RTX5090 与当前 PyTorch
不兼容的枚举警告；这不证明实际使用的 GPU1/2 有相同问题，也不能用它解释
所有分数差距。后续应记录实际设备，优先使用已验证兼容且空闲的设备。

## 6. 下一步：只先补一组纯风险

保持当前代码、数据、缓存、seed1、k35、B4096和KD0，只把选择改为 `risk`。
这用于区分“覆盖项拖累”与“当前风险代理缺乏收益”，不承诺会提升。
已有 original 和 random 无需为这一步重新训练。

先进入实际用于这些实验的 dgar_env；下面从当前环境取得 Python，避免
服务器上环境安装路径不同。GPU1只是已成功使用的示例，需确认空闲。

```bash
cd /data1/zihao/DGAR-code/src
export DGAR_PYTHON="$(command -v python)"
mkdir -p /data1/zihao/DGAR-code/logs
setsid bash -c '
set -o pipefail
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
bash /data1/zihao/DGAR-code/scripts/run_fahc_experiment.sh \
  Risk4096 1 ICEWS14s 1 risk 4096 0.1 0 \
  2>&1 \
| tee /data1/zihao/DGAR-code/logs/ICEWS14s-risk-B4096-seed1-v3.log
' < /dev/null > /dev/null 2>&1 &
```

看日志：

```bash
tail -f /data1/zihao/DGAR-code/logs/ICEWS14s-risk-B4096-seed1-v3.log
```

汇总时继续只读验证：

```bash
"$DGAR_PYTHON" /data1/zihao/DGAR-code/scripts/summarize_fahc_runs.py /data1/zihao/DGAR-code/experiment_runs --csv /data1/zihao/DGAR-code/logs/ICEWS14s-validation-summary-next.csv
```

决策：

1. 纯 risk 的完整推理旧验证 MRR 若明显好于组合，并优于 random，才试
   eta0.03或保留纯风险，然后做与 random 配对的seed2/3确认。
2. 纯 risk 只好于组合、仍低于 random：说明不能只靠去掉覆盖解决；先核对
   阶段风险分数与实际历史退化的相关性，必要时补一个 coverage 诊断，不加KD。
3. 纯 risk 也没有收益：暂停扩大eta、候选池、k网格。优先检查风险代理：
   一步探测、只128当前事实、历史模式实体损失，与完整diffusion路径不完全相同。
   阶段JSONL可用于查看风险/覆盖增益、选择稳定性和事实分布。现在这些文件
   没有复制到本工作区，尚不能定位具体是哪一项造成下降。
4. 出现明确正向配置后，冻结再做多种子。KD作为辅助路径随后考虑，并同时
   保留random+KD对照；当前组合没有收益，不优先用KD掩盖选择问题。

## 7. 严格复现与实现修复应独立归档

对当前代码静态复核，现有复查记录中的共同继承问题仍存在：

- `diffusion/model_21.py:277/301`：关系条件从实体表 `initial_h` 按关系ID读取，
  而不是使用已定义的 `emb_rel`。
- `diffusion/difffu_21.py:385–393`：时间步嵌入计算后未进入Transformer。
- `diffusion/difffu_21.py:637`：生成只运行最大时间步的一个posterior更新。

这些出现在共享框架中，不能单独解释FAHC相对random的差距；也不能因为
历史平均接近论文，就认定这些实现符合完整设计或不存在性能影响。
49项CPU测试通过仅覆盖对应的检查，不能证明完整论文复现正确。
更多既有证据见 `docs/FAHC_REVIEW_20261008.md`，本轮没有重跑其数值探针。

若要按论文模型语义修复，应单独建立新版本，核对公式、数据与参数映射，
再统一重跑original/random/FAHC。不能只改FAHC后继续用本轮基线做正式比较。
论文DAR的层数L应核对其与时间演化单元的映射，不能直接等同于 `--n-layers`；
当前将GCN层数直接设3还会遇到固定两项关系向量的索引问题。

## 结论

当前基线的历史平均数值已接近论文，重点不是盲目追高查询汇总MRR。
FAHC本次恢复了此前大幅下降，但仍弱于random，未显示选择收益。
最节省算力的下一步是保持版本不变，只补一组纯risk；同时核对论文Current
的多次均值、数据实体统计和共享实现。现有三组结果应保留，不能宣称已得到
FAHC稳定提升或严格完成论文复现。
