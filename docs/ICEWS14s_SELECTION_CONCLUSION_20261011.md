# ICEWS14s：补齐 coverage 与机制检查后的阶段结论

日期：2026-10-11。分析时工作区 HEAD：`58ad709`。
本次只读取日志、代码和已有结果，执行标准库统计并归档；没有启动训练、
模型评估或 GPU 任务。训练由用户在实验服务器上完成。

## 1. 结论与工作安排

当前实现、当前设置下，risk、coverage、risk+coverage 都未超过 random B4096
或同批次本地 original DGAR 的最终旧快照宏平均 MRR。覆盖单独选择的收益
也未得到支持。风险并非完全没有信号，但两个小批量平均风险对正式 DGAR
更新后历史模式 CE 退化的预测，在三个诊断阶段中只有一个达到预定相关门槛。

因此，结束这版选择器的参数探索，保留 original/random 基线和负结果。
不根据本轮结果继续扫 eta、k、预算、候选倍数，也不安排两批平均选择器的
完整训练。KD 没有在本轮启用，这些结果不能用于否定 KD 或所有可能的
遗忘感知方法。如果以后研究 KD，应独立使用 random+KD 对照，不能把
KD 的收益直接归给本轮没有显示收益的选择器。

这项决定是依据现有结果停止投入的工程决定，不是多种子显著性检验，也不
宣称已经查明退化的唯一原因。

## 2. 输入、完成状态和可比性

完整训练日志：

- `logs/ICEWS14s-dgar-seed1-v3.log`
- `logs/ICEWS14s-random-B4096-seed1-v3.log`
- `logs/ICEWS14s-fahc-B4096-eta0.1-seed1-v3.log`
- `logs/ICEWS14s-risk-B4096-seed1-v3.log`
- `logs/ICEWS14s-coverage-B4096-seed1-v3.log`

五份日志均包含一次最终 TEST、一次最终 VALIDATION 和 `Completed:`，没有
Traceback。coverage 日志的参数与之前 random 相比，除选择模式、GPU、日志
路径外，仅新增空的机制阶段参数和默认参考步数；没有启用机制诊断。
主要参数仍为 seed1、k35、共同上限 B4096、候选倍数2、实体覆盖权重1、
eta0.1、KD0、lr0.001、时间线遍历1次、RGCN/diffusion 上限10/10。
coverage 不执行风险探测，只按关系、实体、时间覆盖选择。

original 使用继承的不限额随机 HCP 路径；它的 CLI 中 B4096 和 eta0.1
不作用于该路径。共同上限 B 不等于逐阶段实际事实数相同。

本工作区仍没有实验服务器的 `experiment_runs`。因此没有独立核对各次运行
的源码/数据/缓存指纹、实际事实量、对应检查点或 coverage 的阶段 BWT。
不同 GPU 编号也不构成受控的设备/耗时对比。上述限制必须随结果保留。
本轮只使用同批次 v3 基线；不混入旧 `ICEWS14s-modify_run.log` 的分数。

## 3. 五组最终结果

以下为百分数；历史宏平均为输出的时间索引1..T-2，Current 仅为最后快照。
验证集用于筛选，TEST 用于描述已完成实验，不用于反复调参。

| 方法 | 验证历史 MRR | 测试历史 MRR | 验证 Current MRR | 测试 Current MRR | 测试查询汇总 MRR |
| --- | ---: | ---: | ---: | ---: | ---: |
| original DGAR（本地实现） | 50.2472 | 50.2162 | 66.1097 | 53.9699 | 49.7292 |
| random B4096 | 50.1924 | 50.1166 | 71.3330 | 57.1572 | 49.7434 |
| coverage B4096 | 49.7801 | 49.6831 | 66.2971 | 53.8240 | 49.2754 |
| risk+coverage B4096 | 49.5972 | 49.6452 | 68.5610 | 54.6475 | 49.2632 |
| risk B4096 | 48.1305 | 47.8290 | 67.2582 | 50.6887 | 47.4124 |

coverage 相对 random：验证历史 -0.4123、测试历史 -0.4335、验证 Current
-5.0359、测试 Current -3.3332 个百分点。相对 original：验证历史 -0.4671、
测试历史 -0.5331 个百分点，未达到已建议的验证历史 +0.5 个百分点筛选门槛。

coverage 相对组合：验证历史 +0.1829、测试历史 +0.0379 个百分点。
组合加入风险后没有改善主历史指标，但 Current 比 coverage 高（验证 +2.2639、
测试 +0.8235 个百分点）。不能概括为风险项对所有指标都造成损害。
coverage 与组合测试历史差距很小，不据此声称稳定优劣或统计显著性。

在当前历史 MRR 上，original ≈ random > coverage ≈ risk+coverage > risk。
纯 coverage 和组合都比纯 risk 好，这与覆盖缓解选择集中相符，但没有证明
覆盖本身优于随机；不同方法还可能有不同实际事实量。

可复核数值保存在 `docs/ICEWS14s_SELECTION_CONCLUSION_20261011.csv`。

## 4. 机制诊断的实际输出

输入：`logs/ICEWS14s-mechanism-seed1.log`。
模式为 random B4096、seed1、GPU2，诊断阶段60/90/120，参考更新10步。
控制台完整打印了三个阶段的 current_only 和 dgar_history_only 结果，随后出现
`Mechanism diagnostic complete at snapshot 120` 和 `Completed:`；没有 Traceback。
它完成的是前120阶段诊断，不是完整365阶段训练，也没有最终 TEST/VALIDATION。

| 阶段 | A/B 风险排序 Spearman | 预测 current-only 退化 Spearman | 预测实际 DGAR 退化 Spearman | 实际 DGAR top-k lift |
| --- | ---: | ---: | ---: | ---: |
| 60 | 0.240795 | 0.432437 | 0.185622 | 0.000741 |
| 90 | 0.724434 | 0.673939 | 0.219071 | 0.008552 |
| 120 | 0.945202 | 0.759076 | 0.570116 | 0.027384 |

这里的预测量是 A/B 两批风险的平均；lift 是所选 top-k 的正向截断实体 CE
增量均值减去候选均值，不是 MRR 增益。预测目标为相同当前阶段历史图下的
历史模式实体 CE，不经过完整 diffusion 推理。实际 DGAR 目标使用正常随机
回放训练后恢复的本阶段最佳模型；回放本身可能已经保护部分候选。

已打印的值支持以下判断：

1. 不能说风险分数全是噪声：90/120阶段排序一致性较高，对 current-only
   多步更新后的损失增量有正向预测；60阶段排序一致性较弱。
2. 三个阶段对实际 DGAR 目标的预测相关都低于 current-only 目标。代理的
   预测能力受到正式回放训练/最佳检查点选择等过程影响；这只是与代理适配
   问题相符，不能从相关系数直接认定某个过程是唯一原因。
3. 预定门槛要求至少两个可用阶段同时满足稳定相关>=0.5、重合率高于随机
   期望、实际 DGAR 预测相关>=0.3、lift>0。60/90的实际预测相关已经低于0.3，
   因而无论尚未看到的重合率如何，本次最多一个阶段通过。第120阶段满足
   已打印的相关性/lift条件，但不能据此宣称所有门槛已经通过。
4. 结合完整训练结果，当前没有安排两批平均选择器完整验证的依据。

## 5. 尚未测到的细节与可选 CPU 汇总

控制台没有输出原配置单次 min(128,N) 探测的相关性、fixed-BN结果、top-k
重合率、实际非空候选数、零风险比例、实际/最佳训练步数和逐候选损失。
因此不能直接用 A/B 的较小批量诊断否定原来的128事实探测，也不能说已
确认 BN 是根因或所有阶段的风险都无效。

`docs/ICEWS14s_MECHANISM_CONSOLE_20261011.csv` 只提取控制台的6条已打印
记录；不冒充标准 JSONL 汇总的9行，也不填造 `complete_run` 等字段。

若希望补全解释，只需在实验服务器读取已有文件，无需任何新训练：

```bash
cd /data1/zihao/DGAR-code
python scripts/summarize_fahc_mechanism.py \
  experiment_runs/ICEWS14s_Mechanism_s1_20261011_113716_223058328_475086/logs/stages.jsonl \
  --csv logs/ICEWS14s-mechanism-summary.csv
```

若要同时核对五组正式训练的阶段量：

```bash
python scripts/summarize_fahc_runs.py experiment_runs \
  --csv logs/ICEWS14s-validation-summary-final.csv
```

CPU 汇总可完善机制解释和运行可比性，不是延迟给结论的前置条件。
依照现有完整训练成绩和已打印的预定门槛，本轮先结束当前选择器探索。
