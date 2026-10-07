# FAHC-DGAR 实验计划与执行命令

日期：2026-10-07。适用代码版本：`full-hcp-v3`。
以下命令由你在 GPU 空闲后执行；编写本计划时没有启动训练。

## 1. 先跑哪组，为什么

优先候选是 **risk+coverage，k=35，B=4096，eta=0.1，KD=0**。
保持学习率、模型、实体覆盖权重、候选池和训练轮次不变。
这是依据已有日志提出的假设，尚无新训练结果证明它能超过 DGAR。

已有 ICEWS14s、seed=1 的完整推理测试结果：

| 方法 | filtered entity MRR，按查询汇总 | 旧快照宏平均 | 最后快照 |
| --- | ---: | ---: | ---: |
| original DGAR | 0.497329 | 0.501341 | 0.612836 |
| risk+coverage，B=128，eta=1，KD=0 | 0.384696 | 0.387304 | 0.397612 |

从旧 FAHC 日志实际选中的 HCP 重建，完整唯一事实数中位数为 1895，
范围为 40–2603；B=128 的保留比例中位数仅 6.75%，裁剪了 360/363 个选择阶段。
原 DGAR 正常 HCP 路径没有这个事实限额。
后期阶段的风险项占总贪心增益的中位数约 14.10%，eta=1 下覆盖项更强。
因此先消除强裁剪，再降低覆盖权重，比立即加 KD 或改学习率更有依据。
这些差异不能分别解释多少 MRR 损失；详见 `docs/ICEWS14s_FAHC_DIAGNOSIS.md`。

4096 大于上述**旧运行**的最大联集，但新运行可能选择不同 HCP，必须查看
`retained_fraction_median` 和逐阶段数量，不能保证永远完全不限额。

v3 修复了一个消融混杂：对同一组选中 HCP，先按时间索引统一顺序，再做
带种子的预算裁剪。同一个集合和种子现在产生相同回放事实，不受贪心选中
顺序影响。旧 v2 结果用于诊断；正式表格中的所有方法都使用 v3。

## 2. 固定设置与公平比较范围

运行器统一使用：

| 设置 | 值 |
| --- | --- |
| seed，初筛 | 1 |
| k / `--reply_batch` | 35 |
| 候选池倍数 | 2 |
| 探测步数 / 当前探测事实数 | 1 / 128 |
| 探测学习率 | 未指定 probe-lr，使用主学习率参数 lr |
| 风险打分批大小 | 256 |
| 实体覆盖权重 / 时间分箱大小 | 1 / 10 |
| 主学习率 / task-weight | 0.001 / 0.7 |
| mu / mu_r | 1 / 1 |
| RGCN / diffusion 每阶段轮次上限 | 10 / 10，保留原早停 |
| 完整时间线遍历次数 | 1 |
| 隐藏维度 / 历史长度 | 200 / 3 |
| KD，初筛 | 0；启用时 T=2，正确且 T=1 置信度至少 0.5 |

其他参数保持当前代码默认，实际值完整保存到每次运行的 `config.json`。
不要用减少轮次、更换静态图、修改 replay loss 权重来加速某一个对比方法。
`mu=0` 仍保留原实现的一项实体回放 CE，不能作为 FT。
`B` 是逆关系查询扩展**之前**的唯一 HCP 三元组数，不是总训练查询数。

所有对比保持同一代码、数据划分、过滤规则和 `history_snap_v2` 缓存。
运行器保存源码、数据、缓存 SHA256、Git 修订号、参数和依赖版本。
比较同一数据集时确认这些指纹一致；运行过程中不要改代码或重建缓存。

**同一 k 和预算上限 B，不必然有同样实际事实数。** HCP 不足、重复事实和
不同选择集合都可能造成数量不同。B=4096 的实验主要诊断恢复较完整回放后的
性能；它不能单独证明严格等事实数量下选择器有效。正式预算实验需要报告
每阶段实际事实数；只有逐阶段数量相等，才称为“相同实际事实预算”。
B=1024 也只是更可能饱和的共同上限，不能预先假定所有阶段恰好使用 1024。

当前缓存是历史文件复用，B 不限制历史总存储；不能据此声称固定字节内存。
当前数据协议把原 train/valid/test 时间片序列合并，再逐快照按行切为
80/10/10；应报告为快照内留出评估。不要称为完全留出的未来预测协议。
当前 filtered ranking 使用对应评估切分的真答案，应保持所有方法一致。

## 3. 最省运行次数的顺序与决策

每个阶段独立启动，完成后先看验证集汇总，再决定是否执行后续阶段。
`core` 不是自动执行所有消融的大网格。

| 阶段 | 方法 | 新运行数，seed=1 | 目的 |
| --- | --- | ---: | --- |
| **core，必做** | original；random B4096；risk+coverage B4096 eta0.1 | 3 | 判断能否恢复性能，以及相对随机是否有价值 |
| budget，定位下降或写预算表 | random、risk+coverage eta0.1，各 B128/B1024 | 4 | 隔离随机方法的预算效应，并做共同预算选择比较 |
| selection，按需要补 | risk+coverage eta1；risk；coverage；recent，均 B4096、KD0 | 4 | 分清风险、覆盖和最近性作用 |
| KD，选择器有效后 | random+KD；risk+coverage+KD，均 B4096、lambda0.03 | 2 | 与已有无 KD 两组构成 2×2 消融 |
| 多种子确认 | original、random、冻结的最佳方案，seed2/3 | 6 | 与相同配置 seed1 合成 3 种子结果 |
| 第二数据集 | 固定配置的 original、random、最佳方案 | 先 3，再补种子 | 检查跨数据集效果 |

### core 后按什么决定

配置筛选只读取**验证集**，主指标预先定为完整 DGAR 推理的最终旧快照
宏平均 `val_full_past_MRR`，主要比较同预算 random。
同时查看完整推理的最后快照 `val_full_current_MRR` 和按查询汇总的
`val_full_MRR`。历史模式 BWT 是独立诊断，不代替完整推理结果。

建议在运行前固定一个实用筛选门槛：旧快照相对 random 至少 +0.005
（0.5 个百分点），最后快照下降不超过 0.01（1 个百分点）。这是节省算力
的筛选规则，不是统计显著性标准，也不能跑完后按结果更改门槛。

1. **random4096 接近 original，FAHC 又优于 random**：先补纯 risk 消融。
   确认覆盖是否额外有用，再考虑 KD 和多种子。
2. **FAHC 优于 random，但仍低于 original**：只能先说共同上限下表现更好；
   还不能说超过原 DGAR。检查逐阶段事实数量，并补 B1024 的匹配比较。
3. **random4096 本身远低于 original**：检查保留比例、实际数量、缓存及
   训练/推理输入规模。先做 budget；不要把全部差距归给风险选择。
4. **FAHC 低于 random4096**：先跑纯 risk，然后 eta0.03 或 eta0.3
   中有依据的一组。先不加 KD。若风险和组合均无改善，停止扩张参数网格，
   分析风险代理与正式 diffusion 路径的对应性。
5. **提升小于筛选门槛但为正**：视算力做 2 个额外种子确认；不能因为 seed1
   有小幅正值就声称稳定提升。

原日志训练约 35 分钟、最终测试约 1–2 分钟/组。core 可先按约 2 小时排期；
大预算、额外验证、GPU 负载和 KD 会改变耗时，这不是新运行的时间保证。

## 4. 如何启动、查看日志与汇总

下面命令从仓库根目录运行。GPU 0 只是示例，选择空闲且当前进程可见的 GPU。
运行器不会因为 CUDA 不可用而悄悄改成 CPU 训练。它会阻止本运行器同时
向同一个 GPU 提交两组，但不会识别其他人直接启动的训练。

### 4.1 先检查命令，完全不训练

```bash
cd /data1/zihao/DGAR-code
bash scripts/run_fahc_experiment.sh --dry-run A4096_E01 0 ICEWS14s 1 risk+coverage 4096 0.1 0
```

`--dry-run` 只校验参数、文件和缓存并打印命令，不创建实验目录、不导入
模型或访问 CUDA。默认 Python 是 `/data1/zihao/miniconda3/envs/dgar_env/bin/python`；
换环境时设置 `DGAR_PYTHON` 为另一个可执行 Python 的绝对路径。

### 4.2 推荐首先执行：core 三组串行

```bash
cd /data1/zihao/DGAR-code
mkdir -p experiment_runs
setsid bash scripts/run_fahc_stage.sh core 0 ICEWS14s 1 < /dev/null > experiment_runs/core-ICEWS14s-seed1.launch.log 2>&1 &
echo $!
```

这条后台命令依次跑 original、random4096、FAHC4096 eta0.1；任一组失败会
停下，不继续下一组。它不需要在多行参数末尾写续行反斜杠。

实时查看：

```bash
tail -f /data1/zihao/DGAR-code/experiment_runs/core-ICEWS14s-seed1.launch.log
```

按 `Ctrl+C` 退出 tail 不会停止后台实验。每组还有独立的 `logs/train.log`。
Python 使用 `-u`，缓解后台重定向造成的日志缓冲问题。

汇总验证集：

```bash
cd /data1/zihao/DGAR-code
/data1/zihao/miniconda3/envs/dgar_env/bin/python scripts/summarize_fahc_runs.py experiment_runs --csv experiment_runs/summary.csv
```

只比较 `state=finished`、`full_validation_complete=True` 的完整推理结果。
BWT 还要求 `past_complete=True`。未完成的运行不能进入最终结果表。
汇总程序不读取 stdout 中的测试集 MRR 来选参数。

### 4.3 需要预算消融时

```bash
setsid bash scripts/run_fahc_stage.sh budget 0 ICEWS14s 1 < /dev/null > experiment_runs/budget-ICEWS14s-seed1.launch.log 2>&1 &
```

该阶段四组的 eta 都是 0.1，因此 A128/A1024/A4096 之间只改变预算。
要复现旧配置的 v3 对照，可另外跑一组；它不在默认队列里：

```bash
bash scripts/run_fahc_experiment.sh A128_E1 0 ICEWS14s 1 risk+coverage 128 1 0
```

### 4.4 先只补一组纯 risk，比直接跑四组更省

```bash
setsid bash scripts/run_fahc_experiment.sh Risk4096 0 ICEWS14s 1 risk 4096 0.1 0 < /dev/null > experiment_runs/risk-ICEWS14s-seed1.launch.log 2>&1 &
```

如果要完整选择消融，`selection` 一次跑 eta1、纯 risk、纯 coverage、recent：

```bash
setsid bash scripts/run_fahc_stage.sh selection 0 ICEWS14s 1 < /dev/null > experiment_runs/selection-ICEWS14s-seed1.launch.log 2>&1 &
```

如果已单独跑过 Risk4096，避免重复，手动只补缺少的三组：

```bash
bash scripts/run_fahc_experiment.sh A4096_E1 0 ICEWS14s 1 risk+coverage 4096 1 0
bash scripts/run_fahc_experiment.sh Coverage4096 0 ICEWS14s 1 coverage 4096 0.1 0
bash scripts/run_fahc_experiment.sh Recent4096 0 ICEWS14s 1 recent 4096 0.1 0
```

### 4.5 确认选择器有效后，再试 KD

以下队列假定暂时最佳无 KD 配置仍为 B4096、eta0.1：

```bash
setsid bash scripts/run_fahc_stage.sh kd 0 ICEWS14s 1 < /dev/null > experiment_runs/kd-ICEWS14s-seed1.launch.log 2>&1 &
```

两组分别是 random+KD 和 FAHC+KD，lambda=0.03、T=2、置信阈值=0.5。
查看 `KD_active_fraction_median` 和 `kd_mean_loss`。如果几乎全部门控为零，
先不要扫 lambda；教师实际没有提供有效信号。可把阈值降至 0.2 做**单独**消融，
同时关注错误教师风险。门控正常且 KD 有改善时，才补 lambda=0.1。
KD 教师 logits 的 CPU 缓存规模随实际回放事实数×实体数增长，记录额外内存和时间。

### 4.6 冻结配置，补 seed2/3

如果最佳方案仍是 core 中的 B4096、eta0.1、KD0，执行：

```bash
setsid bash -c 'for seed in 2 3; do bash scripts/run_fahc_stage.sh core 0 ICEWS14s "$seed" || exit $?; done' < /dev/null > experiment_runs/confirm-ICEWS14s-seed2-3.launch.log 2>&1 &
```

这样新增 6 组，合并同版本、同配置的 seed1 后得到三组方法各 3 种子。
如果最佳参数已经改变，**不要用 core 来验证另一个最佳配置**。
把每个 seed 的候选调用换成冻结的实际参数。例如最佳若加入 lambda0.03，
用以下串行队列（这里只是条件示例，不推荐提前选 KD）：

```bash
setsid bash -c '
for seed in 2 3; do
  bash scripts/run_fahc_experiment.sh DGAR 0 ICEWS14s "$seed" original 4096 0.1 0 || exit $?
  bash scripts/run_fahc_experiment.sh R4096 0 ICEWS14s "$seed" random 4096 0.1 0 || exit $?
  bash scripts/run_fahc_experiment.sh Best_KD003 0 ICEWS14s "$seed" risk+coverage 4096 0.1 0.03 || exit $?
done
' < /dev/null > experiment_runs/confirm-KD-ICEWS14s-seed2-3.launch.log 2>&1 &
```

正式论文建议 5 个种子（再补 4/5），每个方法采用相同种子列表。
冻结方案后再汇总完整推理测试结果，报告均值±样本标准差及逐种子配对差值。
seed1 用于筛选，不能单独作为确认性证据；也应单独报告新增种子的结果。
快照来自同一时间线且相关，不能把数百快照当作数百个独立随机重复。
三种子只提供初步稳定性证据；不要仅凭均值正向就声称统计显著。

### 4.7 再迁移到 ICEWS18

若冻结配置仍为 core 配置，先固定参数执行：

```bash
setsid bash scripts/run_fahc_stage.sh core 0 ICEWS18 1 < /dev/null > experiment_runs/core-ICEWS18-seed1.launch.log 2>&1 &
```

若最佳已改变，按上一节逐方法命令将数据集改为 ICEWS18，并使用冻结参数。
先不依据 ICEWS18 测试集调参；同样需要补成 3/5 种子才讨论稳定性。
重新检查预算保留比例，4096 对 ICEWS18 未必是宽松上限。
论文多数据集比较应事先规定同一方案与预算规则，不能按测试结果挑数据集。

## 5. 可选调参：一次只改一个变量

单组调用格式：

```text
bash scripts/run_fahc_experiment.sh LABEL GPU DATASET SEED MODE BUDGET ETA KD [额外 main.py 参数]
```

以下都相对 A4096_E01，只选有诊断依据的实验，不自动全部跑：

```bash
bash scripts/run_fahc_experiment.sh A4096_E003 0 ICEWS14s 1 risk+coverage 4096 0.03 0
bash scripts/run_fahc_experiment.sh A4096_E03 0 ICEWS14s 1 risk+coverage 4096 0.3 0
bash scripts/run_fahc_experiment.sh A4096_Entity0 0 ICEWS14s 1 risk+coverage 4096 0.1 0 --fahc-entity-weight 0
bash scripts/run_fahc_experiment.sh A4096_Pool4 0 ICEWS14s 1 risk+coverage 4096 0.1 0 --fahc-candidate-multiplier 4
bash scripts/run_fahc_experiment.sh A4096_Probe2 0 ICEWS14s 1 risk+coverage 4096 0.1 0 --fahc-probe-steps 2
bash scripts/run_fahc_experiment.sh A4096_KD01 0 ICEWS14s 1 risk+coverage 4096 0.1 0.1
bash scripts/run_fahc_experiment.sh A4096_KD003_C02 0 ICEWS14s 1 risk+coverage 4096 0.1 0.03 --fahc-kd-confidence 0.2
```

eta：若纯风险较好，优先 0.03；若纯风险欠缺但覆盖有用，再试 0.3。
实体权重：若实体频率偏向明显，先试 0，分清实体覆盖是否有害。
候选池：只在有用选择器基础上从 2 扩到 4；候选打分成本会提高。
probe2 表示增加连续探测更新步数，**不是**“从相同 theta_old 出发的两个独立
探测分数取平均”；若要后者，需要另行实现，不能把两者等同。

k 扫描最后再做，例如 20/35/50，并给每个 k 配同预算 random：

```bash
bash scripts/run_fahc_experiment.sh R4096_K20 0 ICEWS14s 1 random 4096 0.1 0 --reply_batch 20
bash scripts/run_fahc_experiment.sh A4096_K20 0 ICEWS14s 1 risk+coverage 4096 0.1 0 --reply_batch 20
bash scripts/run_fahc_experiment.sh R4096_K50 0 ICEWS14s 1 random 4096 0.1 0 --reply_batch 50
bash scripts/run_fahc_experiment.sh A4096_K50 0 ICEWS14s 1 risk+coverage 4096 0.1 0 --reply_batch 50
```

当前时间覆盖默认 bin=10，若要时间分箱消融，应单独改
`--fahc-time-bin-size`，不与 eta/实体权重同时变化。
不要做 k×池倍数×eta×lambda×probe 的全笛卡尔网格。

## 6. 评估、遗忘定义和最终表格

运行器每阶段记录一次当前历史模式验证；最后阶段评估全部旧验证快照。
另在训练结束后评估一次**完整 DGAR 推理**的全部验证快照，结果写入
`event=final_validation`。主任务配置筛选看后者。
完整验证和测试均使用最终模型，不能当作各旧阶段模型的峰值。

设 R(t,j) 为阶段 t 模型对快照 j 的历史模式验证 MRR，T 为最后训练阶段，
当前代码不评估快照 0，则汇总中的：

```text
val_past_MRR = mean_{j=1..T-1} R(T,j)
val_BWT = mean_{j=1..T-1} [R(T,j) - R(j,j)]
val_arrival_forgetting_pos = mean_{j=1..T-1} max(R(j,j) - R(T,j), 0)
```

BWT 越大越好；正向截断的到达至最终遗忘越小越好。两者不互为简单相反数，
因为改善阶段会在截断遗忘中记零。
这**不是**常用的“历史最好表现减去最终表现”：后者需要每个 j 在所有
t≥j 的 R(t,j)。默认高效方案只存到达与最终，不能声称精确得到峰值遗忘。
如论文确实需要完整历史模式遗忘矩阵，仅对最终少数配置使用：

```bash
bash scripts/run_fahc_experiment.sh A4096_Dense 0 ICEWS14s 1 risk+coverage 4096 0.1 0 --fahc-eval-every 1 --fahc-past-eval-samples 0
```

original/random 也要用同样评估参数。该模式历史评估约 O(T²)，应从 JSONL
矩阵计算峰值遗忘；它仍是历史模式指标，不是 diffusion 完整路径 BWT。
不能把 sampled_forgetting 在稀疏评估下称为完整峰值遗忘。

最终表格至少报告：

| 类别 | 指标 |
| --- | --- |
| 主要任务 | 完整推理测试 filtered entity MRR；旧快照宏平均；最后快照 MRR |
| 次要任务 | Hits@1/3/10；关系 MRR 如需报告 |
| 遗忘诊断 | 同口径历史模式 BWT、到达至最终正向遗忘；明确协议 |
| 资源 | 全流程 wall time，候选打分/探测时间，实际 replay 事实数分布和保留比例 |
| 重复性 | 种子列表、均值±样本标准差、逐种子配对增益、代码/缓存指纹 |
| 预算 | k、B、候选倍数；是否每阶段实际数量相等；历史总存储范围 |

不要把按查询汇总 MRR 与快照宏平均混用。主张“超过原 DGAR”需要冻结配置
的完整推理测试在相同条件、多种子下支持；仅超过 budget-matched random
或改善历史模式指标，都不足以支持这项主张。

原程序最终测试输出在日志中位于 `Final full-pipeline VALIDATION` 标题
**之前**，标题之后是验证集输出。读取最后一个 MRR 会读到验证集，不能
作为测试集成绩。最终测试表应取正确区段；汇总脚本则专门只读验证 JSONL。

冻结方案后，导出含完整推理测试成绩的表：

```bash
cd /data1/zihao/DGAR-code
/data1/zihao/miniconda3/envs/dgar_env/bin/python scripts/summarize_fahc_runs.py experiment_runs --include-test --csv experiment_runs/final-results.csv
```

只保留事先冻结的配置和种子，不从该表反向挑新的参数。默认汇总不会加入
`test_full_*`；测试成绩和验证成绩在 JSONL 中分别为 `final_test` 和
`final_validation`，便于准确提取。

## 7. 输出、对比方法与验证边界

每次运行生成唯一目录：

```text
experiment_runs/ICEWS14s_<label>_s<seed>_<time>_<pid>/
  config.json            参数、版本、源码/数据/历史缓存指纹
  command.txt            实际命令
  status.json            Python/tee 退出码、训练及评估总秒数
  logs/train.log         无缓冲 stdout/stderr
  logs/stages.jsonl      阶段诊断和完整验证结果
  models/                本次 RGCN checkpoint
  diffu_models/          本次 diffusion checkpoint
  data -> 仓库 data      只复用同一数据与历史缓存
```

模型和日志均与仓库现有 `models/`、`diffu_models/`、`logs/` 隔离。
无需为每个 seed 改 GPU 来区分模型文件。后台启动日志按上述名字写入根
`experiment_runs/`；重复启动同一阶段时另取日志名，避免 shell 重定向覆盖
之前的启动日志。

`original` 是当前已修复代码下的 inherited DGAR 配置，不承诺逐位复现论文
作者环境。已实现的正式消融有 random/recent/risk/coverage/组合/HCP-KD。
普通 ER、History Repeats 和 MA-DAR 尚无对应实现，不能用现有模式改个名字
就声称完成这些文献基线。需要严格论文对比时另行实现与核验。

可追加一个当前模型的 FT 消融，保留相同结构和静态输入，关闭 diffusion
和特征 replay；这不是已经核验的论文 FT 实现：

```bash
bash scripts/run_fahc_experiment.sh FT_local 0 ICEWS14s 1 original 4096 0.1 0 --diffuc False --delete_feature_reply True
```

本次已通过 CPU 控制流、参数、选择器、探测恢复、KD、评估、输出隔离和
失败日志检查。真实训练、GPU 内存峰值与任何参数的提升效果尚未验证。
CPU 测试中的 optimizer step 使用状态模拟，不执行训练更新。
