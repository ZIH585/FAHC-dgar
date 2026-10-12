# ICEWS14s Original / ECRI seed1 结果分析（2026-10-12）

当前 ECRI 没有达到超过 Original 的实体预测目标：TEST pooled filtered entity
MRR 下降 0.4398 个百分点，旧时间片 macro MRR 下降 0.4468 个百分点；关系预测
MRR 上升。最新 TEST 时间片有收益，最新 VALIDATION 时间片则下降，不能把局部
收益解释成稳定提升。当前版本应记为主指标未改善，先核对机制，再决定是否继续。

## 数据与可比性

来源为两份完整日志：

- [Original](../../logs/ICEWS14s-original-s1-20261011_210151.log)
- [ECRI](../../logs/ICEWS14s-ecri-s1-20261012_100243.log)

都完成 364 个训练阶段，以及各 365 片的最终 TEST、VALIDATION 循环；第 0 片按
原协议跳过。没有 Traceback。日志参数只有 ecri 开关和 run_dir 不同，其余一致：
ICEWS14s、seed1、gpu1、task-weight0.7、reply_batch35、mu/mu_r1、10/10 轮等。
ecri-window30、min-support10、sample-cap128、permutations20、kappa20。
sample-cap 是统计采样上限，不是 FAHC fact budget。

本地没有这两次运行的 config.json、ecri.jsonl 和模型权重，尚不能核对远端
source/data/history 哈希、学习到的门控参数和实际 HCP 命中覆盖。Namespace 一致
是现有日志支持的可比性证据，不代表已经核验所有运行环境。

## 同口径结果

以下均为 filtered 指标。差值为 ECRI−Original 的原始指标值。

| TEST | Original | ECRI | 差值 |
|---|---:|---:|---:|
| pooled entity MRR | 0.494907 | 0.490509 | -0.004398 |
| pooled entity Hits@1 | 0.385782 | 0.382694 | -0.003088 |
| pooled entity Hits@3 | 0.556892 | 0.550715 | -0.006177 |
| pooled entity Hits@10 | 0.697822 | 0.695655 | -0.002167 |
| past macro entity MRR, t=1..T-2 | 0.499344 | 0.494876 | -0.004468 |
| Current entity MRR | 0.567219 | 0.602220 | +0.035001 |
| Current entity Hits@1 | 0.500000 | 0.555556 | +0.055556 |
| Current entity Hits@10 | 0.666667 | 0.666667 | 0 |
| pooled relation MRR | 0.474480 | 0.480006 | +0.005526 |

| VALIDATION | Original | ECRI | 差值 |
|---|---:|---:|---:|
| pooled entity MRR | 0.491407 | 0.489153 | -0.002254 |
| pooled entity Hits@1 | 0.381774 | 0.382493 | +0.000719 |
| pooled entity Hits@3 | 0.551316 | 0.545510 | -0.005806 |
| pooled entity Hits@10 | 0.698020 | 0.693597 | -0.004423 |
| past macro entity MRR, t=1..T-2 | 0.499225 | 0.497548 | -0.001677 |
| Current entity MRR | 0.667391 | 0.634534 | -0.032857 |
| Current entity Hits@1 | 0.583333 | 0.527778 | -0.055555 |
| Current entity Hits@10 | 0.805556 | 0.861111 | +0.055555 |
| pooled relation MRR | 0.470049 | 0.474967 | +0.004918 |

完整 raw/filtered、entity/relation 指标保存在 metrics.csv，避免把日志最后一段
VALIDATION 当作 TEST。数值由日志自动解析，未手工挑选区段。

主实体 MRR 在 TEST 与 VALIDATION 上方向一致地下降，TEST 的实体 Hits@1/3/10
也全部下降。关系 MRR 和 relation Hits@1/3/10 在两个 split 上均上升，说明预测
表现发生了任务方向分化；这些结果还不能证明分化的具体优化原因。

最后一片本地划分为 18 条正向验证事实、18 条正向测试事实，各含逆 query 后只有
36 个 query。Current TEST Hits@1 提升对应多 2 个 query 排第 1；Current
VALIDATION 则少 2 个。单片权重很小，不足以支持“适应新知识稳定改善”。

Average 是最终模型对旧片的 macro 表现，不是完整 forgetting 指标；没有各阶段
当时/最佳旧片性能矩阵，不能从这两个终值推导灾难性遗忘的变化。

## 耗时

训练进度记录 Original 为 1:09:18，ECRI 为 1:18:01，增加 8:43（约 12.6%）。
最终 TEST 约 1:35 / 2:32，VALIDATION 约 1:32 / 2:21。时间是这两次实际墙钟记录，
不是隔离机器负载后的精确模块基准。按此结果，新增时间没有换来实体主指标收益。

两组都有同类 DGL、beta reciprocal 除零及 PyTorch GPU 架构兼容警告，运行仍完整
结束，日志没有给出 NaN 指标或异常。这些共同警告不能单独解释 ECRI 相对下降。
GPU 架构警告涉及枚举到的 5090，不足以断定日志 gpu1 实际选择了哪块物理 GPU。

## 低成本机制复核

用本地 ICEWS14s、同一 Original 划分和默认 ECRI 参数，纯 CPU 重算六个阶段的
过去窗口统计，约 0.8 秒，CUDA 未初始化。这里只分析数据统计，没有训练模型。
这不是从远端 checkpoint 恢复出的实际运行记录；远端数据哈希尚未核对。

| 阶段 | 可信实体数 | 当前训练独立实体覆盖 | 当前训练端点出现次数覆盖 |
|---|---:|---:|---:|
| 60 | 138 | 33.083% | 53.226% |
| 120 | 132 | 35.632% | 51.250% |
| 180 | 143 | 31.933% | 55.761% |
| 240 | 127 | 35.443% | 51.299% |
| 300 | 161 | 34.524% | 56.415% |
| 364 | 158 | 37.059% | 51.773% |

可信实体占总词表约 1.8%～2.3%，但多为常出现的实体，端点频次覆盖超过一半。
因此不能拿“只有 2% 词表”断言 ECRI 完全不起作用。实际融合仅在 HCP 涉及的实体
上发生，表里的覆盖不等于真实 HCP 融合覆盖，也不等于 query 改善比例。

校准后 role excess 均值约 0.066～0.090，churn excess 均值约 0.042～0.050；
可信度均值约 0.52～0.56。分数非零，但它们是否预测回放收益没有被这两份日志证明。
不能直接把结构演化的存在当作历史表示不再有用的证据。

代码还存在一个需验证的机制混杂：当 d=u=0 但 c>0 时，系数仍是
`(1-c)*alpha0+c*sigmoid(b)`，会偏离原 scalar alpha。故性能变化可能来自普通偏置/
支持度门控，而非演化辨别。需要零演化或匹配频次的 shuffle 对照，不能仅用开关
对照就宣称“演化感知机制”成立。这里是待检验解释，不是已确定的失败原因。

## 下一步决策

暂不继续扫 window、kappa、sample-cap，也不加入 KD 或新 selector。优先用已经
训练好的状态做一次无需训练的机制检查：

1. 取两组 run_dir 的 config.json，核对 source/data/history 哈希。
2. 取 ECRI 的 ecri.jsonl 和两组 model.pt，读出可信实体、演化统计、b/beta/gamma、
   原 alpha，检查非预热阶段是否形成有效而非饱和的融合差异。
3. 在预选的验证片上固定 checkpoint、历史图、HCP、生成表示及随机噪声，只比较
   学到的 ECRI gate、原 alpha、零演化 gate 和匹配频次的 shuffled evolution。
   分高/低演化及支持度统计逐 query RR 差值；原模型的固定表示融合收益也应对照。

这个检查回答“过去结构变化是否真的能识别历史表示的预测效用”，而不是再次
训练一套模型后猜原因。若高演化信号稳定对应降低历史融合更有益，才值得做一组
有针对性的修正与复验；若信号与功能收益没有关系或反向，本方向应停止。方向尚未
通过机制证据时，不把多 seed 重训或大量调参作为首要下一步。

文件清单：metrics.csv 为完整数值对照，prefix_audit.csv 为本地六阶段统计，
provenance.json 为日志 SHA256、配置差异、耗时和复核范围。未改训练代码。
