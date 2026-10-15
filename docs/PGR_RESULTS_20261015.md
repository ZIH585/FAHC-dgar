# PGR v1 实验结果与停止决定

分析日期：2026-10-15。数据集 ICEWS14s，seed=1。只分析已完成实验；本次没有启动训练或 GPU 推理。

## 结论

**停止当前 PGR geometry 主线，不继续扫描 rank/strength，也不启动预留的 random/isotropic 完整训练。**

PGR 对 Original 的 Average MRR 有小幅正增益，但没有超过最强简单控制 mean+Shared；
后者在 TEST 和补充 VALIDATION 都更好。未通过实验前写在 [PGR.md](PGR.md) 的继续门槛。
不能将 mean+Shared 的正结果归功于 predictive geometry，也不能将当前 PGR 摘要写成已验证的方法结论。

值得保留的是 **DCR mean + Shared Adapter 这一固定组合**。它是当前这轮的正向工程结果，
还不构成已经验证的新论文创新，也没有证明最新时间片预测同步改善。

## 1. 实验身份与可比性

代码分支 `codex/pgr`，训练实现提交 `b9c8ec6`，从
`codex/original-baseline@23dd43dd914a783731c7f39e3fd66491d2aba429` 建立。

运行目录均位于本 worktree 的 `experiment_runs/`：

| 组别 | 运行目录 |
|---|---|
| Original | `ICEWS14s-pgr-original-seed1-20261015_133222_118013` |
| mean+Shared | `ICEWS14s-pgr-mean-shared-seed1-20261015_191853_919554` |
| PGR geometry | `ICEWS14s-pgr-geometry-seed1-20261015_220719_1332220` |

三组配置排除方法参数后完全一致，均记录 GPU 0、seed 1、完成阶段 364。
同一 launcher 顺序运行；没有记录可用于严格比较吞吐的 GPU 外部负载，不能把墙钟差异解释为方法加速。

共同指纹：

- source：`5e56dd4ae17f5ff08a0ad072f1bfeb2187f840f75adaf5d73bb0922bc63a481c`
- data：`a81dea25dfbbd5b7420563e9291e356d8e14971cab73a409ed2efbcf514aa038`
- history：`6185eadbb881aacd9cddf566fd1cfcae9b0adacaf3f81a5adc8db032fd6eb804`

方法参数固定：adapter rank=16，geometry rank=4，geometry facts=当前/历史各最多64，strength=1。
没有改变 HCP 预算、mu、task loss 或 evaluation protocol。

## 2. 论文口径的最终 TEST

单位为百分数。Average 是历史时间片 1..363 的等权宏平均，不是 pooled。
Hits 列为 **Average** Hits。Current 是最后时间片，只有36条双向 query。

| 方法 | Current MRR | Average MRR | Average Hits@1 | Average Hits@10 |
|---|---:|---:|---:|---:|
| 论文 DGAR，参考值 | 58.59 | 50.12 | 39.36 | 70.48 |
| 本轮 Original | 50.1566 | 49.9197 | 38.9931 | 70.1574 |
| mean+Shared | 54.0313 | **51.4958** | **40.7646** | **71.4503** |
| PGR geometry | **61.9580** | 50.3124 | 39.4527 | 70.4377 |

论文参考：[DGAR，ACL 2025，Table 1](https://aclanthology.org/2025.acl-long.537.pdf)。
论文报告多次运行汇总；本轮是一个固定 seed，方法增益首先按本轮 Original 计算。

- mean+Shared 对 Original：Average MRR **+1.5761pp**。
- PGR 对 Original：Average MRR **+0.3927pp**。
- PGR 对 mean+Shared：Average MRR **−1.1833pp**。
- Original Average MRR 与论文参考差约 −0.2003pp，未出现过去某些公共修改造成的大幅历史指标下降。
  Current 的差异仍需保留，不能声称本轮完整复现了论文四项指标。

PGR 的 Current TEST 较高不能替代历史指标与验证集检查；36条 query 的结果对少数 rank 变化敏感。

## 3. 补充 VALIDATION

| 方法 | Current MRR | Average MRR | Average Hits@1 | Average Hits@10 |
|---|---:|---:|---:|---:|
| Original | **68.8483** | 49.8903 | 39.1502 | 70.4610 |
| mean+Shared | 63.0925 | **51.8086** | **41.0792** | **71.9998** |
| PGR geometry | 63.1121 | 50.0893 | 39.1854 | 70.8650 |

mean+Shared 对 Original 的 Average MRR +1.9182pp；PGR 仅 +0.1989pp，
对 mean+Shared **−1.7193pp**。
实验前要求 geometry 比 mean+Shared 至少 +0.2pp 且 TEST 方向一致，本轮明确未通过。

补充 VALIDATION 按既定程序在 TEST 后报告，没有重新启动独立随机流。
这里用于核对趋势，不把它描述为训练前未查看的独立模型选择流程。

mean+Shared 的 Current VALIDATION 比 Original 低5.7557pp。
因此现阶段能说“历史 Average 提升”，不能说“当前与历史均提升”。

## 4. 逐 query 重算：收益有明显方向差异

下表为方向内 pooled filtered MRR，**不能替代论文宏平均**。TEST 每个方向9228条 query。

| 方法 | 正向 | 逆向 | 正向变化 vs Original | 逆向变化 vs Original |
|---|---:|---:|---:|---:|
| Original | 43.1364 | 55.7960 | — | — |
| mean+Shared | 42.6183 | 59.5491 | −0.5181pp | +3.7531pp |
| PGR geometry | 41.7533 | 58.0257 | −1.3831pp | +2.2297pp |

VALIDATION 同方向：mean+Shared 正向 −0.4619pp、逆向 +4.2090pp；
PGR 正向 −1.7265pp、逆向 +2.0675pp。
geometry 比 mean+Shared 在两个方向都差，不能解释成只是在两个方向间交换收益。

TEST 已重复/未重复 query MRR：

| 方法 | 历史已出现 triple | 历史未出现 triple |
|---|---:|---:|
| Original | 70.5382 | 34.9242 |
| mean+Shared | 72.3221 | 36.4269 |
| PGR geometry | 70.7551 | 35.4900 |

“未出现”按当前阶段之前的训练前缀判断，不表示 entity 未见或 relation 未见。
mean+Shared 的总体未重复组也改善，因此不能断言它只会记忆旧事实。
但正向未重复组从27.8769降至27.2350，逆向未重复组从41.9716升至45.6188。
更具体的研究问题是**回放表示如何支持双向排序，尤其正向新 triple**。

## 5. 模块确实生效，但代理下降不代表预测更好

CPU 审计实际保存的权重和阶段 catalog：

- geometry 在363个有回放阶段均有有效 Q/w，统计与 checkpoint 逐项一致；第一阶段空投影正常。
- Shared 与 PGR 两组原本零初始化的 U 均已学成非零，V 也非零，checkpoint、阶段记录无 NaN/Inf。
- `model.pt`、`diffusion.pt` 与 `latest.pt` 对应最佳权重逐 tensor 完全相同。
- 六份 queries.csv 的指标、方向标记、逆关系配对、三组 query key、历史重复标记均核对通过。
  独立重建了原始数据的逐片80/10/10划分及历史训练前缀；没有把未来事实用于 repeated 标记。
- 训练与评估均经过同一 adapter/聚合/decoder 调用，最终直接评估使用训练期间保存的阶段投影。
  本轮没有发现“新增模块训练了、预测没接入”的证据。

geometry 的 batch CE 噪声代理 after/before 均值为 **0.05638**，下降约94.36%。
这只是构造 Q/w 的同一批数据上的一阶敏感度代理，不是独立验证过的 rank risk。
它下降而 Average MRR 不如 mean+Shared，直接说明该代理目前不能支撑方法收益主张。

原始末端扰动能量平均保留98.37%，不能仅据此说“干预太小”：
对真实最终 adapter A 与逐阶段 S，`||A S||_F² / ||A||_F²` 均值约55.24%。
这个量也只是最终线性映射后的条件方差比例，不是 decoder 风险，不等于 MRR 变化。

TEST replay 的事实级尾特征平均范数：Original14.13、mean+Shared0.76、PGR14.05；
adapter 后 mean+Shared4.07、PGR19.68。
两套方法同时改变扰动与学习结果，不能据此单独证明是范数、方向或某个因素导致差距。

## 6. 失败原因：按目前证据排序

1. **几何目标尚未对齐最终排序。** 正式结果不支持“降低 batch CE 噪声敏感度就能改善回放排序”的核心假设。
   代理下降是事实，迁移到 MRR 的失败也是事实；不能继续只以代理下降解释方法有效。
2. **保留大部分采样扰动的复杂方案没有胜过确定性均值。** 当前简单组合同时在 TEST/VALIDATION 更好。
   这不证明所有扩散噪声都无用；缺少同轮 unprojected Shared，不能把全部差距归因于投影方向本身。
3. **局部几何可能与后续使用阶段失配。** Q 在 predictor 正式更新前、均值处、少量事实上估计；
   随后的 adapter/图传播/decoder 都会更新，最终 retrospective 评估又使用最终模型。
   这是代码支持的潜在机制，不是已由本轮实验唯一定位的原因。方向收益失衡提供了更具体的诊断入口。

不能据此断言整个 predictive geometry 研究方向错误；足以判定当前 PGR v1 不值得继续参数搜索。

## 7. 新发现的真实 checkpoint 恢复问题

`src/pgr.py` 的 `PGRRuntime.load_state_dict` 使用 `atol=2e-5` 检查 Q 的正交性。
实际 CUDA 保存的363个阶段基在 CPU 上均未通过；双精度重算 `max|QᵀQ−I|` 最大约
**9.55e−5**，主要为列范数的微小偏差。CPU toy roundtrip 没覆盖这个真实误差范围。

后果：当前代码重新加载 geometry checkpoint 的独立 `--test` / `--resume` 会报
`PGR basis is not orthonormal`。mean+Shared 的空 geometry catalog 校验通过。

**它不解释本轮得分下降**：三组无 resume，训练结束后的评估使用内存 catalog，
没有调用该恢复检查，TEST 和 VALIDATION 都已完整完成。不能以此为由重跑整轮 PGR。
本次仅做结果分析，未修改训练源码或 checkpoint。若将来确需复用 geometry checkpoint，
需单独修复数值容差和源码指纹兼容恢复，不能直接改源码后声称旧 run 已可严格恢复。

## 8. 下一步有限计划

1. 封存 PGR geometry；不补 seed，不扫参数，不继续预留的 random/isotropic 完整训练。
2. 把 **mean+Shared 固定配方**作为这轮最强简单参考，保留已有 Original，后续满足源码/OFF行为、
   科学参数、seed、data/cache、评估协议和环境可比性时复用，不按分支名机械重跑基线。
3. 下一项正式验证优先做 **ICEWS18 的 mean+Shared 固定组合**，参数仍不扫描。
   已有 ICEWS18 Original 可先审计复用；此前 DCR mean 在阶段114曾 OOM，需先处理资源/日志问题，
   不因 GPU 空闲就盲目提交同配置长训练。
4. 研究机制集中于**回放表示进入预测器后的尺度、方向与双向排序失衡**。
   在继续增加模块前，用固定生成特征和冻结预测器比较 learned adapter、逐实体范数匹配 identity、
   固定 gain 与保谱随机方向，观察正逆向 rank/margin。现有 `diagnose_pgr.py` 只支持已审计的旧 RAHA Shared，
   不能把它误标为本轮 mean+Shared 诊断；迁移到本轮组合须显式适配并保留严格 checkpoint 校验。
   这里不再次提出角色 adapter 或全局 KD，已有实验不足以支持这类重复尝试。

如果固定组合在第二数据集也没有优势，就不继续包装 mean+Shared 为方法主线；
若 Average 优势存在而正向持续下降，下一方法必须针对这项可测的失衡提出机制，
以“超过固定 mean+Shared 且不靠逆向掩盖正向损失”作为新的继续门槛。

## 9. 审计产物

本机派生文件位于 `experiment_runs/pgr-analysis-20261015/`，不提交训练输出到 Git：

- `analyze.py`：CPU 只读分析脚本；不执行神经网络 forward、不训练。
- `metrics.csv`：TEST/VALIDATION 的论文指标及 pooled 指标。
- `group_metrics.csv`：方向、重复性及交叉分组统计。
- `comparisons.csv`：匹配 query 与历史时间片的差值/胜负数。
- `geometry_stages.csv`：真实逐阶段矩阵误差、代理与能量比例。
- `checkpoint_audit.json`：权重一致性、finite、实际恢复问题。
- `summary.json`、`input_audit.json`：配置、门槛、输入文件指纹。

所有指标由 queries.csv 独立重算后与 summary.json、最终日志对齐；分析结束确认 CUDA 未初始化。
