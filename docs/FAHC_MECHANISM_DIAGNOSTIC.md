# FAHC 小规模机制诊断

日期：2026-10-11。目标是检查风险是否稳定、能否预测退化；本次新增诊断代码，
尚未执行真实训练或使用 GPU。该模式不用于报告完整数据集的最终 MRR。

## 1. 为什么需要重新跑一段时间线

现有训练只覆盖保存阶段模型，最佳 Adam 状态在内存中，没有保留每个 t-1
检查点。不能用训练完365阶段的最终模型回头诊断早期阶段，那会混入未来信息。

默认用 random B4096、seed1、k35、KD0 正常顺序训练，诊断零基索引
60、90、120，完成第120阶段后停止。共进行120次正常阶段训练（跳过索引0），
加上三个阶段的临时探测，成本包含这段正常训练，不只是几次梯度更新。
诊断结果保存在独立 experiment_runs 目录；所有临时更新都恢复，不写入正常
模型检查点。结束时跳过最后快照的 TEST 打分和最终全数据 TEST/VALIDATION。
正常阶段的验证早停仍沿用原流程。

新增参数未启用时不执行临时诊断；诊断参数排除训练配置签名，正常训练算法
仍为 full-hcp-v3。新增源码会改变 source_sha256，需如实保留本轮 config；
旧结果继续作为旧版本记录，不冒称源指纹完全相同。

## 2. 每个诊断阶段的四支临时更新及实际更新

候选池与正常选择使用同一池，最多2k个时间片；每个候选完整评分。空HCP
保留在记录中，但从排序统计中排除。所有临时分支从同一个旧模型、BatchNorm
状态和匹配的 Adam moments 出发，匹配 dropout seed，结束后恢复模型权重、
buffers、梯度、训练模式、缓存、主优化器和 RNG（主优化器本身未参与临时更新）。

| 分支 | 数据与更新 | 用途 |
| --- | --- | --- |
| A | 不放回抽样的第一批当前训练事实，1步 | 第一组风险排序 |
| B | 与A行位置不重叠的第二批事实，1步 | 第二组风险排序 |
| 原配置探测 | min(128,N)条当前事实，1步 | 检查原来批量大小下的预测能力 |
| current_only | 全部当前训练事实，正常lr/task_weight/裁剪，固定10步，无回放和KD | 测量没有回放保护时的多步退化 |
| dgar_history_only | 临时分支恢复后，实际random DGAR训练得到的本阶段最佳权重 | 测量正式更新后的实际退化 |

A/B批量大小为 min(128,floor(N/2))，不是强行重复抽128条。当前本地ICEWS14s
三个阶段 N=124/221/160，因此两批各62/110/80条；实际服务器数值以日志为准。
A/B是不重叠行的无放回抽样，统计上并非完全独立；若数据有重复事实，也可能
发生事实重叠，日志记录 batch_fact_overlap_count。原配置探测另行记录，避免
因较小批量的诊断结果直接否定原来的128条探测。

所有风险和观察量使用同一公式与同一当前阶段历史图：

```text
score(P) = mean_q max(CE_after(q) - CE_old(q), 0)
```

另记录未截断的平均损失变化。事实来自历史训练缓存和当前训练集。
actual_rgcn_steps记录实际执行轮数，best_rgcn_step记录所恢复最佳权重的轮次；
二者可能不同。current_only固定10步，不使用早停选最佳权重。
`dgar_history_only` 虽在实际DGAR训练后测量，评分依然是历史模式实体CE，
不是完整diffusion历史MRR，也不能单独证明选择产生因果收益。随机回放可能
已经保护某些候选，不能把其残余退化直接等同于无回放时的真实遗忘。

## 3. BatchNorm 诊断

模型探测会更新BatchNorm统计量。A、B、current_only在评分时各追加一次
“保持更新后的权重、将BN running buffers换回旧值”的测量，形成
fixed_bn_current_only。它用于观察评分统计量变化对风险排序的影响。
这没有移除训练时mini-batch BN对梯度的影响，不能称完全隔离BN。

## 4. 执行命令

先激活原来成功使用的 dgar_env，并取得当前Python。

仅检查命令与数据缓存，不训练：

```bash
cd /data1/zihao/DGAR-code
export DGAR_PYTHON="$(command -v python)"
bash scripts/run_fahc_mechanism.sh --dry-run 1 ICEWS14s 1 60,90,120
```

GPU空闲后，由用户启动实际实验。GPU1是此前成功使用的示例，可换成空闲设备。

```bash
cd /data1/zihao/DGAR-code
export DGAR_PYTHON="$(command -v python)"
mkdir -p /data1/zihao/DGAR-code/logs
setsid bash -c '
set -o pipefail
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
bash /data1/zihao/DGAR-code/scripts/run_fahc_mechanism.sh \
  1 ICEWS14s 1 60,90,120 \
  2>&1 \
| tee /data1/zihao/DGAR-code/logs/ICEWS14s-mechanism-seed1.log
' < /dev/null > /dev/null 2>&1 &
```

查看进度：

```bash
tail -f /data1/zihao/DGAR-code/logs/ICEWS14s-mechanism-seed1.log
```

结束时会出现 `Mechanism diagnostic complete` 和 launcher 的 `Completed:`。
控制台在三个诊断阶段打印 `Mechanism snapshot=...`，详细数据在该运行目录的
logs/stages.jsonl。汇总只读JSON文件，不训练：

```bash
cd /data1/zihao/DGAR-code
python scripts/summarize_fahc_mechanism.py experiment_runs \
  --csv logs/ICEWS14s-mechanism-summary.csv
```

可将输入改为日志的RUN_DIR或单个stages.jsonl，以只汇总本次运行。
默认三个阶段、三个评分口径，共9行；complete_run须为True。
普通实验汇总器会把这种结束状态标成 diagnostic_finished，避免混作完整训练。

## 5. 看哪些指标，怎么决定下一步

| CSV指标 | 含义 |
| --- | --- |
| spearman_ab | A/B风险排序稳定性，越接近1越一致 |
| topk_overlap_fraction | 两批top-k交集数量/k |
| topk_overlap_random_baseline | 随机选择的期望重合比例k/m，m为实际非空候选数 |
| spearman_mean_observed | 两批风险平均分对更新后退化排序的预测能力 |
| observed_topk_lift | 平均风险top-k的观察退化均值减候选全体均值；大于0才优于随机期望 |
| configured_probe_spearman_observed | 原配置大小单次探测的退化预测能力 |
| configured_probe_observed_topk_lift | 原配置大小单次探测的top-k退化捕获增益 |
| risk_*_zero_fraction/tied_item_fraction | 是否多数风险为零或并列 |

常量向量的相关系数为未定义，输出空值，不当作0或1；并列top-k使用独立
确定性随机打破并列，防止全零风险被同一排序规则伪装成完美稳定。若m<=k，
所有候选全选，selection_has_choice=False，此阶段不提供选择收益证据。
第60阶段最多59候选，剔除空候选后还会更少，随机基线不能一律写0.5。

可预先固定如下工程筛选门槛，避免看结果后不断调整标准：在至少两个可用
阶段，稳定相关>=0.5，重合率高于对应随机期望，预测相关>=0.3且lift>0。
这些不是公认统计阈值；三个阶段仅用于机制筛选，HCP共享事实也不能被当作
独立样本直接做显著性推断。先看current_only和dgar_history_only，再结合
fixed_bn与原配置探测结果解释。

决策顺序：

1. A/B不稳定而原配置探测尚能预测：先检查较小批量效应，不直接否定原配置。
2. 原配置和两批平均都缺乏预测力：暂停当前风险公式，不继续大范围扫参数。
3. current_only能预测而实际DGAR目标不能：核查评分代理与正式回放训练的差异。
4. fixed_bn明显改变结果：优先检查BN评分统计量对风险的影响。
5. 稳定性和预测性都达到预定门槛：才实现两批平均选择器，并与random做完整
   验证对照。这个诊断通过仍不代表旧快照MRR一定提升。

本次验证限于CPU统计、状态恢复及mock控制流，优化器step被mock，未执行
真实梯度参数更新；dry-run不导入模型/CUDA，不构成GPU训练正确性的证明。
相关47项CPU检查通过，文档命令的bash语法检查和启动器dry-run通过。
