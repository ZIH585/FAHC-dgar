# CUR-DGAR v1: Counterfactual Utility Routing

## 起点与研究目标

2026-10-16 首次冻结诊断未通过预设门槛：学习路由 236/236 条 query 全选 R，
平均增益 0，改善阶段 0/5。暂停三组完整训练计划；详见
[冻结诊断报告](CUR_FROZEN_RESULTS_20261016.md)。下文保留实现和原始筛选方案。

独立分支 `codex/cur`，从纯净 `codex/original-baseline` 的
`23dd43dd914a783731c7f39e3fd66491d2aba429` 创建；作者参考为
`c5bbc87467bb22d266b664d30eefb8184c1d3104`。没有合并此前的方法分支。
Original 的兼容修复及科研行为约束继承自 [BASELINE.md](BASELINE.md)。
只重新实现 CUR 所需的 mean+Shared 控制：保留已抽取的 HCP 和其事实数，
保留噪声随机数调用，用 guided conditional mean 替换末端尾表示；
头/尾表示在原均值聚合前使用共享 A(z)=z+UVz。U 为零、V 为非零初始化，
并隔离新增模块的 CPU 初始化 RNG。没有引入 PGR geometry、FAHC、KD、
risk/coverage、事实预算或其他旧方法的优化规则。

已有正结果主要集中在 inverse query，forward 并未一致改善。CUR 的可检验
假设是：同一预测器中，生成表示注入对不同 query 的即时排序效用有差异，
且能从合法 query/预测统计预测这种差异。它仍是待验证假设。

## 路径、效用与训练目标

同一组参数、同一训练历史窗口、同一份 HCP 和一次生成：

* R：mean+Shared 特征注入，使用 DGAR 原可学习 alpha 融合。
* 0：完全关闭生成特征注入；**仍使用相同的历史事实标签训练**。

阶段 t 先沿 Original 流程生成回放，然后在正式 predictor 更新前，用已有
刚完成 best 权重 rollback 的 predictor model 计算两条路径的 TRAIN 排名：

```
u_t(q) = RR_R(q; theta_(t-1)) - RR_0(q; theta_(t-1))
y_route = 1[u_t(q)>0], w=|u_t(q)|
L_route = sum(w * BCEWithLogits(g_phi(x_q), y_route)) / sum(w)
```

使用全部候选实体，raw midrank 处理并列：rank=1+#greater+0.5*(#equal-1)。
此训练标签不是论文的 filtered MRR，也不是“未来遗忘”或长期因果效应。
效用为零的 query 权重为零；训练部分全部为零则跳过该次 router 更新。
阶段 1/空回放同样跳过，评分退回普通历史图预测。

严格遵循实际时序：测量对象是主模型刚恢复的上一阶段 best 参数及 BN buffers，
尚未执行本阶段任何 predictor SGD。生成仍按 Original 使用 model_repre；
效用测量在 rollback 后对主模型临时 eval，完整恢复混合 modes、buffers、
h/h_0 和 RNG。这样效用对应正式更新起点；没有 probe SGD，也不复制额外 old model。

当前 TRAIN 最多 128 条 forward facts，按私有 seed 抽取并增加 inverse。
事实数 >=4 时末尾 1/4 作为同阶段观察集，forward/inverse 始终成对划分；
余下事实拟合路由器 10 步。这个留出集只做诊断，不选超参数或 checkpoint。
router 历史参数与独立 Adam 状态跨阶段保存，没有新 replay memory。

正式 predictor 训练为：

```
L = task_weight * (L_entity^R + L_entity^0)/2
    + (1-task_weight)*L_relation^R + L_static^R
```

两项 entity loss 都包含 Original 的 current CE、单向 replay CE 和 mu 倍
双向 replay CE。只平均 entity loss，避免简单加倍；relation/static 仍使用
R 支路及原系数。diffusion 使用原目标和 R 的 embedding 输出。
两次 forward 会额外消耗训练 RNG、更新 entity decoder 的 BN；这是明确的
`cur-dual-*-v1` 科研变化，由 dual-replay 控制，不作为 Original。

router 有单独 Adam，只在阶段开始拟合。其输入/权重标签都 detach，
router 参数排除在 backbone Adam 外，拟合后清除梯度，避免污染全局梯度裁剪。
生成器的 guidance 仍直接调用原 forward/decoder，路由只作用于 entity 评分。

## 评分、输入与无泄漏约束

```
p_q = sigmoid(g_phi(x_q))
s(q, :) = s_R(q, :) if p_q >= 0.5 else s_0(q, :)
```

整行候选分数选路，不混合不同尺度 logits。relation prediction 继续使用
R 支路；CUR 不用 entity query 已知的 relation ID 去路由未知关系任务。
训练时的早停验证、最终 TEST、最终 VALIDATION、独立 `--test` 均走同一
`RecurrentRGCN.predict -> cur.predict` 接口，adapter/router 都在 model.pt 内。

router 输入为已知 directional relation ID 的 8 维 embedding 加 10 个数：
两支路各自归一化预测熵及 tanh(top1-top2 gap)、top10 重合率、subject 在
当前训练历史图/HCP 的支持数、relation 在这两处的支持数、inverse 标志。
支持数为 log1p 后按对应总数缩放。没有 target entity、真值排名、test
label、未来频次或未来图输入。query_features 强制接收 [subject, relation]
两列；测试改变 target 后路由输入/选择/实体分数应完全不变。

评估产生的 RR_R/RR_0 只写诊断，不更新参数、不进入 router；oracle 只能
解释候选路径的互补上限，不能作为可部署性能。主结果仍为原 filtered
Current MRR、历史 macro Average MRR/Hits@1/3/10。

最终评估原样保留 Original 的 retrospective guidance history 与共用 RNG，
其局限见 BASELINE.md。不能把它宣称成全支路严格在线预测。

## 模式与默认参数

| 参数 | 默认 | 含义 |
|---|---:|---|
| --cur-mode | original | original / mean-shared / dual-replay / dual-direction / utility |
| --cur-adapter-rank | 16 | Shared 低秩维度，d=200 时 6400 参数 |
| --cur-router-hidden | 32 | 小 MLP 隐层；另有每个方向关系 8 维 embedding |
| --cur-probe-facts | 128 | 仅用于训练效用测量，**不限制实际 replay 事实数** |
| --cur-score-batch-size | 256 | probe/eval decoder 分块，保留完整候选实体 |
| --cur-router-steps | 10 | 阶段开始的小路由器优化步数 |
| --cur-router-lr | 0.01 | router 独立 Adam 学习率 |
| --cur-threshold | 0.5 | 使用 R 的概率阈值 |
| --cur-diagnostics | false | 保存阶段观察及最终 CSV/JSON |

default original 不创建新参数，不消耗新增初始化 RNG，不测量效用、不增加
forward/loss；OFF 回归与固定基线比较 loss、梯度、分数、BN 与 RNG。
mean-shared 单支路作为当前强控制；dual-replay 同样训练 router 和两条预测
支路，但推理始终 R；dual-direction 固定 forward 用 0、inverse 用 R；
utility 使用学习的选择。前三个方法模式同用 35 个历史时间片和原事实全集。

没有新增独立 teacher；新增参数量为 6400 + (16*R+641)，R 是正向关系数。
dual 相比 mean-shared 多一次图表示/实体 loss forward，probe 与评分也需要
两条路径。训练/推理都会更慢，显存和耗时不能在没有 CUDA 实测时给固定倍数。
24GB GPU 可在启动脚本前设置 DGAR_CPU_OFFLOAD=1，外层使用 FP32 save_on_cpu；
代价是 RAM 和传输耗时，不能保证任意快照均不 OOM。默认该执行选项为 0。

## 文件与恢复

* src/cur.py：adapter、router、配对表示/效用、双路径 loss、路由、独立 Adam 状态。
* src/rrgcn.py：创建模块，select_entity 插入 adapter，get_loss 区分历史标签与
  特征注入，predict 接入实体路由；Original 默认行为对照测试固定。
* diffusion/difffu_21.py：原随机数抽取后，方法分支替换末端表示为均值。
* src/main.py：rollback 后 fit、正式双路径训练、最终报告、恢复与保存辅助状态。
* src/cli.py、src/runtime.py：独立参数验证、协议名与 provenance。
* src/cur_diagnostics.py：原 filtered 指标、方向/重复分组、query 决策和被动 RR。
* scripts/run_cur_experiment.sh、run_cur_entry.py：单组启动、配置 dry-run、CPU offload。
* scripts/diagnose_cur.py：只读加载已审计 PGR mean+Shared 的低成本候选路径诊断。
* tests/test_cur.py 与 reference：因果输入、真实训练/评分连接、恢复、OFF 对照。

latest.pt 保存 best predictor/diffusion、最后更新 Adam、RNG，外加 CUR 的
完成 stage 和 router Adam。adapter/router 参数随 best model 保存。恢复时先
执行原有 weights+Adam+RNG 恢复，再恢复 router Adam；检查配置/源码/数据/cache
完全匹配。不要用本分支强行续跑 PGR 的训练 checkpoint。冻结诊断仅显式迁移
两个 adapter 键名，其他 backbone 键逐个匹配，不改写源 checkpoint。

## 实验顺序与停止条件

先跑冻结模型诊断：现有最终 mean+Shared 权重不训练，只在时间片
60/120/180/240/300 的 TRAIN 上拟合小 router，再测对应 VALIDATION。
只采样每阶段最多 128 条 TRAIN 和 VALIDATION forward facts，不评分 TEST。
保留 generator 的 Original TRAIN 模式；没有 backbone/diffusion optimizer step。
**这是使用最终 backbone 的事后诊断，不是在线证明，也不是正式 Average MRR。**
0 支路尚未经过 dual objective 训练，负结果只用来决定是否值得投入，不能
推导出所有双路径方法都无效。

查看 summary.json 的 macro_stage_probe：gain_vs_replay 为 learned router
相对始终 R 的 raw midrank MRR 差；oracle_gain_vs_replay 是知道标签的上限；
direction_gain_vs_replay 是固定方向选择。数值乘 100 才是百分点。
建议预先采用筛选门槛：oracle 平均增益 >=0.005、学习路由增益 >=0.002、
至少 3/5 阶段增益为正，且学习路由不低于固定方向。达不到则先停 v1；
不要直接进行大量 threshold/hidden/rank 搜索。这是研发取舍门槛，不是统计显著性。

通过后只跑 ICEWS14s seed1 三组：mean-shared、dual-replay、utility。
Original 可复用同 GPU/环境/data/cache、审计确认等价的现有结果；需要新的
Original 时脚本也提供入口。不要无条件补 seed 或为了每个参数重跑 Original。
现有其他分支 mean+Shared 可作为参考；正式归因推荐本分支同源码的三组。

正式继续门槛：utility 的最终 VALIDATION macro MRR 比 mean-shared 和
dual-replay 两者都高至少 0.002，再检查 TEST 是否同向及 forward/inverse、
relation 分组。若仅胜 dual-replay、仍不如 mean-shared，当前完整方法仍不成功；
若只胜 mean-shared 而不胜 dual-replay，不能把收益归因于路由。之后再考虑
ICEWS18 一组迁移验证和 fixed-direction 对照，不在首轮横向扫超参数。

## 可复制命令（不由代码修改过程执行）

激活 dgar_env 后，先运行诊断：

```bash
cd /data1/zihao/DGAR-code/.worktrees/cur
export DGAR_PYTHON="$(command -v python)"
export DGAR_DATA_ROOT=/data1/zihao/DGAR-code/data
export CUR_GPU=0
export CUR_ID="$(date +%Y%m%d_%H%M%S)_$$"
mkdir -p logs experiment_runs

setsid bash -c '
set -euo pipefail
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
"$DGAR_PYTHON" -u scripts/diagnose_cur.py \
  --gpu "$CUR_GPU" --data-root "$DGAR_DATA_ROOT" \
  --run-dir /data1/zihao/DGAR-code/.worktrees/pgr/experiment_runs/ICEWS14s-pgr-mean-shared-seed1-20261015_191853_919554 \
  --output "experiment_runs/ICEWS14s-cur-frozen-$CUR_ID" \
  --stages 60,120,180,240,300 --facts 128 \
  2>&1 | tee "logs/ICEWS14s-cur-frozen-$CUR_ID.log"
' < /dev/null > "logs/ICEWS14s-cur-frozen-launcher-$CUR_ID.log" 2>&1 &
```

诊断达到门槛后，手动执行三组完整训练；CPU offload 有较高开销：

```bash
cd /data1/zihao/DGAR-code/.worktrees/cur
export DGAR_PYTHON="$(command -v python)"
export DGAR_DATA_ROOT=/data1/zihao/DGAR-code/data
export DGAR_CPU_OFFLOAD=1
export CUR_GPU=0
export CUR_ID="$(date +%Y%m%d_%H%M%S)_$$"
mkdir -p logs experiment_runs

setsid bash -c '
set -euo pipefail
for CUR_GROUP in mean-shared dual-replay utility; do
  bash scripts/run_cur_experiment.sh "$CUR_GROUP" "$CUR_GPU" ICEWS14s 1
done
' < /dev/null > "logs/ICEWS14s-cur-screen-launcher-$CUR_ID.log" 2>&1 &
```

同一终端 tail -f 对应 launcher 文件即可；每组正式 log 同时保存在 logs，
config/checkpoint/stages.jsonl 和 TEST/VALIDATION queries.csv、summary.json
在各自 experiment_runs 子目录中。脚本支持 `--dry-run utility 0 ICEWS14s 1`，
只验证参数、不创建 run-dir、不触碰 CUDA。

## 2026-10-16 实现验证

`scripts/check_baseline.sh`：29 项 synthetic CPU 检查全部通过，29 个 Python
文件编译通过，主入口/外层入口/冻结诊断入口的 --help 均通过。
实际双支路 loss backward、guided mean generation、CPU saved-tensor offload、
router 翻转导致评分切换、权重 reload 后同分、optimizer 下一阶段更新一致、
空 replay/首阶段/单事实/小样本、逆关系、无标签输入和真实主循环 hook 均检查。
OFF 与固定基线的 loss、梯度、prediction、BN、RNG 逐元素一致。

此外，冻结诊断使用 PGR 已完成的真实 ICEWS14s mean+Shared checkpoint
执行 `--check-only --gpu -1`：配置/数据/cache/权重键及两处 adapter 迁移通过。
没有真实数据 forward 或真实数据参数更新；没有启动正式训练、占用 GPU。

仍未验证：真实 CUDA/DGL 完整训练的峰值显存、运行时间、长序列数值稳定性，
以及预更新效用能否迁移到正式更新后的评分。冻结诊断和后续三组实验正是
用来检验后者；当前没有 CUR 性能提升的结论。
