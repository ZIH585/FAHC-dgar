# RAHA：角色分离的历史表示适配

## 起点和范围

独立分支 `codex/raha` 从 `codex/original-baseline` 的
`23dd43dd914a783731c7f39e3fd66491d2aba429` 新建。没有继承 FAHC、ECRI、
QCDR、GRB、BCR、DCR 或 RTER。默认模式 `original` 不创建适配器。
作者协议及其回顾式评估限制仍见 [BASELINE.md](BASELINE.md)。

DCR mean 在既有 ICEWS14s 完整训练上是积极结果，但尺度诊断未证明独立
去噪收益。RAHA 测试另一条假设：生成历史中头/尾表示的分布或坐标差异，
使直接平均不够合适；分别进行任务驱动的适配可能优于共同适配。
这是待验证假设，不声称之前的日志已经证明了头尾混合有害。

## 方法和数据流

对每条已由 Original HCP 检索、扩散生成的历史事实，取头/尾向量：

```
A_H(z) = z + U_H V_H z
A_T(z) = z + U_T V_T z
b[e] = (sum_{head=e} A_H(z_head) + sum_{tail=e} A_T(z_tail)) / (n_head[e] + n_tail[e])
```

`shared` 对头、尾共用一个 `A`。保持按出现次数加权，不改成两个角色各占一半。
`role` 用两个独立适配器。这里的角色是历史 triple 中的位置，不是当前预测
方向；若原缓存有逆关系，仍按该条历史 triple 的编码处理，不额外翻转。

数据流：Original 随机时间片 → 原 subject 检索 → 原事实顺序 → 原扩散
采样和引导 → 头/尾适配 → 原 `select_entity` 均值聚合 → 原每步 RGCN
历史混合 `alpha*h + (1-alpha)*b` → 原 time gate → 原 decoder 和全候选排名。

训练调用：`main.py` 原 chronological loop → `model.get_loss` → `forward`
→ `select_entity` → 适配器。仅使用 Original current/replay CE 与静态约束
更新适配器及预测器；无新 loss，无 KD，无 teacher 筛选。Original 返回但未
进入总 loss 的 replay MSE 也未被新加入优化。`get_loss` 为此辅助返回值计算
的历史 bank 同样适配，但不会额外影响总目标。

下一阶段的 Original 冻结特征预测器加载 best 权重时也加载适配器，因此
原扩散引导能够通过已学习适配器计算输入梯度，参数自身仍冻结。
这保持“训练用的融合”和“引导用的融合”一致；不是新增旧模型蒸馏。
最终评估使用训练得到的适配器，不继续学习适配器。

无新跨阶段事实缓存、importance 分数或未来统计。适配器权重及 Adam 状态
已包含在现有 checkpoint 中；仍是 best 权重回退、last Adam 状态保留。
阶段 1/空 replay 不调用适配器。relation token、inverse ID、HCP 范围、
候选集合、原关系条件索引、train/eval 模式和最终评估顺序均不修改。

## 初始化与成本

`V` 随机初始化，`U` 初始化为零，因此初始 `A(z)=z`；两者同时为零会使
梯度无法启动，所以没有这样做。角色两支的初始权重相同，但 Parameter
独立。构造模块时临时保存并恢复 CPU RNG；之后训练/验证仍共用原随机流。
所有 Original 参数先按原顺序初始化，diffusion 的后续初始化也不被移位。
不同 rank 的低秩投影不同，但两者初始输出与主干权重均相同。

共享适配器参数量为 `2*d*r`，角色适配器为 `4*d*r`。默认实验：
`d=200`，shared r16 与 role r8 均为 6,400 参数。FP32 参数约25 KiB；
Adam 两个状态约50 KiB，梯度约25 KiB，另有 Original 冻结预测器中的
适配器副本和临时激活。实际额外显存/耗时未通过 GPU 训练测量。
无需复制整个新模型；无新增候选采样或额外 decoder 预测。

## 参数

| 参数 | CLI 默认 | 用途 |
|---|---|---|
| `--raha-mode` | `original` | `original / shared / role` |
| `--raha` | 不启用 | `--raha-mode role` 的别名 |
| `--raha-rank` | 8 | 单个适配器的瓶颈维度 |
| `--raha-diagnostics` | 关闭 | 被动记录统计和最终逐查询排名 |

方法开启要求独立 run-dir、单次 chronological pass、原 add feature fusion。
拒绝禁用 replay、grid search 或非法 rank。脚本对 shared 默认选16、role
默认选8，这是等参数量对照；CLI 通用默认值仍是8。
Original 的有效配置不包含 RAHA rank/mode；观察开关不参与 checkpoint
科学配置。源码指纹包含新增文件，因此不能把别的分支旧 run-dir 直接拿来
续跑或 --test。OFF 路径数值对照通过不代表旧 checkpoint 来源无需核对。

## 输出与检查

启用观察时，run-dir 中新增：

- `diagnostics/stages.jsonl`：每阶段 best 适配器参数范数、原/适配后头尾
  范数、残差范数、事实数。resume 时追加；中断后重做阶段可能有重复记录。
- `diagnostics/test-<时间戳>/` 和 `validation-<时间戳>/`：`queries.csv`、
  `replay.jsonl`、`summary.json`。每次最终评估创建新目录，不覆盖旧结果。
- CSV 包含正/逆向、原始/过滤 rank、是否在 `train[:t]` 出现过该事实。
  标签仅在预测和排名完成后用于分组，绝不回传采样、引导或 loss。
- summary 记录 `1..T-2` 宏 MRR、全 query pooled MRR/Hits、Current，
  以及正向/逆向、重复/未见事实的 pooled 分组指标。

默认路径不记录这些文件。观察器不开关 train/eval，不产生随机数或保留
计算图；只在阶段结束或评估片结束做一次统计，不在每次适配器 forward
里 `.cpu()`。观察有额外统计开销，正式三组都开启。
观察器最终评估期间的 CPU 前缀事实集合随历史事实数增长，仅用于分组，
不作为模型 replay memory，不声称带观察器的全流程总存储固定。

CPU 核验命令（只运行 synthetic 小模型）：

```bash
export DGAR_PYTHON="$(command -v python)"
bash scripts/check_baseline.sh
bash scripts/run_raha_experiment.sh --dry-run role 0 ICEWS14s 1
```

tests 中冻结了该基线的 `rrgcn.py` 和 blob SHA，用真实小型 loss/backward/
predict 检验 OFF 参数、数值、梯度、buffer、RNG；另检验恒等初始化、真实
适配器更新、引导输入梯度、checkpoint/Adam 恢复、空 replay、单事实、
逆关系、dtype、观察器无干预及 CSV 汇总。CUDA 性能与实际提升未验证。

本次检查结果：29项 CPU 测试全部通过（最后一次1.418秒），28个 Python
文件编译、两个 --help 和 runner Bash 语法检查通过；三组 dry-run 参数
校验通过。17个原 src/diffusion/rgcn 模块逐字节未变，RecurrentRGCN 除
构造函数和 select_entity 外的所有方法 AST 未变。没有加载真实数据 epoch、
启动训练、使用 GPU 或生成本工作区的实验输出。

## 首轮正式实验

只固定 ICEWS14s、seed1、同一个 GPU 顺序跑三组，不先扫 rank/lr。Original
和方法从头初始化，不加载 FAHC/DCR 或以前其他方法的权重。

| 组别 | rank | 新参数量 | 回答的问题 |
|---|---:|---:|---|
| Original | 无 | 0 | 是否真的超过当前基线 |
| Shared | 16 | 6400 | 通用表示适配是否已经足够 |
| Role | 每支8 | 6400 | 同容量下角色分离是否有额外价值 |

在激活 `dgar_env` 后执行。下面是命令说明，代码实现期间不启动训练。

```bash
cd /data1/zihao/DGAR-code/.worktrees/raha
export DGAR_PYTHON="$(command -v python)"
export DGAR_DATA_ROOT=/data1/zihao/DGAR-code/data
mkdir -p logs
setsid bash -c '
set -euo pipefail
bash scripts/run_raha_experiment.sh original 0 ICEWS14s 1
bash scripts/run_raha_experiment.sh shared   0 ICEWS14s 1 16
bash scripts/run_raha_experiment.sh role     0 ICEWS14s 1 8
' < /dev/null > logs/ICEWS14s-raha-launcher-"$(date +%Y%m%d_%H%M%S)".log 2>&1 &
```

脚本为每组创建唯一 log/run-dir，使用 `python -u`、`tee`、`pipefail`；
某组失败就停止，不让后续组悄悄接着跑。GPU改为当前空闲的实际索引，
无需额外加 seed。其他服务器将 cd 替换为实际 RAHA checkout 路径。

```bash
ls -t logs/ICEWS14s-raha-*.log
tail -f logs/某个具体实验日志.log
```

三组结束后，用脚本打印的实际 run-dir 替换下列占位符：

```bash
python scripts/summarize_raha.py \
  experiment_runs/实际original目录 \
  experiment_runs/实际shared目录 \
  experiment_runs/实际role目录 \
  --output experiment_runs/raha-summary-"$(date +%Y%m%d_%H%M%S)"
```

汇总只用标准库读文件，不加载模型、不占 GPU。检查同源码、数据、cache、
主干配置和 seed，核对 query 身份/顺序及 CSV 与 summary，再输出 metrics.csv。
不允许把旧协议日志加入三组比较。

## 预先约定的继续/停止标准

以下是研究筛查门槛，不是显著性检验。以完整 VALIDATION 为决定依据：

1. role 相对 Original 宏 MRR 至少 +0.3pp，pooled MRR 也为正；
2. role 胜过 shared16；若差距不足0.2pp，暂不认为角色创新值得扩展；
3. 逆向和未见事实 pooled MRR 相对 Original 的损失各不超过0.3pp；
4. 适配器 residual/up 权重有更新且有限，收益不能只出现在极小 Current。

正式脚本同时保留最终 TEST，作为冻结配方的结果报告，不据此反复改 rank。
只有 VALIDATION 支持继续，才考虑 ICEWS18 的同三组；TEST 若方向相反，
结论应收缩为不稳定，不能只报告 VALIDATION。role 与 shared 同样好时
收缩为通用表示适配；两者均无收益就停止。DCR mean 是已完成的强参照，
本版未组合；角色方法独立成立后再做组合对照，不能把 mean 收益算成 RAHA。

另一台服务器同步：

```bash
cd /data1/zihao/DGAR-code
git fetch origin
git worktree add --track -b codex/raha .worktrees/raha origin/codex/raha
```

若已有该分支工作区，在其目录先检查 `git status --short`，确认没有源码
修改后再 `git pull --ff-only`；不对旧实验工作区 reset/clean。
