# PGR v1：Predictive-Geometry Replay

## 起点和证据边界

独立分支 `codex/pgr`，从审计后的 `codex/original-baseline`
`23dd43dd914a783731c7f39e3fd66491d2aba429` 建立。
保留作者上游 `c5bbc87467bb22d266b664d30eefb8184c1d3104` 的科研流程和
本地 CPU/device/空样本兼容修复，见 [BASELINE.md](BASELINE.md)。没有合并
FAHC、ECRI、QCDR、BCR、DCR、RTER 或 RAHA 方法分支。
为同代码对照，重新实现 mean 和 Shared 两个简单控制；独立诊断脚本仅允许
只读加载已审计的旧 Shared checkpoint，并明确迁移两个 adapter 键名。

DCR mean 和 Shared 的已有正结果共同提示：应研究生成表示进入预测器的方式。
它们没有证明“末端噪声是主要错误来源”，也没有证明两者组合有收益。
此前的 PGR 只有框架，没有固定学习规则。本次 **v1 是明确指定的新研究实现**：
阶段开始时估计训练损失对回放噪声的敏感子空间，只衰减这些方向的末端扰动，
同时训练共享线性残差适配器。尚无本方法的真实训练成绩，不保证提升。

## 数学定义

保持原始 HCP 抽样、subject filtering、事实数量、denoiser、guidance 和均值聚合。
固定一份 HCP 和一次 guided generation，头表示为 a_i，尾表示为
m_i + sigma_i epsilon_i。关系通道不被 PGR 修改。

按 Original 的出现次数 n_e = n_head[e] + n_tail[e] 聚合：

```
mu_e = (sum_{h_i=e} a_i + sum_{o_i=e} m_i) / n_e
v_e  = sum_{o_i=e} sigma_i^2 / n_e^2
Cov_terminal[b_e | fixed HCP, denoiser, guidance] = v_e I
```

v_e 只描述这一次采样的**末端条件方差**；不代表事实真实性、关系演化、
遗忘程度或全部生成过程的不确定性。不按 v_e 选事实或预算。

总映射为 A_t = I + U_t V_t，U 初始为零，V 非零随机初始化，无 bias、
activation、dropout 或额外归一化。只隔离新模块的 CPU 初始化 RNG；
不隔离 Original 训练/验证的随机流。d=200、rank=16 时增加 6400 个训练参数。

在阶段 t 的正式 predictor 更新之前，用已有冻结特征模型的上一阶段最佳参数，
在当前训练历史窗口上预测当前 train 和本次历史 HCP 中各最多 64 条事实。
HCP 已有逆关系先还原为 forward，再统一构造正向/逆向 query。**它们只用于
几何估计，不改变正式 replay 样本**。对两个方向分别计算平均 CE：

```
g_d,e = d L_d(A_{t-1} B) / d B_e | B=mu, d in {forward,inverse}
C_t   = 1/2 sum_d sum_e v_e g_d,e g_d,e^T
Q_t, lambda_j = top-r eigenvectors/eigenvalues of C_t
tau_t = trace(C_t) / d
w_j   = strength * lambda_j / (lambda_j + tau_t)  (zero geometry => zero w)
S_t   = I - Q_t diag(w) Q_t^T
z_tail_i = m_i + S_t (sigma_i epsilon_i)
bank_e = A_t (mu_e + S_t epsilon_aggregate,e)
```

梯度穿过现有 adapter、图传播、融合、归一化和 entity decoder；不构造巨型
Jacobian。C 是**方向分开的 batch CE 梯度外积代理**，不是逐 query Fisher、
Hessian 或精确 margin covariance。方向内梯度仍可能抵消。无需探测更新、
新 teacher、KD、额外训练 loss 或新的 replay memory。

选择衰减高敏感方向，是在保持条件均值时减少该梯度代理的扰动方差；
不是关于 CE、rank、MRR 必然改善的定理。v1 不解决其他生成噪声来源。

## 数据流和时序

1. Original 构造 t 之前的训练历史图，得到当前阶段 train facts。
2. `main.py` 调用 `PGRRuntime.begin_fit`，暂存本阶段训练上下文。
3. Original `gereration_feature` 原样选择过去 HCP，执行一次 denoiser/guidance，
   并原样抽取末端噪声。均值控制也消耗该次噪声，保证随机调用数一致。
4. `DiffuRec.p_sample` 在非 `vl` 的 replay 分支调用 PGR。geometry 首次调用
   计算 C、Q、w，然后将 detached CPU 张量存入本阶段 catalog。
   估计时模型临时 eval，恢复原有混合 train/eval 状态、BN buffers、RNG、
   `select_entity/h/h_0`；不写参数 `.grad`。使用的是 Original guidance 刚调用后
   的 buffers、尚未进行本阶段 predictor 更新的参数，不是假设 pristine buffers。
5. PGR 作用于已抽取的同一份末端扰动；头通道、guided mean 和关系通道保持原样。
6. `end_fit` 清除训练图和标签引用；没有 replay 的阶段记录空投影。
7. Original predictor loop 的 `select_entity` 在原均值聚合前应用共享 A；
   正式 task/replay CE、mu、mu_r、早停、best 权重回退和 Adam 流程保持原样。
   A 在此阶段正常训练；Q/w 在整个阶段冻结。
8. Original diffusion 训练保持原样。`args.vl=True` 的扩散验证也绕过 PGR sampler。
9. 阶段末保存 best 特征权重、last Adam、RNG 和 `auxiliary.pgr` catalog。
10. 验证和最终评估按 query 时间索引读取**训练时保存的 Q_t/w_t**。
    不重估方向，不读取验证/test 目标来修改 projector。没有 catalog 会报错，
    不借用最后时间片的方向。阶段 1 若训练时无 replay，则使用 identity 投影。

无未来训练事实参与某个训练阶段的 PGR 估计。**保留 Original 的 retrospective
最终评估协议**：首个被评估 slice 的扩散 guidance 仍来自最终训练历史；不能
声称整个 DGAR 推理流程变成严格在线。predictor 历史窗口、filter 和指标口径不改。
末尾 VALIDATION 仍是 TEST 后的补充报告，保持与同批 Original 一致。

## 代码和状态

| 文件 | 改动 |
|---|---|
| `src/pgr.py` | adapter、方差、batch 梯度几何、收缩、阶段 catalog、恢复校验 |
| `src/rrgcn.py` | 在全部 Original 参数初始化后添加 adapter；`select_entity` 接入 |
| `diffusion/difffu_21.py` | `p_sample` 原始采样后处理，仅非 vl replay |
| `diffusion/model_21.py` | 传递生成的阶段编号，Original 分支保留 |
| `src/main.py` | 初始化、begin/end、checkpoint catalog、可选被动报告 |
| `src/cli.py` | 独立参数和合法配置校验 |
| `src/runtime.py` | OFF 仍 `official-dgar-v1`；ON 使用独立版本，记录配置/哈希 |
| `src/pgr_diagnostics.py` | 无额外推理的阶段统计和最终逐 query 输出 |
| `scripts/run_pgr_experiment.sh` | 独立 run-dir/log、参数校验、dry-run、pipefail |
| `scripts/diagnose_pgr.py` | 已有 Shared 的一次生成、多控制、末端噪声诊断 |
| `tests/test_pgr.py`、`tests/reference/pgr_baseline*` | CPU 回归及基线源码指纹 |

catalog 只保存 Q[d,r]、w[r]、标量统计，不保存图、标签、计算图或全实体梯度。
默认 d200/r4/T365 的 Q/w 约 1.2 MB（未计 Python/序列化开销），在 CPU 保存，
GPU 只缓存当前阶段。无需另外存 replay 数据或 teacher 参数。
额外训练开销：每个有回放阶段一次图 forward、两次关于 bank 的反向，以及
d×d eigendecomposition；每个 replay 的低秩乘法 O(facts*d*r)。实际 GPU 时间和
峰值显存尚未测量，不能用 CPU smoke test 估算提升和成本。

`latest.pt` 可通过原 `--resume .../latest.pt --run-dir ...` 恢复；同一源码、
科学参数、seed、data/cache 是前提。catalog 检查完整阶段集合、shape、finite、
权重范围、正交性；`--test` 也从 latest 恢复目录。不要把 RAHA checkpoint 用作
PGR 训练起点；旧 checkpoint 只用于独立诊断脚本。

## 参数和控制

| 参数 | 默认值 | 含义 |
|---|---:|---|
| `--pgr-mode` | `original` | 总开关及实验组 |
| `--pgr-adapter-rank` | 16 | 所有带 adapter 组的 U/V rank |
| `--pgr-risk-rank` | 4 | 被衰减子空间维度 |
| `--pgr-geometry-facts` | 64 | 当前训练和历史 HCP **各自**最多抽多少条作估计；不是回放预算 |
| `--pgr-risk-strength` | 1.0 | 收缩强度上限，范围 [0,1]；首轮固定，不扫描 |
| `--pgr-diagnostics` | false | 仅记录；不参与训练决策 |

| mode | 条件均值映射 | 末端扰动 | 目的 |
|---|---|---|---|
| original | I | 原始 | Original；不建 runtime 或 adapter |
| mean | I | 0 | DCR mean 简单对照 |
| shared | A | A epsilon | Shared 简单对照 |
| mean-shared | A | 0 | 检查两个旧正结果的组合是否已足够 |
| geometry | A | A S epsilon | 完整 PGR v1 |
| geometry-random | A | A S_random epsilon | 相同收缩谱、私有随机正交方向 |
| geometry-isotropic | A | A c epsilon | c²=trace(S²)/d，相同原始扰动总能量的均匀缩小 |

等能量指**期望意义上、adapter 之前**的末端噪声总能量，不是逐次样本范数，
也不保证经非正交 A 或 decoder 后仍然等风险。随机方向组仍计算训练几何来
取得相同 w，避免把计算预算和谱同时改变。它们是机制控制，不是默认 Original。

关闭方法时不创建新参数、不消耗新随机数、不改 HCP 和 loss。新增参数从
Original 的科学 settings 中排除；仅 source hash 因新增源码而变化。
OFF 与 `23dd43d` 的实际参数、loss、梯度、预测、随机流，以及作者循环/最终
评估/Adam 的回归由 CPU 测试核对，不能据此保证某次 CUDA 分数恰好等于论文均值。

## 有限实验方案

### 0. 先看机制：不重训 Shared

独立诊断只加载已完成的 Shared，默认 slice 60/120/180/240/300。
每个 slice 一次生成，固定 HCP/denoiser/guided mean，比较：

- learned A；
- 恒等方向、逐实体聚合 bank 范数匹配 learned；
- A 的随机正交共轭（保留 A 的奇异谱），再匹配 bank 范数；
- HCP 训练事实拟合的固定头/尾 norm gains；
- mean+Shared。

保持已学 alpha。额外只重采四份末端噪声，输出正逆向 top2 score-gap 方差、
top1 翻转、filtered rank 范围、经验/解析 bank 方差比。竞争者由 mean scores
决定，目标只用于评分后统计。四次 MC 只能粗查数量级，不是精确协方差估计。
使用每片自身过去训练窗口进行 guidance、隔离每片随机流；**不是论文 Average**。
旧最终 checkpoint 已见全训练时间轴，本诊断也不能充当在线泛化证明。

若范数匹配或固定 gain 复制了学习映射的收益，就削弱“特定学习方向必不可少”
的解释。若 learned 胜过控制，也只能说明冻结预测器依赖其方向，不能独立
证明噪声收缩必然有效。最终须看下面的训练干预和随机/等能量对照。

### 1. 主筛选：三组，单一 seed，固定参数

同服务器、同 GPU、同源码、同 seed=1、同 data/cache，依次跑：
`original -> mean-shared -> geometry`。不要先扫描 rank、strength、seed。
旧结果可帮助判断数量级，正式差值用本轮同源控制。首要比较 geometry 是否
超过 mean-shared，而不是仅超过 Original 或失败过的 FAHC。

预先采用的工程判断门槛（不是显著性检验）：

- 补充 VALIDATION macro MRR 比 mean-shared 至少 +0.2pp；
- 同批最终 TEST 的改善方向一致，forward/inverse 都报告，不能隐藏单侧下降；
- 如果 macro 改善主要来自逆向、正向继续明显下降，不算解决双向利用问题；
- 若 mean-shared 已经更好，停止 PGR 几何模块，只保留简单正结果。

所有命令按 Original 流程先 TEST 后补充 VALIDATION，不再从 TEST 挑参数。

### 2. 仅主筛选有希望才做四项确认

`mean`、`shared`、`geometry-random`、`geometry-isotropic`。
完整方法应超过最强简单控制，且优于随机/等能量控制；否则不能把收益归因
于 predictive geometry。多个指标等价或低于简单组合就停止，不继续无界调参。
ICEWS14s 通过以后再在 ICEWS18 跑 Original / mean-shared / geometry 三组。

### 输出

每次训练自动创建：

```
logs/<dataset>-pgr-<mode>-seed1-<time>_<pid>.log
experiment_runs/<same-id>/config.json, model.pt, diffusion.pt, latest.pt
experiment_runs/<same-id>/diagnostics/stages.jsonl
experiment_runs/<same-id>/diagnostics/test-*/queries.csv, summary.json, replay.jsonl
experiment_runs/<same-id>/diagnostics/validation-*/queries.csv, summary.json, replay.jsonl
```

summary 区分宏 MRR（1..T-2）、pooled MRR/Hits1/3/10、Current、
forward/inverse、repeated/unseen、逐片成绩。阶段 JSONL 中 trace 和 after 是
**batch CE 噪声代理**，不能直接解释为预计 MRR 增益。

### 命令（只提供，代码修改时不启动）

```bash
cd /data1/zihao/DGAR-code/.worktrees/pgr
export DGAR_PYTHON="$(command -v python)"
export DGAR_DATA_ROOT=/data1/zihao/DGAR-code/data
mkdir -p logs experiment_runs

# 零训练 dry-run，检查环境中的 Python、路径及参数。
bash scripts/run_pgr_experiment.sh --dry-run geometry 0 ICEWS14s 1

# 主筛选，三组串行。GPU 0 空闲后再提交；此脚本不会等待 GPU。
setsid bash -c '
set -euo pipefail
for PGR_GROUP in original mean-shared geometry; do
  bash scripts/run_pgr_experiment.sh "$PGR_GROUP" 0 ICEWS14s 1
done
' < /dev/null > "logs/ICEWS14s-pgr-screen-launcher-$(date +%Y%m%d_%H%M%S)_$$.log" 2>&1 &

# 主筛选通过后才运行，不要和上一条同时提交到同一 GPU。
setsid bash -c '
set -euo pipefail
for PGR_GROUP in mean shared geometry-random geometry-isotropic; do
  bash scripts/run_pgr_experiment.sh "$PGR_GROUP" 0 ICEWS14s 1
done
' < /dev/null > "logs/ICEWS14s-pgr-controls-launcher-$(date +%Y%m%d_%H%M%S)_$$.log" 2>&1 &

# 可选冻结诊断：路径须指向实际已有、完整的 Shared 目录。
export PGR_SHARED_RUN=/data1/zihao/DGAR-code/.worktrees/raha/experiment_runs/ICEWS14s-raha-shared-r16-seed1-20261015_005305_1320520
export PGR_DIAG_ID="$(date +%Y%m%d_%H%M%S)_$$"
setsid bash -c '
set -euo pipefail
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "$DGAR_PYTHON" -u scripts/diagnose_pgr.py \
  --run-dir "$PGR_SHARED_RUN" --data-root "$DGAR_DATA_ROOT" --gpu 0 \
  --stages 60,120,180,240,300 --noise-trials 4 \
  --output "experiment_runs/PGR-frozen-$PGR_DIAG_ID" \
  2>&1 | tee "logs/PGR-frozen-$PGR_DIAG_ID.log"
' < /dev/null > "logs/PGR-frozen-launcher-$PGR_DIAG_ID.log" 2>&1 &
```

另一台服务器先获取分支（已有同名工作区时优先使用它）：

```bash
cd /data1/zihao/DGAR-code
git fetch origin codex/pgr
git worktree add -b codex/pgr .worktrees/pgr origin/codex/pgr
```

如果本地已有 `codex/pgr`，不要再次 `-b`；在其干净工作区执行
`git pull --ff-only origin codex/pgr`。无需复制旧方法日志/模型来训练新方法；
冻结诊断是唯一需要已有 Shared checkpoint 的独立入口。

## 本轮验证与局限

不运行真实训练或真实数据 neural forward。CPU 验证包括：作者循环/Adam/RNG，
实际 Original 源码的参数/loss/grad/score 对照，新 adapter 两阶段梯度，
方差/空集/单事实/self-loop/逆关系，低秩收缩和随机谱/等能量控制，
实际 guided generator + RGCN + PGR + backward，真实 main 的阶段 hook，
静态图，扩散验证 bypass，恢复 catalog，异常清理，以及冻结诊断小张量路径。
另只读加载真实 Shared 权重与 data/cache，执行 `--check-only`，不调用 forward。

本轮 `scripts/check_baseline.sh`：26 项 CPU 测试通过，含 13 项既有 baseline
检查；`--help`、28 个 Python 文件编译、两个 shell 脚本语法和全部七组 dry-run
均通过。真实 Shared 的只读 `--check-only` 通过。没有启动正式训练、真实数据
forward 或 GPU 作业。

仍须实测：CUDA/DGL 反向和显存；真实数据 C 是否常为零；batch 代理能否预测
实际排序波动；阶段冻结 Q 在 A 与最终模型变化后是否过时；是否需要保留噪声
作为正则化；性能能否超过 mean+Shared。低秩扰动方向和时间演化不是一回事，
本实现不据此声称识别 stable/evolving knowledge。
