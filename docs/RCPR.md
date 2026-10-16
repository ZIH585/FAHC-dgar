# RCPR-DGAR：关系条件原型回放

实现日期：2026-10-16。分支 `codex/rcpr`，工作区
`/data1/zihao/DGAR-code/.worktrees/rcpr`。
直接从 `codex/original-baseline@23dd43dd914a783731c7f39e3fd66491d2aba429` 建立。
没有继承旧方法分支的训练实现。原工作区、现有实验及活跃作业均保留。

本次完成代码、CPU 检查及极小 synthetic smoke test；没有启动真实训练、GPU forward 或正式筛查。
方法收益、实际显存和耗时均待实验。Original 的作者协议及回顾式评估限制见 [BASELINE.md](BASELINE.md)。

## 方法与实现选择

以 mean+Shared 为强简单控制。仅将非 `vl` 生成路径的末端采样替换为其引导后条件均值；
仍消耗同一次 Original 随机噪声，上游 dropout/初始噪声等随机性仍存在。
Shared 使用 `A(z)=z+U V z`：U初始化0，V非零随机，保持初始 identity 并允许梯度启动。
新增模块初始化在私有 fork_rng 中，不移动 Original 主干、后续模型与扩散模型的初始化随机流。

沿用 Original HCP 时间片抽样、按当前主体检索的事实集合、事实顺序及全候选集合。
对其每条生成 triple 的两个 occurrence，定义：

- 头 occurrence：实体 s，上下文 r，特征 A(z_head)。
- 尾 occurrence：实体 o，上下文 inverse(r)，特征 A(z_tail)。

inverse(r)在 `[0,2R)` 内作 `r+R mod 2R`，允许原缓存包含逆事实，自环保留两个 occurrence。
原来的实体 occurrence 均值 b 不变，同时构造稀疏的实体–关系原型：

\[
 p_{e,c}=\frac{\sum_{i:e_i=e,c_i=c}z_i}{n_{e,c}},\qquad d_{e,c}=p_{e,c}-b_e.
\]

原图传播继续使用 b。额外信息只通过候选实体评分残差进入预测：

\[
 s(q,o)=s_{\mathrm{MS}}(q,o)
 +\frac{(W_q\phi(q))^\top W_p d_{o,\operatorname{inverse}(r)}}{\sqrt{k}}.
\]

phi 是原 ConvTransE 最后一次 ReLU 后、候选矩阵乘法之前的查询向量。
同一次 decoder 调用同时返回 score/phi，不重复 BN、dropout 或卷积。
W_q=0，W_p非零随机，因此初始新评分等于 mean+Shared；后续二者都可学习。
没有对应候选原型时加0；第一阶段及空回放不使用新评分。

原实体 CE、两次原有历史实体 CE、关系 CE 与静态约束的组成和权重保持原配置。
新评分参与所有实体 CE，关系 decoder 沿用原接口。没有新增 KD、old model 副本、辅助 loss 或回放事实预算。
原有 feature teacher 依然存在；其权重也包含 adapter/score head，并按原流程载入前阶段 best 参数。

## 训练、评估与引导的接线

```text
Original 时间划分 / HCP 检索
  → 原 diffusion 与原 frozen feature predictor 引导
  → 末端条件均值（消耗原同一次噪声）
  → detach 生成结果
  → Shared Adapter
  → 原实体均值 bank → 原 RGCN / relation GRU / time gate
  → 原 ConvTransE → base scores + phi
                     ↘ 稀疏实体–关系原型 residual → 新 score correction
  → current / HCP / 双向 HCP 的原实体 CE
  → 原任务权重、clip、Adam、验证早停
  → 原 diffusion 训练与验证
  → 保存 best feature weights + last Adam + RNG
```

`get_loss`、`loss_of_reply`、`predict` 共用 `RecurrentRGCN.score_entities`。
`diffusion/difffu_21.py:p_mean_variance` 的 predictor 引导也使用该评分，
并通过原型和均值保留对当前生成尾表示的梯度；不 detach 原型后再求引导梯度。
引导只求输入梯度，不更新冻结 feature predictor 的参数。

这相当于让训练、最终 TEST、补充 VALIDATION、独立 `--test` 和生成引导识别同一个新评分。
`vl=True` 的 diffusion 自身验证路径仍沿用其原始预测流程。

## 因果边界、状态和开销

新原型只来自本次 Original HCP 的历史训练事实，不使用 query target 选原型或提供查询特征。
评分头只接收 `[subject,relation]` 两列。标签只用于原任务监督及排名计算。
训练 HCP 仍使用作者既定边界；没有让当前 TRAIN target 进入过去的图或原型。

最终评估保留 Original 的回顾式历史传递：首个评估片的 diffusion guider 可能使用最终训练阶段的图。
这项作者既有局限没有被本方法修正，不能宣传所有支路都是严格逐 query 因果评估。

原型每次 forward 重建，不存入 checkpoint；学习到的 adapter/head 和对应 Adam 在原 checkpoint 中保存。
只缓存一个整数 HCP layout，检查 tag 对象和 in-place version；不跨 forward 缓存可微特征。
loss/predict/引导后清除临时 context，防止新分支持有上一次计算图。
resume 后根据重新生成的合法 HCP 重建原型，不恢复旧坐标原型。

h=200、adapter rank=16、score rank=16：adapter 6400 参数，head 6400 参数，共12800。
mean-head/prototype/shuffled 参数量完全相同。
稀疏 pair 数最多为 occurrence 数；没有分配 `entities × relations × hidden` 稠密 bank。
score-chunk 只分块修正矩阵，不改变 Original decoder 的 BN batch。
训练 autograd 仍会保存需要的分块中间量及原全候选 logits；分块不保证整个模型适配24GB显存。
没有实测 CUDA 显存/速度，不给出固定百分比的开销承诺。

## 模式与参数

|参数|默认值|作用|
|---|---|---|
|`--rcpr-mode`|`original`|original / mean-shared / mean-head / prototype / shuffled|
|`--rcpr`|不启用|`--rcpr-mode prototype` 的便捷别名|
|`--rcpr-adapter-rank`|16|Shared 的低秩维度|
|`--rcpr-rank`|16|额外评分头低秩维度|
|`--rcpr-score-chunk`|256|修正矩阵的 query 行数分块|
|`--rcpr-shuffle-seed`|0|打乱关系控制的私有 RNG 偏移|
|`--rcpr-diagnostics`|关闭|被动阶段统计、最终逐 query 和 summary|

`original` 不创建 adapter/head，不消耗新增初始化 RNG；方法选项从 Original 科学设置中排除。
它通过与固定基线 fixture 的真实 loss/梯度/预测/RNG 数值对照。
新模式要求独立 run-dir、原 add fusion、HCP/feature replay、单次 chronological pass。
不兼容配置直接报错。新 run 的 provenance 是 `rcpr-{mode}-v1`，源代码/数据/cache 仍严格检查。

`mean-head`：与 prototype 相同支持位置及同参数量，评分头输入实体均值，排除只增加容量的解释。
`shuffled`：在每个实体、每个原头/尾角色内部打乱 occurrence 的关系标签。
保留实体均值、entity-role-context 计数、候选覆盖、模块参数量；私有 RNG 不移动训练随机流。
这保留了角色层面信息，能更严格检验细粒度关系标签的价值。

## 文件

|文件|作用|
|---|---|
|`src/rcpr.py`|adapter、稀疏布局/原型、评分残差、scatter统计|
|`src/rrgcn.py`|模型状态、原均值适配、context生命周期、统一实体评分|
|`src/decoder.py`|可选返回同一次 forward 的 phi；默认返回值不变|
|`diffusion/difffu_21.py`|均值终点、同评分的可微引导|
|`src/main.py`|可选被动阶段/最终评估记录|
|`src/cli.py`|独立参数与合法性检查|
|`src/runtime.py`|方法协议标识与科学配置；原 checkpoint 流程不变|
|`src/rcpr_diagnostics.py`|宏平均、Hits@1/3/10、方向、重复性、JSONL/CSV|
|`src/rcpr_probe.py`|冻结诊断的 TRAIN 拟合与预设 gate|
|`scripts/diagnose_rcpr.py`|读取已审计 mean+Shared，跑5阶段四组诊断|
|`scripts/run_rcpr_diagnostic.sh`|单个前台诊断，自动生成日志/输出目录|
|`scripts/run_rcpr_experiment.sh`|单个正式实验；含仅解析的 dry-run|
|`tests/test_rcpr.py` 与 `tests/reference/rcpr_baseline*`|CPU 功能、基线、接线、恢复对照|
|`scripts/check_baseline.sh`|运行必需 CPU 检查，并编译新增 scripts|

## 首轮：先做冻结诊断

只读取已完成的 PGR mean+Shared 正控制，严格接受已审计 source SHA
`5e56dd4ae17f5ff08a0ad072f1bfeb2187f840f75adaf5d73bb0922bc63a481c`。
核对 config、completed stage、best/latest 一致、dataset 与 HCP 指纹；只迁移两项 adapter key 名。
这是受限的只读迁移，不是绕过正式训练 `--resume/--test` 的来源检查。

每个阶段只生成一次 **Original 当前 TRAIN subjects 检索的 HCP**，四组共用其全部特征。
只从当前 TRAIN 与这份历史 TRAIN HCP 各最多128条事实，加逆关系，拟合新的评分头。
主干、adapter、diffusion 参数均冻结。每组固定100步，不用 VALIDATION 选步数或 checkpoint。
验证使用该阶段全部 VALIDATION queries 和原始 integer filtered-rank 函数；不评分 held-out TEST。

|组|特征/参数|
|---|---|
|mean-shared|无额外评分头|
|mean-head|实体均值＋同容量评分头|
|prototype|真实关系残差＋评分头|
|shuffled|实体和角色内打乱关系的残差＋同容量评分头|

final backbone 已经过全时序训练，因此这是回顾式表征筛查，不能作为在线 CL 效果。
不把筛查 MRR 和论文完整 Average MRR 直接对比。

```bash
cd /data1/zihao/DGAR-code/.worktrees/rcpr
export DGAR_PYTHON="$(command -v python)"
export DGAR_DATA_ROOT=/data1/zihao/DGAR-code/data
export RCPR_GPU=0
export RCPR_LAUNCH_ID="$(date +%Y%m%d_%H%M%S)_$$"
mkdir -p logs experiment_runs

setsid bash -c '
set -euo pipefail
bash scripts/run_rcpr_diagnostic.sh \
  /data1/zihao/DGAR-code/.worktrees/pgr/experiment_runs/ICEWS14s-pgr-mean-shared-seed1-20261015_191853_919554 \
  "$RCPR_GPU" 60,120,180,240,300
' < /dev/null > "logs/rcpr-diagnostic-launcher-$RCPR_LAUNCH_ID.log" 2>&1 &

tail -f "logs/rcpr-diagnostic-launcher-$RCPR_LAUNCH_ID.log"
```

四组在一次命令内顺序完成。输出 `experiment_runs/rcpr-frozen-*/`：
`config.json`、`stages.jsonl`、`queries.csv`、`summary.json`。
summary 同时给真实/打乱关系解释的特征散度、TRAIN拟合CE、验证四项指标、方向、耗时与峰值allocated显存。
逐query的 seen_before 可用于报告历史未重复 triple；该标签不进入头训练。

继续标准（summary 的 `gate.pass`）：prototype 的阶段宏平均 MRR 相对其余三组均≥+0.3pp，
至少3/5阶段高于 mean+Shared，正向改善且逆向下降≤0.2pp。它是研发止损规则，不是显著性检验。
如果没通过，停止此版本，不继续扫 rank/steps/学习率。

## 第二轮：通过 gate 后正式训练一次

下面命令仅供后续执行，本次未启动：

```bash
cd /data1/zihao/DGAR-code/.worktrees/rcpr
export DGAR_PYTHON="$(command -v python)"
export DGAR_DATA_ROOT=/data1/zihao/DGAR-code/data
export RCPR_GPU=0
export RCPR_TRAIN_ID="$(date +%Y%m%d_%H%M%S)_$$"
mkdir -p logs experiment_runs

setsid bash -c '
set -euo pipefail
bash scripts/run_rcpr_experiment.sh prototype "$RCPR_GPU" ICEWS14s 1
' < /dev/null > "logs/ICEWS14s-rcpr-prototype-launcher-$RCPR_TRAIN_ID.log" 2>&1 &

tail -f "logs/ICEWS14s-rcpr-prototype-launcher-$RCPR_TRAIN_ID.log"
```

先复用已完成同口径 Original 与 mean+Shared 结果，核对显式seed、硬件/软件、数据/cache及设置。
无需因新分支机械重跑 Original。若环境改变或无法证实控制等价，单独跑匹配控制并明确原因。
同一 launcher 的控制命令是 `bash scripts/run_rcpr_experiment.sh mean-shared GPU ICEWS14s 1`；
Original 是 `... original GPU ICEWS14s 1`。

若正式 prototype 确实超过 mean+Shared，再按同参数运行 mean-head/shuffled 两个归因控制：

```bash
for RCPR_MODE in mean-head shuffled; do
  bash scripts/run_rcpr_experiment.sh "$RCPR_MODE" "$RCPR_GPU" ICEWS14s 1
done
```

此循环前台执行，若需后台可包在与上面相同的 setsid 模板中。
所有组固定 seed1，不默认追加 seeds。首轮不做多个数据集或参数网格。
正式看 `Final full-pipeline TEST` 的 Current MRR 与历史 macro Average MRR/Hits@1/Hits@10；
Hits@3 同时保留在 summary。方向与未重复 triple 分组是辅助解释，不能替代总体指标。
要求超过强简单控制并避免仅靠逆向收益掩盖正向损失，再考虑 ICEWS18。

## 静态验证、恢复和限制

已执行：

- `scripts/check_baseline.sh`：28项 CPU 测试全部通过，29个 Python 文件编译、主入口help通过。
- 真正 Original fixture 的初始化/参数/BN/loss/梯度/预测/RNG精确对照。
- 真实 tiny RGCN＋guided diffusion，以及 static graph forward/backward；新head实际被调用。
- 零初始输出与首/后续矩阵梯度；稀疏与稠密公式及输入梯度一致；chunk结果一致。
- 空回放、第一阶段、单事实、自环、逆关系、越界、query target独立性、私有shuffle。
- 训练—预测—权重reload、新head改变评估分数；checkpoint/Adam恢复下一步完全一致。
- 临时context清理、整数cache无计算图、被动记录不改变参数/RNG。
- 完整冻结诊断脚本在数条合成数据上通过；没有真实数据训练。
- 真实 ICEWS14s mean+Shared `--gpu -1 --check-only` 通过来源、权重、数据/cache及key迁移验证。
  仅加载，无forward/更新。构造原扩散模型时出现其既有 schedule divide-by-zero warning；未改变作者schedule。
- 两个 launcher dry-run、bash语法和Git空白检查通过。

CPU检查只能验证实现和连接，不能验证CUDA运行、真实显存、训练稳定性或MRR增益。
同源 run 恢复仍用完整科学参数及 `--resume RUN/latest.pt --run-dir RUN`；
独立评估用相同方法参数加 `--test --run-dir RUN`。不要把别的分支run-dir直接传给正式恢复入口。

迁移到另一台服务器时，Git只传代码。冻结诊断还需要只读源run的
`config.json/model.pt/diffusion.pt/latest.pt` 及同一data/cache；这些不提交Git。
