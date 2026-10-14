# BCR-DGAR：双向历史上下文检索

## 来源与设计范围

独立分支 `codex/bcr` 从 `codex/original-baseline@23dd43dd914a783731c7f39e3fd66491d2aba429`
建立。它继承已经核对作者固定提交 `c5bbc87467bb22d266b664d30eefb8184c1d3104`
的 Original 恢复工作。没有从任何旧方法分支开发。已有工作区、历史缓存、模型和日志均保留。

当前 ICEWS14s 实际缓存的 365 个文件包含 72,437 条正向记录，均以头实体索引，
没有逆关系记录。代码中的 get_history 支持逆关系，不代表现有缓存具有逆关系。
Original 用正向 query 主体检索 HCP，但同时计算正向和反向实体排名。此前的
推理消融显示生成历史对反向排名贡献较大；这支持检验历史覆盖，不能证明补全一定提升。

BCR 第一版只改变历史事实的检索与评估时的上下文组织。原 RGCN、生成器、实体聚合、
融合 alpha、decoder、当前 CE、两个实体 replay CE、关系 replay CE、静态约束均沿用。
没有 KD、额外 old model、importance/forgetting score 或新增可学习参数。

## 检索规则与预算

设阶段 t 选中的历史片集合为 I_t。对已知查询实体集合 Q：

\[
O(Q,I_t)=\{(s,r,o)\in D_i^{train}:s\in Q, i\in I_t\},
\]
\[
A(Q,I_t)=\{(s,r,o)\in D_i^{train}:s\in Q\lor o\in Q, i\in I_t\}.
\]

`outgoing` 从 O 取；`incident` 从 A 取；`random` 从相同时间片的全体训练事实取。
超过预算时均匀无放回采样；采样后以 canonical 正向三元组排序。没有关系硬过滤、
入边配额、风险权重或频率权重。小于预算时全取，预算零时不生成历史表示。

**训练**：Q 为当前已观察训练事实的头、尾实体并集。预算 B 为 Original 在同一组
历史时间片、用当前正向训练主体从原 cache 检索所得的唯一 canonical 正向事实数。
当前 forward-only cache 下，训练事实数量与 Original 匹配。标签属于当前已观察
train[t]，允许用于训练；validation/test 不参与训练索引。三个 enabled 对照使用同一个
预算规则。包含逆记录的其他缓存会先还原、去重；此时 canonical 数不一定等于作者
原 raw/inverse 行数，必须据实际日志报告，不能声称每个数据集都原始行数完全相同。

**评价**：正向分支 Q_f 只含正向已知主体，反向分支 Q_i 只含反向已知主体。
两分支各用 B_d=|O(Q_d,I_t)|，预算不读取相反方向的实体或隐藏答案。
outgoing/incident/random 每方向的生成事实数一致（池不足会记录实际数）。没有
出边容量且预算为零的分支回退无生成表示；不偷偷增加事实预算。

这不是“总推理事实数与 Original 相同”的承诺：Original 共用一次生成，enabled
通常生成两次。对照 outgoing/incident/random 有相同的分支、预算、排序、RNG 和
buffer 隔离机制。incident 相对 outgoing 是补入边价值的关键比较；Original 是
最终端到端性能参考。必须同时报告额外调用/耗时，不能把 2 次生成当成免费收益。

## 时序与调用流程

```text
原始 load_data → 原 split_by_time / 80%-10%-10% 划分 → 只读 cache 校验
若 enabled：建立 BCR 管理器（仅引用训练数组，索引按所选历史片惰性建立）
for t:
    按作者范围和随机策略抽历史片 I_t
    当前训练实体两端 → 取历史池 → 匹配 Original 事实数 → 生成一次 detached 特征
    原 get_loss / backward / Adam / 验证早停
        验证：分别取已知正向/反向头 → 两份历史上下文 → 两次生成/预测 → 合并方向行
    原 diffusion 当前事实训练/验证、最佳权重回退、最后 Adam 保留
    原 latest.pt + RNG；BCR 无额外学习状态
最终 TEST / 可选 VALIDATION：方向分开取历史 → 原排名过滤和统计
```

训练及阶段验证保留作者候选范围 `[0,t-1)`，最终评估保留 `[0,t)`；
第一训练阶段不生成，空池安全回退。检索索引访问时检查片 ID 严格小于边界。
入边索引由对应过去训练事实建立，不写回原 cache，不产生逆 ID diffusion targets。
预测逆关系 ID 仍由原 predict/get_all_triple 处理。

保留原最终模型回看时间片的评估流程及 guidance 历史。**作者的首个最终评估片
guidance 仍来自最终训练阶段的历史，这是继承协议的局限**，不能描述成严格在线
因果评估。BCR 新增检索本身只看该 query 的过去训练片，没有新增未来片检索。
Current/Average macro/pooled、过滤答案、评估片范围及模型选择标签来源保留。

## 答案隔离与显式协议版本

训练前向标签是已观察训练数据。评估检索只接收两组已知 heads，不接收 target 标签
或过滤答案。在正向预测中不使用对应真实尾实体来检索；逆向预测合法使用已知尾实体，
但不将其生成表示传入该事实的正向评分。

作者生成器保持 TRAIN，有 BatchNorm/dropout；若直接顺序调用两个方向，前一分支
可以通过 running buffer 改变后一分支。因此 enabled 的每个生成调用临时保留并
恢复生成器/guider buffers，并在 fork_rng 中使用分支 seed。模式不改成 eval。
每次验证消耗一次共享 Python seed draw；两个方向在自己的 torch 随机流生成，随后
恢复共享 torch RNG。训练生成仍沿用共享随机流。

这是 enabled 的必要适配：配置 protocol 明确记为 `bcr-directional-v1`。outgoing、
incident、random 都使用它；关闭仍是 `official-dgar-v1`，不更改作者 OFF 行为。
不得把 enabled 的验证 RNG/buffer 轨迹说成与作者完全相同，也不得只用 Original
和 incident 的差值来断言入边机制独立有效。

decoder 仍接收完整 2B query batch，每个分支只取自己方向的 B 行，避免 batch=1
触发作者 ConvTransE 的特殊 BN 路径。eval 的 BatchNorm 使用 running statistics。
实体预测分数对另一方向隐藏答案不变已用实际模型和实际 guided diffusion 测试。
关系预测本身需要已知实体对；标签仅用于该关系任务和排名计算，不参与实体检索。

## 参数、日志与恢复

| 参数 | 默认 | 含义 |
|---|---|---|
| --bcr-mode | original | original 关闭；outgoing 双向出边对照；incident BCR；random 随机对照 |
| --bcr | 未启用 | `--bcr-mode incident` 的简写 |
| --reply_batch | 35，Original 默认 | 历史时间片数，所有组相同 |

没有新增 budget、eta、probe、KD 参数。enabled 要求 run-dir、一次时间遍历、
feature replay 和 subject prompt，不支持同时关闭这些路径或使用 grid-search。

run-dir：原 config.json/model.pt/diffusion.pt/latest.pt/fitlog；enabled 额外写 bcr.jsonl。
后者记录阶段、用途、历史片、预算、实际选中数、入边新增数、选中入边数、无关事实数、
采样 seed。断点恢复时部分阶段可能重复追加日志；训练统计按同阶段最后一条 train
记录分析。学习状态由原 checkpoint 保存，索引由 train 重建，不保存计算图/特征缓存。
源、配置、数据和 cache 指纹不匹配时拒绝恢复/独立评估。

CPU 索引随被访问历史片增加，额外存储约 O(过去事实和实体索引)，不是固定字节缓存。
训练 GPU 特征数量与匹配预算一致；评估保存两份 detached replay，增加特征内存、
buffer 临时复制和两次预测成本。没有额外模型副本、optimizer 或学习 loss。真实 GPU
峰值、耗时和数值稳定性尚未实测，不能以 CPU smoke test 承诺提升或开销百分比。

## 文件与验证

修改：src/main.py 接线；src/cli.py 参数；src/runtime.py enabled protocol；
scripts/check_baseline.sh 增加 scripts 的编译检查。新增：src/bcr.py、tests/test_bcr.py、
scripts/diagnose_bcr.py、scripts/run_bcr_experiment.sh、scripts/summarize_bcr.py、本说明。
src/rrgcn.py、decoder、diffusion、rgcn 相对 23dd43d 均无 diff。

CPU 检查包含原 13 项官方回归与 BCR 测试：边界/空池/首阶段/小预算/逆 ID、真实
单事实 loss backward 和双向 prediction、实际扩散引导、buffer/RNG/异常恢复、
真实训练 caller 停在首个 synthetic loss、真实最终 eval caller、重建状态一致性、
临时 toy checkpoint 的四组诊断和输入逐字节不变。未启动真实 epoch 或 GPU。
真实 Original checkpoint 的 `--gpu -1 --check-only` 已成功，只加载后停止。

```bash
cd /data1/zihao/DGAR-code/.worktrees/bcr
export DGAR_PYTHON="$(command -v python)"  # 先激活 dgar_env
bash scripts/check_baseline.sh
bash scripts/run_bcr_experiment.sh --dry-run incident 0 ICEWS14s 1
```

## 首先运行一次配对筛查（推荐）

已有 GRB 分支 Original seed1 checkpoint 已审核为恢复基线的等价 OFF 路径。
诊断只接受这些已知 Original 来源、完整 latest 与一致的 best weights/data/cache，
也接受本分支 OFF checkpoint。拒绝旧方法和非 Original 的权重。

使用五个片的全部 VALIDATION query、同一模型、同一历史时间片、配对噪声。
四组同时输出，不训练、不覆盖原 checkpoint。Original 组保留其共用 HCP 和
list(set) 顺序；enabled 控制统一排序与分支生成。诊断不复制完整最终评估 RNG
轨迹，其分数不是论文 Average MRR。方法变化带来的噪声/顺序影响也需要承认。

```bash
cd /data1/zihao/DGAR-code/.worktrees/bcr
mkdir -p logs experiment_runs
export DGAR_PYTHON="$(command -v python)"
export BCR_SCREEN_ID="ICEWS14s-bcr-screen-$(date +%Y%m%d_%H%M%S)_$$"
setsid bash -c '
set -euo pipefail
cd /data1/zihao/DGAR-code/.worktrees/bcr
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
"$DGAR_PYTHON" -u scripts/diagnose_bcr.py \
  --gpu 0 --seed 1 --stages 60,120,180,240,300 \
  --data-root /data1/zihao/DGAR-code/data \
  --run-dir /data1/zihao/DGAR-code/.worktrees/grb/experiment_runs/ICEWS14s-original-seed1-20261013_215614_2096490 \
  --output "experiment_runs/$BCR_SCREEN_ID" \
  2>&1 | tee "logs/$BCR_SCREEN_ID.log"
' < /dev/null > "logs/$BCR_SCREEN_ID-launcher.log" 2>&1 &
echo "launcher PID: $!"
tail -F "logs/$BCR_SCREEN_ID.log"
```

输出 summary.json、queries.csv、bcr.jsonl、终端日志。summary 含总体、分方向、分片
MRR/H1/H3/H10、实际生成数量、运行时间、粗筛门槛。预算在启用三组中应逐方向一致。
预设资源门槛：incident 超过三个对照至少 0.3pp，五片至少四片不低于三个对照；
再从 query 表核对收益集中度和正向是否明显受损。自动 provisional_gate 只做前两项，
不是显著性结论。诊断失败则暂停本设计；成功才执行下面正式训练，不自动启动训练。

## 通过后正式训练

最低机制对照为 Original、outgoing、incident，单 seed1，同一 GPU/environment，
依次运行。outgoing 与 incident 的比较检验入边补全，Original 比较端到端收益。
先不扫参数、不补模型 seed；random 完整训练作为确认机制的后续消融。

```bash
cd /data1/zihao/DGAR-code/.worktrees/bcr
mkdir -p logs experiment_runs
export DGAR_PYTHON="$(command -v python)"
export DGAR_DATA_ROOT=/data1/zihao/DGAR-code/data
export BCR_BATCH_ID="$(date +%Y%m%d_%H%M%S)_$$"
setsid bash -c '
set -euo pipefail
cd /data1/zihao/DGAR-code/.worktrees/bcr
for BCR_GROUP in original outgoing incident; do
  bash scripts/run_bcr_experiment.sh "$BCR_GROUP" 0 ICEWS14s 1
done
' < /dev/null > "logs/ICEWS14s-bcr-launcher-$BCR_BATCH_ID.log" 2>&1 &
echo "launcher PID: $!"
```

每组自动新建时间戳 run-dir 和 logs 文件，命令在启动前打印。pipefail 保证失败时
停止后续组。如需直接启动 BCR 一组（与前述顺序运行二选一，勿重复同时启动）：

```bash
cd /data1/zihao/DGAR-code/.worktrees/bcr
mkdir -p logs experiment_runs
export BCR_RUN_ID="ICEWS14s-bcr-incident-seed1-$(date +%Y%m%d_%H%M%S)_$$"
setsid bash -c '
set -euo pipefail
cd /data1/zihao/DGAR-code/.worktrees/bcr
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python -u src/main.py --gpu 0 --dataset ICEWS14s --seed 1 \
  --bcr-mode incident --reply_batch 35 --mu 1 --mu_r 1 \
  --data-root /data1/zihao/DGAR-code/data \
  --run-dir "experiment_runs/$BCR_RUN_ID" --final-validation \
  2>&1 | tee "logs/$BCR_RUN_ID.log"
' < /dev/null > "logs/$BCR_RUN_ID-launcher.log" 2>&1 &
```

完成后用实际目录名称汇总（同分支 source、seed、参数和 GPU 会被校验）：

```bash
python scripts/summarize_bcr.py \
  --original experiment_runs/实际original目录 \
  --outgoing experiment_runs/实际outgoing目录 \
  --incident experiment_runs/实际incident目录 \
  --output experiment_runs/ICEWS14s-bcr-summary-新的唯一名称
```

产出 metrics.csv/summary.json；看 `Final full-pipeline TEST` 和 VALIDATION 的 macro
MRR、pooled MRR 与 H1/H3/H10。不要把中间 last-stage TEST diagnostic 当最终 Current。
如果只超过 Original 而未超过 outgoing，尚不能认定入边补全有效。

中断恢复保持同一 mode/seed/科学配置：

```bash
python -u src/main.py --gpu 0 --dataset ICEWS14s --seed 1 --bcr \
  --data-root /data1/zihao/DGAR-code/data \
  --run-dir "$BCR_RUN" --resume "$BCR_RUN/latest.pt" --final-validation
```

## 另一台服务器

```bash
cd /data1/zihao/DGAR-code
git fetch origin
git worktree add --track -b codex/bcr .worktrees/bcr origin/codex/bcr
```

若本机已有此 worktree/分支，在该工作区检查 status 后 `git pull --ff-only`，
勿从旧分支 HEAD push 到 BCR。代码由 Git 拉取；数据、匹配 cache 和完整 Original
checkpoint 不在 Git 中。筛查需要另行复制整组 config/model/diffusion/latest；
正式训练从零开始，不加载旧方法 checkpoint。GPU 被占用时只运行 CPU check-only/dry-run。
