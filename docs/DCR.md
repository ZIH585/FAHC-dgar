# DCR-DGAR：去噪终点上下文回放

## 基线与范围

分支 `codex/dcr` 从 `codex/original-baseline@23dd43dd914a783731c7f39e3fd66491d2aba429`
独立建立。固定作者来源为 `c5bbc87467bb22d266b664d30eefb8184c1d3104`。
没有从其他方法分支继承训练代码；没有改变运行中的 BCR 工作区。原缓存只校验、不重建。

Original 默认 linear/35步/scale50/beta_end.02 的末端 beta=1；作者实际只执行最高噪声
片的一次 posterior 更新，并返回带新噪声的结果。该配置下 x0 尾项的直接系数约
1.4552e-7、新噪声标准差1，还有 predictor 引导梯度；头/关系 token 仍来自网络。
这不是“生成器无效”的证明，系数也不等于贡献百分比。DCR 验证终点选择是否限制
历史表示质量，不把作者路径直接定性为 bug，也不修改全局 Original 来强制采用新方法。

## 公式和对照

预测终点 X0=D_phi(XT,T,c)。原输出为 `[X0^s,X0^r,m_post+kappa*g_post+sigma*eps]`。

| 参数 | 输出 | 用途 |
|---|---|---|
| --dcr-mode original | 作者原输出 | 默认关闭 |
| --dcr-mode mean | 原 guided posterior mean，去掉最后新增噪声 | 噪声去除对照 |
| --dcr-mode x0 | 完整 X0，无 predictor 引导 | 终点对照 |
| --dcr-mode guided-x0 / --dcr | X0 尾项 + 在 X0 处重新求得的引导 | DCR |

\[
g_0=\nabla_u\log p_{\theta^{guide}}
(o\mid s,r,\mathcal G_{past},[X_0^s,X_0^r,u])\big|_{u=X_0^o},
\]
\[
H_{DCR}=[X_0^s,X_0^r,X_0^o+\kappa g_0].
\]

头/关系 token 在求此偏导时固定。历史事实标签 o 来自训练 cache，绝不来自当前
validation/test 的隐藏答案。guidance 只对输入 tail 求梯度，不累加参数 .grad。
使用原来的 guider：阶段训练/验证为已有 frozen previous predictor，最终评估为
原流程指定模型；没有增加 old model、KD、新 loss、learned adapter 或 memory。

**重要：在 X0 处重新算梯度。** mean 模式复用作者 p_mean_variance，保留原梯度位置；
guided-x0 不能直接搬用 g_post。每种模式只调用一次去噪网络，不增加反向采样步数。
输出和指标标量 detach；guidance 留在 guider 上的临时 h/h0 也 detach，避免留住输入图。

新增唯一科学开关为 --dcr-mode，默认 original；--dcr 是 guided-x0 简写。
kappa 沿用 `--classifier_scale` 默认1，不增加另一套同义系数。首轮不扫引导强度。
enabled 要求 replay、引导、run-dir、一次时间遍历、非 grid-search、初始 vl=False。
扩散验证循环内部可以临时设 vl=True，原采样完全保留。

## 实际训练流和时序

```text
原 load/split/cache 校验与 provenance
阶段 t：原随机 HCP 时间片和主体检索 → 历史训练 tags
    原 Att_Diffuse_model.forward(False) 构造条件、初始噪声和一次网络输出
    DiffuRec.reverse_p_sample：关闭时走原代码；开启且 vl=False 时调用 dcr.sample
    新终点特征 detach → 原 select_entity 均值聚合 → 原 alpha 融合
    原 current/replay/entity/relation/static loss → 原 predictor Adam/验证早停
    原 diffusion 当前事实训练（不改）；vl=True 的 diffusion 验证采样（不改）
    原 best 权重回退、最后 Adam、teacher 更新、RNG/checkpoint
最终 TEST/VALIDATION：原检索、原历史/guider流，回放采样使用所选 endpoint
```

HCP 候选范围、事实集合与 list(set) 顺序、数量、逆 ID 编码和所有损失权重不变。
原 feature generator 的 TRAIN 模式保留，不在正式训练中隔离/重播验证随机流。
enabled 仍抽取但丢弃最后噪声，避免单纯删除 randn 导致随机流偏移；这不保证跨模式
整个随机轨迹相同，尤其 x0 不执行 guidance 及其随机/buffer 操作。所有正式组都从零
训练，不能把 Original checkpoint 直接加开关当作已经训练好的 DCR。

默认 OFF protocol 保持 official-dgar-v1；开启为显式 dcr-endpoint-v1。
当前 CE、两个实体 replay CE、关系 replay CE、静态约束、decoder 均不变。
diffusion 自身训练仍预测原目标，没有顺便加入一致性 loss 或改噪声 schedule。

新增分支只使用原流程提供的历史训练 tags/图。保留作者最终模型回看各片的协议：
首个最终评估片的 guidance 来自最终训练历史，这是已有基线局限；不称作严格在线
无未来信息协议。Current、历史 macro Average、pooled、过滤答案和评估片范围保留。
已有中间 last-stage TEST diagnostic 仍只打印、不用于模型选择。

## 状态、日志和成本

没有新增模型参数或跨阶段学习状态；原 latest.pt 足够恢复。新 mode/protocol/source
记入 config，恢复/独立 test 会拒绝错配。Recorder 只是 detached 标量聚合，不保存
到模型 checkpoint；阶段完成时清空。中途重跑可能留下重复日志，按同阶段最后 fit
记录分析；完整 TEST 后的附加 VALIDATION 也会追加 evaluation 记录。

enabled 写 run-dir/dcr.jsonl：stage/mode/phase、生成调用和累计 facts、x0 head/tail
范数、scaled guidance 范数、返回 tail 范数、丢弃 noise 范数、posterior x0 系数。
范数按每次生成的事实数加权，并在阶段边界一次传回 CPU，不逐事实同步或保存图。
首阶段、空 HCP 保留原 guard，可能没有生成记录；这不是模块未连接。

mean/guided-x0 的网络与指导调用量接近 Original，x0 省去 guidance。没有额外模型、
永久 GPU 表或回放预算。额外成本主要为少量 norm/5标量聚合/日志；具体 GPU 峰值、
耗时和完整训练稳定性未实测，不能把 CPU 检查当作性能/开销保证。

## 文件与检查

| 文件 | 修改 |
|---|---|
| diffusion/dcr.py | endpoint、重新求引导、detach、阶段标量记录器 |
| diffusion/difffu_21.py | reverse_p_sample 的默认关闭分派 |
| diffusion/model_21.py | 回放输出后记录 detached 指标 |
| src/cli.py | --dcr-mode/--dcr 和组合校验 |
| src/main.py | protocol 打印、记录器初始化和阶段 flush |
| src/runtime.py | enabled protocol 写入 provenance |
| scripts/diagnose_dcr.py | 配对 VALIDATION 筛查，四组共用一次 x0 |
| scripts/run_dcr_experiment.sh | fresh run/log、dry-run、pipefail |
| scripts/summarize_dcr.py | final 区段汇总和配置/来源校验 |
| tests/test_dcr.py | 实际采样/训练接口的 CPU 回归 |
| tests/reference/official_difffu_21.py.txt | 作者原始 sampler，Git blob SHA 校验 |
| scripts/check_baseline.sh / README.md | 编译脚本及入口说明 |

RGCN、decoder、图构建、HCP selector、loss 函数没有 diff。原13项回归以及新增 CPU
测试覆盖：关闭时值/buffer/RNG与作者 sampler一致；mean准确移除末噪声；x0跳过guide；
X0处偏导与手工实际模型一致；线性末端beta=1有限；单事实/空历史/首阶段/逆关系；
真实 predictor 与 diffusion loss backward（无optimizerstep）；实际主循环停在首个
synthetic loss；四组共享网络输出；toy checkpoint端到端且输入字节不变；日志detach。
真实 Original checkpoint --gpu -1 --check-only 只读加载成功，在推理前停止。

```bash
cd /data1/zihao/DGAR-code/.worktrees/dcr
export DGAR_PYTHON="$(command -v python)" # 先激活 dgar_env
bash scripts/check_baseline.sh
bash scripts/run_dcr_experiment.sh --dry-run guided-x0 0 ICEWS14s 1
```

## 第一组实验：复用 Original 的配对筛查

先跑这组，不立即跑完整四组训练。不补模型 seed、不用 TEST 选择 endpoint。
五个预先选定片的全部 VALIDATION query；每片同 HCP、同初始噪声、**同一次去噪网络
输出**。保存原引导前的 RNG/buffers，在相同状态重算 X0 引导；各片独立诊断状态不
复现完整官方随机轨迹。分数不是论文 Average，不代表在线遗忘或重训结果。

设置 DCR_GPU 为已确认空闲的设备。只有一张 GPU 时，等 BCR 完成再运行。

```bash
cd /data1/zihao/DGAR-code/.worktrees/dcr
mkdir -p logs experiment_runs
export DGAR_PYTHON="$(command -v python)"
export DCR_GPU=0
export DCR_SCREEN_ID="ICEWS14s-dcr-screen-$(date +%Y%m%d_%H%M%S)_$$"
setsid bash -c '
set -euo pipefail
cd /data1/zihao/DGAR-code/.worktrees/dcr
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
"$DGAR_PYTHON" -u scripts/diagnose_dcr.py \
  --gpu "$DCR_GPU" --seed 1 --stages 60,120,180,240,300 \
  --data-root /data1/zihao/DGAR-code/data \
  --run-dir /data1/zihao/DGAR-code/.worktrees/grb/experiment_runs/ICEWS14s-original-seed1-20261013_215614_2096490 \
  --output "experiment_runs/$DCR_SCREEN_ID" \
  2>&1 | tee "logs/$DCR_SCREEN_ID.log"
' < /dev/null > "logs/$DCR_SCREEN_ID-launcher.log" 2>&1 &
echo "launcher PID: $!"
tail -F "logs/$DCR_SCREEN_ID.log"
```

输出 summary.json、queries.csv、generation.jsonl 和终端日志。包含四组 MRR/H1/H3/H10、
正逆方向、分片、逐 query 排名、生成/聚合范数、时间和改善/退化计数。
Original 来源只接受已审核的 23dd43d/GRB Original 或本分支 OFF，完整 best/latest
一致且源/数据/cache 匹配；拒绝旧方法 checkpoint。input run 永远不写入。

继续门槛：guided-x0 对 Original、mean 至少 +0.3pp；不低于 raw x0；五片至少四片
不低于 Original/mean；再检查正向损失和少数事实收益集中度。自动 provisional_gate
是资源筛查，不是显著性结论。若 x0 更好，应优先研究简单终点而非坚持额外引导；
若只有 mean 有效，结论是去除末噪声有收益；都无收益则暂停，不堆新模块。

## 通过后完整训练：最低三组

同一个 source、同一 seed1、同一 GPU/environment、data/cache，顺序运行 Original、
mean、guided-x0。x0 的完整训练是需要确认引导独立作用时的后续消融。
下面只是命令，不在实现阶段执行。BCR与DCR单独验证后才考虑组合。

```bash
cd /data1/zihao/DGAR-code/.worktrees/dcr
mkdir -p logs experiment_runs
export DGAR_PYTHON="$(command -v python)"
export DGAR_DATA_ROOT=/data1/zihao/DGAR-code/data
export DCR_GPU=0
export DCR_BATCH_ID="$(date +%Y%m%d_%H%M%S)_$$"
setsid bash -c '
set -euo pipefail
cd /data1/zihao/DGAR-code/.worktrees/dcr
for DCR_GROUP in original mean guided-x0; do
  bash scripts/run_dcr_experiment.sh "$DCR_GROUP" "$DCR_GPU" ICEWS14s 1
done
' < /dev/null > "logs/ICEWS14s-dcr-launcher-$DCR_BATCH_ID.log" 2>&1 &
echo "launcher PID: $!"
```

每组自动生成独立时间戳日志与 run-dir。查看 launcher 能找到具体路径，pipefail
使任一组失败时停止。只启动 DCR 一组的等价直接命令（与顺序命令二选一）：

```bash
cd /data1/zihao/DGAR-code/.worktrees/dcr
mkdir -p logs experiment_runs
export DCR_RUN_ID="ICEWS14s-dcr-guided-x0-seed1-$(date +%Y%m%d_%H%M%S)_$$"
export DCR_GPU=0
setsid bash -c '
set -euo pipefail
cd /data1/zihao/DGAR-code/.worktrees/dcr
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python -u src/main.py --gpu "$DCR_GPU" --dataset ICEWS14s --seed 1 \
  --dcr-mode guided-x0 --reply_batch 35 --mu 1 --mu_r 1 \
  --data-root /data1/zihao/DGAR-code/data \
  --run-dir "experiment_runs/$DCR_RUN_ID" --final-validation \
  2>&1 | tee "logs/$DCR_RUN_ID.log"
' < /dev/null > "logs/$DCR_RUN_ID-launcher.log" 2>&1 &
```

完成后用实际目录汇总：

```bash
python scripts/summarize_dcr.py \
  --original experiment_runs/实际original目录 \
  --mean experiment_runs/实际mean目录 \
  --guided-x0 experiment_runs/实际guided-x0目录 \
  --output experiment_runs/ICEWS14s-dcr-summary-新的唯一名称
```

产出 metrics.csv/summary.json。看 Final full-pipeline TEST 与 VALIDATION 的历史macro
MRR、pooled MRR、H1/H3/H10，区分小样本 Current 与旧阶段 Average。不把中间 TEST
诊断替代最终 Current。如果不超过 mean，不能归因为去噪终点引导独立有效。

恢复保持全部原参数和 mode，DCR 不需要另行恢复额外状态：

```bash
python -u src/main.py --gpu 0 --dataset ICEWS14s --seed 1 --dcr \
  --data-root /data1/zihao/DGAR-code/data \
  --run-dir "$DCR_RUN" --resume "$DCR_RUN/latest.pt" --final-validation
```

另一服务器的全新工作区：

```bash
cd /data1/zihao/DGAR-code
git fetch origin
git worktree add --track -b codex/dcr .worktrees/dcr origin/codex/dcr
```

已有工作区则在其中检查 status 后 git pull --ff-only。代码由Git提供，匹配的完整
Original run（config/latest/model/diffusion）和 data/cache 需另行复制，不能用旧方法
checkpoint代替。数据、模型、日志、锁、缓存和输出目录均不提交Git。
