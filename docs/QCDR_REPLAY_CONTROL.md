# Query/Candidate 解耦与整体实体回放缩放对照

2026-10-13：新增 --qcdr-replay-entity-weight（默认 1.0），仅为机制对照。
该系数同时乘两项 historical entity CE，current entity CE、relation CE、static loss、
原 HCP/diffusion、前向特征融合均保持原流程。没有增加模型状态或新损失。

设 w 为此系数：

L_entity = CE(current forward+inverse)
         + w * CE(replay forward)
         + w * mu * CE(replay forward+inverse)

w=1 直接绕过乘法，保持原计算路径。
--qcdr-mode original --qcdr-replay-entity-weight .5 是整体实体 CE 缩放对照，
不是 Original baseline。只把 mu=.5 会漏掉第一项，不构成这个对照。

| 组别 | candidate alpha | entity replay weight | Query / Candidate replay 梯度 |
|---|---:|---:|---|
| Original | 1 | 1 | 100% / 100% |
| QCDR alpha=.5 | .5 | 1 | 100% / 50% |
| Entity replay half | 1 | .5 | 50% / 50% |

上述比例是分支裁剪前的梯度，不是最终 Adam 更新比例。其他损失和共享路径仍存在。
两种缩放可组合，但本次只跑表中三种设置，避免扩大实验矩阵。

新增校验：finite 且 0<=w<=1；启动打印 QCDR entity replay weight。
w 进入配置指纹，保持同一值恢复即可；新源码/新增参数使旧 latest.pt 无法严格 resume。
旧实验已完成，新增对照直接使用新 run-dir，不续训旧权重。

验证：19 项 CPU 测试通过（1.541 秒），包含：
- 默认 w=1 对独立 Original 损失/梯度/一次 Adam 更新的完全等价检查；
- 真实 get_loss 中两项 replay CE 均乘 w，使用非默认 mu=.3 排除误改单项；
- 完整参数 raw gradient 随 w 呈仿射关系，relation/static loss 及其梯度保持一致；
- 真实 ConvTransE 的独立 query/candidate 梯度在 w=.5 下都精确缩放为一半。
未启动真实训练或 CUDA。

## 准备

当前 4090 服务器，先激活原 dgar_env，使用已有 seed2。命令一次执行一组。

```bash
cd /data1/zihao/DGAR-code/.worktrees/qcdr
mkdir -p logs experiment_runs
export DGAR_PYTHON="$(command -v python)"

```

最省：先只跑 A，用它与已有 seed2 Original / QCDR .5 作探索性比较。
现有两组的 GPU 分别为 0/1；新代码默认 w=1 的 CPU 等价已验证，但运行差异仍可能存在。
严格同版本、同 GPU 的三组表：顺序运行 A、B、C，全部 GPU0、seed2。
这里不增加随机种子，也不扫描其他参数。

## A. 新增的整体实体回放减半对照（优先）

```bash
setsid bash -c '
set -euo pipefail
cd /data1/zihao/DGAR-code/.worktrees/qcdr
run_id="ICEWS14s-entity-replay-half-seed2-$(date +%Y%m%d_%H%M%S)_$$"
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
"$DGAR_PYTHON" -u src/main.py --gpu 0 --dataset ICEWS14s \
  --seed 2 \
  --data-root /data1/zihao/DGAR-code/data \
  --run-dir "experiment_runs/$run_id" \
  --qcdr-mode original \
  --qcdr-candidate-alpha 1.0 \
  --qcdr-replay-entity-weight 0.5 \
  --qcdr-target entity \
  --mu 1 --mu_r 1 \
  --final-validation \
  2>&1 | tee "logs/$run_id.log"
' < /dev/null > /dev/null 2>&1 &

```

## B. Original（严格三组比较时运行）

```bash
setsid bash -c '
set -euo pipefail
cd /data1/zihao/DGAR-code/.worktrees/qcdr
run_id="ICEWS14s-original-replay-control-seed2-$(date +%Y%m%d_%H%M%S)_$$"
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
"$DGAR_PYTHON" -u src/main.py --gpu 0 --dataset ICEWS14s \
  --seed 2 \
  --data-root /data1/zihao/DGAR-code/data \
  --run-dir "experiment_runs/$run_id" \
  --qcdr-mode original \
  --qcdr-candidate-alpha 1.0 \
  --qcdr-replay-entity-weight 1.0 \
  --qcdr-target entity \
  --mu 1 --mu_r 1 \
  --final-validation \
  2>&1 | tee "logs/$run_id.log"
' < /dev/null > /dev/null 2>&1 &

```

## C. QCDR alpha=.5（严格三组比较时运行）

```bash
setsid bash -c '
set -euo pipefail
cd /data1/zihao/DGAR-code/.worktrees/qcdr
run_id="ICEWS14s-qcdr-a05-replay-control-seed2-$(date +%Y%m%d_%H%M%S)_$$"
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
"$DGAR_PYTHON" -u src/main.py --gpu 0 --dataset ICEWS14s \
  --seed 2 \
  --data-root /data1/zihao/DGAR-code/data \
  --run-dir "experiment_runs/$run_id" \
  --qcdr-mode candidate-scaled \
  --qcdr-candidate-alpha 0.5 \
  --qcdr-replay-entity-weight 1.0 \
  --qcdr-target entity \
  --mu 1 --mu_r 1 \
  --final-validation \
  2>&1 | tee "logs/$run_id.log"
' < /dev/null > /dev/null 2>&1 &

```

## 看结果

```bash
cd /data1/zihao/DGAR-code/.worktrees/qcdr
ls -lt logs/ICEWS14s-*.log

```

找到实际日志名后 tail -f。config.json 与启动日志中的 entity replay weight
是区分 Original 与 half 的关键：两者 mode 都是 original，不能仅按 mode 分类。
训练完比较 Final full-pipeline VALIDATION 的 filter_ent pooled / past macro MRR
及 Hits，再单独记录同口径 TEST。

- QCDR .5 优于 half：支持保留 query replay 的价值。
- 两者相近：尚不能证明 branch routing 超过回放强度调整。
- half 优于 QCDR：不继续把 candidate-specific 干预视为必要设计。

上述三组仍不能穷尽所有 gradient-norm 匹配解释；half 不是与 QCDR 自动等总梯度范数。
本轮先用这个最低成本对照筛机制，后续再决定是否做更细的梯度归因。
