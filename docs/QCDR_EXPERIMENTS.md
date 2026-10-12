# QCDR 第一阶段实验命令

这里只提供命令；实现期间未执行任何正式训练。先在 ICEWS14s、seed=1 判断方向，
不要同时扩展数据集。五组使用同一份代码、数据、history_snap_v2 和所有 Original
超参数，仅改变 QCDR 模式/系数。Original 的 reply_batch=35、mu=mu_r=1、
task_weight=0.7、regcn_epochs=diffu_epochs=10 等默认值未改，没有 B4096 参数。

## 1. 另一台服务器首次取分支

前提：该服务器已经有原 DGAR 仓库、数据与匹配的历史缓存，且还没有本 QCDR 分支/
工作树。以下创建独立工作树，不切换那里的已有实验分支。先确认本分支已推到 origin。

```bash
cd /data1/zihao/DGAR-code
git fetch origin
git worktree add --track -b codex/qcdr /data1/zihao/DGAR-code/.worktrees/qcdr origin/codex/qcdr
```

本机工作树已经存在，不要重复 worktree add。若另一台服务器已经有同名工作树，
在确认它没有自己的未提交改动后，在该工作树使用 git pull --ff-only 即可。

## 2. 每次打开终端后的准备

先激活该服务器已有的 dgar_env；Python 绝对路径随服务器环境而变，所以从当前
激活环境捕获，不硬编码其他服务器的 conda 路径。

```bash
cd /data1/zihao/DGAR-code/.worktrees/qcdr
mkdir -p logs experiment_runs
export DGAR_PYTHON="$(command -v python)"
```

数据来自 /data1/zihao/DGAR-code/data，不需要复制进新 worktree。
其中 ICEWS14s/history_snap_v2 必须是与当前 split 一致的原缓存。
不要在组间重新生成缓存；不匹配时程序在训练前报错。
使用 GPU 0 的命令仅供未来 GPU 空闲后执行；按顺序一次启动一组，勿把五段一起提交。

每组 run_id 含时间戳与进程号，日志为 logs/<run_id>.log，
模型/配置/恢复点为 experiment_runs/<run_id>/，不会覆盖上次运行。
python -u 保证及时刷新输出，pipefail 保留 Python 失败的退出语义。
反斜杠必须位于行末，其后不要添加空格。

## 3. 五组完整命令

### A. Original DGAR

```bash
setsid bash -c '
set -euo pipefail
cd /data1/zihao/DGAR-code/.worktrees/qcdr
run_id="ICEWS14s-original-seed1-$(date +%Y%m%d_%H%M%S)_$$"
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
"$DGAR_PYTHON" -u src/main.py --gpu 0 --dataset ICEWS14s \
  --seed 1 \
  --data-root /data1/zihao/DGAR-code/data \
  --run-dir "experiment_runs/$run_id" \
  --qcdr-mode original \
  --qcdr-candidate-alpha 1.0 \
  --qcdr-target entity \
  --final-validation \
  2>&1 | tee "logs/$run_id.log"
' < /dev/null > /dev/null 2>&1 &
```

### B. QCDR candidate-stopgrad：主版本

```bash
setsid bash -c '
set -euo pipefail
cd /data1/zihao/DGAR-code/.worktrees/qcdr
run_id="ICEWS14s-qcdr-a0-seed1-$(date +%Y%m%d_%H%M%S)_$$"
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
"$DGAR_PYTHON" -u src/main.py --gpu 0 --dataset ICEWS14s \
  --seed 1 \
  --data-root /data1/zihao/DGAR-code/data \
  --run-dir "experiment_runs/$run_id" \
  --qcdr-mode candidate-stopgrad \
  --qcdr-candidate-alpha 0.0 \
  --qcdr-target entity \
  --final-validation \
  2>&1 | tee "logs/$run_id.log"
' < /dev/null > /dev/null 2>&1 &
```

### C. QCDR candidate-scaled：25%

```bash
setsid bash -c '
set -euo pipefail
cd /data1/zihao/DGAR-code/.worktrees/qcdr
run_id="ICEWS14s-qcdr-a025-seed1-$(date +%Y%m%d_%H%M%S)_$$"
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
"$DGAR_PYTHON" -u src/main.py --gpu 0 --dataset ICEWS14s \
  --seed 1 \
  --data-root /data1/zihao/DGAR-code/data \
  --run-dir "experiment_runs/$run_id" \
  --qcdr-mode candidate-scaled \
  --qcdr-candidate-alpha 0.25 \
  --qcdr-target entity \
  --final-validation \
  2>&1 | tee "logs/$run_id.log"
' < /dev/null > /dev/null 2>&1 &
```

### D. QCDR candidate-scaled：50%

```bash
setsid bash -c '
set -euo pipefail
cd /data1/zihao/DGAR-code/.worktrees/qcdr
run_id="ICEWS14s-qcdr-a05-seed1-$(date +%Y%m%d_%H%M%S)_$$"
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
"$DGAR_PYTHON" -u src/main.py --gpu 0 --dataset ICEWS14s \
  --seed 1 \
  --data-root /data1/zihao/DGAR-code/data \
  --run-dir "experiment_runs/$run_id" \
  --qcdr-mode candidate-scaled \
  --qcdr-candidate-alpha 0.5 \
  --qcdr-target entity \
  --final-validation \
  2>&1 | tee "logs/$run_id.log"
' < /dev/null > /dev/null 2>&1 &
```

### E. QCDR candidate-scaled：100%，等价性核验

```bash
setsid bash -c '
set -euo pipefail
cd /data1/zihao/DGAR-code/.worktrees/qcdr
run_id="ICEWS14s-qcdr-a1-seed1-$(date +%Y%m%d_%H%M%S)_$$"
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
"$DGAR_PYTHON" -u src/main.py --gpu 0 --dataset ICEWS14s \
  --seed 1 \
  --data-root /data1/zihao/DGAR-code/data \
  --run-dir "experiment_runs/$run_id" \
  --qcdr-mode candidate-scaled \
  --qcdr-candidate-alpha 1.0 \
  --qcdr-target entity \
  --final-validation \
  2>&1 | tee "logs/$run_id.log"
' < /dev/null > /dev/null 2>&1 &
```

Original 与 a=1 是有意保留的重复控制，CPU 上已证明它们完全等价。如果完整跑这组
重复控制，其作用是检查实际 CUDA/依赖/运行环境下的一致性，不代表另一项算法。
不要根据一次末快照 Current MRR 决定方案成败。

## 4. 看日志与恢复

```bash
cd /data1/zihao/DGAR-code/.worktrees/qcdr
ls -lt logs/ICEWS14s-*.log
```

找到对应完整文件名后使用 tail -f；例如下面变量需替换为 ls 输出的真实文件名：

```bash
run_log="/data1/zihao/DGAR-code/.worktrees/qcdr/logs/填入实际文件名.log"
tail -f "$run_log"
```

启动时应打印 QCDR mode / QCDR candidate alpha / QCDR target。
stopgrad 的 effective alpha 为 0；Original 的为 1。config.json 还记录源码、
数据、缓存哈希和完整训练参数。日志路径在 **qcdr 工作树的 logs**，不是根目录
logs 或 ECRI 的 logs。

恢复例子（a=.25）：把 RUN_ID 替换为已中断的真实目录名；不得改模式、alpha、
数据或源码。恢复会回到最近完成的时间阶段，而不是中断的阶段内部某个 epoch。

```bash
setsid bash -c '
set -euo pipefail
cd /data1/zihao/DGAR-code/.worktrees/qcdr
run_id="RUN_ID"
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
"$DGAR_PYTHON" -u src/main.py --gpu 0 --dataset ICEWS14s \
  --seed 1 \
  --data-root /data1/zihao/DGAR-code/data \
  --run-dir "experiment_runs/$run_id" \
  --resume "experiment_runs/$run_id/latest.pt" \
  --qcdr-mode candidate-scaled \
  --qcdr-candidate-alpha 0.25 \
  --qcdr-target entity \
  --final-validation \
  2>&1 | tee -a "logs/$run_id.log"
' < /dev/null > /dev/null 2>&1 &
```

这里的 latest.pt 是主模型、diffusion、两个 Adam、RNG、阶段号与配置的匹配恢复点。
model.pt / diffusion.pt 为评估权重；QCDR 没有新 tensor state。

## 5. 比较口径和停止条件

- 比较同一区段的 filter_ent MRR / Hits@1 / Hits@3 / Hits@10。TEST 与
  VALIDATION 分开记表，不能混合；输出中 raw_ent、filter_rel 也不能混进实体过滤指标。
- 调系数以 **Final full-pipeline VALIDATION** 为准；原协议仍会输出 TEST，
  但不要用 TEST 在这五组里挑系数再报告那个最佳 TEST 作为无偏结论。
- 同时记录 pooled MRR 和 Average MRR（macro, time_idx=1..T-2），以及末快照
  Current；最后一个快照样本量可能很小。macro 和 pooled 不可直接互换。
- Original 和 scaled1 的源码、数据、缓存哈希必须一致，除 QCDR 字段外配置一致。
  若它们明显分离，先检查运行环境、seed、resume、checkpoint 和 GPU 非确定性，
  不先解读 0/.25/.5 的算法效果。
- 若 a<1 稳定改善验证集且没有明显牺牲另一主要指标，再选一个系数补 seed 2/3；
  单 seed 只用于筛方向。随后才考虑其他数据集。
- 若 0/.25/.5 都退步，不继续无边界扫描细碎 alpha，更不叠回 FAHC/ECRI/KD。
  停止将“候选 replay 有害”作为当前项目的优先假设。
- 若仅 .25/.5 改善而 0 退步，解释是候选 replay 仍有价值、其梯度强度可能过大。
  若后续要声称 branch-specific 机制收益，还需与整体 replay 梯度/损失缩放对照，
  以排除只是减小总 replay 强度的解释。
