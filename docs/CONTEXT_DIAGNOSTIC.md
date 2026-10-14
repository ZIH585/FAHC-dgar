# 历史表示聚合诊断

分支 `codex/context-diagnostic` 从恢复后 Original `23dd43d` 创建。
只新增独立诊断入口和 CPU 测试，src/、diffusion/、rgcn/ 完全不改。
这个实验复用已有 Original 权重，不进行训练或 optimizer.step。

## 要验证什么

DGAR 的 select_entity 将一个实体作为头、尾出现在不同关系事实中的表示
平均为一个向量。本诊断考察这种平均是否丢失当前查询的关系条件。
不能由“代码采用平均”直接推出平均有害，因此先做配对推理实验。

同一批 validation query、相同训练历史图、相同 HCP、相同已生成特征，对比：

- original：原平均聚合。
- matched：只替换查询已知主体 s 的聚合向量，采用与已知关系 r 匹配的历史角色。
- control：同一个 s 的历史出现向量随机取样，数量与 matched 相同。
- no_replay：关闭本次推理的生成表示融合，作为额外定位对照。

头出现的角色为 r，尾出现的角色为 inverse(r)，支持缓存本来就含逆关系。
无匹配时回退 Original；全匹配时两种条件聚合相同。其他实体的原平均向量保留。
同数量随机对照不是与 matched 自动匹配表示分布，只用于排除数量差异。
查询条件选择只接收 (s,r)，不接收待预测答案。标签仅用于原过滤排名。

每个预测臂均向 decoder 传入完全相同的完整双向 query batch，再取对应行。
不能直接使用单 query decoder：作者 ConvTransE 的 batch=1 分支会跳过 bn2，
造成额外的预测变化。

每个时间片只生成一次扩散表示，再给各臂复用。预测器 eval，参数冻结，
生成器保留 Original train 模式；扩散内部引导允许对生成特征计算梯度，
没有对模型参数优化。输入缓存只来自 t 之前的训练片。t>=2 的引导图重建为
上一个评估片的历史窗口，而不是上一个被抽样片；传入 triples=None，
有效引导实现只使用历史 tag 标签，没有将 held-out 答案传入引导。

使用最终训练后的 Original checkpoint 回看抽样 validation 时间片。
这是 paired-context-diagnostic-v1，随机流与生成器 buffer 经过的片数不同于
完整评估。输出不能与论文/完整 TEST 的 Average 直接对比，也不能解释成逐阶段遗忘。
推理时改聚合有分布变化；负结果不能排除重新训练后的关系条件模型，但应降低优先级。

## 数据与 checkpoint 检查

只允许已经审核的 baseline@23dd43d 或 grb@caf2193 且 grb_mode=original。
检查 official-dgar-v1、upstream、数据/cache SHA、完整阶段、latest 与 best 权重。
不接受 Half、adaptive、FAHC/ECRI/QCDR 或未知源码版本。
已有训练目录只读，诊断必须写入另一个新目录。

默认时间片 60,120,180,240,300，每片最多12条forward validation事实，
自动加 inverse，共最多120个 query；不会进入 epoch。
如果 GPU 空闲可以用 GPU0，CPU用 --gpu -1但真实扩散推理可能明显更慢。

2026-10-14 验证：18项CPU测试全部通过（1.137秒，不含解释器导入），包含实际
小型guided diffusion端到端诊断，检查Original目录字节不变、参数/RNG不被
预测干预修改、逆关系/空池/同数量对照及固定decoder batch。基线25个Python
文件和新增脚本编译通过；脚本help及文档Bash语法通过。真实ICEWS14s Original
checkpoint通过CPU --check-only，stage=364，7128实体、230关系、365片；没有
进入真实数据推理或训练。源文件src/diffusion/rgcn与23dd43d逐字不变。

## 本机命令

先激活已有 dgar_env，再执行：

```bash
cd /data1/zihao/DGAR-code/.worktrees/context-diagnostic
mkdir -p logs experiment_runs
export DGAR_PYTHON="$(command -v python)"
export CONTEXT_RUN_ID="ICEWS14s-context-$(date +%Y%m%d_%H%M%S)_$$"

setsid bash -c '
set -euo pipefail
cd /data1/zihao/DGAR-code/.worktrees/context-diagnostic
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
"$DGAR_PYTHON" -u scripts/context_diagnostic.py \
  --gpu 0 \
  --run-dir /data1/zihao/DGAR-code/.worktrees/grb/experiment_runs/ICEWS14s-original-seed1-20261013_215614_2096490 \
  --data-root /data1/zihao/DGAR-code/data \
  --stages 60,120,180,240,300 \
  --facts-per-stage 12 --seed 1 \
  --output "experiment_runs/$CONTEXT_RUN_ID" \
  2>&1 | tee "logs/$CONTEXT_RUN_ID.log"
' < /dev/null > /dev/null 2>&1 &
```

```bash
tail -f "/data1/zihao/DGAR-code/.worktrees/context-diagnostic/logs/$CONTEXT_RUN_ID.log"
```

输出 summary.json、queries.csv 和日志。先看 supported/pool_changed 的 query 数量，
再比较同一批 query 的 matched vs Original、matched vs control，并检查各时间片。
建议继续条件：匹配覆盖不是极少数，matched 相对两种对照有同向 MRR 提升，
并在多数抽样时间片出现，H1没有明显受损。单次120query只是方向筛选。
若matched不超过同数量随机控制，停止把关系匹配视为收益来源。
若仅no_replay更好，应研究生成特征的可靠性；若覆盖极低则当前诊断不足以下结论。
不扫描这个脚本的 seed 来选最高分。

## 另一台服务器

```bash
git -C /data1/zihao/DGAR-code fetch origin
git -C /data1/zihao/DGAR-code worktree add --track -b codex/context-diagnostic \
  /data1/zihao/DGAR-code/.worktrees/context-diagnostic origin/codex/context-diagnostic
```

须同时复制完整 Original 运行目录（config/latest/model/diffusion）及同一份数据、
history_snap_v2。脚本不能通过只拉源码自动获得训练权重；现有同名干净worktree
可用 git pull --ff-only。无需重训 Original。
