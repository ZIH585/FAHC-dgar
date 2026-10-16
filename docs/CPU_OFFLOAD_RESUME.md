# ICEWS18 mean+Shared 显存不足后的续跑

`scripts/resume_pgr_cpu_offload.py` 从已有 `config.json` 恢复全部科研参数，
调用已有 `--resume latest.pt` 流程，在 PyTorch `save_on_cpu` 上下文中执行。
反向传播要保存的中间张量暂存 CPU，用到时传回原设备；FP32、完整候选实体、
HCP 数量、loss、BatchNorm、dropout 和 optimizer step 均使用原实现。
没有修改 `src/`、`diffusion/` 或 `rgcn/`，旧 checkpoint 的源码指纹仍可校验。
本入口适用于 Original/mean/Shared/mean-shared，不处理已停止的 geometry 实验。

它会增加主机内存用量和 CPU/GPU 传输时间，不能事先保证具体显存降幅或耗时。
仍在 GPU 上常驻的模型、图、输出等不会全部卸载。当前验证限于 CPU；
真实 CUDA/DGL 的内存峰值和性能需由正式续跑确认。

先在激活 dgar_env 后设置 DGAR_DATA_ROOT，运行 `--dry-run`。该检查只核对
配置、源码、数据和历史缓存指纹，不加载数据集、不初始化 CUDA、不训练。

```bash
export DGAR_DATA_ROOT=/data1/zihao/DGAR-code/data
python -u scripts/resume_pgr_cpu_offload.py --gpu 1 \
  --run-dir experiment_runs/ICEWS18-pgr-mean-shared-seed1-20261016_002715_1682535 \
  --dry-run
```

去掉 `--dry-run` 即从最新完整阶段继续；未完成阶段会重做。保留原日志，
把本次输出写入带新时间戳的日志。恢复器会按 latest.pt 重写未完成阶段可能
覆盖的 best 权重文件，这是原有断点恢复流程。

2026-10-16 的本地检查确认 latest.pt 完成阶段为 142，因此该断点从 143 开始。
原 OOM 进程已退出后再启动，确保同一个 run-dir 只有一个训练进程。
CPU smoke test 对实际 mean+Shared predictor 比较卸载开关两种情况下的
loss、梯度、BatchNorm buffers 和 RNG，要求逐元素相同；原基线检查也需通过。

## 全量 CPU 暂存太慢时：解码器激活重计算

2026-10-16 增加 `--memory-mode`，原有命令的默认值仍是 `cpu`。
本次检查时全量暂存的日志显示 40 分钟完成 10 个阶段，latest.pt 已完成 152；
这只是检查时进度，续跑入口始终自动读取最新完整阶段。

| 模式 | 行为 | 使用场景 |
|---|---|---|
| checkpoint | GPU 重算 entity decoder 激活，无 CPU 激活暂存 | 24GB GPU 优先尝试 |
| checkpoint-selective | 解码器重算 + 只暂存大 CUDA 中间张量 | 重算模式仍 OOM |
| selective | 只暂存大 CUDA 中间张量 | 单独验证选择性暂存 |
| cpu | 全量 CPU 暂存 | 原有保底方式 |
| none | 原生 GPU 执行 | 更大显存 GPU |

checkpoint 使用 PyTorch non-reentrant checkpoint，保留 dropout RNG；
重算时给 decoder 的 BN 分配同设备临时 buffer，结束后恢复原 buffer 引用，
避免每次 backward 重复更新 running_mean、running_var 和 num_batches_tracked。
保持 TRAIN 的 batch statistics，不把重算改成 eval，不拆分训练 batch。
实体候选全集、logits、CE、HCP、mu、adapter、精度和 optimizer step 保留原实现。
验证/测试的 entity decoder 走原路径；guidance 的 autograd.grad 可正常使用。

checkpoint 只作用于 `src.decoder.ConvTransE`，不重算完整图编码器，因此还会
保留图表示和 CE 等显存。是否能覆盖整个 ICEWS18 的峰值、实际速度及 CUDA
数值行为，当前环境无法测量。增加 GPU 计算通常比全量 CPU 搬运更适合此问题，
但不承诺具体倍数或所有阶段均不 OOM。

选择性暂存默认阈值 64 MiB，小张量和参数保留原设备；`--pin-memory` 可用于
暂存模式。该设置不影响 checkpoint 模式，后者不搬运激活到 CPU。
全部实现位于 `scripts/pgr_memory.py`，未更改 src/diffusion/rgcn，仍兼容旧断点。

**先停止原续跑 Python 进程，确认退出；同一个 run-dir 只能有一个训练进程。**
可用 `pgrep -af 'python.*[r]esume_pgr_cpu_offload.py'` 查找对应 ICEWS18 进程，
对其 PID 执行普通 kill 后等待退出。未完成阶段会重做；原恢复器会从 latest
重写未完成阶段可能覆盖的 best 权重。

```bash
cd /data1/zihao/DGAR-code/.worktrees/pgr
export DGAR_PYTHON="$(command -v python)"
export DGAR_DATA_ROOT=/data1/zihao/DGAR-code/data
export PGR_GPU=1
export PGR_FAST_ID="$(date +%Y%m%d_%H%M%S)_$$"
mkdir -p logs

setsid bash -c '
set -euo pipefail
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
"$DGAR_PYTHON" -u scripts/resume_pgr_cpu_offload.py \
  --gpu "$PGR_GPU" --memory-mode checkpoint \
  --run-dir experiment_runs/ICEWS18-pgr-mean-shared-seed1-20261016_002715_1682535 \
  2>&1 | tee "logs/ICEWS18-mean-shared-checkpoint-$PGR_FAST_ID.log"
' < /dev/null > "logs/ICEWS18-mean-shared-checkpoint-launcher-$PGR_FAST_ID.log" 2>&1 &
```

若再次 OOM，在旧进程退出后，改为
`--memory-mode checkpoint-selective --offload-min-mib 64 --pin-memory`，
使用新日志名称继续读取同一个 latest.pt。更大显存 GPU 可用 `--memory-mode none`。
先加 `--dry-run` 可只验证指纹和参数。

本次 30 项 CPU 检查通过，31 个 Python 文件编译通过；实际小型 mean+Shared
模型五种策略的 loss、梯度、BN buffers 和 RNG 逐元素相同，确认 checkpoint
确实多执行了 decoder forward，且实际 guided diffusion 的 autograd.grad 正常。
没有执行真实训练，也没有中断现有训练进程。
