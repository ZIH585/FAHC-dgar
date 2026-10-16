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
