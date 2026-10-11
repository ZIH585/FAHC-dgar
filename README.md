# DGAR + ECRI

独立分支 `codex/ecri` 直接源自 Original DGAR `a454564`，支持同一套代码的
DGAR 基线与 **Evolution-Calibrated Replay Integration (ECRI)** 对照。
默认关闭 ECRI；没有 FAHC selector、probe、fact budget 或新增 KD。

- [方法、参数、数据流、验证与正式实验命令](docs/ECRI.md)
- [Original 来源和两组共同使用的必要修复](docs/BASELINE.md)

另一台服务器使用独立目录，保留旧实验目录：

```bash
cd /data1/zihao
git clone --single-branch --branch codex/ecri git@github.com:ZIH585/FAHC-dgar.git DGAR-ECRI
cd DGAR-ECRI
git log -3 --oneline
```

数据、缓存和模型不随此分支提供。使用原环境，传入原数据目录：
`--data-root /data1/zihao/DGAR-code/data`。必须含数据文件、静态图和匹配的
`history_snap_v2`；启动时会拒绝错配缓存，不自动改变缓存的正向/逆向约定。

只检查代码，不训练：

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 python -B -m unittest discover -s tests -v
CUDA_VISIBLE_DEVICES='' python -B src/main.py --help
```

单次运行的形式（正式执行前请阅读方法文档）：

```bash
# Original DGAR（共同修复后的对照）
python -u src/main.py --gpu 0 --dataset ICEWS14s --seed 1 \
  --data-root /data1/zihao/DGAR-code/data --run-dir experiment_runs/original-s1

# DGAR + ECRI
python -u src/main.py --gpu 0 --dataset ICEWS14s --seed 1 \
  --data-root /data1/zihao/DGAR-code/data --run-dir experiment_runs/ecri-s1 --ecri
```

每次使用新的 `--run-dir`；已有运行需用 `--resume .../latest.pt` 显式恢复。
本分支尚无正式训练结果，不表示 ECRI 已获得性能提升。
