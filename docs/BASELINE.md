# DGAR Original 基线审计与恢复

## 基线来源

本分支 `codex/original-baseline` 从 `a4f00eb` 新建，路径是
`a454564 → f7bf2d8 → a4f00eb`。这个起点包含运行兼容修复，但没有
FAHC、ECRI、QCDR 的方法实现。随后单独恢复被公共修改改变的科研行为。
没有在已有方法分支上修改，也没有覆盖主工作区的数据或实验。

逐文件核对的作者仓库是 [zyzhang11/DGAR](https://github.com/zyzhang11/DGAR/tree/c5bbc87467bb22d266b664d30eefb8184c1d3104)，
固定提交 `c5bbc87467bb22d266b664d30eefb8184c1d3104`。其 19 个顶层核心
Python 文件的 Git blob SHA 记录在 `tests/reference/manifest.json`。
最初本地提交 `a454564` 的 18 个核心文件与作者完全相同，只有 `src/main.py`
已不同：本地冻结特征模型并切换到 eval，同时修复评估报告和独立测试入口。
因此 `a454564` 也不能直接称为逐行相同的作者源码。

官方 `main.py` 和 `diffusion/model_21.py` 原文保留为 `.py.txt` 测试参照，
测试检查其 Git blob SHA。参照文件只用于小型 CPU 对照，不作为训练入口。
`.gitattributes` 保留这些原文的行尾和空白，不以格式化破坏其官方 SHA。

## 已证实的行为差异与本次处理

| 位置 | 以前的行为 | 本次恢复的作者行为 | 为什么会影响结果 |
|---|---|---|---|
| 特征生成模型 `model_repre/model_diffpre` | 本地最初提交已调用 eval | 保持 train；参数仍冻结 | dropout、RReLU、BatchNorm 的行为及缓冲量不同 |
| 阶段衔接 | f7 同时回退 best 权重及 best Adam | 下一阶段仅加载 best 权重；Adam 保留最后一次更新的状态 | 动量及步数改变后，后续优化轨迹不同 |
| 验证及最终评估 RNG | f7 固定验证种子、隔离随机流、最终评估另行播种 | 与训练共用 Python/NumPy/Torch 随机流 | 验证 HCP、扩散噪声、早停和后续随机样本都会变化 |
| HCP 事实顺序 | f7 使用 sorted(set) | 官方 list(set) | 事实集合相同，但随机噪声与行的对应关系不同 |
| 最终 TEST 扩散引导 | f7 使用当前 query 的历史图 | 官方传递 last_history/last_output | 当前 query 的 decoder 历史仍是自身窗口；扩散引导历史不同 |
| 最后阶段训练顺序 | f7 删除中间 TEST 诊断 | 恢复 predictor 后、diffusion 更新前的一次官方诊断 | 会消耗随机数和更新 train 模式生成模型缓冲量 |
| 单事实关系 decoder 的 eval | 公共修复在 train/eval 都跳过 bn2 | 仅 train 单事实时跳过；eval 保留作者 bn2 | eval 的 BatchNorm 原本可以处理单事实 |

这些是已经从代码确认的差异。没有正式训练，不能给出各项造成多少 MRR
下降的因果分解，也不能保证恢复后某一个运行恰好达到论文数值。

### 官方随机种子的事实

作者 CLI 声明 `--seed`，但 `run_experiment` 的 `set_seed` 调用被注释。
NumPy 在 `rgcn/knowledge_graph.py` 导入时固定为 123；Python/Torch 没有统一
实验播种。本分支不传 `--seed` 时保留这个策略，Namespace 显示 `seed=None`。
显式 `--seed 1` 则同时播种三种 RNG，是公开的可复现选项。之后验证、训练和
最终评估仍共用一个随机流，绝不再按阶段隔离。未来 Original 与方法必须用
同一个显式 seed 或同样的省略策略，不能混用。

### 官方最终评估历史的局限

本分支明确复现作者的**训练完成后回看所有时间片**流程：首个被评估的
时间片 1 的扩散引导来自最终训练阶段的历史；随后使用上一个评估时间片的
历史。当前 query 的 predictor 图窗口仍按它自己的 time_idx 构造。
这不是严格在线预测的统一历史协议，不应将其描述成所有支路都只看 query
之前的数据。以后若研究严格因果评估，应新增独立协议版本与对应基线。
恢复这一流程是为了与作者代码一致，不是为了选择更高的分数。

`last_output` 虽被传入，但作者的有效引导逻辑没有读取它；本次没有发现
利用 test label 优化参数或决定早停的证据。最后阶段的中间 TEST 指标只打印，
不用于模型选择，其标题与最终指标区分。对论文表格应看 `Final full-pipeline TEST`。

## 保留下来的运行修复

- CPU/指定 device 安全的图元数据、张量及零初始化；空 replay 的处理。
- 单事实 diffusion CE 保持一维标签；只在不合法的 train 单事实关系 BN 时跳过。
- 字符串比较由 `is` 改成 `==`；零范数静态约束避免 NaN。
- 路径不依赖 cwd；布尔与浮点参数正确解析。科学参数默认值对照官方 parser。
- 保留官方时间拆分、原 HCP 抽样范围、subject filtering、扩散目标、replay CE、
  可学习全局 alpha、decoder 和默认静态约束。不引入 KD、risk、预算或新方法参数。
- 原缓存必须存在，只校验、不重建。允许缓存原有 forward-only 或含逆关系形式，
  但其内容必须对应当前训练快照；不根据测试集或未来数据修改缓存。
- 启用 replay 时仍拒绝非零 `start_reply`：作者这个选项依赖循环覆盖变量，
  会造成早期阶段没有 diffusion checkpoint。默认 0 与作者执行条件相同。
- `--run-dir` 独立保存 config、best 权重及 latest；源代码/数据/缓存均记录 SHA256。
  新 checkpoint 标识 `official-dgar-v1`、version=2，与此前实验 checkpoint 不兼容。
- latest 保存 best 特征权重、最后 Adam 状态及阶段结束 RNG。恢复时重写可能被
  未完成下一阶段覆盖的 best 文件；CPU 对照证明续跑与完整循环相同。
- 完成训练后用相同参数 `--test --run-dir`，从 latest 恢复最终 TEST 入口随机流，
  并检查 best 文件一致性。没有 run-dir 的独立测试无法恢复历史 RNG，不能保证
  与训练结束时的随机扩散结果相同。
- `--final-validation` 是可选补充报告，在 TEST 后执行，因而不会影响已输出的 TEST。

作者 diffusion 通过 entity table 索引 relation condition 的原有语义保留。
本分支没有借修复基线改变模型结构，也没有修改任何已有数据/cache/checkpoint。

## 本次验证及证据边界

执行 `scripts/check_baseline.sh`：无可见 GPU，仅数条 synthetic triples。

1. 参照源码 Git blob SHA 验证。
2. 实际主程序的 chronological loop 与实际官方 loop，在 scalar models + real Adam
   下对比调用、权重、Adam 步数、teacher 缓冲量及 RNG；故意让第一轮 best、后两轮
   较差，确保检测出“回退 Adam”这种偏差。
3. 实际 test() 的 HCP 抽样、顺序、扩散引导历史、最终 RNG 和 rank 对照。
4. 实际 HCP selector 与官方 selector 的事实行顺序及 RNG 对照。
5. 断点恢复与连续循环对照，并模拟下一阶段部分覆盖 best checkpoint。
6. 独立 TEST 的 RNG 恢复、阶段完整性和 stale 权重检查。
7. 实际 RGCN + guided diffusion 的小型 CPU 前向：冻结/no_grad 与官方 train 模式
   数值逐元素相同，BatchNorm 缓冲量相同，内部引导梯度仍可计算。
8. 一个 synthetic loss backward 和 prediction：单事实输入、inverse、shape、有限数值。
9. 单事实 diffusion CE、训练 cache 不变、官方参数默认值、无方法参数。
10. import、--help、所有 Python 文件 py_compile。

这些检查验证所恢复的行为和代码连接。没有运行真实数据 epoch、GPU训练或任何
完整数据集训练，因此当前没有“修复后 MRR 已达论文”的实验结果。
真实 CUDA/DGL 数值差异、运行显存和论文性能仍要由后续一次正式复现确认。

## 后续运行

基线工作区没有复制大数据。共享原数据根目录即可，输出必须使用新 run-dir。
下面只给命令，不在代码修复时执行。

```bash
cd /data1/zihao/DGAR-code/.worktrees/original-baseline
mkdir -p logs
setsid bash -c '
set -o pipefail
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python -u src/main.py --gpu 0 --dataset ICEWS14s \
  --data-root /data1/zihao/DGAR-code/data \
  --run-dir /data1/zihao/DGAR-code/.worktrees/original-baseline/experiment_runs/ICEWS14s-official-v1 \
  2>&1 | tee /data1/zihao/DGAR-code/.worktrees/original-baseline/logs/ICEWS14s-official-v1.log
' < /dev/null > /dev/null 2>&1 &
```

该命令省略 seed，以作者的初始化策略核对复现。未来做同一分支 Original/新方法
对照时建议双方显式传入同一个 `--seed 1`；这不意味着要求补跑多 seed。

```bash
tail -f /data1/zihao/DGAR-code/.worktrees/original-baseline/logs/ICEWS14s-official-v1.log
```

如果对应 run-dir 已存在且有内容，换名称；勿删除旧结果或把旧方法 checkpoint
作为本次起点。看最终 TEST 中 Current 和 Average macro，同时保留 pooled。
论文 [ACL 2025 DGAR](https://aclanthology.org/2025.acl-long.537/) 的 ICEWS14s
Current MRR 为 58.59、Average MRR 为 50.12（百分数，五次运行平均）；
不把 pooled MRR、另一种协议的分数或单次运行与其混为同一统计量。

## 以后新方法的 Git 起点

```bash
git -C /data1/zihao/DGAR-code worktree add \
  -b codex/new-method \
  /data1/zihao/DGAR-code/.worktrees/new-method \
  codex/original-baseline
```

未来只在新分支增加默认关闭的方法，并验证关闭时与此基线相同。共享修复若
改变训练/评估协议，先单独记录，不混入方法收益。`AGENTS.md` 和官方参照测试
用于防止基线再次无意漂移。
