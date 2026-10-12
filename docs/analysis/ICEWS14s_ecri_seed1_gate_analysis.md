# ECRI seed1 阶段统计复核与下一步（2026-10-12）

现有记录排除了“配置不同”与“ECRI 根本没接上”的主要疑点。更明确的发现是：
最终门控主要受 bias 驱动，实体演化项的额外系数影响较小。实体 forecasting 的
现有 TEST/VALIDATION 主指标仍下降，当前版本没有超过 Original 的证据。
下一步仅做一次现有 checkpoint 的验证融合干预，不重新训练、扫参数或加入模块。

## 1. 可比性与数据时序

两组 config.json 的 source_sha256、data_sha256、history_sha256 完全相同，settings
只有 ecri=false/true 不同。364 条 ECRI 记录覆盖阶段 1..364，没有缺失、重复、
非有限值；prefix_last_snapshot=t-1，窗口上界为 t。1..59 均为预热，60 开始生效，
共 305 个非预热阶段。数据边界符合实现所规定的历史前缀顺序。

非预热期可信实体为 123～173 个，均值 145.75；平均可信度为 0.5367。平均 role
excess 为 0.0694，churn excess 为 0.0469。本地七个选定阶段重算的统计与实际
JSONL 一致，数据与历史缓存哈希也与训练记录一致。

额外发现：两次训练源码哈希均为
`c32134fd804b15db3c38a511749373ecfd66561b5e5be2b12b0f6e8bbd2d97aa`，
当前本地 918573f/工作区源码集合哈希为
`a5eebf500e0ddc62136d9c46ba07d731fcfee79039a1ef2998290aed60ef5858`。
这不影响“两次运行使用相同记录源码”的判断，但不能假定训练目录的源文件集合
与本地完全相同。额外 .py 文件或本地编辑都可能导致差异；现有配置未记录逐文件
哈希，不能据此定位。功能诊断应先在原训练目录 --dry-run 核对，不能默认绕过。

## 2. 门控确实学习，但 bias 占主要作用

这些参数是在各阶段训练/最佳恢复之后记录的。最终完整回看评估统一使用最后
checkpoint 的门控参数，不能把早期行的参数当作最终模型评估早期 query 的参数。

| 阶段 | bias | role weight | churn weight | 可信实体 |
|---|---:|---:|---:|---:|
| 1 | 0.0000 | 1.0000 | 1.0000 | 0 |
| 59 | 0.0000 | 0.8354 | 0.8354 | 0 |
| 60 | 0.0293 | 0.8520 | 0.8520 | 138 |
| 120 | 0.7344 | 1.2811 | 1.2832 | 132 |
| 240 | 0.9953 | 1.3686 | 1.4199 | 127 |
| 364 | 1.2790 | 1.3066 | 1.4393 | 158 |

最终自适应分支是 `sigmoid(1.279 + 1.307*d + 1.439*u)`。
即使 d=u=0，也输出 0.7823；根据实际过去特征重算，七片可信实体上的平均输出为
0.8057～0.8114。并未饱和到 1，也不是严格常量，但偏置贡献明显强于演化项。

以 t=364 为例，演化 logit 平均约 0.1901，而 bias 为 1.2790。固定其余值，
把演化项置零使最终融合系数在可信实体上平均降低约 0.0152；把 bias 置零则平均
降低约 0.1443。两项是分别定义的反事实差值，不是可相加的唯一贡献分解。

这里的 0.782/0.811 是 sigmoid 自适应分支，不是最终所有实体的 alpha。
实际系数还有 `(1-c)*alpha0`；本地未收到 model.pt，不能恢复 alpha0 或判断实际
历史权重相对单独训练的 Original 是增是减。差值是系数单位，不是 MRR 变化。
重算仅涵盖可信实体，不能当作实际 HCP/query 加权的效用证据。

## 3. 一个工程细节：预热期斜率仍被 weight decay 更新

阶段 59 时 c=0，没有有效演化融合，但两个斜率已从 1 降到 0.8354（约 -16.5%）。
模型在有 HCP 时仍计算 gate，任务对它的梯度为零；Adam 使用耦合 weight_decay，
可以更新具有零梯度的参数。这与日志中的对称衰减、bias 保持零一致。

这一点应在下一版中修正，例如门控单独不使用 weight decay 或预热时冻结。它影响
有效初值与实验解释，但没有证据证明它单独造成了 MRR 下降。当前不改训练优化器，
避免在既有 checkpoint 的功能验证前引入新变量。

## 4. 已准备的固定 checkpoint 诊断

脚本：`scripts/diagnose_ecri_gate.py`。只读取 ECRI run 的 config/model/diffusion，
不调用 optimizer/backward，不保存或覆盖 checkpoint。原扩散采样仍需对潜变量
求梯度，这是引导生成过程，不是参数训练。诊断默认只评估 VALIDATION 的
90、180、300 三片，各最多 16 条正向 query，保留逆向 query 和完整候选实体集。

每片先抽固定验证 query、随机 HCP，再生成一次回放；后续所有模式复用相同的
图、query、标签过滤、HCP 和生成表示：

| 模式 | 检验内容 |
|---|---|
| full | 已学习 ECRI 门控 |
| bias_only | 保留 bias、可信度，令 d=u=0：演化项是否有额外价值 |
| scalar_alpha | 关闭 gate，使用同一个 ECRI checkpoint 的 alpha0 |
| alpha025 / alpha075 | 固定 0.25 / 0.75：更偏当前是否在高演化 query 上更有益 |
| current_only | 固定 alpha=1：不在此预测融合接口注入 H |
| shuffle_0..4 | 20/40/80 支持数量分箱内联合打乱 d/u，保留 c；记录实际打乱比例 |

scalar_alpha **不是单独训练的 Original DGAR**，所有模式沿用 ECRI 学出的参数。
因此该实验识别最终 checkpoint 融合的功能作用，不能代替重训的算法消融，也不能
排除演化信号曾在训练阶段影响参数。query 分组按 subject 的演化强度，排名还会
受候选表示影响，因此高/低组也不是精确的因果归因。逆向 query 与正向 query 成对，
不能作为完全独立样本计算统计置信度。

## 5. 运行命令（没有执行真实 checkpoint 诊断）

在之前实际训练的服务器、保留两份模型权重的目录执行。先激活同一个 dgar_env。
当前同步到本地的 run_dir 只有 config.json/ecri.jsonl，没有模型权重，不能在
这个仅含记录的目录执行真实诊断，也不需要为此重新训练。

```bash
cd /data1/zihao/DGAR-ECRI
git pull --ff-only origin codex/ecri

python -u scripts/diagnose_ecri_gate.py \
  --run-dir experiment_runs/ICEWS14s-ecri-s1-20261012_100243 \
  --data-root /data1/zihao/DGAR-code/data \
  --output-dir experiment_runs/ecri-gate-diagnostic \
  --stages 90,180,300 --query-limit 16 --shuffle-repeats 5 \
  --dry-run
```

dry-run 不读取模型权重或数据、不初始化 CUDA；会检查文件存在并输出逐源码文件
哈希和 source_matches_config。若 false，先检查训练目录的 Git 改动和额外 .py
文件，保留训练时代码版本。脚本提供显式 forensic override，但常规对照不建议用它。
数据或缓存哈希错配始终拒绝；真实运行还会校验 checkpoint gate 与末条 JSONL。

确认 source_matches_config=true 后，运行一组有限验证诊断：

```bash
cd /data1/zihao/DGAR-ECRI
mkdir -p logs
export DGAR_DIAG_ID="$(date +%Y%m%d_%H%M%S)"

setsid bash -c '
set -euo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python -u scripts/diagnose_ecri_gate.py \
  --run-dir experiment_runs/ICEWS14s-ecri-s1-20261012_100243 \
  --data-root /data1/zihao/DGAR-code/data \
  --output-dir "experiment_runs/ecri-gate-diagnostic-$DGAR_DIAG_ID" \
  --gpu 1 --stages 90,180,300 --query-limit 16 --shuffle-repeats 5 \
  2>&1 | tee "logs/ICEWS14s-ecri-gate-diagnostic-$DGAR_DIAG_ID.log"
' < /dev/null > "logs/ecri-gate-launcher-$DGAR_DIAG_ID.log" 2>&1 &
```

只有这个诊断的推理使用指定 GPU；不进行完整 epoch 或测试集评估。GPU 被占用时
先只执行 dry-run。CPU 可显式指定 --gpu -1，但真实扩散生成可能较慢，不把它称为
几秒内 smoke test。query-limit 限制验证 query，不篡改 HCP 事实集合。

输出 summary.json 含各模式 pooled/分片 MRR、checkpoint alpha、门控权重、哈希、
HCP 数量与 shuffle 比例；queries.csv 含配对 query 的全部排名。把这两个小文件和
诊断日志同步用于后续判断，不必同步大型权重。

## 6. 如何决定继续或停止

- 若 full 在预选片上持续好于 bias_only 与 shuffle，并且高演化组更受益于
  alpha075 相对 alpha025，才支持进一步投入。此时只做一个有依据的修改：
  零演化严格回退原 alpha，并修正 gate 的预热衰减，再作一次同条件对照。
- 若 full 与 bias_only/shuffle 基本相同，当前结构演化信号没有体现可用的最终
  融合辨别价值。结合现有总体实体 MRR 下降，不再为当前 ECRI 扫参或加 KD。
- 若 scalar_alpha 更好，说明已学 checkpoint 的当前 gate 会损害这些验证 query。
  不能声称关开 switch 就能恢复单独训练的 Original 成绩，也不据此直接重训。
- 若片间混合、差值仅涉及少数 query，则是机制证据不足。这个小样本诊断用于资源
  决策，不能当论文显著性结论；保持当前版本“未改善”的结论，不扩大训练搜索。

本次验证：23 项纯 CPU 单元检查通过，新增三个检查覆盖状态/异常恢复、分箱 shuffle
和一片合成数据的真实模型验证流程。没有运行真实模型权重评估、没有占用 GPU，
没有修改 src/diffusion/rgcn 的训练代码。
