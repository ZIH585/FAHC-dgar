# ECRI 实现与复现实验说明

## 1. 来源与范围

- 真正的 Original 起点：`a454564`，`master` / `origin/master` 的 Initial commit。
- 独立分支：`codex/ecri`；共同修复的基线提交：`f7bf2d8`。
- 没有继承 FAHC 分支的提交，没有复制其 selector、risk/coverage、预算、probe、
  KD、mechanism diagnostic 或 staged launcher。
- 原工作区及其数据、日志、模型保持原状。本分支取消跟踪原提交中包含的运行产物，
  因而初始清理提交显示大量删除；旧工作区文件和 Git 历史仍保留。
- ECRI 是此前提出的 Evolution-Calibrated Replay Integration 的 DGAR 实现，
  不是已验证有效的方法，也不是某篇论文实现的直接复刻。

必要的共用修复见 [BASELINE.md](BASELINE.md)。关掉 ECRI 回到的是这一共同修复后的
Original，而不是承诺逐位复现 `a454564` 的所有缺陷或既有日志。

## 2. 原始数据流与插入位置

本仓库的原始流程先按时间切分原 train/valid/test 文件，然后连接这些时间片，
**每片按原行序再划分 80%/10%/10%**。实现保留这个协议；ECRI 中的 `train[t]`
指重新划分后的训练部分，不是该片的全部数据。

1. 第 0 片只作为历史；原训练循环从第 1 片开始。
2. `main.run_experiment` 为阶段 t 构造过去 `train_history_len` 片的 RGCN 图。
3. `Att_Diffuse_model.gereration_feature` 按原随机策略选历史时间片，按当前 query
   的 subject 取相关历史事实。默认训练路径的候选片为 `[0,t-1)`；测试为 `[0,t)`。
   保留这个原始差异，不重新定义 HCP、不加入事实预算。
4. 旧阶段扩散模型和已有 RGCN guider 生成回放表示。
5. `RecurrentRGCN.select_entity` 对 HCP 中涉及的实体聚合生成表示 H。
6. `RecurrentRGCN.forward` 每个历史图更新后，在原 `alpha*C+(1-alpha)*H`
   位置用 ECRI 的实体条件系数替换 alpha；其后的 time gate、decoder 保留。
7. 原实体 CE、关系 CE、回放 CE、静态约束训练预测模型，然后按原流程训练扩散模型。
8. 验证选最佳参数；恢复匹配的 Adam 状态及已有 feature teacher，保存阶段状态。

ECRI **不改变所选时间片、事实集合、回放 CE 权重或候选实体全集**。它改变的是
历史表示在预测融合中的比例，没有新增 loss，没有新增 old model。

## 3. 演化分数：训练前、仅过去、与模型预测无关

阶段 t 开始时使用两个完整、相邻、等长窗口：

\[
 W_o=[t-2W,t-W),\qquad W_r=[t-W,t).
\]

每条正向训练事实 `(s,r,o)` 为 s 贡献有向角色邻居 `(r,o)`，为 o 贡献
`(r+R,s)`。关系范围是 `[0,R)`，逆角色范围 `[R,2R)`；统计输入禁止重复加入逆事实。

对实体 e，从两窗口各用独立于训练随机流的 RNG 抽取
`n=min(n_old,n_recent,sample_cap)` 个角色邻居。计算：

- 角色变化：有向关系频率分布的 JSD，以 2 为对数底。
- 邻域变化：有向 `(relation-role, neighbor)` 集合的 Jaccard distance。
- 采样 null：混合两组样本后，重复 P 次等量随机划分，计算相同统计。

\[
 d_e=[JSD_e-\operatorname{mean}(JSD_{null,e})]_+,\quad
 u_e=[churn_e-\operatorname{mean}(churn_{null,e})]_+.
\]

\[
 c_e=\mathbf1[t\ge2W\land n\ge m]\frac{n}{n+\kappa}.
\]

窗口未完整、两侧任一缺证据或支持数不足时，`d=u=c=0`。默认前 60 片处于窗口
预热期。均值 null 扣除是采样校准，**不是显著性检验或 p-value**。
每个实体的抽样种子由实验 seed、阶段和实体 ID 决定，不消耗 HCP 或模型 RNG。

这些分数不使用当前模型、旧模型或二者 loss 差值；它们描述**更新之前已观察到的
训练结构变化**。`train[t]` 在阶段结束后才加入统计，validation/test 不加入统计。

## 4. 门控公式与梯度

\[
 \beta=softplus(b_r)>0,\quad\gamma=softplus(b_u)>0,
\]
\[
 \alpha_{e,t}=(1-c_e)\alpha_0+
 c_e\sigma(b+\beta d_e+\gamma u_e),\qquad
 h_{e,t}=\alpha_{e,t}C_{e,t}+(1-\alpha_{e,t})H_{e,t}.
\]

`alpha_0` 是 Original 自己学习的、未受约束的标量。回退时不 sigmoid、不 clamp。
新增可学习参数只有 b、b_r、b_u 三个标量，由原任务 loss 端到端更新。
`beta/gamma` 采用正值参数化，明确表达“更强演化偏向当前表示”的假设。
开启时不保证所有实体与 Original 恒等初始化；低支持回退保证成立。

每阶段 `[num_entities,3]` 特征只传到模型设备一次、detach 并缓存；forward 只索引
涉及的实体，无新增逐 batch 的 CPU/NumPy 同步。缓存不保留计算图、不进入模型
state_dict。运行日志每阶段传回三个参数一次。

**适配选择：扩散引导不使用 ECRI 门控。** `DiffuRec.p_mean_variance` 调用
`RecurrentRGCN.forward(..., apply_ecri=False)`，沿用原 scalar alpha。
原因是该 guider 使用上阶段模型与历史图；把当前阶段的演化上下文直接塞进去会混淆
时序、同时改变生成机制。第一版只研究预测端融合，避免两个改动位置混杂。
原引导仍会对生成表示求梯度，评价中不会因此更新 ECRI 参数或调用 optimizer。

## 5. 调用顺序、评价和恢复

```text
加载 Original 数据划分 → 校验/指纹化历史缓存 → 初始化两个原模型及其既有 teacher
若 ECRI：observe(0, train[0])
for t = 1 ... T-1:
    ECRI.features(t)                    # 只见 train[:t]
    model.set_ecri_context(features,t)  # 在生成/训练之前固定本阶段统计
    Original HCP → diffusion replay     # 不选择、不重排、不新增预算
    原 get_loss / backward / optimizer  # forward 中实体条件融合
    原 validation / 原 diffusion training
    恢复最佳 model/diffusion 和对应 Adam，更新原 teacher
    写 ecri.jsonl → observe(t,train[t])
    原子写 latest.pt
最终 TEST / 可选 VALIDATION
```

最终评估保留 Original 的**最终模型回看各时间片**协议，不改成 prequential 协议。
第 0 片仍跳过；Current 是最后一片，Average 是 1..T-2 的宏平均，pooled MRR/Hits
按全部评估 query 汇总并含逆向 query。过滤答案只用于指标计算。

每次完整评估重新构造统计队列，在预测时间片 t 时只使用训练前缀 `train[:t]`。
不用训练结束时的统计替早期 query 计算特征。图、缓存事实也只来自 query 的过去。
模型参数本身是最终训练参数——这是原始回看评估的定义，不能把它称为逐时间片在线
预测。测试标签不进入训练 loss、演化统计或模型选择。

`--run-dir` 包含：

| 文件 | 内容 |
|---|---|
| config.json | 所有配置、源文件/数据/历史缓存 SHA256 |
| model.pt / diffusion.pt | 最近已选择的最佳模型权重 |
| latest.pt | 已完成阶段、两模型、两 Adam、Python/NumPy/torch RNG、ECRI 队列 |
| ecri.jsonl | 仅 ECRI：时间边界、预热标志、有效实体数、分数均值、门控参数 |
| fitlog/ | 原日志系统输出 |

`latest.pt` 是恢复训练的依据；`model.pt` 不能代替完整恢复。恢复后从下一阶段重新
计算特征，同一 run_dir、配置、源码、数据和缓存不匹配时拒绝恢复。若在一个阶段
中途退出，重跑该阶段，不支持 mid-epoch 恢复。追加式 ecri.jsonl 在极少数“已写
日志但还没原子保存 latest.pt”中断时可能出现重复阶段行，以同阶段最后一行为准。

同一训练配置下，加 `--test` 可只加载权重评估；ECRI 状态从训练前缀重建，不从
test 内容恢复。不能把 enabled 权重静默当作 disabled 模型加载。

## 6. 参数和开销

| 参数 | 默认值 | 含义 |
|---|---:|---|
| --ecri | 关闭 | 启用 ECRI |
| --ecri-window | 30 | 两个统计窗口各自的时间片数 |
| --ecri-min-support | 10 | 两窗口均需达到的角色样本数 |
| --ecri-sample-cap | 128 | 每实体每窗口统计采样上限；不是回放事实预算 |
| --ecri-permutations | 20 | null 随机划分次数 |
| --ecri-kappa | 20.0 | 支持可信度收缩强度 |
| --ecri-bias-init | 0.0 | 门控 bias 初值 |
| --ecri-role-init | 1.0 | 正值角色斜率初值 |
| --ecri-churn-init | 1.0 | 正值邻域斜率初值 |

通用新参数为 `--data-root`（默认项目 data）、`--run-dir`（默认无）、`--resume`
（默认无）、`--final-validation`（默认关闭）。Original 所有原有参数的默认值已
通过 AST 提取比较，未改变。开启 ECRI 要求 run_dir、单次时间顺序遍历、add 融合；
不接受关闭 replay feature 或 grid-search，防止名义开启但没有融合可用。

额外 GPU 常驻量约为 `3*E*sizeof(float32)` 加三个门控参数，ICEWS14s 约 84 KiB
特征表；teacher 多三个无训练参数。还会有每层门控的少量临时张量及梯度。
CPU 保存最近 `2W` 片的正向训练 triples，用于窗口移动与恢复；**时间窗口有界，
总字节没有固定上界**。sample-cap 不限制这个原始统计队列大小。
每阶段 CPU 校准成本约为 `O(窗口事实数 + 有效实体数*P*sample_cap)`，未测实际
完整训练耗时。没有多一次模型探测、teacher forward 或新的回放预算。

## 7. 验证结果与尚未验证事项

实现环境：Python 3.11、torch 2.4.0+cu121、DGL 2.4.0+cu121、NumPy 2.4.6。
执行时屏蔽 CUDA、单 CPU 线程。20 项 unittest 约一秒测试时间，进程总时长约四秒：

- off 路径对照 f7bf2d8：state_dict、loss、梯度、一次 Adam 更新逐项相等。
- 实际 RGCN+DGL forward/backward、静态图、ECRI 参数梯度及 detached context。
- 实际扩散引导绕过 gate；单条事实扩散 CE forward/backward 有限。
- 第一片、空回放、缺上下文、单条 query、低支持、样本少于 cap、正逆角色与非法 ID。
- 两个完整窗口预热、严格时间游标、固定 seed、独立 RNG、队列恢复。
- 主训练函数停在第一个合成 batch loss 后，证实设置上下文后才调用 loss；没有 epoch。
- 最终评估使用各自过去的图和特征，无 ECRI 参数梯度。
- 参数拒绝旧 FAHC 标志；最佳 Adam 与完整阶段恢复；错配缓存拒绝。
- 当前 ICEWS14s 的 365 个真实历史缓存与训练划分一致，只读加载校验；没有模型训练。
- 全部 Python 文件 py_compile、import、--help、git diff --check 和默认参数比较。

仍未验证：GPU 算子/显存峰值、真实长时间训练数值稳定性、跨机器完全确定性、MRR
提升和多 seed 方差；其他数据集的缓存一致性需在各自启动时校验。ECRI 的统计变化
能否预测回放效用仍需正式实验，不能以 smoke test 代替科研结论。

保留两个 Original 已知问题：扩散条件用实体表索引关系 ID；默认 beta 末端为 1，
会在未被当前 x-start 路径调用的 reciprocal 数组构造时提示除零。极小实测路径有限，
本次未改采样 schedule。详见 BASELINE.md。

## 8. 推荐第一组正式实验（此处仅给命令，不执行）

在另一服务器 clone 完成并激活原 dgar_env 后运行。用同一个新分支、同一 seed 和
默认 Original 参数，**依次**跑关开两组；原来的 FAHC/B4096 日志不能替代这组对照。
代码下的 logs 保存终端输出，experiment_runs 保存配置、状态和 fitlog。

```bash
cd /data1/zihao/DGAR-ECRI
export DGAR_PYTHON="$(command -v python)"
export DGAR_ECRI_ROOT="$PWD"
export DGAR_DATA_ROOT=/data1/zihao/DGAR-code/data
export DGAR_RUN_ID="$(date +%Y%m%d_%H%M%S)"
mkdir -p logs

setsid bash -c '
set -euo pipefail
cd "$DGAR_ECRI_ROOT"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

"$DGAR_PYTHON" -u src/main.py --gpu 0 --dataset ICEWS14s --seed 1 \
  --data-root "$DGAR_DATA_ROOT" \
  --run-dir "experiment_runs/ICEWS14s-original-s1-$DGAR_RUN_ID" \
  --final-validation 2>&1 \
| tee "logs/ICEWS14s-original-s1-$DGAR_RUN_ID.log"

"$DGAR_PYTHON" -u src/main.py --gpu 0 --dataset ICEWS14s --seed 1 \
  --data-root "$DGAR_DATA_ROOT" \
  --run-dir "experiment_runs/ICEWS14s-ecri-s1-$DGAR_RUN_ID" \
  --final-validation --ecri 2>&1 \
| tee "logs/ICEWS14s-ecri-s1-$DGAR_RUN_ID.log"
' < /dev/null > "logs/launcher-$DGAR_RUN_ID.log" 2>&1 &
echo "launcher PID: $!"
```

`python -u` 保证及时输出；launcher 日志记录启动错误，pipefail 使第一组失败时
停止而不是继续第二组。GPU 被占用时不要执行。后续看日志：

```bash
tail -f "logs/ICEWS14s-original-s1-$DGAR_RUN_ID.log"
# 第二组开始后：
tail -f "logs/ICEWS14s-ecri-s1-$DGAR_RUN_ID.log"
```

中断后的恢复示例（替换 RUN 为真实目录，保留原训练参数）：

```bash
python -u src/main.py --gpu 0 --dataset ICEWS14s --seed 1 \
  --data-root /data1/zihao/DGAR-code/data --ecri \
  --run-dir "$RUN" --resume "$RUN/latest.pt" --final-validation
```

先比较同区段 Final TEST/VALIDATION 的 Current、Average、pooled filtered MRR 与
Hits。第一组是可行性对照，未完成多 seed 与消融前不能宣称方法成立。不要加入 KD、
改变 HCP 数量或同时扫多组窗口参数，保持首组实验可解释。
