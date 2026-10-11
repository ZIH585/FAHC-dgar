# DGAR 数据演化与本地 HCP 缓存核验

日期：2026-10-11。本目录是 CPU 数据诊断补充，不是新方法训练结果，也不是模型表示漂移实验。

## 可复现命令

独立工具仅使用 Python 标准库，不导入 torch、DGL，不使用 GPU：

```bash
cd /data1/zihao/DGAR-code
python scripts/analyze_dgar_evolution.py \
  --data-dir /data1/zihao/DGAR-code/data/ICEWS14s \
  --output-dir /data1/zihao/DGAR-code/logs/ICEWS14s-evolution-cpu \
  --split current-dgar --windows 5 --seed 1 \
  --permutations 200 --min-count 80 --max-sample 200
```

本次同参数运行约 16.45 秒。先前的20次置换试算仅用于确认工具运行，归档的 `summary.json` 和 `relation_windows.csv` 已换成200次置换结果。

`--prefix-snapshots 120` 可只分析前120个训练快照；窗口边界由所给前缀重新划分。`--split canonical-train` 改用原始 train.txt 的304天，适用于核对源文件协议，不能将其统计直接当成当前365阶段训练的统计。

## 分割与时间因果性

当前 `src/main.py:343–351` 和 `rgcn/utils.py:476–487` 实际执行：依次读取原 train/valid/test 文件的日快照（304/30/31天），串接365天，再将每一天按原文件行顺序的首 `floor(0.8*N)` 条分配为训练。没有 shuffle。剩余后缀的验证、测试事实不进入本诊断。原文件名叫 valid/test 的部分事实已被当前程序重新分配为训练，这与传统保持原时间划分的协议不同。

本次72437条实际训练事实按时间分为5个73天窗口。每个转移指标仅使用两个已完成的训练窗口，指标在新窗口结束后才能用于后续更新，不能反过来用于该窗口中尚未观察到的事实。源文件 SHA256、配置、缓存字节指纹都在 summary.json 中。

## 数据演化结果

Jensen–Shannon divergence 单位为 bits，范围0–1。尾实体检验要求每关系两窗口至少80条事实，各侧等量抽取最多200条；条件检验对相同 `(subject, relation)` 的重复 query 各侧匹配数量，并在该 query 内置换时间窗口标签。

| 转移 | 关系频率 JS | 实体频率 JS | 支持充分的正向关系数：尾分布 / 同主体条件分布 | 同主体条件 JS 减置换 null 均值的中位数 |
|---|---:|---:|---:|---:|
| D1→D2 | 0.011890 | 0.157238 | 27 / 13 | 0.065216 |
| D2→D3 | 0.010284 | 0.155152 | 26 / 14 | 0.068216 |
| D3→D4 | 0.010804 | 0.145873 | 27 / 15 | 0.074607 |
| D4→D5 | 0.007876 | 0.139275 | 28 / 16 | 0.061537 |

230类关系与7128类实体的空间维度不同，不能据两列 JS 直接断言“实体比关系漂移更严重”。这些是数据分布差异，不是 embedding drift。

关系4 `Make_a_visit` 的同主体条件分布超出 null 均值约0.2453（D1→D2）、0.3012（D2→D3）、0.2480（D4→D5）；关系10 `Use_unconventional_violence` 对应约0.0034（D1→D2）、0.0074（D4→D5），这两项置换 p 分别约0.438和0.338，没有同等强度的变化证据。

这支持“关系之间的时间分布变化程度不同”的探索性判断。不能把没有拒绝 null 当成已证明稳定，也不能把尾实体变动直接当成旧知识不再成立。许多稀有关系没有足够支持，CSV明确保留 unreliable 标记。200次置换未进行多重比较校正，p仅用于筛查，不作论文显著性结论。

正向关系r和构造的逆向关系r+230分开统计，CSV共有1840行。默认构造逆查询用于分析方向差异，未修改任何 HCP 文件。

## 365个本地 HCP 缓存全量核验

`cache_audit.json` 通过读取本地 pickle（使用已有环境中的 NumPy 解析其数组；无模型运行）逐文件核验：

- 365/365缓存的唯一事实集合，精确等于当前重新分割的当日正向训练事实。
- 逆向事实总数为0；缓存 key 与三元组 subject 不一致的数量为0。
- 没有发现当前实际验证/测试后缀事实混入本地缓存。
- 缓存清单摘要：`9b3e3aff24cd49af979d07b61a226a35c86589dc27885d47d5f111f5708bdff8`。

当前 `rgcn/utils.py:get_history` 会生成正向与逆向事实，已有缓存只含正向事实。因此重新生成缓存会改变 HCP 协议。跨服务器比较需要核对指纹；本次没有删除或重建缓存。独立标准库工具只计算缓存字节指纹，不执行 pickle 语义核验。本地核验也不能代替实验服务器缓存的核验。

## 表示架构与尚缺的证据

| 对象 | 当前代码中的实际张量 | 能否由终态文件判断阶段漂移 |
|---|---|---|
| 基础实体 | `RecurrentRGCN.dynamic_emb`，rrgcn.py:103 | 不能，缺少对应旧阶段状态 |
| 基础关系 | `emb_rel`，rrgcn.py:99 | 不能 |
| 演化关系 | `relation_cell_1` 输出 `h_0`，rrgcn.py:245–256 | 需固定历史图后重提取 |
| 历史实体表示 | RGCN与time gate的 `self.h`，rrgcn.py:257–272 | 同上 |
| HCP生成表示 | diffusion/model_21.py:205；冻结上一阶段特征教师生成 | 需固定HCP与扩散随机性 |
| query与candidate | ConvTransE.forward，decoder.py:78–100；两者共用演化实体表 | 并非独立候选参数，需要功能评分对照 |
| decoder状态 | 卷积、fc及BatchNorm参数/缓冲区，decoder.py:67–74 | 需旧状态及同一query输入 |

训练按日快照更新；`EarlyStopping.save_checkpoint`（utils.py:689）反复写入同一 state_dict 路径。当前本地没有对应 D1/D2/D3 的成对 checkpoints，不能说 entity、relation、decoder 中哪一项漂移最大。

基础 cosine/L2 变化可能包含坐标系变化。要定位性能损失，仍需固定query、固定历史图和HCP的功能评估；优先执行本轮主报告提出的融合反事实诊断。本目录数据工具只是已完成的低成本补充。
