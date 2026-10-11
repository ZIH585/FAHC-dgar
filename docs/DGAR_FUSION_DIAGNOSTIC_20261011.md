# DGAR 固定特征融合诊断

脚本：`scripts/diagnose_dgar_fusion.py`。只读取指定已有检查点，评价 VALIDATION，
不创建优化器，不训练，不生成或重建 `history_snap_v2`，不修改 main/rrgcn。

## 命令

在已有模型的服务器激活 dgar_env，并指定**同一次 original 实验**的两个检查点。
从隔离实验目录读取模型时也应指定该目录的 `config.json`；它提供训练时架构，
如果配置包含数据/缓存 SHA，诊断要求与本地文件一致。

```bash
cd /data1/zihao/DGAR-code
DGAR_MODEL='/绝对路径/原始DGAR实验/models/模型文件'
DGAR_DIFFUSION='/绝对路径/同一次实验/diffu_models/模型文件'
DGAR_CONFIG='/绝对路径/同一次实验/config.json'

python scripts/diagnose_dgar_fusion.py \
  --model-checkpoint "$DGAR_MODEL" \
  --diffusion-checkpoint "$DGAR_DIFFUSION" \
  --run-config "$DGAR_CONFIG" \
  --gpu 0 --dataset ICEWS14s --seed 1 \
  --snapshots 60,90,120,180,240,300,360 --snapshot-radius 4 \
  --alphas checkpoint,0.25,0.5,0.75,1.0 \
  --output-dir /data1/zihao/DGAR-code/logs/dgar-fusion-diagnostic-seed1 \
  --dry-run
```

检查通过后去掉 `--dry-run`，其余不变，即执行有限的模型评估。没有 `config.json`
的旧模型可以不传 `--run-config`，使用当前 main.py 默认架构；strict state_dict
加载必须匹配。此时不能独立确认旧模型的全部训练设置。不要自动挑选“最新模型”。
输出目录必须为空；dry-run 不导入 torch/DGL、不建输出目录，不申请 GPU。

正式设计预先固定七个中心的相邻九日窗口，合计63个不同时间片；这是为了增加
验证事实支持数，尚未执行GPU诊断，不是根据结果反复扩大窗口。
`--snapshot-radius`默认0保留原单日行为；展开时裁剪到合法索引2..T−1，并去重。
metadata记录中心、半径、每个中心对应的实际时间片以及各窗口验证事实数量。

| 中心 | 时间片范围 | 验证事实数量（扩逆前） |
| --- | --- | ---: |
| 60 | 56..64 | 246 |
| 90 | 86..94 | 216 |
| 120 | 116..124 | 204 |
| 180 | 176..184 | 230 |
| 240 | 236..244 | 230 |
| 300 | 296..304 | 224 |
| 360 | 356..364 | 169 |

合计1519个原始验证事实，两个预测方向合计3038个查询。前三个中心用于预选
全局alpha和简单效用路由阈值，后四个中心冻结评估；分析仍须按事实cluster和
时间block报告，不因两个方向扩展而把支持数翻倍。这个设计增加筛查功效，
不能保证每个演化分组都有足够事实；不足时如实标记不支持判断。

## 对照与输出

- 重建原 main.py 的时间片80/10/10切分，仅用其中验证后缀评价与过滤。
  读取 raw test.txt 只是重建原程序后段时间片；不会评价 TEST 后缀。
- 按最终 VALIDATION 的随机时间片 HCP 协议，保留验证主体关联的缓存事实。
  对每个指定时间片只生成一次 HCP 特征；后续 alpha 对照共享查询、历史图、
  特征和噪声。随机种子固定，但没有逐位复制原完整评估的 RNG 消费序列。
- `checkpoint` 使用检查点学得的 alpha；alpha=1关闭最终 RE-GCN 特征混合，
  但 HCP 特征仍生成，并固定不变。这里测的是融合环节的局部效应，不能等同
  于重新训练一个“无回放模型”。alpha 在 finally 中恢复。
- `metadata.json`：显式模型来源、模型/数据/缓存/运行源码 SHA 与协议限制。
- `snapshot_metrics.csv`：filtered MRR/Hits@1/3/10、raw top10重合率、
  HCP实体均值特征/最新RGCN聚合表征的范数比中位数，以及最终实体表征的
  fusion perturbation。layer_norm开启时范数比分母也使用归一化RGCN输出。
- `query_metrics.csv`：每个正向/逆向查询的 rank变化、RR变化、raw top10、
  raw rank、filtered RR/Hits@1/3/10、主体/真实候选表征扰动、
  `hcp_subject_touched`、主体 HCP/RGCN 范数比与余弦，以及仅来自 t 以前训练
  事实的频次、recency、相邻过去窗口tail分布JS和支持数。
  未被HCP触及的主体，其范数比/余弦留空，不与实际零值混在一起。
- 核心演化代理只针对已知查询主体：入/出relation角色JS、
  `(方向, relation, 邻居)`集合churn、两个窗口支持数。默认最多128事件的
  等样本比较与20次实体内置换提供null mean、excess和p。
  支持不足10的窗口标为不可靠，平衡代理与null字段留空；full churn只是描述。
  20次置换的p较粗，不能作为正式显著性证明。relation tailJS仅为辅助组成分析，
  不作为实体演化的替代证据。真实目标实体的频次/扰动仅用于结果描述，不能
  用作可部署门控特征。逆向查询的主体是该查询已知条件，按相同规则统计。
- `fact_cluster_id`为`时间片:原始验证事实索引`，两个预测方向共享该ID；
  bootstrap必须按事实cluster重采样，不能把两个方向当独立事实。
- `relation_metrics.csv`：按方向关系分别汇总的查询加权指标；不是时间宏平均。
- `status.json`：只有全部指定时间片完成后才写入 complete=true。

## 解释边界

用最终检查点查看早期时间片是 retrospective mechanism analysis：模型已经学过
后来训练数据。这不是在线预测复现，不是 D1→D2 的时间遗忘/漂移实验。
上述表征变化来自同一检查点更改融合系数，不能叫 temporal representation drift。
按每个query选择某个全局alpha分支的结果只能检查效用是否可预测；实体共享
候选矩阵和多步图更新，真正的实体门控模型不会等价于这种输出路由。

若多阶段、足够样本的关系在弱化融合时验证RR改善，而另一些关系持续受益于
原融合，说明统一alpha可能掩盖异质性，值得进一步研究。若只在18事实的
最后片有差异，或收益不重复、仅与稀疏JS相关，不足以推进自适应演化方案。
应先按原设定看 checkpoint 对照，不能使用 TEST 选择alpha。

本次只执行标准库 dry-run 和 py_compile；实际模型评估留给空闲服务器。

CPU后处理脚本 `scripts/summarize_dgar_fusion.py` 固定前三窗口选择阈值/方向和共享
alpha，在后四窗口冻结评价；运行接口见主复盘报告第12节。输出主指标为等日macro，
控制分层的current-preference为query-pooled描述均值。按时间窗口block及每日原始
事实cluster bootstrap；不会把正逆方向或多个alpha当成独立事实，不自动批准路线。
