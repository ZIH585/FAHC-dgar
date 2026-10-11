# DGAR / FAHC 复盘、诊断与下一路线

日期：2026-10-11；分析起点：`0d51667`。本阶段没有训练、没有 GPU 模型评估，
没有实现最终新方法。新增内容是离线统计、独立诊断工具及研究方案。

## 0. 决策

1. **结束当前 FAHC 风险选择器主线。** 它未显示超越同版本 random/original 的收益；
   继续扫 forgetting score / eta / lambda 不值得优先投入。
2. **不把 KD 写成完全无效。** random+KD 的完整历史测试 MRR +0.4406 个百分点，
   Current -2.5380；验证对应 +0.4285 / -8.8457。本配置不满足历史/当前折中标准。
3. 目前优先研究 **结构演化条件下，历史生成表示的边际预测效用及融合**；
   次优问题是按关系/实体角色区分有益后向迁移、无益漂移与适应。
4. 最推荐候选：**Evolution-Calibrated Replay Integration DGAR，ECRI-DGAR**。
   先运行同 checkpoint、同 HCP/噪声的融合反事实诊断；诊断不通过就放弃 ECRI，
   不把“高演化应少回放”预先当成结论。

## 1. 检查范围、Git 与可用证据

已读初始提交 `a454564`、当前 `src/main.py` / `rrgcn.py` / `decoder.py`、
`diffusion/model_21.py` / `difffu_21.py` / `fahc_selector.py`、`src/fahc*`、
`rgcn/utils.py`、实验脚本和配置验证、现有日志/汇总CSV，并检查 Git diff/log。

关键提交：

| 提交 | 作用 |
| --- | --- |
| a454564 | 本仓库 original 初始版本 |
| 76c2c35 | 初版 FAHC 选择与 KD |
| e6bf8ce | 完整候选 HCP、探测恢复、评估修复 |
| a6e905b | 回放预算回归诊断与保留率 |
| 13f47f9 | 隔离运行、配置、验证汇总 |
| 58ad709 | 有限机制诊断 |
| 6d82d5e | 完整机制 CSV 结论 |
| 0d51667 | 六组结果及 KD 收益/代价存档 |

工作区未提交的差异主要是用户权重、pycache、旧日志；未改动/归档这些权重。
源码相对初始版有共同修复：种子、确定性验证、最佳 Adam 状态恢复、
阶段边界模型同步、设备处理、单例 BN 等。因此同批次 v3 original 才是模块增量
比较基线，旧 modify 日志不能与 v3 分数直接作单因素比较。

所有明确记录 seed 的 ICEWS14s 日志均为 seed1；未发现匹配条件的 seed2+。
旧文件名中的 run1/run2 或日期不能替代明确的 seed 配置。
本地未收到 v3 `experiment_runs/.../config.json/status.json/stages.jsonl`；其配置
主要由 stdout Namespace 验证，诊断聚合由用户CSV提供，不能独立重算全部阶段。
存在一份旧 B128 JSONL，可用于旧运行离线重建。

## 2. 性能：两个口径分别列出

### 2.1 最终 TEST 历史宏平均（%）

| 模式 | Current MRR | 历史 MRR | 历史 H1 | 历史 H10 |
| --- | ---: | ---: | ---: | ---: |
| original v3 | 53.9699 | 50.2162 | 39.7307 | 70.2630 |
| random B4096 | 57.1572 | 50.1166 | 39.2073 | 70.7197 |
| random + KD0.03 | 54.6192 | 50.5572 | 39.6891 | 70.9372 |
| coverage | 53.8240 | 49.6831 | 38.7175 | 70.0360 |
| risk+coverage | 54.6475 | 49.6452 | 38.8021 | 70.1890 |
| risk | 50.6887 | 47.8290 | 36.8676 | 68.6867 |

原代码虽然计算 H3，但 supplied stdout 和 final JSONL 没有保存历史宏平均H3；
**不能补造这个指标**。下面只使用日志可恢复的全部查询混合指标。

### 2.2 最终 TEST query-pooled（%）

| 模式 | MRR | H1 | H3 | H10 |
| --- | ---: | ---: | ---: | ---: |
| original | 49.7292 | 39.1580 | 55.6567 | 69.8960 |
| random | 49.7434 | 38.8654 | 55.7380 | 70.3023 |
| random+KD | 50.0829 | 39.1905 | 56.2365 | 70.5462 |
| coverage | 49.2754 | 38.2477 | 55.4454 | 69.7443 |
| risk+coverage | 49.2632 | 38.4211 | 55.0282 | 69.7551 |
| risk | 47.4124 | 36.4868 | 53.4189 | 68.1729 |

完整 VALIDATION 历史/Current：original 50.2472/66.1097；random
50.1924/71.3330；KD 50.6209/62.4873；coverage 49.7801/66.2971；
combo 49.5972/68.5610；risk 48.1305/67.2582。

最早 B128 combo 历史测试 MRR 38.7304%，不能与 v3 表当作严格单因素预算实验。
那一轮 HCP 去重事实中位数1895、B128保留率中位数6.75%、360/363阶段裁剪。
v3 各受控组 B4096达到上限比例0，回放中位数1604–1920，裁剪问题已经解除。
这意味着B是相同上限，**不是每组实际回放事实数严格相同**；coverage中位数1920、
random/risk/KD约1604，仍有剂量差异。现有结果足以决定当前主线不值得继续投入，
不足以单独估计选择器每条事实的因果收益。

六组 history-only BWT 都为正（2.8655–5.2017个百分点），同时有部分快照正遗忘。
**有后向迁移和局部退化并存，尚不能称全模型严重灾难性遗忘。** KD 的 history-only
旧MRR略降0.1115pp、BWT略降0.3512pp，正遗忘量略减0.1451pp，诊断改善不一致。
Current TEST仅18条事实/36个正逆query，需避免把其单次变化解释为普遍因果。

## 3. HCP：原始行为与 FAHC 的真实差别

### 3.1 原始选择

`diffusion/model_21.py:177–191`：训练时从历史 `0..t-2` 随机抽最多k=35个时间片；
在各时间片字典中取当前训练批主体实体对应的历史事实，合并去重。
最终评估 `src/main.py:125–147` 也是随机时间片+测试query主体筛选，历史范围略有差别。

原始方法有**主体实体相关性筛选**，没有 relation relevance、语义相似度、recency
的事件top-k排序。故不能说 FAHC “破坏了原来的相关性排序”。其筛选是否已有效，
需要独立同口径消融，现有数据不能证明“原筛选已经最优”。

### 3.2 FAHC

`diffusion/fahc_selector.py:141–172` 继续按相同当前主体过滤。
`src/main.py:674–732` 在候选时间片上计算风险/覆盖，先选k片，再按B裁剪去重联集。
候选仍来自原历史索引，未绕过主体过滤重新加入被该过滤排除的实体。
风险/覆盖可能重新分配哪些时间片进入HCP；“有用事实被挤出”“噪声增多”未做因果测量。
过去发生的事件在旧时间仍是真实标签，不能因近期不复现就宣布它应被淘汰或为噪声。

### 3.3 离线结构相关性审计（本轮新增，CPU）

来源：旧 B128/eta1/KD0 JSONL，328个有选择空间阶段、22365个候选，
候选事实数与缓存重建逐一一致，0不匹配。输出见
`FAHC_HCP_RELEVANCE_AUDIT_20261011.json` 和逐阶段CSV。

| risk 与什么比较 | 逐阶段 Spearman 中位数 |
| --- | ---: |
| 历史事实(s,r)在当前训练片出现的比例 | -0.015628 |
| 历史relation在当前训练片出现的比例 | +0.000744 |
| 时间片年龄 | +0.093290 |
| HCP事实数量 | +0.004475 |

候选池/(离线纯风险top35)/旧组合实际选择的(s,r)匹配率为
24.7513% / 24.7106% / 24.2690%。平均年龄100.55 / 105.85 / 95.21。
**旧风险分数没有显示提升这些结构相关性代理的能力；组合并非总体选择更旧事件。**
这些统计不等价于真实预测效用/importance，不是v3重测，也没有训练离线top35。

importance应通过功能效用定义，例如固定其余条件：

\[
 U_t(P)=L_{current}(M\setminus P)-L_{current}(M)
\]

或者下一阶段可预测的回放边际收益。高风险并不按定义保证这个量为正。
importance、当前relevance、旧任务保留价值是不同目标，不能由同一个CE增量替代。

### 3.4 风险机制结果

风险在当前history图、无diffusion特征的 entity decoder 上评分（`src/fahc.py:98–174`）；
probe优化current-only实体/关系/静态损失，而正式更新包含diffusion融合与replay。
FAHC只改训练选片；阶段验证与最终预测仍随机HCP。

| 阶段 | A/B风险排序rho | 预测current-only退化rho | 预测正式DGAR历史模式退化rho |
| --- | ---: | ---: | ---: |
| 60 | .240795 | .432437 | .185622 |
| 90 | .724434 | .673939 | .219071 |
| 120 | .945202 | .759076 | .570116 |

风险不是纯噪声，但只能在1/3阶段满足先前综合门槛。正式DGAR退化列仍是
history-only CE，不是完整pipeline MRR；不能据此认为主任务收益已经验证。

## 4. KD：保护对象、力度与可塑性

- `model_repre`（main:527）冻结/eval；阶段开始加载前一最佳模型，阶段内不更新。
- 仅所选且预算保留的回放query，不覆盖未选HCP或全部新query；每个query的类别
  分布覆盖全7128实体。没有relation-logit KD（rrgcn:346–399）。
- 旧/新logits用同一当前history图、同一类别排列、无diffusion路径，因此KL合法；
  但它不是旧时间完整DGAR预测分布，存在保护代理与主任务错配。
- (T=2,T^2)；教师在T=1 top1正确且概率>=.5才蒸馏。
- 除以所有回放query数量，而不是门控通过数量；通过比例低会降低有效总贡献。
- 本次lambda=.03，适用阶段门控比例中位数7.158%。门控不是全零，也不是所有
  candidate都被蒸馏。正确top1仍不保证尾部分布正确或新阶段继续遵循相同模式。
- supplied stdout/CSV没有KD/task loss比例或梯度比例；原JSONL代码仅记录
  `kd_mean_loss`，没有base task loss，且v3原JSONL未传入。**无法断言KD主导优化、
  lambda过大或T不合理。** 损失尺度也不能替代梯度干预强度。

设计上旧事件仍需正确回答，但未来预测可能需要新邻域/新角色。旧分布保护与
适应之间存在潜在权衡。random+KD观测支持“本配置历史收益+Current代价”，
未确定唯一原因。五组选择器均KD=0，KD不能解释它们的失败。

## 5. idea / implementation / hyperparameter：排序

### v3 选择器主线最核心三个原因候选

1. **代理目标错位（证据最强）**：局部current-only CE增加不能代表完整DGAR
   回放的预测效用，也不代表当前相关性。机制相关性与结构代理审计均不支持
   从风险直接推导收益。
2. **没有区分该保留的旧角色与需要适应的模式变化（机制假设，尚非因果结论）**：
   在当前图上预测旧事实的损失增大可来自遗忘、邻域组成变化、BN和局部噪声。
   正向截断平均和覆盖奖励没有辨别这些原因；risk/coverage单独都未优于random。
3. **选择偏差通过实体聚合和共享融合传播，且训练/评估prompt分布不同（代码事实+
   待测因果）**：不同relation生成特征按实体均值合并，所有实体共享alpha；
   选片变化不保证在最终随机prompt推理中有益。

初版巨大下降的首因另列：B128强裁剪。KD最新任务代价另列：它是独立辅助机制
结果，不能替代上面三个选择器原因。

当前probe恢复已包括参数/BN、RNG、混合train/eval、缓存、grad，probe Adam复制
主optimizer状态；最佳checkpoint还原相配Adam。未找到足以解释v3相对下降的
残留FAHC独有运行bug。不能以没有发现bug证明任意GPU运行无bug。
仍有配置和预算影响，但不是继续扫参数的首选依据。

### 共享的 original 实现/协议疑点

1. 本地365缓存全为正向，总72437条，与重分割训练事实完全一致；0逆事实、0主体键错配、
   未见本地val/test泄漏。但 `get_history`（utils:41–42）会生成正向+逆向；已有目录
   不升级（main:354）。换机器重建会改变HCP协议，必须固定缓存指纹。
2. diffusion声明 `emb_rel`，训练/生成relation条件却取实体表
   `initial_h[tag[:,1]]`（model_21:277/301）。这是初始版继承疑点。
3. `time_embeddings`定义后未使用；`ct`未进入实际计算；另需核验denoiser timestep
   `emb_t`计算后丢弃及lossaware sampler未更新。时间作用主要来自按序图/GRU和当前
   历史窗口，不应把未使用time参数当成已测的temporal representation。

这些可影响复现与后续机制设计，但同版本共享，不能归因FAHC独有失败。
若修正它们，应作为独立correctness baseline重新比较，不能包装为创新或混进新模块。

## 6. Original DGAR：变化究竟在哪里

### 6.1 真正的时间任务

main:342–349先合并原train/valid/test的逐日片，再每片按文件顺序80/10/10分割。
当前ICEWS14s是365个日任务；D1..D5可作5×73天分析窗口，不是已有5个task checkpoint。
静态图、实体/关系ID空间固定；日图、邻域、频次、history窗口和HCP集合不断变化。

### 6.2 架构与可观测量

| 对象 | 代码位置 | 需要区分的变化 |
| --- | --- | --- |
| 原始entity参数 | rrgcn:103 dynamic_emb；diffusion/model_21 emb_ent | 参数更新 vs 当前图传播 |
| 原始relation参数 | rrgcn:99 emb_rel | 参数更新 vs GRU后的relation h_0 |
| relation历史表示 | rrgcn:245–256 relation_cell_1 | 邻域实体均值和序列共同驱动 |
| history entity表示 | rrgcn:210–273 forward | static/RGCN/GRU式time gate+DAR融合 |
| HCP生成表示 | model_21 forward/generation；rrgcn:190 select_entity | checkpoint、所选prompt、噪声、旧guider均会改变 |
| temporal作用 | rrgcn:268–271 time gate；图序列 | 时间窗口变化不等于time embedding参数漂移 |
| candidate表示 | decoder:78–100 | query与所有candidate共用同一evolved entity表；没有独立candidate参数模块 |
| decoder/score | ConvTransE/ConvTransR；decoder:78–100 | conv/fc/BN状态 + query侧 + candidate表一起决定排名 |
| 融合alpha | rrgcn:176、263–264 | 可学习、样本无关、未sigmoid限界的全局标量 |

current_h在融合前归一化（257–259），generated均值表示在融合处没有对称归一化。
后面的time gate还会混合，故alpha=0/1不是完整模型切成两个独立专家。

现有模型文件每阶段覆盖同一路径，只保存state_dict，未保留D1/D2权重。
本地无逐阶段表示/logits/梯度。因此**不能给entity drift大、decoder drift小这样的
虚构排序**；也不能从聚合MRR推断哪个参数块遗忘最严重。

### 6.3 本轮CPU数据统计

重建365片真实训练流，按5×73天窗口；不使用重新分割的val/test标签。
相邻窗口的关系频次JSD(bits)：.011890/.010284/.010804/.007876；
实体频次JSD：.157238/.155152/.145873/.139275。

200次等样本置换的补充统计已归档。控制同主体query组成后，充分支持的正向关系
仅13/14/15/16种；条件分布JSD超null均值中位数为
.065216/.068216/.074607/.061537。Make_a_visit的变化信号较强，部分关系接近
置换噪声。所有p值是探索筛查，尚未做多重比较校正；未检测到变化不等于已证明稳定。
这支持“实体参与组成在变、部分同主体关系条件分布也有变化”。
**不证明relation语义发生翻转，也不证明embedding漂移导致遗忘。** 两类频次JSD的
支持空间分别230/7128，不能按其比值判定哪类神经表示漂移更大。主体组成变化可解释
p(o|r)的变化，需另看有重复支持的p(o|s,r)；稀有事件采用采样null与支持标记。
原始统计和输入SHA见DGAR_EVOLUTION_DIAGNOSTIC_20261011/summary.json。

### 6.4 低成本诊断菜单（执行优先级只选第1项）

| 诊断 | 控制与统计 | 支持什么后续方向 |
| --- | --- | --- |
| 固定HCP的融合反事实×前缀演化 | 同checkpoint/query/history/生成特征，改变alpha；RR/rank/top10、norm比、前缀角色JSD/freq/recency | ECRI主路线是否值得做 |
| Representation Drift | 在阶段边界采样同实体/关系；固定历史图，原始与contextual表示cos/L2/norm，按频次关系分组 | 漂移是否与功能退化相关；需保存旧checkpoint |
| Prediction Drift | 固定旧验证query及同一原属历史图/HCP/noise、所有candidate/过滤；阶段前后rank/MRR/top10 | 真遗忘 vs history变化 |
| Encoder/Decoder功能交换 | old/new encoder×old/new decoder，BN随decoder；candidate表单独固定为old | 定位score输入/候选排序/decoder哪侧损害 |
| Gradient Conflict | 同一参数状态下32对旧/新训练batch；旧/新各用固定所属历史，dynamic_emb/emb_rel/GRU/RGCN/decoder逐块cos；零范数排除，共同实体支持行另算 | 仅局部稳健负cos才考虑投影；原始全局cos可能被不相交实体稀释 |
| Relation-wise Forgetting | 关系方向分开，按原标签关系分组RR变化/正遗忘/BWT；支持数及bootstrap | relation异质性保护 |
| Frequency/Recency | 训练前缀计数/最后出现时间划分，匹配relation、频次、年龄，配对预测变化 | 区分稀有采样噪声/真正过时历史 |

上述漂移/梯度实验是设计，不是已经跑出的结果。Raw cosine/L2受坐标重参数化影响，
排名稳定的大位移不能直接作为anchor理由。只在固定上下文下的功能变化才支持保护。

## 7. 文献机制匹配：原结论与迁移分开

下列迁移均是本项目建议；没有论文保证在DGAR上有效。检索日期2026-10-11。

| 论文（primary） | 原论文机制 | DGAR迁移建议与边界 |
| --- | --- | --- |
| [DGAR ACL2025](https://aclanthology.org/2025.acl-long.537/) | 生成历史实体表示并融合当前表示 | 主体HCP保持完整；应检查how-to-integrate的效用，不能把原方法描述为普通ER |
| [History Repeats ACL2023](https://aclanthology.org/2023.findings-acl.490/) | 参数重要性时间正则+聚类回放 | 作为代表性memory对照；迁移需保留HCP上下文，非风险topk |
| [TiRGN IJCAI2022](https://www.ijcai.org/proceedings/2022/299) | local recurrent/global重复历史与周期decoder | 借双时间尺度前缀统计，不能把重复率视为稳定性的充分条件 |
| [CENET AAAI2023](https://ojs.aaai.org/index.php/AAAI/article/view/25601) | 学历史/非历史依赖及query分类 | 检查历史replay贡献是否按query有正负；不直接照搬candidate硬mask |
| [IncDE AAAI2024](https://arxiv.org/abs/2405.04453) | 结构层级学习、增量实体表示蒸馏、两阶段训练 | 可迁移角色原型/分阶段保护；静态KG新知识不等于TKG事件演化 |
| [CCD KDD2024](https://arxiv.org/abs/2405.19046) | 推荐模型协作蒸馏，慢/快稳定与可塑代理 | 借长短期双原型思想；不默认给DGAR再加两整模型 |
| [CaT ICDM2023](https://arxiv.org/abs/2309.09455) | 图凝缩memory及memory中训练处理不平衡 | 借受预算分布保真原则；生成图凝缩对TKG关系/时间一致性成本高，暂不推荐 |
| [TRGP ICLR2022](https://arxiv.org/abs/2202.02931) | 层级旧子空间相关性选择和权重缩放复用 | 先测模块冲突/迁移，不能称cos<0简单投影就是TRGP |
| [SGP AAAI2023](https://arxiv.org/abs/2302.01386) | 历史重要子空间按重要程度软缩放更新 | 仅在局部冲突存在时借软保护，避免锁死演化知识 |
| [Loss of plasticity Nature2024](https://www.nature.com/articles/s41586-024-07711-7) | 实证可塑性丧失并提出continual backprop | 提醒检查新任务学习能力；不能据其结论断言DGAR已有同样神经元失活 |
| [DiMNet Findings2025](https://aclanthology.org/2025.findings-acl.658/) | 多跨度邻域演化、active/stable分解 | stable/evolving拆分已有近邻；本项目需独立演化证据与效用校准 |
| [DECRL NeurIPS2024](https://proceedings.neurips.cc/paper_files/paper/2024/file/63460cef7f8e76fb2479c2f1500ecbfb-Paper-Conference.pdf) | 动态聚类、跨时间簇匹配与表示平滑 | 若群体迁移明确可做temporal prototypes；先验证簇稳定性 |
| [SEEKR EMNLP2024](https://aclanthology.org/2024.emnlp-main.190/) | 选择重要attention heads作蒸馏 | 借选择保护粒度，不能把LLM头重要性原结论转为relation保护结论 |
| [MA-DAR 2026预印本](https://arxiv.org/abs/2607.21949) | 范数/流形对齐、逐维动态gate、polarization | 最直接融合近邻；归一化+MLP gate本身创新性不足，必须作方法对照 |

候选方案的数学表达是本项目设计，不是上述论文原公式。没有经过完整新颖性审查，
评分表示初步判断，不能据此声称方法原创或给出数值成功概率。

## 8. 八个候选：问题、假设、公式、位置

约定：C=当前RGCN表示，H=生成回放表示，d=训练前缀演化信号，c=支持可信度。
所有默认保留原随机HCP、k及事实集合；只有候选7研究memory采样。最终方法尚未实现。

### 1. Norm-Balanced DAR（NBD，Route A）

- 问题：融合前C已归一化、H未对称归一化，norm domination可能影响融合。
  FAHC只挑选时间片，未控制融合输入尺度。
- 假设：归一化H可改善排名，且无需演化模型。诊断若不支持norm影响则停止。
- 机制/公式：\(h=\alpha C+(1-\alpha)H/(\|H\|+\epsilon)\)，保留原alpha。
- 位置：`src/rrgcn.py:261–264`, `RecurrentRGCN.forward`。
- 新参数：`--dar-normalize-replay`；必要时epsilon固定。无新增memory/old model。
- 风险：H范数可能编码可信度；删掉它会丢信号。它是直接近邻控制/工程改进，
  不能单独作为新论文主创新。

### 2. Evolution-Calibrated Replay Integration（ECRI，Route B）

- 问题：所有实体共享alpha，缺少历史表示对当前预测是否有益的演化条件。
  FAHC按风险保护旧事实，未优化融合效用。
- 假设：在匹配freq/recency/norm后，前缀角色/邻域演化仍预测H的正负效用。
- 机制：等长训练前缀窗口统计入/出关系角色分布、邻域churn；扣除采样null，
  低支持回退原alpha；把统计映射到实体融合强度。
- 公式：\(d_e=[JSD(p_e^{recent},p_e^{ref})-JSD_{null,e}]_+\),
  \(c_e=\mathbf1[\min(n_{old},n_{recent})\ge m]n_{eff}/(n_{eff}+\kappa)\),
  \(\alpha_e=(1-c_e)\alpha_0+c_e\sigma(b+\beta d_e+\gamma\,churn_e)\),
  \(h_e=\alpha_e C_e+(1-\alpha_e)H_e\)。主假设是高演化更偏向当前表示；
  诊断若显示反向，应判本方向未通过，不能事后更改假设。这里
  \(n_{eff}=\min(n_{old},n_{recent},128)\)。
- 位置：`rrgcn.forward:261–264`；新增拟议 `src/temporal_evolution.py`；
  `main:625–632` 更新只来自已到达训练事实的统计。
- 新参数：window=30、min_support=10、kappa、gate_mode；小规模只验证一种window，
  不同时加KD/anchor/新selector。
- memory：原HCP外少量稀疏计数；无新增old model。
- 回退：低支持令c=0，关闭时直接原代码，均不对原alpha作sigmoid/clamp。开启时
  并不保证所有实体恒等初始化，尤其原alpha在(0,1)外时不能用有限sigmoid bias复现。
- 风险：频次/采样噪声解释一切；统计演化不预测效用；未来query信息被误用；
  与MA-DAR/DiMNet差异不足。需以采样校准、时间外推和功能效用证据形成研究故事。

### 3. Regime-Conditioned Temporal Prototypes（RCTP，Route C）

- 问题：实体HCP均值混合长期角色与近期角色转换；FAHC只换被平均的样本。
- 假设：有可复现的双时间尺度角色/邻域变化，单一历史均值抹掉转变。
- 机制：维护(s,r,direction)或relation簇的慢/快原型，用演化证据选择当前角色载体。
- 公式：\(p^S\leftarrow(1-\rho_S)p^S+\rho_S z\),
  \(p^F\leftarrow(1-\rho_F)p^F+\rho_F z\), \(\rho_S<\rho_F\),
  \(p=(1-g(d))p^S+g(d)p^F\)。只对稳定且高支持组加anchor。
- 位置：`rrgcn.select_entity:190–206`、`forward:261`；新增拟议prototype模块。
- 参数：slow/fast rate、支持数、prototype容量、稳定anchor权重。
- memory：两组prototype和元数据；无新增old model。
- 风险：组身份不稳/对齐混乱、稀疏角色、增加容量引起不公平比较、双代理近邻多。

### 4. Relation-Conditioned Functional Anchors（RCFA）

- 问题：可能存在与旧query排名下降关联的relation-context漂移；目前未证实。
  FAHC保留损失高的事实而未确定发生功能退化的表示块。
- 假设：稳定关系的功能退化与固定上下文表示变化相关，evolving关系不应被同等约束。
- 公式：\(L=L_{DGAR}+\lambda\sum_r c_r(1-d_r)\|\bar z_{r,t}-p_{r,t-1}\|^2\)。
  使用归一化或对齐表示，避免处罚坐标等价变换。
- 位置：`rrgcn.get_loss:400`、`forward:245–256`。
- 参数：anchor_weight、prototype EMA、min_support。需relation prototype memory；
  无需全old model。风险：平均prototype掩盖多模态，表示位移并非遗忘。

### 5. Selective Functional Distillation（SFD）

- 问题：旧KD代理没有diffusion且没有演化条件；FAHC的confidence gate仅查旧标签。
- 假设：完整预测中旧分布对稳定角色仍有功能价值；只保护该部分可减少Current代价。
- 公式：\(L_{KD}=\frac{T^2}{N}\sum_q I_{old\ correct}c_q(1-d_q)
  KL(p_{old}^{full,T}\|p_{new}^{full,T})\)。固定原属history/HCP/noise，不用当前图
  偷换过去上下文。low-support回退不作强保护。
- 位置：`src/fahc.py:210–257`、`rrgcn.predict:275`、`main:796–841`。
- 参数：stable_min_support、KD weight、temperature固定初筛。需replay query/context
  memory及old predictor；原teacher可复用，可能需old diffusion版本。
- 风险：历史正确不等未来稳定；完整路径成本高；不值得先于融合诊断。

### 6. Conflict-Localized Updates（CLU）

- 问题：可能只有部分参数块发生旧/新目标冲突，FAHC全局代理未定位。
- 假设：共同实体支持行和特定relation/GRU/decoder有可复现负gradient cosine。
- 公式：当\(g_n^\top g_o<0\)，
  \(g'_n=g_n-\frac{g_n^\top g_o}{\|g_o\|^2+\epsilon}g_o\)，仅作用于已验证块；
  无冲突保留更新。这是候选投影式，不是TRGP原算法。
- 位置：`main:843–850` backward/clip/step接口；新增拟议gradient工具。
- 参数：block名单、检测间隔、old batch数。需少量old训练batch；无需old model。
- 风险：梯度噪声/缺少共同支持、分块投影仍损害有益演化、额外backward显著。

### 7. Transition-Balanced HCP Replay（TBHR）

- 问题：随机时间片可能欠覆盖transition边界和稀有角色；FAHC没有训练分布转换证据。
- 假设：历史转变附近的代表HCP比高风险事实更能保留可迁移变化模式。
- 公式：过去时刻u的\(b_u=[JSD(p_u,p_{u-1})-null_u]_+\)，
  memory采样\(p(P_u)\propto\epsilon+\xi b_u+(1-\xi)diversity(P_u)\)，同时保留
  稳定窗口配额，固定逐阶段事实预算。
- 位置：`diffusion/fahc_selector.py:245`或独立selector、`main:674`。
- 参数：transition_fraction、rare_relation quota、固定字节容量。
- memory：新增有限HCP索引/缓存；无需old model。
- 风险：最接近旧选片主线，边界是数据采样异常、事件不再相关、存储定义失真；
  不优先实施，必须先有transition功能证据。

### 8. Role-Conditioned Historical Pooling（RCHP）

- 问题：同实体不同relation的生成条件表示最后被均值混合；FAHC仅改变均值输入。
- 假设：存在互相干扰的entity-role历史模式，分角色聚合能改善query排名。
- 公式：\(H_{e,r}=\sum_{q:\,role(q)=(e,r)}w_qz_q/\sum_qw_q\)，
  query表示\(h_{e,r}=\alpha C_e+(1-\alpha)H_{e,r}\)。candidate共享表先不改变，
  防止每query重建全实体表。
- 位置：`rrgcn.select_entity:190`、`decoder.ConvTransE.forward:78–100`。
- 参数：role_min_support、fallback、pooling type。需稀疏role聚合缓存；无old model。
- 风险：稀疏角色、扩散relation token继承疑点、query侧与candidate侧不对齐，
  显存随batch active roles增长；不是直接在现有E表加r索引就完成。

### 成本和评分（全部是工程估算/主观评级，不是测得的运行收益）

FP32下实体E=7128,d=200的一个表约5.44MiB；relation双向460×200约0.351MiB。
GPU激活与Adam可能另增加数倍；稀疏角色数量远大于relation数时按实际容量核算。
时间列以F=一次额外RGCN forward、D=decoder pass、B=backward表示，避免伪称百分比。

| 候选 | 新memory / 新old model | 额外显存（估计） | 额外训练计算 | 创新/难度/风险/预期收益，1–5 |
| --- | --- | --- | --- | --- |
| NBD | 无/无 | 通常0–5.44MiB临时norm表 | norm操作 | 1 / 1 / 2 / 2 |
| ECRI | 稀疏统计/无 | gate向量<0.1MiB，若中间表示复制则+5.44MiB | CPU计数+逐实体标量；若监督效用另+F/D | 3 / 2 / 3 / 3 |
| RCTP | 两prototype/无 | relation级约0.70MiB；entity级约10.88MiB；role级按容量 | EMA+gate；可能+F | 4 / 4 / 4 / 3 |
| RCFA | prototype/无 | relation级<1MiB及聚合激活 | 聚合+anchor梯度 | 2 / 2 / 3 / 2 |
| SFD | query/context/复用旧model，可能旧diffusion | CPU teacher logits可缓存；GPU分块C×chunk及旧diffusion权重 | +F/D与可能完整diffusion采样 | 2 / 3 / 4 / 2 |
| CLU | old batch/无 | 选定块额外gradient buffer；最坏模型一份grad | 检测阶段+old F/B | 2 / 3 / 4 / 2 |
| TBHR | 固定字节HCP缓存/无 | CPU metadata；GPU事实预算同原 | CPU转变统计+选片 | 3 / 3 / 4 / 2 |
| RCHP | 活跃role聚合/无 | O(active_roles×d)，需设容量 | 分组聚合+query构造 | 3 / 3 / 3 / 3 |

“超过Original可能性”按预期收益列作条件式相对评级，不给未经实验的概率百分比。
没有对应诊断问题的方案，其当前实施优先级为0；如无局部冲突，不实施CLU。

## 9. 只保留三条路线

### Route A：NBD

小改融合输入范数。目标先排除尺度偏差、接近/超过original；一天内可用同checkpoint
反事实筛查及一组短prefix试验判定方向。一天不是完成全部365片和多seed的承诺。
归一化已有MA-DAR直接近邻，只作工程路线和方法对照。

### Route B（最推荐）：ECRI

故事：**回放效用随可观测的历史结构演化而变，融合应受证据与不确定性控制。**
重点不是忘了多少，而是历史信息对当前预测是否仍有用。先诊断前缀信号能否预测
固定HCP的功能收益，再让融合自适应。必须比较global alpha、普通gate、MA-DAR、
只freq/recency/norm gate、shuffle evolution及可信度校准消融。

### Route C：RCTP

如果诊断显示稳定/变化角色分成双时间尺度且单scalar无法解释，研究慢/快原型与
regime-conditioned表示。创新性潜力较高，但成本和歧义更多；此时才扩展表示载体。

## 10. 唯一优先实验：固定HCP的融合反事实×前缀演化

先用已有original训练checkpoint，无新训练。冻结权重，每个预选时间片的
验证query只生成一次HCP特征；相同history图、过滤答案、全部candidate、噪声，
仅改变临时alpha，结束恢复。比较checkpoint原alpha与.25/.5/.75/1.0；
alpha=1是该融合接口不注入生成表示，仍保留原RGCN/time gate。

这个实验定位DAR融合的功能效用，不测阶段间parameter drift。若使用最终checkpoint
去评价早时间片，这是retrospective诊断，该checkpoint已经学过后面的训练片；
不能称在线forecast或据此报告新方法泛化提升。演化特征必须只用query时间之前的
重分割训练事实，不能使用当片valid/test标签；验证标签仅计算功能指标。

### 输出与判断规则

- 每query：snapshot、fact ID/direction、relation、各alpha的rank/RR/H1/H3/H10、
  Top10 overlap、query实体频次/recency/支持数、角色JSD、邻域churn、回放是否触达。
- 表示：current/replay norm与cos（能记录时）；alpha造成的最终表示变化，明确
  是fusion perturbation，不是时间学习引起的representation drift。
- 汇总：按relation、频次、演化分位/支持分组，
  \(u_q=RR_q(.75)-RR_q(.25)\)；同一fact的正反query作为cluster，按时间片/
  fact bootstrap，而不是把每alpha或正反query当独立样本。
- oracle最佳alpha只衡量上界，不能当可实现成绩。中心60/90/120各自前后4日窗口
  拟合最多一个简单规则，中心180/240/300/360的对应窗口冻结评价；global alpha和
  freq/recency/norm-only规则也仅在前三窗口选择。共63天、1519独立验证事实，
  fit666/held-out853；用四个后期窗口block及窗口内事实cluster作bootstrap。
  从全局alpha的多个预测pass逐query挑选，是信号可预测性的诊断性路由，
  **不等价于拟议实体gate的单次forward，也不计为新方法性能**。
- 控制normratio/cos时固定使用checkpoint-alpha参考行；反事实alpha下最后历史步
  的current表示已受前面融合影响，不能把该条件下范数当成预处理控制。
- 稳定/演化不是永久relation标签。先以entity-role统计为主，relation作为分组解释，
  有支持的重复(s,r)条件分析与p(o|r)组成变化分开报告。

**继续条件（新工程screen，在看诊断结果前固定）**：至少2个不同预选后期窗口中，
高支持高演化组更偏向当前，低演化组更偏向回放，每组至少30个独立事实；
在汇总的频次/recency/norm分层中，有足够事实的分组仍保持方向；
冻结的前缀信号规则在后阶段验证macro MRR比最佳共享alpha至少+0.2pp，且任一
预选组下降不超过1pp。样本不足时判“证据不足”，不能缩窄分组直到通过。

**放弃条件**：各组最佳alpha几乎一样；演化与效用无关联或方向反转；收益仅由
norm解释；有oracle空间但过去可得信号预测不了；支持足够的后阶段规则不优于共享alpha。
诊断通过仍只授权候选开发，不等于已超过Original；最终需prefix online训练与多seed。
后处理的频次/recency/norm分别二分是探索分层，不等于联合消除混淆。
控制分层的current-preference均值是query-pooled描述统计；主规则成绩是等日macro，
二者权重不同。1519事实只解决总量问题，支持过滤后单组是否达到30仍需检查。
拟合允许反向规则作为诊断，但若高演化更依赖历史，则原预设方向未通过，
不能仅凭反向规则+0.2pp改写screen。若剩余混淆明确，再做一项预定联合分析，不扩大模型网格。

### 文件与命令

独立评估脚本：`scripts/diagnose_dgar_fusion.py`（不改main/rrgcn训练路径），
CPU数据统计：`scripts/analyze_dgar_evolution.py`。实际可运行接口及验证记录见
本文下一节补充。若已有v3隔离run目录，必须显式选择其中original的两份checkpoint，
不能误用根目录被其他实验覆盖的权重。无阶段checkpoint时先做上述融合定位，
以后再按main:637/1003增加阶段快照用于严格old/new漂移诊断。

## 11. 对七个最终问题的明确答复

1. **FAHC最可能三个主因**：代理效用错位；没有区分适应/遗忘；选片偏差经过
   无关系区分的实体均值+共享融合，且训练和评估prompt分布不同。后两项仍待因果实验。
2. **最值得研究两个问题**：历史生成表示在知识演化下何时有益/有害；不同实体角色/
   关系的有益迁移与无益漂移如何区分。
3. **避免**：把旧事实损失增长直接等同importance；一律保护旧logits；只看raw drift；
   没固定HCP/噪声就比较；混用宏平均/pooled/旧新协议；把所有低频波动都称regime change。
4. **优先模块**：`src/rrgcn.py` 的 `RecurrentRGCN.select_entity` 与 `forward` DAR接口；
   最小ECRI先改forward融合，不先改HCP selection或decoder。
5. **只能选一个**：ECRI-DGAR；核心公式在候选2，先统计past结构演化→校准支持与
   sampling null→实体融合alpha→原decoder/原损失训练。不加全局KD。
6. **更有希望的具体原因**：直接干预generated/current表示相遇的接口，保留原HCP
   上下文和生成器；验证的是full预测的回放效用，避免current-only CE代理；低支持回退。
   这只是结构上更贴合，诊断尚未证明它有收益，成功不能保证。
7. **先做一个实验**：固定生成特征的融合反事实×过去结构演化。它可以用现有模型
   不重新训练，直接判定是否值得改变DAR；未通过即停止这条主线。

## 12. 实际执行命令与本轮验证

这是新增诊断代码，另一台服务器需要先同步本次提交。不要用根目录可能被其他运行
覆盖的权重代替 original 隔离实验权重。以下 original 路径来自现有 stdout 的 RUN_DIR；
本工作区没有该隔离目录，应在保留该实验文件的服务器上执行。

### 设置同一次 original 的输入和新的输出目录

激活 dgar_env，然后：

```bash
cd /data1/zihao/DGAR-code
export DGAR_BASE_RUN='/data1/zihao/DGAR-code/experiment_runs/ICEWS14s_DGAR_s1_20261010_205225_196503824_142706'
DGAR_BASENAME='ICEWS14s-uvrgcn-convtranse-ly2-dilate1-his3-weight:0.5-discount:1-angle:10-dp0.2|0.2|0.2|0.2-gpu1'
export DGAR_MODEL="$DGAR_BASE_RUN/models/$DGAR_BASENAME"
export DGAR_DIFFUSION="$DGAR_BASE_RUN/diffu_models/$DGAR_BASENAME"
export DGAR_CONFIG="$DGAR_BASE_RUN/config.json"
export DGAR_DIAG_OUT="/data1/zihao/DGAR-code/logs/ICEWS14s-fusion-s1-$(date +%Y%m%d_%H%M%S)"
mkdir -p /data1/zihao/DGAR-code/logs
```

权重名中的 gpu1 只标记原运行设备，不要求本次使用GPU1。

### 不占GPU的文件/配置检查

```bash
python -B scripts/diagnose_dgar_fusion.py \
  --model-checkpoint "$DGAR_MODEL" \
  --diffusion-checkpoint "$DGAR_DIFFUSION" \
  --run-config "$DGAR_CONFIG" \
  --gpu 0 --dataset ICEWS14s --seed 1 \
  --snapshots 60,90,120,180,240,300,360 --snapshot-radius 4 \
  --alphas checkpoint,0.25,0.5,0.75,1.0 \
  --output-dir "$DGAR_DIAG_OUT" --dry-run
```

dry-run不加载checkpoint张量，因此只确认路径、架构配置和指纹；真正的strict权重
加载发生在正式评估。路径错误请指向同一次original run，而非换成任意最新权重。

### GPU空闲后执行一次有限评估及CPU后处理

```bash
setsid bash -c '
set -o pipefail
{
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python -u /data1/zihao/DGAR-code/scripts/diagnose_dgar_fusion.py \
    --model-checkpoint "$DGAR_MODEL" \
    --diffusion-checkpoint "$DGAR_DIFFUSION" \
    --run-config "$DGAR_CONFIG" \
    --gpu 0 --dataset ICEWS14s --seed 1 \
    --snapshots 60,90,120,180,240,300,360 --snapshot-radius 4 \
    --alphas checkpoint,0.25,0.5,0.75,1.0 \
    --output-dir "$DGAR_DIAG_OUT" \
  && python -u /data1/zihao/DGAR-code/scripts/summarize_dgar_fusion.py \
    --input-dir "$DGAR_DIAG_OUT" --output-dir "$DGAR_DIAG_OUT/analysis" \
    --fit-snapshots 60,90,120 --eval-snapshots 180,240,300,360 \
    --snapshot-radius 4 --bootstrap 500 --seed 1
} 2>&1 | tee "$DGAR_DIAG_OUT.log"
' < /dev/null > /dev/null 2>&1 &

tail -f "$DGAR_DIAG_OUT.log"
```

输出目录包含query/snapshot/relation三类CSV、metadata及完成status；analysis包含
summary.json、window_comparison.csv、controlled_strata.csv。输出失败时不继续后处理。
四个held-out窗口的bootstrap区间只是描述性不确定性，不能代替多seed最终验证。

本次已验证：三个新脚本的标准库语法/CLI；真实本地模型路径dry-run；前缀不受未来
训练片修改影响；低支持留空；63日窗口计数与裁剪；后处理正逆事实配对和缺失拒绝；
held-out rank改变不影响拟合阈值/规则；按窗口及事实cluster重采样。
未执行torch/DGL模型加载、GPU评估或训练，正式模型路径的运行结果仍需服务器验证。
