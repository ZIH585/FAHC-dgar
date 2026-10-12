# QCDR 实现与核验报告

## 1. 来源与隔离

工作树：`/data1/zihao/DGAR-code/.worktrees/qcdr`；分支：`codex/qcdr`。

提交关系：

```text
a454564  Initial commit（master / origin/master，Original 源码）
  └─ f7bf2d8  Prepare clean Original DGAR baseline with shared runtime fixes
      └─ a4f00eb  Retain shared cache and singleton fixes on Original DGAR
          └─ Implement query-candidate decoupled replay on DGAR
```

分支从 `f7bf2d8` 新建，后者直接继承 `a454564`。根工作区与 ECRI 工作树没有
reset、切分支或清理操作，已有改动、日志及权重保留在原处。

没有继承 FAHC/ECRI 的实验提交，也没有复制它们的 selector、risk/coverage、
fact budget、probe、KD、relation statistics、ECRI gate、diagnostic、staged
experiment 或实验启动脚本。这里保留的是 **Original 原有 temporal gate 和标量
融合参数 `model.alpha`**；它们不是 ECRI 的 gate，也不是 QCDR 的梯度系数。

`f7bf2d8` 保留的公共修复见 [BASELINE.md](BASELINE.md)，包括 CPU/device 兼容、
参数类型、空 replay/单样本处理、路径、随机数隔离、匹配的模型/Adam 恢复，以及
最终评估的历史图使用时序修复。部分修复会改变未修复初始代码的实验结果，故
**对照必须使用本分支的 `--qcdr-mode original`，不能直接拿旧日志作为严格对照。**

另从混有 ECRI 的 `918573f` 中逐项提取了三项公共修复，单独提交为 `a4f00eb`：

1. diffusion CE 的 target 保持 [batch]，避免单样本被 squeeze 成标量。
2. 逐快照验证既有 HCP 字典只含该快照 train facts 及其合法逆关系，并包含全部
   forward train facts。接受原缓存已有的 forward-only / forward+inverse 形式，
   不重建、不重采样；错误缓存明确报错。
3. 最终评估仅传入 train history。原来附加的 validation 后缀实际未被索引，
   删除它以明确输入边界；查询、过滤集合与指标定义不变。

源码搜索确认 src/、diffusion/、rgcn/ 无先前方法专属标识或调用。
原来就有的通用名称（例如 importance sampler）不能等同于 FAHC selector。

## 2. 沿真实 tensor 追踪 Original

设实体数为 N、正向关系数为 R、隐藏维度为 d，某个实体 CE batch 大小为 B。

| 张量 | 来源 | 形状 |
|---|---|---|
| dynamic_emb | 可训练实体初始表 | N × d |
| emb_rel | 正向和逆向关系初始表 | 2R × d |
| model_output | 冻结的原有 feature/diffusion 模型生成 | M × 3 × d |
| ent_embeding | select_entity 按实体平均生成事实的头尾表示 | N × d |
| H | 历史图递归编码后的最终实体表示，按原配置 normalize | N × d |
| R_t | relation GRU 的最后输出 | 2R × d |
| V | tanh(H)，全实体候选矩阵 | N × d |
| Z | ConvTransE 的 subject/relation 卷积查询输出 | B × d |
| S | Z @ V.T，未 softmax 的实体 logits | B × N |

训练阶段 t 的输入图为 `train_list[max(0,t-history_len):t]`。当前 D_t 提供监督
triples，不将含当前 target 的 G_t 加到这段历史图中。stage 0 作为初始历史，
原训练循环从 stage 1 开始；QCDR 不增加对 stage 0 的训练。

`RecurrentRGCN.forward` 的实际顺序：

1. 从 dynamic_emb 开始；若开启 static graph，则先经过原静态 RGCN。
2. 每张过去图聚合相应关系的邻居实体，送入 relation_cell_1，得到关系表示。
3. self.rgcn 编码该图，产生 current_h。
4. 若有生成 replay，将 select_entity 得到的实体表示按原有可学习标量
   model.alpha 融入命中实体行。
5. 原 temporal gate 合并旧状态与 current_h，推进到下一张历史图。
6. 最后得到 H、R_t，送入原 decoder。

HCP 与 diffusion 路径保持 Original：在严格过去的时间索引中随机抽取历史，
按当前查询 subject 匹配历史字典，合并 HCP；原有冻结模型生成特征。
gereration_feature 在主预测器训练调用处位于 torch.no_grad() 中，因此生成
特征不会通过预测器损失训练该冻结模型，但特征融合之后的预测器参数仍可训练。
QCDR 不改变历史样本、预算、历史索引、diffusion 目标或 diffusion 的训练阶段。

在 ConvTransE.forward 中，先取 V[subject] 与 R_t[relation]，沿用原
BN/dropout/Conv1d/FC/ReLU 构造 Z，然后与 **同一个 V 的全实体矩阵**做矩阵乘法：

```text
historical graphs + original generated replay features
                      ↓
          shared RGCN / relation GRU / temporal gate
                      ↓ H, R_t
              V = tanh(H)
                ↙        ↘
    V[s], R_t[r]           full candidate bank V
         ↓                          ↓
 ConvTransE query Z       QCDR routing ONLY for replay entity CE
                ↘        ↙
                    Z Vᵀ
                      ↓
                  entity CE
```

关系预测仍使用原 ConvTransR 和 2R 个关系候选；它没有被接入 QCDR。
原有 partial_embeding 调用方式也保留：显式给候选矩阵时只在这个矩阵上 routing，
不额外 tanh，不改变 subject 的来源。

## 3. 耦合是否真实存在

**存在双路径梯度，这是代码和 autograd 测试共同证实的事实。耦合导致性能下降，
则仍是待实验检验的假设。**

令 P=softmax(ZVᵀ)、Y 为 one-hot target。忽略 batch 平均系数时：

```text
∂L/∂Z = (P-Y)V
∂L/∂V | candidate = (P-Y)ᵀ Z
```

Original replay 的 candidate 路径会对全候选空间产生梯度，query 路径也通过
subject/context 访问共享 H。二者都可能沿 H 更新 dynamic_emb、RGCN、关系
GRU/emb_rel、temporal gate、原融合标量及开启时的静态编码器。query 路径还更新
ConvTransE 的 conv/fc/BN 参数。并非每个声明的参数在每个 batch 都必有非零梯度。

Original 已冻结特征生成模型，**但没有切断实体 replay CE 中这两条评分路径**。
冻结生成特征和限制主预测器的 candidate 梯度，是两个不同位置。

## 4. QCDR 的精确定义及插入点

将 QCDR 梯度系数记作 a，避免与 Original 的 model.alpha 混淆：

```text
V_a = stopgrad(V) + a * (V - stopgrad(V))
S_replay = Z_replay V_aᵀ
```

有限数值下 V-stopgrad(V) 为零，前向数值不变，反向只有 candidate 路径乘 a。
比把两个缩放后的副本相加更直接地避免前向舍入差异。端点单独处理：

```python
if alpha == 1.0:
    return candidate_repr        # Original：直接返回，不 detach
fixed = candidate_repr.detach()
if alpha == 0.0:
    return fixed                 # stopgrad
return fixed + alpha * (candidate_repr - fixed)
```

位置在 `src/decoder.py::ConvTransE.forward` **最后的 candidate matmul 之前**。
此时 subject 已从共享 H 取出并形成查询计算图；不能在 RGCN 输出 H 处整体 detach，
否则会同时切断 query。

`src/rrgcn.py::loss_of_reply(mode='e')` 是唯一启用入口，向 decoder 传入系数。
get_loss 中的 current entity CE 不传这个参数，走默认 1；relation CE 不变；
predict、diffusion guidance 和 loss_diffu_ce 均沿原路径。

在相同权重、batch 和 dropout/RNG 状态下，各模式 logits、CE 数值相同。
**更新参数后分数可以不同，这是训练方法改变的预期；并不要求跨训练轨迹相同。**

共享参数上的未裁剪梯度（各项含原 loss 系数）满足：

```text
g = g_current + g_replay_query + a*g_replay_candidate + g_relation + g_static
```

a=0 仍允许 query subjects、图邻居、逆关系查询中的原 object 通过共享编码器
获得梯度。relation replay、当前 CE 以及融合过历史特征的当前 H 也保留更新。
因此 QCDR 不等于冻结实体表，也不等于清除历史对 candidate representation 的
全部影响。a 是**直接候选分支、裁剪前的梯度比例**，不是最终 Adam 参数更新比例。

## 5. 原 loss 和训练调用链

下面的两个实体 replay CE 都是 Original 已有项，QCDR 同时覆盖，未删减其一：

```text
L_entity = CE(current forward+inverse)
         + CE(replay forward)
         + mu * CE(replay forward+inverse)
L_relation = CE(current forward+inverse relations)
           + mu_r * CE(replay forward+inverse relations)
L_total = task_weight*L_entity + (1-task_weight)*L_relation + L_static
```

每项 CE 保持原 reduction。loss_Reply MSE 虽然由模型返回，但原 main 没有将它
加入最终 loss；这里也没有新增它。没有新 KD、正则或额外 loss。

完整阶段流程：

1. run_experiment 校验参数，打印一次 mode/effective alpha/target；按原 seed
   初始化模型与 RNG，读取数据、验证既有 HCP、记录运行指纹。
2. 阶段 t 构造严格过去的 train history；原冻结 feature 模型生成 replay。
3. 原主预测器内循环调用 get_loss：一次原共享图编码，current CE 正常反传，
   两个历史实体 CE 在 loss_of_reply 中启用 QCDR，关系与静态项保持原样。
4. 沿用总 loss、梯度裁剪、Adam、validation early stopping，以及随后的原 diffusion
   阶段训练；二者分别恢复最佳模型与匹配 Adam，再同步各自的原有冻结模型。
5. 保存阶段 checkpoint、推进下一时间步；最终沿用完整 TEST，并可附加 VALIDATION。

QCDR 不读取标签来生成新分数或做选择，不增加任何未来统计量。缓存校验只能验证
相对于当前代码 train split 的边界，不能把原论文协议转换成全新的数据划分。
原代码将源 train/valid/test 时间段拼接后按快照 80/10/10 切分；最终模型回测所有
时间片、添加逆关系、采用原过滤口径。该协议保持，不应把最终回测误称为逐时在线预测。

## 6. 参数、状态、开销

| 参数 | 默认值 | 含义 |
|---|---|---|
| --qcdr-mode | original | original / candidate-stopgrad / candidate-scaled |
| --qcdr-candidate-alpha | 1.0 | finite 且在 [0,1]；scaled 使用指定值 |
| --qcdr-target | entity | 第一版仅支持 entity；relation/both 明确拒绝 |

| 模式/系数 | historical entity candidate 梯度 | current / replay query |
|---|---:|---|
| original（无论指定的合法 alpha 为何） | 100% | 原样 |
| candidate-stopgrad | 0% | 原样 |
| candidate-scaled, a=0 | 0%，等同 stopgrad | 原样 |
| candidate-scaled, a=0.25 | 25% | 原样 |
| candidate-scaled, a=0.5 | 50% | 原样 |
| candidate-scaled, a=1 | 100%，等同 Original | 原样 |

启动日志显示的是**生效系数**。无新增可学习参数、buffer、old model、memory 或
跨阶段状态；已有 Original 冻结模型继续按原流程使用。配置指纹记录 QCDR 参数，
同一模式/系数恢复即可；checkpoint 的模型 tensor key 不变，无 QCDR auxiliary。
严格 resume 会拒绝不匹配的源码/数据/缓存/配置，包括改 alpha 或直接续训旧分支。
本方法不把不同分支的 checkpoint 当作同一实验接着训练。

helper 不移动 device/dtype，不调用 CPU/NumPy，不抽随机数，不保存计算图。
stopgrad 为 detach view；scaled 增加 O(Nd) 的临时张量/逐元素运算，没有第二次
图编码或 diffusion。以 7128×200 FP32 为例，每张 N×d 张量约 5.44 MiB，
实际峰值显存和耗时尚未测量，不能据此宣称训练零开销。

## 7. 文件清单

| 文件 | 改动 |
|---|---|
| src/qcdr.py（新增） | 35 行无状态参数解析/候选梯度 routing |
| src/decoder.py | ConvTransE 接收默认 1 的系数，最后 matmul 前应用 |
| src/rrgcn.py | 保存普通 Python 系数；只在历史实体 CE 传入 |
| src/cli.py | 三个 QCDR 参数及有效性校验 |
| src/main.py | 导入解析函数，启动时打印三行；QCDR 不改训练循环 |
| tests/test_qcdr.py（新增） | 15 项 synthetic CPU 梯度、集成、兼容、resume 测试 |
| tests/test_baseline_fixes.py（新增于公共修复提交） | 2 项缓存与 singleton diffusion 测试 |
| src/runtime.py（公共修复提交） | 验证既有 HCP 与阶段 train facts 一致 |
| diffusion/model_21.py（公共修复提交） | singleton CE target shape 修复 |
| src/main.py（公共修复提交） | 接入缓存校验、移除未使用的 validation history 后缀 |
| docs/BASELINE.md | 记录原始来源、保留修复与方法边界 |
| docs/QCDR.md（新增） | 本实现与核验报告 |
| docs/QCDR_EXPERIMENTS.md（新增） | 完整实验矩阵命令、日志与恢复说明 |
| README.md | 方法入口与 CPU 检查说明 |

既有 .gitignore 已排除 data、weights、logs、experiment_runs、缓存、lock 等，
无需追加无关规则。没有把运行产物加入本次提交。

## 8. 验证及结果

环境：现有 dgar_env，Python 通过其绝对路径调用；PyTorch 2.4.0+cu121、
DGL 2.4.0+cu121。所有测试显式隐藏 CUDA，使用 CPU synthetic tensors/graphs。

| 检查 | 结果 |
|---|---|
| 24 个 Python 文件 py_compile | 通过，编译产物只写临时目录 |
| src.main / helper import | 通过；CUDA 未初始化 |
| python -B src/main.py --help | 通过；三参数可见 |
| git diff --check | 通过 |
| 实验文档中 10 个 Bash 代码块 bash -n | 通过，仅解析语法，未执行命令 |
| runtime 源码旧方法专属标识搜索 | 无匹配 |
| CPU unittest | 17 / 17 通过，最终复跑 1.389 秒（不含解释器启动） |

17 项测试具体覆盖：

1. 独立 query/candidate tensors：前向完全一致；候选梯度 0/.25/.5/1，查询梯度相同且非零。
2. helper 的 FP16/BF16/FP32/FP64、device、空 tensor、no_grad、非法 alpha；1 直接返回原对象。
3. 实际 ConvTransE：带 dropout/BN、batch 1 和 3、独立候选输入，前向和 query/decoder
   梯度一致，candidate 按比例缩放；每次恢复同一权重与 RNG。
4. 共享实体 tensor：显式分离 query-only/candidate-only；验证 g_a=g_query+a*g_candidate。
5. 实际 ConvTransE 的 CPU BF16 autocast：同样的前向/反向合同通过。
6. 从 Git a4f00eb 独立加载**旧 decoder 和旧 RGCN**，与当前 Original、scaled1
   比较初始化、损失、梯度、静态图及一次梯度裁剪/Adam 后状态：rtol=atol=0 全部通过。
7. 真实 get_loss 调用记录：current scale=1、两个 replay scale=0；逆关系 ID 正确，
   relation decoder 不接收 QCDR 参数。
8. 完整模型 raw gradient 对 a 呈仿射关系；固定状态下 relation loss 梯度完全相同。
9. replay-only stopgrad 后 entity/relation 表、RGCN、关系 GRU、time gate、decoder
   conv/fc 仍收到 query 路径非零梯度。
10. 无 replay、空 replay、初始无旧回放阶段、单事实 batch；current 梯度不变，scaled0=stopgrad。
11. 实际 predict 不调用 routing，各模式 logits 相同且没有梯度。
12. CLI 默认值与旧版除 QCDR 外全部一致，非法模式/alpha/target 拒绝。
13. checkpoint 恢复模型、原 teacher、Adam 和 RNG；无新增 QCDR tensor/auxiliary；改 alpha 拒绝恢复。
14. 真实 run_experiment 调用链注入 synthetic 数据/生成器，只完成一个极小 replay
    batch forward/backward 后立即中断；禁止 optimizer.step，未进入完整 epoch，未写模型 checkpoint。
15. CUDA 始终未初始化。
16. 缓存校验：接受原 forward-only/forward+inverse 且不改文件，拒绝错误时间/subject/缺失 facts。
17. 实际 diffusion CE 在单事实 batch 上 forward/backward 正常。

静态命令和 toy 测试不等于正式训练验证。尚未验证 CUDA/DGL 长时间运行、CUDA AMP、
多进程/跨硬件复现、整轮指标、候选空间是否改善及实际性能开销。没有启动真实数据训练。

## 9. 研究判断与实验决策

这是一项有真实计算图依据的 **replay 梯度分配假设检验**，目前不能称为已证明有效的
遗忘缓解方法。形式上就是 stop-gradient/梯度缩放；价值取决于 DGAR 是否存在稳定的
候选分支负迁移证据，而不是这几行公式本身有多新。

支持继续的结果：a<1 在相同预算/同代码的验证集上优于 Original，多个阶段与
关系群体能复现，后续多个 seed 仍保持；而 a=1 符合 Original 对照。若 .25/.5
优于 0，说明适度削弱可能有益，不能宣称 candidate replay 完全无用。

主要失败风险：全实体候选梯度同时提供正目标与负候选学习，切掉它可能损伤少见
实体和旧事实的可分性；共享 query 与关系路径仍会改变实体空间；原 replay 的前向
特征融合若才是问题来源，只改 CE backward 可能无效。

若 0/.25/.5 全部不如 Original，应停止把“候选 replay 有害”当作既定方向。
如果只是缩小所有 replay 梯度也能同样改善，则研究贡献可能只是有效 replay 强度
变化，未来需要等计算量的整体 replay 权重对照来区分；**本次不叠加这个机制**。
五组第一阶段命令与指标口径见 [QCDR_EXPERIMENTS.md](QCDR_EXPERIMENTS.md)。
