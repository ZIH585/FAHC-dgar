# mean 正结果后的实验

先做固定模型的噪声/尺度诊断，再固定 mean 配方在 ICEWS18 检查可迁移性。
不追加训练seed、不调guided-x0、不混合RTER来替代独立效果判断。

## 第一组：噪声与尺度配对诊断

`scripts/diagnose_dcr_scale.py` 分别读取已完成的 Original 和 mean checkpoint，
仅做预先选定11片的全部 VALIDATION 推理，不更新参数。

| Arm | 尾表示 | 目的 |
|---|---|---|
| original | x=m+terminal_noise | 原采样终点 |
| mean | m=原guided posterior mean | 当前有效配方 |
| scaled-original | x*norm(m)/norm(x) | 带噪表示缩到mean范数 |
| scaled-mean | m*norm(x)/norm(m) | mean放大到原表示范数 |

四组共用每片的一次实际denoiser/guide输出、HCP、初始噪声、头/关系token和完整
decoder batch。范数按每条历史事实尾向量计算，不用当前query答案决定比例。零
范数时无法匹配非零向量，安全返回零并记录。匹配生成尾范数不保证聚合实体bank
范数相同，因为平均、头尾混合与方向抵消仍有影响；同时报告聚合范数。因此这是
可操作的机制控制，不是所有尺度因素的唯一因果分解。

主要比较每份checkpoint内部四组。两模型权重不同，不能把其绝对MRR差当单因素
效应。加载mean必须显式--checkpoint-mode mean；控制生成仅局部用Original采样
取得x和m，不改配置/权重。原diagnose_dcr.py默认仍只接受Original。

11片为30..330每30片。每片诊断RNG/buffer隔离，generator仍TRAIN；不是完整
官方评估随机轨迹，绝对分数不能与论文Average直接比较。正向主体的历史检索与
过去训练tags、decoder全候选范围保留，标签仅在预测之后用于排名统计。

```bash
cd /data1/zihao/DGAR-code/.worktrees/dcr
export DGAR_PYTHON="$(command -v python)" # 先激活 dgar_env
export DCR_GPU=1                        # 改为空闲GPU
export DCR_SCALE_ID="$(date +%Y%m%d_%H%M%S)_$$"
mkdir -p logs experiment_runs
setsid bash -c '
set -euo pipefail
for DCR_CHECKPOINT in original mean; do
  if [[ "$DCR_CHECKPOINT" == original ]]; then
    DCR_INPUT=/data1/zihao/DGAR-code/.worktrees/dcr/experiment_runs/ICEWS14s-dcr-original-seed1-20261014_143019_4046517
  else
    DCR_INPUT=/data1/zihao/DGAR-code/.worktrees/dcr/experiment_runs/ICEWS14s-dcr-mean-seed1-20261014_152147_4151455
  fi
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  "$DGAR_PYTHON" -u scripts/diagnose_dcr_scale.py \
    --gpu "$DCR_GPU" --seed 1 --checkpoint-mode "$DCR_CHECKPOINT" \
    --stages 30,60,90,120,150,180,210,240,270,300,330 \
    --data-root /data1/zihao/DGAR-code/data --run-dir "$DCR_INPUT" \
    --output "experiment_runs/ICEWS14s-dcr-scale-$DCR_CHECKPOINT-$DCR_SCALE_ID" \
    2>&1 | tee "logs/ICEWS14s-dcr-scale-$DCR_CHECKPOINT-$DCR_SCALE_ID.log"
done
' < /dev/null > "logs/ICEWS14s-dcr-scale-launcher-$DCR_SCALE_ID.log" 2>&1 &
```

summary.json、queries.csv、generation.jsonl包含四组MRR/H1/H3/H10、片/方向分组，
各contrast的pooled/macro差、胜负平、正收益片数、top5收益集中度、生成/聚合范数、
零范数数量、norm误差、HCP时间片、噪声seed及来源指纹。

判读：mean胜过scaled-original且scaled-mean胜过original，支持去噪方向；
scaled-original复制mean收益，则优先研究尺度/头尾比例；两份模型表现不同，说明
训练适应不可忽略，不能用Original固定模型结果否定mean重训收益。看跨片/方向
和集中度，差异很小时标为机制未决。保留mean已观察到的总体收益，不无止境搜索。

本次28项CPU回归、29文件编译、--help通过；覆盖范数匹配、固定头/关系token、
detach、空/零范数、一次网络输出、Original值/RNG、hook恢复，两种toy配置加载
与输入字节不变。真实mean CPU check-only只读验证，在推理前停止。神经训练核心
文件未修改，核心源码指纹保持313c60fe...。本次没有启动真实数据推理或训练。

## 第二组：ICEWS18，仅Original与mean

同seed1、同GPU/环境、同数据/cache，双方固定配置各训练一次。第一组解决机制
归因，第二组检查性能能否迁移。无需以第一组所有对比都赢作为跨数据验证的条件。
执行前确保ICEWS18原数据/cache在共享data目录。以下为正式训练命令，本次未执行。

```bash
cd /data1/zihao/DGAR-code/.worktrees/dcr
export DGAR_PYTHON="$(command -v python)"
export DGAR_DATA_ROOT=/data1/zihao/DGAR-code/data
export DCR_GPU=1
export DCR_18_ID="$(date +%Y%m%d_%H%M%S)_$$"
mkdir -p logs experiment_runs
setsid bash -c '
set -euo pipefail
for DCR_GROUP in original mean; do
  bash scripts/run_dcr_experiment.sh "$DCR_GROUP" "$DCR_GPU" ICEWS18 1
done
' < /dev/null > "logs/ICEWS18-dcr-mean-launcher-$DCR_18_ID.log" 2>&1 &
```

launcher记录两组具体路径。比较最终TEST/VALIDATION的macro及pooled MRR、H1/H3/H10。
两数据集方向一致则发展mean；ICEWS18下降则记录适用范围，不调到TEST变好。

另一服务器只更新代码，不覆盖本地数据/outputs：

```bash
cd /data1/zihao/DGAR-code/.worktrees/dcr
git pull --ff-only origin codex/dcr
```

未创建DCR工作区时先按DCR.md的Git步骤创建。两份完整checkpoint需要单独复制，
config.json/latest.pt/model.pt/diffusion.pt必须配套；模型和数据不通过Git传输。
