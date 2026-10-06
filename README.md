# DGAR and FAHC-DGAR

Run the inherited DGAR configuration from `src`:

```bash
cd src
python main.py --gpu 0 --dataset ICEWS14s
```

## FAHC-DGAR

FAHC changes which historical context prompts (HCPs) DGAR replays. At stage
`t`, it uses the previous best model as a frozen teacher, makes a short
current-only probe update, and scores each candidate HCP with the **mean of
positive per-fact entity cross-entropy increases**. Candidate facts are scored
under DGAR's current history graph, without diffusion feature injection. The
probe sees at most `--fahc-probe-facts` current facts (128 by default). It
restores model parameters, BatchNorm buffers, module train/eval flags,
pre-existing gradients, forward caches, and Python/NumPy/PyTorch RNG states
after scoring. The separate probe Adam can copy the main Adam's parameter
groups and moments from the matched best checkpoint; only its learning rate
is changed. The main optimizer and its moments are not updated by the probe.

Each candidate HCP contains the historical slice's cached facts whose subject
appears in the current snapshot. Risk and coverage use this **complete
subject-filtered HCP** for every candidate slice. Greedy selection chooses
`k = --reply_batch` slices using risk plus normalized
harmonic coverage of base relation IDs, entities, and time bins. The next
occurrence of a feature already covered `n` times earns `1/(n+1)`; the
coverage families are normalized by their candidate-pool universe sizes.
Inverse relations share the base relation's coverage count. Set
`--fahc-entity-weight 0` for relation/time-only coverage. The default
candidate pool contains up to `2k` eligible historical slices; change it
with `--fahc-candidate-multiplier` (for example, 2–4).

Only **after** selecting slices does FAHC cap the unique replay union at
`B = --fahc-fact-budget`. It shuffles each selected slice with a fixed seed,
randomizes slice order, then takes unique facts in round-robin order. The cap
can underfill when slices have too few relevant facts or share duplicate
triples. The JSONL log records the actual fact count, so equal `k` and `B`
do **not** imply exactly equal facts consumed in every run. `B` counts unique
HCP triples before DGAR's inverse-query expansion; it does not bound all
historical files in memory.

Optional old-model entity distillation uses teacher and student history-only
logits on the selected HCP facts. A fact contributes to the temperature-scaled
KL only when the teacher's ordinary (`T=1`) top entity is correct and its
confidence meets `--fahc-kd-confidence`. KL is averaged over **all** replay
facts, including those gated out, then multiplied by `--fahc-kd-weight`.
This is an auxiliary path; DGAR's diffusion-conditioned replay loss remains.

### Run comparable modes

Use the same `--reply_batch`, `--fahc-fact-budget`, seed, and training
configuration when comparing selection modes:

```bash
cd src
python main.py --gpu 0 --dataset ICEWS14s --hcp-selection random --reply_batch 35 --fahc-fact-budget 128
python main.py --gpu 0 --dataset ICEWS14s --hcp-selection recent --reply_batch 35 --fahc-fact-budget 128
python main.py --gpu 0 --dataset ICEWS14s --hcp-selection risk --reply_batch 35 --fahc-fact-budget 128
python main.py --gpu 0 --dataset ICEWS14s --hcp-selection coverage --reply_batch 35 --fahc-fact-budget 128
python main.py --gpu 0 --dataset ICEWS14s --hcp-selection risk+coverage --reply_batch 35 --fahc-fact-budget 128
python main.py --gpu 0 --dataset ICEWS14s --hcp-selection risk+coverage --reply_batch 35 --fahc-fact-budget 128 --fahc-kd-weight 0.1
```

`original` keeps the inherited random, uncapped DGAR HCP path. Use
`--hcp-selection random --fahc-kd-weight 0.1` for a budget-capped HCP-KD
ablation. Controlled modes use DGAR's eligible snapshots `0..t-2`; `recent`
takes the newest `k` of these. For an original-DGAR history-only diagnostic
with the same metric as controlled runs, use:

```bash
python main.py --gpu 0 --dataset ICEWS14s --hcp-selection original --fahc-diagnostics
```

The inherited `get_loss` replay weighting is more specific than the schematic
`L_current + mu L_HCP + lambda KD` description: when a nonempty HCP produces
diffusion features, entity replay includes **unweighted CE on the `B` HCP
triples** plus `mu * CE` on their **`2B` inverse-expanded queries**. Relation
replay includes `mu_r * CE` on those `2B` queries. Current entity/relation
losses and the static constraint also apply. Thus `mu=0` does **not** disable
all replay, and inverse-expanded query count is not the unique-fact budget.
The same inherited loss is used across selection modes; KD adds its separate
weighted term.

### Logs, checkpoints, and evaluation

FAHC JSONL logs contain selected/candidate indices, full-HCP risks, greedy
marginal selection scores, the actual unique replay fact count, KD loss, and
current/past history-only filtered MRR. `--fahc-diagnostics` enables these history-only
metrics for `original` too. Validation and evaluation use the same fixed,
isolated RNG policy in every mode, so their sampling does not consume
training randomness. RGCN and diffusion models reload their optimizer
moments with the matching best model checkpoints at stage boundaries.
FAHC checkpoints carry a **v2 hash of the full effective training
configuration** to separate runs with different settings.
Controlled FAHC runs and `--fahc-diagnostics` require `--n-epochs 1`
(the default). Repeating the entire timeline from its final checkpoint would
give earlier stages knowledge from later snapshots.

Budget diagnostics also record `selected_full_fact_count`,
`replay_retained_fraction`, `selected_risk_sum`,
`selected_weighted_coverage_gain_sum`, and `risk_share_of_greedy_gain`.
`fact_budget_active=false` identifies the uncapped `original` path. A small
budget such as 128 can remove most of DGAR's normal replay facts; compare with
`random` at the same budget before attributing a drop to forgetting-aware
selection. The first ICEWS14s run is analyzed in
[docs/ICEWS14s_FAHC_DIAGNOSIS.md](docs/ICEWS14s_FAHC_DIAGNOSIS.md).

History-only MRR is separate from DGAR's diffusion-conditioned validation
MRR. By default, the diagnostic samples four evenly spaced older validation
snapshots per stage. Set `--fahc-past-eval-samples 0` to evaluate all older
snapshots, or `--fahc-eval-every N` to change how often the past is evaluated.
The sampled set changes as time advances, so use all older snapshots for a
comparable forgetting curve. The log's actual replay count should be checked
when interpreting budget-matched ablations.

### Data and storage scope

The inherited loader reads the dataset's train, valid, and test files as time
snapshots, concatenates those snapshot sequences in that order, then divides
**each snapshot's rows** into 80% training, 10% validation, and 10% test,
without reshuffling. It is therefore a within-snapshot holdout protocol,
**not** a held-out-future forecasting evaluation. Filtered ranking uses
answers from the corresponding validation or test split, rather than every
true fact in the entire snapshot.

The code reuses `data/<dataset>/history_snap_v2` whenever that directory
already exists. A newly generated cache from `get_history` contains both
forward and inverse triples; an older forward-only cache can produce a
different HCP population. Keep cache provenance and contents consistent
across compared runs. The per-stage `B` cap does **not** make total historical
storage finite. A fixed-byte online HCP cache requires a separate experiment.
Teacher-logit storage with KD grows with replay facts times entity count.

Run focused CPU checks from the repository root with PyTorch, DGL, and NumPy
installed. These checks do not perform optimizer updates:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s tests -p 'test_fahc*.py'
```

The code-only review and its verification limits are recorded in
[docs/FAHC_REVIEW.md](docs/FAHC_REVIEW.md).
