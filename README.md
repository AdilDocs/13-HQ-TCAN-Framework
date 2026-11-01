# HQ-TCAN: Hybrid Quantum–Classical Temporal Convolutional Attention Network

Source code for the paper *"HQ-TCAN: A Hybrid Quantum–Classical Temporal Convolutional
Attention Network for Early Warning of Severe Adequacy Violations in Renewable-Rich Power
Systems"*.

HQ-TCAN gives a **six-hour-ahead early warning** of severe energy shortages
(ENS ≥ 90th percentile) in a renewable-rich power system with battery storage. A dilated causal
TCN extracts short-term warning signs from a 24-hour window, a 4-qubit quantum self-attention
layer relates them across the window, and a variational quantum readout returns the shortage
probability. The model has 538 trainable parameters.

## Installation
```bash
pip install -r requirements.txt
```
No GPU and no quantum SDK are required: the 4/6-qubit circuits are simulated exactly in
`qsim.py` and differentiated with `autograd` (equivalent to parameter-shift gradients).

## Data
Download `time_series_60min_singleindex.csv` from the OPSD Time Series package
(version 2020-10-06) and place it in `data/` (see `data/README.md`).

## Reproduce everything
```bash
sh run_all.sh
```
or run the steps individually:

| Step | Command | Output |
|---|---|---|
| 1. Adequacy simulation, ENS, labels | `python data.py` | `sim.pkl` |
| 2. Simulator check vs PennyLane | `python test_qsim.py` | max. differences ≈ 1e-16 |
| 3. Base study (comparison, ablation, sensitivity) | `python run.py main \| ablation \| reup \| lam \| pct \| horizon` | `results/*.json` |
| 4. Validation-only tuning | `python tune.py search hq \| twin \| rnn \| gbm` | `results_v2/search_*.json` |
| 5. Final 3-seed test evaluation | `python final_all.py hqtcan2 twin2 lstm transformer vqc qsann hqtcn lgbm logreg persistence` | `results_v2/final_*.json` |
| 6. Tables and figures | see `run_all.sh` | `figs/` |

All runs are cached; re-running a step skips finished configurations.

## Files
| File | Content |
|---|---|
| `data.py` | PV–wind–BESS adequacy model (energy and power limits, firm capacity), ENS |
| `qsim.py` | Exact batched state-vector simulator (angle encoding, ZZ feature map, RealAmplitudes) |
| `models.py` | HQ-TCAN, ablation variants and baselines (LSTM, Transformer, VQC, QSANN, quantum TCN) |
| `train.py` | Windowing, risk-aware severity-weighted loss, training loop, metrics, threshold calibration |
| `run.py` | Base-configuration experiments |
| `tune.py`, `final_all.py` | Validation-only hyperparameter search and final test evaluation |
| `analyze.py`, `final_analysis.py`, `make_tables.py` | Tables, block-bootstrap tests, figures |
| `fig_framework.py` | Framework diagram (Fig. 1) |
| `test_qsim.py` | Verification of the simulator against PennyLane |

## Evaluation protocol
* Chronological split by target year: train 2015–2017, validation 2018, test 2019.
* Feature scaling and the ENS tail threshold are computed on the training years only.
* Early stopping, hyperparameter selection (validation PR-AUC) and the alarm threshold
  (maximum validation G-Mean) use the validation year only; the test year is used once.
* All neural models share the same loss, optimiser, learning-rate schedule and stopping rule.

## Main result (test year 2019, 6 h lead time, 3 seeds)
HQ-TCAN warns of 90.9% of severe shortage hours (persistence: 41.6%); with false alarms
limited to 10% of normal hours it still warns of 88.4% of severe events, using 538 parameters.

## Cached results
`results/` and `results_v2/` contain the outputs of every run reported in the paper
(metrics, training histories and test scores). With these folders present, step 6 regenerates
all tables and figures in a few minutes without retraining; delete them to rerun from scratch.
