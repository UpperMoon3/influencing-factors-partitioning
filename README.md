# Factor-induced partitioning of training samples

Code for the paper *Using Information on Influencing Factors to Partition
Training Samples for Classification under Changing Operating Conditions*
(N. Bazhayev, L. Rzayeva, M. Shayakhmetov, I. Lebedev, B. Rzayev; submitted to
*Algorithms*, MDPI).

Every number, table, and figure of the paper that comes from a simulation or
from the real-data experiments is produced by the scripts in `code/`. The
results of the runs reported in the paper are included in `results/`, so the
tables and figures can be regenerated without re-running the experiments.

## Layout

```
code/      scripts (run them from this directory)
data/      input streams (not included, see data/README.md)
results/   raw results of the runs reported in the paper (JSON) and logs
tables/    LaTeX table bodies generated from results/
figures/   figures generated from data/ and results/
```

## Scripts and where their results appear in the paper

| Script | Purpose | Paper |
|---|---|---|
| `sim.py`, `fig_sim.py` | simulation study with two regimes | Table 2, Figure 2 |
| `sim_extra.py` | over-partitioning at a fixed budget; a priori indicators over 300 random configurations | Tables 3, 4 |
| `verify_calendar.py` | recovery and verification of the timestamps of both streams | Section 5.2, Figure 3 |
| `datasets.py`, `partitions.py` | loading, timestamp recovery, calendar factors, nested partitions | Section 5 |
| `protocol.py` | classifiers and the paired randomised and chronological protocols | Section 5 |
| `run_main.py`, `analyze.py`, `make_tables.py` | main paired experiment | Tables 6, 7, 8 |
| `run_selfix.py` | two estimators of the per-regime risk for the selection rule | Table 9 |
| `run_control.py`, `run_control_chrono.py` | negative control with a random partition | Table 10 |
| `run_factors.py` | ablation over the choice of factors | Table 11 |
| `run_size.py` | minimum subsample size (`p_min` sweep and learning curves) | Table 12, Figure 6 |
| `run_stream.py` | prequential comparison with DDM, ADWIN, ARF, and HAT | Table 13 |
| `indicators.py`, `analyze_swc.py` | a priori indicators over 273 real-data cells | Table 14, Figure 8 |
| `timing.py` | measured training cost | Table 15 |
| `fig_data.py`, `fig_results.py` | figures from the data and from the main results | Figures 3, 4, 5, 7, 8 |
| `make_revision_tables.py` | tables and figure for `run_factors.py`, `run_size.py`, `run_stream.py`, `sim_extra.py` | Tables 3, 4, 11–13, Figure 6 |
| `verify_main.py` | re-runs part of `run_main.py` and compares it with `results/main.json` | — |

## Running

```
cd code
python verify_calendar.py
python sim.py; python fig_sim.py
python sim_extra.py
python run_main.py 20                 # long: about an hour on a laptop
python run_control.py 10; python run_control_chrono.py; python run_selfix.py; python timing.py
python analyze.py; python analyze_swc.py; python indicators.py
python make_tables.py; python fig_data.py; python fig_results.py
python run_stream.py
python run_factors.py 10
python run_size.py 10 3
python make_revision_tables.py
```

Random seeds are fixed in the scripts. Re-running `run_main.py` reproduces the
stored randomised-protocol results exactly and the chronological-protocol
results to within 0.02 percentage points (multi-threaded random forests are not
bit-reproducible); `verify_main.py` performs this check on a subset.

## Requirements

Python 3.10, numpy 2.0, scipy 1.15, scikit-learn 1.6.1, matplotlib, and
river 0.22.0 (only for `run_stream.py`).
