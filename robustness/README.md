# Robustness analysis

The dissertation's canonical dashboard run uses random seed 42. This supplementary analysis repeats the same chronological supervised-learning pipeline across 30 independently generated synthetic datasets using seeds 1-30, with model configuration and temporal boundaries held fixed.

Run from the project root:

```bash
python robustness/run_robustness.py
```

Files:
- `run_robustness.py`: reproduction script for the 30-seed experiment.
- `robustness_30_seeds.csv`: per-seed final-holdout before/after metrics and drift-signal outcome.
- `robustness_summary.csv`: summary values used in Section 5.7.
- `robustness_recall_30_seeds.png`: recall plot used as Figure 5.6.
