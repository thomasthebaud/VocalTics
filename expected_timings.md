# Expected runtimes

These are approximate runtimes for the current dataset and compute environment.
Actual times may vary with GPU availability, storage speed, model choice, and
the number and duration of the recordings.

## Individual scripts

| Script | Expected time | Notes |
|---|---:|---|
| `01_data_preprocessing.py` | < 10 seconds | Reads annotations and builds segment metadata. |
| `02_grouping_categories.py` | < 10 seconds | Assigns tic groups and prints summaries. |
| `03_extract_features.py` | ≈ 20 minutes | Extracts WavLM Base Plus embeddings for all recordings. |
| `04_generate_new_split.py` | < 10 seconds | Generates and summarizes the cross-validation folds. |
| `05_train_tic_detection.py` | ≈ 55 minutes | Time for one model on one fold. |
| `06_train_tic_segmentation.py` | ≈ 55 minutes | Time for one model on one fold. |
| `07_training_graphs.py` | < 10 seconds | Produces training-curve figures from saved logs. |
| `10_metrics.py` | < 10 seconds | Aggregates metrics across folds and experiments. |
| `11_make_graphs.py` | < 10 seconds | Produces confusion-matrix figures. |

## Full training launchers

The approximate training time is:

```text
number of models × number of folds × approximately 1 hour
```

With five folds:

| Workload | Expected time |
|---|---:|
| One model across 5 folds | ≈ 5 hours |
| Two models across 5 folds | ≈ 10 hours |
| Three models across 5 folds | ≈ 15 hours |

Therefore, a launcher or sequence of launchers covering two to three model
configurations should take approximately **10–15 hours** when folds run
sequentially. The current launcher scripts process their folds one after
another rather than in parallel.
