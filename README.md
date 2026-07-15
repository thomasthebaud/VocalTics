# VocalTics

This repository contains a preprocessing, feature extraction, training, and evaluation pipeline for vocal-tic detection and tic-group classification. The protected audio and annotation data are not included in the repository.

## Pipeline overview

Run the numbered scripts in this order:

1. `01_data_preprocessing.py` creates tic and non-tic segment metadata.
2. `02_grouping_categories.py` assigns an in-house YGTSS group to each tic segment.
3. `03_extract_features.py` optionally extracts one WavLM-large tensor per full recording.
4. `04_train_tic_detection.py` trains one cross-validation fold.
5. Run script 04 once for every fold.
6. `05_training_graphs.py` plots training curves across folds.
7. `10_metrics.py` aggregates validation and test metrics across folds.
8. `11_make_graphs.py` creates confusion matrices from the test predictions.

## Expected data layout

The preprocessing scripts currently expect:

```text
/projects/vocaltics/data/
├── ticList.csv
└── DET_Audio_Data_16kHz_mono/
    ├── DET0101/
    │   ├── DET0101_V1_HI.wav
    │   ├── DET0101_V1_LO.wav
    │   └── DET0101_V1_NO.wav
    ├── DET0102/
    └── ...
```

Participant IDs can range from `DET0101` through `DET0122`, session numbers from 1 through 10, and phases are `HI`, `LO`, or `NO`.

The input `ticList.csv` must contain:

```text
ID,Sess,Phase,Type,StartTime,EndTime,Duration
```

Paths are constants near the top of each numbered script and can be changed for another environment.

> **Path note:** `03_extract_features.py` currently writes to `/project/vocaltics/data/wavlm_embeddings/` (singular `project`), while the other data paths use `/projects/vocaltics/data/` (plural `projects`).

## Dependencies

The full pipeline uses:

- Python 3
- pandas
- PyTorch
- torchaudio
- Hugging Face Transformers
- Matplotlib
- Seaborn

A minimal installation command is:

```bash
pip install pandas torch torchaudio transformers matplotlib seaborn
```

Install matching PyTorch and torchaudio builds for the target CPU or CUDA environment. WavLM extraction also requires access to download `microsoft/wavlm-large` from Hugging Face unless it is already cached.

## 1. Data preprocessing

Run:

```bash
python 01_data_preprocessing.py
```

The script:

- reads `ticList.csv` with tic IDs preserved as strings;
- discovers valid WAV files and measures their duration with the standard-library `wave` module;
- retains only the 75 vocal tic IDs hard-coded in `KEEP_TYPES`;
- clips annotations to valid recording boundaries;
- merges overlapping or touching annotations;
- joins multiple retained tic IDs with `+`;
- creates consecutive tic and non-tic metadata rows;
- uses `Type=-1` for non-tic rows; and
- saves `/projects/vocaltics/data/metadata.csv`.

Annotations whose type is not in `KEEP_TYPES` are excluded from the final metadata. Their time ranges are not relabeled as non-tic. If an excluded annotation overlaps a retained tic, the complete merged interval is kept and labeled using only its retained types.

The generated columns are:

```text
AudioPath,ID,Sess,Phase,tic/nontic,Type,StartTime,EndTime,Duration
```

The script prints the total tic count and duration, average tics per participant, average tic duration, counts of segments containing two or three types, and counts by tic type.

## 2. Tic-group assignment

Run:

```bash
python 02_grouping_categories.py
```

`TIC_GROUPS` is a hard-coded mapping derived from the `In-house YGTSS category` column of the `Old Master Tic List` sheet in `Master Tic Record.xlsx`. Tic types in the categories `Facial Movements`, `Hand Movements`, and `Upper Body` were removed from both `KEEP_TYPES` and `TIC_GROUPS`.

The script adds a `Group` column to the existing metadata and overwrites `metadata.csv`:

- non-tic rows use `Type=-1` and `Group=-1`;
- merged tic types from the same group use that group once; and
- merged tic types from different groups use a sorted `GroupA+GroupB` label.

The current mapping contains 75 tic types across 15 individual groups. Barking, Coprolalia, DisinhibitedSpeech, and Humming were removed because each contributed less than one minute in the available data. One summary table expands combined labels so each segment contributes to every individual group it contains. A second table reports the `+` combination categories themselves.

### Tics by individual group

| Group | Tics | Minutes | Participants |
|---|---:|---:|---:|
| Atypical Breathing | 146 | 200.71 | 5 |
| Blocking | 123 | 42.89 | 2 |
| Coughing | 156 | 151.14 | 3 |
| Grunting | 97 | 212.85 | 2 |
| Mouth Movements | 163 | 120.17 | 4 |
| Mouth Noises | 471 | 369.60 | 7 |
| Nose Movements | 158 | 151.86 | 2 |
| Other Animal Noises | 70 | 163.72 | 2 |
| Sniffing | 169 | 222.63 | 4 |
| Snorting | 16 | 53.61 | 2 |
| Syllables | 25 | 25.81 | 2 |
| Throat Clearing | 124 | 183.58 | 4 |

## 3. WavLM feature extraction

Run:

```bash
python 03_extract_features.py
```

The script loads `microsoft/wavlm-large`, converts each recording to mono, resamples it to the model sampling rate, and processes it in 30-second windows. The contextual `last_hidden_state` tensors are concatenated over time.

One tensor is saved per full audio file:

```text
/project/vocaltics/data/wavlm_embeddings/{participant}/{audio_stem}.pt
```

All metadata segments belonging to the same recording reference the same tensor. A copied metadata file containing an `embedding_path` column is saved as:

```text
/project/vocaltics/data/wavlm_embeddings/metadata.csv
```

CUDA is used when available; otherwise extraction runs on CPU.

## Dataset

`bin/dataset.py` defines `SpecDataset`. It receives a metadata file, a list of recording tuples, and a torchaudio transform:

```python
import torchaudio

from bin.dataset import SpecDataset

transform = torchaudio.transforms.MFCC(sample_rate=16000, n_mfcc=40)

dataset = SpecDataset(
    metadata_file="/projects/vocaltics/data/metadata.csv",
    participant_phase_sessions=[
        ("DET0101", "NO", 1),
        ("DET0102", "HI", 2),
    ],
    transform=transform,
    win_len=10,
    p_tics=0.5,
    include_multigroup=True,
)
```

The tuple order is always:

```python
(participant, phase, session)
```

For every `__getitem__` call, the dataset randomly returns:

- with probability `p_tics`, a window with a randomly selected tic centered in it; or
- otherwise, a window sampled entirely inside a non-tic interval that is at least `win_len` seconds long.

PCM WAV frames are loaded with Python's standard `wave` module, converted to mono, and zero-padded when a centered window crosses a file boundary. Torchaudio is used for the feature transforms rather than audio decoding, so the dataset does not require a torchaudio I/O backend. Supported transforms are `Spectrogram`, `MelSpectrogram`, and `MFCC`.

Each item is:

```python
features, tic_type, group_target, has_tic
```

`group_target` is a multi-hot vector using the stable `dataset.group_to_index` mapping built from the full metadata. A single-group tic has one active position, while a combined tic has one active position for every component group. Non-tic samples return `tic_type="-1"`, an all-zero group vector, and `has_tic=False`.

Initialization prints the number of unique tic groups and individual tic types available in the selected split. The full output dimension is available as `dataset.num_groups`, the split-specific count as `dataset.num_groups_available`, and the type count as `dataset.num_types`.

The dataset is stochastic: its index is bounds-checked, but sampling is random and can select metadata rows with replacement.

Set `include_multigroup=False` to prevent tic windows with multiple real groups from being sampled. Script 04 enables this option for the test dataset only; training and validation retain multi-group tic samples.

## Cross-validation splits

`bin/make_splits.py` provides:

- `splits_by_participant(dataframe, K=5, seed=42)`;
- `splits_by_session(dataframe, K=5, seed=42)`; and
- `splits_by_file(dataframe, K=5, seed=42)`.

Every fold has `train`, `val`, and `test` lists containing tuples compatible with `SpecDataset`. Test groups are balanced across folds. Approximately 20% of the non-test recordings are assigned to validation, and the remaining recordings are used for training.

- Participant splitting holds out complete participants.
- Session splitting holds out complete `(participant, session)` pairs across phases.
- File splitting holds out individual audio files.

Splits are deterministic for a fixed seed.

```python
from bin.make_splits import save_split, splits_by_session

folds = splits_by_session(metadata, K=5)
save_split(folds, "splits.json")

train_recordings = folds[1]["train"]
```

Use `load_split("splits.json")` to reload the JSON file. Integer fold keys and recording tuples are restored automatically.

## Models

`bin/models.py` contains three temporal architectures inspired by x-vector systems:

- `TDNN`: time-delay convolution layers followed by statistics pooling;
- `ResNet34`: a one-dimensional temporal ResNet-34; and
- `TCNN`: a dilated temporal convolutional network.

All models are initialized with an input feature dimension and number of tic groups:

```python
from bin.models import TDNN

model = TDNN(input_dim=40, num_groups=19)
tic_logits, group_logits = model(features)
```

They accept `[batch, features, time]`, `[batch, time, features]`, or `[batch, 1, features, time]`. Each uses mean and standard-deviation statistics pooling and returns:

- tic-presence logits shaped `[batch, 2]`; and
- independent tic-group logits shaped `[batch, num_groups]` for multi-label prediction.

## Metrics

`bin/metrics.py` implements metrics using PyTorch only:

```python
from bin.metrics import get_group_metrics, get_tic_metrics

group_accuracy, group_macro_f1 = get_group_metrics(group_pred, group_real)

tic_accuracy, tic_f1, tic_auroc, tic_precision, tic_recall = (
    get_tic_metrics(tic_pred, tic_real)
)
```

The functions accept logits or predicted labels. Multi-hot all-zero group targets are excluded from group metrics; legacy group targets equal to `-1` are also supported. Tic class `1` is the positive class. AUROC supports tied scores and returns `NaN` when only one real class is present.

## 4. Model training

`04_train_tic_detection.py` trains one fold at a time. Its main configuration is near the top of the file:

```python
MODEL_NAME = "TDNN"          # TDNN, ResNet34, or TCNN
SPLIT_BY = "session"         # participant, session, or file
FEAT_NAME = "MFCC"           # Spectrogram, MelSpectrogram, or MFCC
GLOBAL_NAME = f"{MODEL_NAME}_{FEAT_NAME}_by{SPLIT_BY}"
K_FOLDS = 5
```

The current defaults use 40-dimensional MFCCs computed from 80 mel bins, 10-second windows, a 50/50 tic/non-tic sampling probability, batch size 16, Adam with learning rate `0.001`, and 10 epochs. Multi-group tic samples are used for training and validation but excluded from test sampling.

Generate a new split definition and run a fold with:

```bash
python 04_train_tic_detection.py --fold 1 --newsplit
```

Without `--newsplit`, the script reloads the existing `splits.json`. Use the flag only when a new cross-validation assignment should be generated and saved.

Repeat for all folds:

```bash
for fold in 1 2 3 4 5; do
    python 04_train_tic_detection.py --fold "$fold"
done
```

The training loss is the sum of:

- tic-presence cross-entropy over every sample; and
- tic-group binary cross-entropy over tic samples only, allowing multiple active groups.

At the end of each epoch, the script prints loss, tic accuracy, tic F1, tic AUROC, tic precision, tic recall, group accuracy, and group macro-F1 for training and validation. It then saves a checkpoint containing model state, optimizer state, label mapping, configuration, fold number, and validation tic AUROC:

```text
models/detection/{GLOBAL_NAME}/fold{fold}/{epoch}.pt
```

The epoch with the highest validation tic AUROC is also saved as:

```text
models/detection/{GLOBAL_NAME}/fold{fold}/best.pt
```

Every epoch's complete train and validation metric dictionaries are also saved in a structured CSV-formatted log:

```text
models/detection/{GLOBAL_NAME}/fold{fold}.log
```

After training, `best.pt` is reloaded. The validation and test prediction tables are regenerated with that checkpoint, so scripts 10 and 11 evaluate the best validation-AUROC epoch rather than the final epoch:

```text
outputs/detection/{GLOBAL_NAME}/fold{fold}_val.csv
outputs/detection/{GLOBAL_NAME}/fold{fold}_test.csv
```

Prediction tables contain:

```text
tic_type,tic_group,tic_real,tic_pred,tic_probability,group_pred,group_probability
```

The cross-validation definition is saved as `splits.json` when `--newsplit` is used. Otherwise, the latest saved JSON is loaded.

## 5. Training curves

After all folds have been trained, configure `GLOBAL_NAME` and `K_FOLDS` in `05_training_graphs.py`, then run:

```bash
python 05_training_graphs.py
```

The script loads every structured fold log and uses Seaborn to plot mean train/validation loss and tic AUROC per epoch with a 95% confidence interval across folds. The figure is saved to:

```text
graphs/training_curve/{GLOBAL_NAME}.png
```

## 10. Metrics across folds

Configure `GLOBAL_NAME` and `K_FOLDS` in `10_metrics.py`, then run:

```bash
python 10_metrics.py
```

The script loads every validation and test prediction table, calculates metrics separately for each fold, and prints their fold-level mean and sample standard deviation in one table.

## 11. Confusion matrices

Configure `GLOBAL_NAME`, `K_FOLDS`, and `SPLIT_NAME` in `11_make_graphs.py`, then run:

```bash
python 11_make_graphs.py
```

The script combines the configured prediction split across folds and draws two count-based confusion matrices:

- tic versus no tic; and
- tic group for samples containing a real tic.

The figure is saved to:

```text
graphs/{GLOBAL_NAME}/confusion_matrices.png
```

## Repository structure

```text
.
├── 01_data_preprocessing.py
├── 02_grouping_categories.py
├── 03_extract_features.py
├── 04_train_tic_detection.py
├── 05_training_graphs.py
├── 10_metrics.py
├── 11_make_graphs.py
├── Master Tic Record.xlsx
├── README.md
└── bin/
    ├── __init__.py
    ├── dataset.py
    ├── make_splits.py
    ├── metrics.py
    └── models.py
```

## Generated artifacts

Generated files are not required to live in the repository. With the default configuration, the main artifacts are:

```text
/projects/vocaltics/data/metadata.csv
/project/vocaltics/data/wavlm_embeddings/
splits.json
models/detection/TDNN_MFCC_bysession/
outputs/detection/TDNN_MFCC_bysession/
graphs/TDNN_MFCC_bysession/confusion_matrices.png
```
