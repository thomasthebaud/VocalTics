# VocalTics

This repository contains a preprocessing, feature extraction, training, and evaluation pipeline for vocal-tic detection and tic-group classification. The protected audio and annotation data are not included in the repository.

## Pipeline overview

Run the numbered scripts in this order:

1. `01_data_preprocessing.py` creates tic and non-tic segment metadata.
2. `02_grouping_categories.py` assigns an in-house YGTSS group to each tic segment.
3. `03_extract_features.py` optionally extracts one WavLM Base Plus tensor per full recording.
4. `04_generate_new_split.py` generates and saves the cross-validation folds.
5. `05_train_tic_detection.py` trains one detection cross-validation fold.
6. `06_train_tic_segmentation.py` trains one segmentation cross-validation fold.
7. Run the relevant training script once for every fold.
8. `07_training_graphs.py` plots training curves across folds.
9. `10_metrics.py` aggregates validation and test detection metrics across folds.
10. `11_make_graphs.py` creates confusion matrices from saved predictions.

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

## Dependencies

The full pipeline uses:

- Python 3
- pandas
- PyTorch
- torchaudio
- Hugging Face Transformers
- tqdm
- SoundFile with libsndfile support
- Matplotlib
- Seaborn

A minimal installation command is:

```bash
pip install pandas torch torchaudio transformers tqdm soundfile matplotlib seaborn
```

The pip package is named `soundfile`; it provides Python access to libsndfile.
On common 64-bit Linux installations, its binary wheel also includes the
libsndfile library. If pip installs from source instead, libsndfile must be
provided by the system package manager.

Install matching PyTorch and torchaudio builds for the target CPU or CUDA environment. WavLM extraction also requires access to download `microsoft/wavlm-base-plus` from Hugging Face unless it is already cached.

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

The script loads `microsoft/wavlm-base-plus`, converts each recording to mono, resamples it to the model sampling rate, and processes it in 30-second windows. The contextual 768-dimensional `last_hidden_state` tensors are concatenated over time.

One tensor is saved per full audio file:

```text
/projects/vocaltics/data/wavlm_embeddings/{participant}/{audio_stem}.pt
```

All metadata segments belonging to the same recording reference the same tensor. A copied metadata file containing an `embedding_path` column is saved as:

```text
/projects/vocaltics/data/wavlm_embeddings/metadata.csv
```

CUDA is used when available; otherwise extraction runs on CPU.

## Dataset

`bin/detection_datasets.py` defines a shared `Detection_Dataset` parent with two
feature-specific children. `SpecDataset` loads PCM audio and computes a
torchaudio transform:

```python
import torchaudio

from bin.detection_datasets import SpecDataset

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

`WavLmDataset` uses the same sampling and target logic but loads the
recording-level tensors created by script 03:

```python
from bin.detection_datasets import WavLmDataset

dataset = WavLmDataset(
    metadata_file="/projects/vocaltics/data/wavlm_embeddings/metadata.csv",
    participant_phase_sessions=[("DET0101", "NO", 1)],
    win_len=10,
    p_tics=0.5,
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

Detection dataset items are:

```python
features, tic_type, group_target, has_tic
```

`bin/segmentation_datasets.py` similarly defines a shared
`Segmentation_Dataset` parent and the children `SpecDataset` and
`WavLmDataset`. Both default to `p_tics=0.2` and return only:

```python
features, frame_labels
```

`frame_labels` is a boolean tensor with the same time length as the features
and marks every tic annotation overlapping the sampled window.

Both WavLM datasets require an `embedding_path` metadata column. They load the
full `(time, embedding_dim)` tensor, select the sampled recording interval,
transpose it to `(embedding_dim, time)`, and pad boundaries with zeros. WavLM
features default to 50 frames per second, producing 500 frames for a 10-second
window; this can be changed with `frames_per_second`.

`group_target` is a multi-hot vector using the stable `dataset.group_to_index` mapping built from the full metadata. A single-group tic has one active position, while a combined tic has one active position for every component group. Non-tic samples return `tic_type="-1"`, an all-zero group vector, and `has_tic=False`.

Initialization prints the number of unique tic groups and individual tic types available in the selected split. The full output dimension is available as `dataset.num_groups`, the split-specific count as `dataset.num_groups_available`, and the type count as `dataset.num_types`.

The dataset is stochastic: its index is bounds-checked, but sampling is random and can select metadata rows with replacement.

Set `include_multigroup=False` to prevent tic windows with multiple real groups
from being sampled. Script 05 enables this option for validation and test
detection datasets.

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

The normal workflow is to generate the split once with:

```bash
python 04_generate_new_split.py --split-by session --k-folds 5 --seed 42
```

The accepted split units are `participant`, `session`, and `file`. The script
saves `splits.json` and prints a table describing the recordings, participants,
participant-session pairs, phases, and participant IDs in every train,
validation, and test split. Training scripts only load this saved file and
never overwrite it.

## Models

`bin/detection_models.py` contains three temporal architectures inspired by x-vector systems:

- `TDNN`: time-delay convolution layers followed by statistics pooling;
- `ResNet34`: a one-dimensional temporal ResNet-34; and
- `TCNN`: a dilated temporal convolutional network; and
- `CNN`: three temporal convolution layers followed by adaptive max pooling
  and separate linear tic-presence and tic-group heads.

All models are initialized with an input feature dimension and number of tic groups:

```python
from bin.detection_models import TDNN

model = TDNN(input_dim=40, num_groups=19)
tic_logits, group_logits = model(features)
```

They accept `[batch, features, time]`, `[batch, time, features]`, or `[batch, 1, features, time]`. Each uses mean and standard-deviation statistics pooling and returns:

- tic-presence logits shaped `[batch, 2]`; and
- independent tic-group logits shaped `[batch, num_groups]` for multi-label prediction.

`bin/segmentation_models.py` provides three frame-level tic segmentation models:

- `BiLSTM`;
- `CNN`; and
- `CNN_BiLSTM`.

They accept features shaped `[batch, feature_dim, time]` and return one raw
tic-presence logit per input frame with shape `[batch, time]`. These logits can
be passed directly to `torch.nn.BCEWithLogitsLoss` during training.

## Metrics

`bin/detection_metrics.py` implements detection metrics using PyTorch only:

```python
from bin.detection_metrics import get_group_metrics, get_tic_metrics

group_accuracy, group_macro_f1 = get_group_metrics(group_pred, group_real)

tic_accuracy, tic_f1, tic_auroc, tic_precision, tic_recall = (
    get_tic_metrics(tic_pred, tic_real)
)
```

The functions accept logits or predicted labels. Multi-hot all-zero group targets are excluded from group metrics; legacy group targets equal to `-1` are also supported. Tic class `1` is the positive class. AUROC supports tied scores and returns `NaN` when only one real class is present.

`bin/segmentation_metrics.py` provides frame-wise accuracy, F1, and AUROC,
along with segment-wise accuracy and F1:

```python
from bin.segmentation_metrics import get_segmentation_metrics

frame_accuracy, frame_f1, frame_auroc, segment_accuracy, segment_f1 = (
    get_segmentation_metrics(frame_logits, frame_targets)
)
```

Frame-wise metrics treat each frame as one prediction. Segment-wise metrics
treat the first dimension as the batch of segments: a segment is positive when
at least one of its frames is positive. Floating-point inputs are treated as
logits by default; pass `from_logits=False` for probabilities.

## 5. Detection training

`05_train_tic_detection.py` trains one fold at a time. The model, split strategy,
and input features can be selected from the command line. For example:

```bash
python 05_train_tic_detection.py --fold 1 \
    --model-name TDNN --split-by session --feat-name MFCC
```

The accepted models are `TDNN`, `ResNet34`, `TCNN`, and `CNN`; split strategies
are `participant`, `session`, and `file`; and features are `Spectrogram`,
`MelSpectrogram`, `MFCC`, and `WavLM`. The existing constants remain the
command-line defaults. The global experiment name is built as
`{MODEL_NAME}_{FEAT_NAME}_by{SPLIT_BY}` from the selected arguments.

When `--feat-name WavLM` is selected, the trainer uses `WavLmDataset`, reads
`/projects/vocaltics/data/wavlm_embeddings/metadata.csv`, and configures the
model for WavLM Base Plus's 768-dimensional embeddings. Script 03 must be run
before WavLM training. Other feature names use `SpecDataset` and compute their
transform from audio during sampling.

The current defaults use 40-dimensional MFCCs computed from 80 mel bins,
10-second windows, a 50/50 tic/non-tic sampling probability, batch size 128,
Adam with learning rate `0.0001`, and 10 epochs. Multi-group tic samples are
used for training and excluded from validation and test sampling.

Generate `splits.json` first, then run a detection fold with:

```bash
python 04_generate_new_split.py --split-by session
python 05_train_tic_detection.py --fold 1 --split-by session
```

The detection trainer always reloads the existing `splits.json` and reports an
error directing you to script 04 when it is missing.

Repeat for all folds:

```bash
for fold in 1 2 3 4 5; do
    python 05_train_tic_detection.py --fold "$fold"
done
```

The provided launcher runs its configured detection experiment over all folds:

```bash
sh launch_all_detection_trainings.sh
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

The cross-validation definition is always loaded from `splits.json`.

## 6. Segmentation training

`06_train_tic_segmentation.py` follows the same fold and command-line structure
for frame-level segmentation. For example:

```bash
python 06_train_tic_segmentation.py --fold 1 \
    --model-name BiLSTM --split-by session --feat-name MFCC
```

After `splits.json` has been created by script 04, run the configured segmentation experiment
over every fold with:

```bash
sh launch_all_segmentation_trainings.sh
```

The available models are `BiLSTM`, `CNN`, and `CNN_BiLSTM`. Spectral features
use the segmentation `SpecDataset`; `--feat-name WavLM` uses `WavLmDataset`
and the embedding metadata created by script 03. Both use `p_tics=0.2` and
`BCEWithLogitsLoss`. Each epoch logs frame accuracy, frame F1, frame AUROC,
segment accuracy, and segment F1.
The checkpoint with the lowest validation loss is saved as `best.pt`.
Models, logs, and validation/test prediction tables are written under:

```text
models/segmentation/{GLOBAL_NAME}/
outputs/segmentation/{GLOBAL_NAME}/
```

Prediction CSV files contain one row per feature frame, including its segment
ID, frame truth/prediction/probability, and the reduced segment truth and
prediction.

Both training scripts use `bin/training_functions.py` for their common command
line arguments, feature transforms, saved-fold loading, DataLoader creation,
metric logging, and elapsed/estimated time reporting. Detection- and
segmentation-specific losses and evaluation remain in their respective scripts.

## 7. Training curves

After training one or more experiments, run:

```bash
python 07_training_graphs.py
```

The script discovers every experiment directory under `models/detection/` and
`models/segmentation/`, then loads every available `fold*.log`. It uses
Seaborn to plot mean train/validation loss and AUROC per epoch with a 95%
confidence interval across folds. Detection plots use tic AUROC and
segmentation plots use frame AUROC. Figures are saved to:

```text
graphs/training_curve/{detection|segmentation}/{GLOBAL_NAME}.png
```

## 8. Network parameter counts

Run:

```bash
python 08_networks_weights.py
```

The script discovers every detection and segmentation experiment with a
`fold1/` directory. It loads `best.pt`, or the latest numbered epoch when no
best checkpoint exists, and prints an aligned table containing each experiment
name and its number of model parameters.

## 10. Metrics across folds

```bash
python 10_metrics.py
```

The script discovers every experiment folder under both `outputs/detection/`
and `outputs/segmentation/`. It automatically uses every available
`fold*_val.csv` and `fold*_test.csv`, so the fold count does not need to be
configured in the script. Every result is formatted as `mean (±std)` across
the contributing folds.

For detection experiments, it prints one table whose columns cover validation
and test results for all, seen, and unseen tic types. It uses `splits.json` and
the source metadata to identify types present or absent in each fold's training
split. A combined TicID is considered seen only when all of its component types
occur in training. Non-tic rows are included in both subsets so binary
tic-detection metrics remain defined. Rows whose real or predicted group
contains `+` are excluded from group metrics but remain part of tic-detection
metrics.

For segmentation experiments, it reconstructs each segment from its frame
rows and reports validation and test frame accuracy, frame F1, frame AUROC,
segment accuracy, and segment F1.

## 11. Confusion matrices

Run:

```bash
python 11_make_graphs.py
```

The script discovers every experiment folder under `outputs/detection/` and
`outputs/segmentation/`, then calls the reusable
`bin.graphs.confusion_matrix.make_confusion_matrix()` helper with the relevant
task. It combines all available validation and test prediction files across
folds.

For each detection experiment, it draws four confusion matrices:

- test and validation tic-versus-no-tic matrices, annotated with sample counts and percentages normalized within each real-label row; and
- test and validation tic-group matrices after excluding rows whose real or predicted group is `-1` or contains `+`. Zero cells remain white, while every positive count is colored using a logarithmic scale.

For each segmentation experiment, it draws test and validation matrices at
both the frame and segment levels. These binary matrices show sample counts and
row-normalized percentages.

Figures are saved to:

```text
graphs/detection/{GLOBAL_NAME}/confusion_matrices.png
graphs/segmentation/{GLOBAL_NAME}/confusion_matrices.png
```

## Repository structure

```text
.
├── 01_data_preprocessing.py
├── 02_grouping_categories.py
├── 03_extract_features.py
├── 04_generate_new_split.py
├── 05_train_tic_detection.py
├── 06_train_tic_segmentation.py
├── 07_training_graphs.py
├── 08_networks_weights.py
├── 10_metrics.py
├── 11_make_graphs.py
├── Master Tic Record.xlsx
├── README.md
├── launch_all_detection_trainings.sh
├── launch_all_segmentation_trainings.sh
└── bin/
    ├── __init__.py
    ├── detection_datasets.py
    ├── graphs/
    │   ├── __init__.py
    │   └── confusion_matrix.py
    ├── make_splits.py
    ├── detection_metrics.py
    ├── detection_models.py
    ├── segmentation_datasets.py
    ├── segmentation_metrics.py
    ├── segmentation_models.py
    └── training_functions.py
```

## Generated artifacts

Generated files are not required to live in the repository. With the default configuration, the main artifacts are:

```text
/projects/vocaltics/data/metadata.csv
/projects/vocaltics/data/wavlm_embeddings/
splits.json
models/detection/TDNN_MFCC_bysession/
outputs/detection/TDNN_MFCC_bysession/
outputs/segmentation/BiLSTM_MFCC_bysession/
graphs/detection/TDNN_MFCC_bysession/confusion_matrices.png
graphs/segmentation/BiLSTM_MFCC_bysession/confusion_matrices.png
```
