# HemoWick

Research code for estimating hemoglobin (Hb, g/dL) and hematocrit (Hct, %)
from time-series blood-spot images.

This package includes model definitions, training, checkpoint evaluation and
experiment settings. Study images, participant split manifests and trained study
checkpoints are not included. Research use only.

## Files

| File | Purpose |
| --- | --- |
| `README.md` | Installation, data format and commands |
| `requirements.txt` | Direct dependency versions used for package checks |
| `config.py` | Shared parameters and built-in Fig. 2, 3 and 4 settings |
| `main.py` | Commands, experiment comparisons and synthetic demonstration |
| `model.py` | Image backbones, patch selection and temporal regression |
| `data.py` | Manifest validation and image-sequence loading |
| `train.py` | Training and validation-based checkpoint selection |
| `test.py` | Checkpoint evaluation and output generation |
| `metrics.py` | Regression, ROC and participant-bootstrap statistics |
| `plots.py` | Regression and ROC plots |

Keep these ten files in the same directory. Figure settings are embedded in
`config.py`; no separate configuration or demonstration files are required.

## Install and try

Use Python 3.10. The package was checked on Windows with Python 3.10.4,
PyTorch 2.6.0 and torchvision 0.21.0, using existing installed dependencies.
A clean installation, other operating systems and GPU training were not tested.
These versions describe package checks, not the historical study environment.

```bash
git clone https://github.com/Jeongsoo-0815/HemoWick.git
cd HemoWick
python -m venv .venv
```

Activate with `.venv\Scripts\activate` in Windows Command Prompt, or
`source .venv/bin/activate` on Linux/macOS, then run:

```bash
python -m pip install -r requirements.txt
python main.py demo
```

The demo generates ten synthetic sequences, trains a small ResNet18-LSTM for
one epoch on CPU, and saves a checkpoint, predictions, metrics and plots under
`demo_output/`. No pretrained weights or study data are downloaded by the demo.
Use a new `--output` directory to repeat it. This is an execution check, not
clinical validation or reproduction of the paper. Installation time was not measured.

## Study data

Supply authorized images and CSV files locally. Each CSV row is one sequence:

```text
sample_id,participant_id,split,hb,hct,frame_dir
```

Use unique `sample_id` values, de-identified `participant_id` values and the
study's fixed `train`, `val` and `test` splits. All measurements of one participant
must belong to one split. Labels are numeric Hb in g/dL and Hct in %.
`frame_dir` is relative to the CSV, or an absolute image-directory path.
Default frames are `0s.jpg`, `10s.jpg`, ..., `120s.jpg` (13 frames).
Alternatively, `frames` contains semicolon-separated file paths in time order,
with the same number of entries as `data.frame_times`.

Optional metadata: `cohort`, `site`, `specimen`, `device`, `sex`, `scd_status`,
`batch`. For nonclinical preparations, use an appropriate independent
preparation/donor identifier and grouping consistent with the study design.

Built-in presets expect `data/fig2_standard.csv`, `data/fig3_korea.csv` and
`data/fig4_senegal.csv` beside these source files. No study CSVs are provided.

## Commands

```bash
# Validate settings and image inputs
python main.py validate-config --experiment fig3
python main.py check-data --experiment fig3

# Train using fixed Korea train/validation/test splits
python main.py run --experiment fig3

# Run the Fig. 2 model-comparison grid
python main.py run --experiment fig2

# Evaluate a frozen study checkpoint on the Senegal test cohort
python main.py evaluate --experiment fig4 --checkpoint runs/STUDY_RUN/best.pt --output runs/senegal_evaluation
```

Replace the example checkpoint path with the actual file. Evaluation accepts
a test-only manifest. Match architecture, frame times and preprocessing to the
checkpoint. `run` trains on the selected cohort's training rows; `evaluate`
performs frozen-model evaluation. Checkpoint-initialized fine-tuning is not implemented.

Individual entry points are also available:

```bash
python train.py --experiment fig3
python test.py --experiment fig4 --checkpoint runs/STUDY_RUN/best.pt
```

Use repeated `--set KEY=VALUE` to override settings, for example
`--set data.batch_size=4 --set data.num_workers=0`. An optional `--config`
external YAML file replaces the built-in preset. Default study training downloads
torchvision ImageNet weights on first use; GPU training is recommended for the
full ViT sequence model. Keep local data, run outputs and checkpoints out of
source uploads unless separately approved for sharing.

## Implementation and outputs

The default is ViT-B/16 with a one-layer, 256-unit LSTM and a 128-unit regression
head predicting Hb and Hct. Loss is ordinary mean squared error in original
target units. The default pruning ratio is 0.4. Adaptive selection ranks patch
tokens by cosine similarity to the class token before the first Transformer
block. No separate learned token-scoring network is implemented. The inherited
name `DynamicViTBackbone` does not identify the official DynamicViT implementation.

Runs save resolved settings, environment versions, epoch history, the best
checkpoint, predictions, metrics, bootstrap intervals and plots. Legacy filenames
containing `image` refer to input sequences. Repeated predictions are averaged
within configured participant groups; bootstrap resampling uses participants.
Subgroup outputs are point estimates. Undefined evaluation metrics are saved as
`null`. Analysis thresholds are research settings, not clinical guidance.

This package combines the supplied code archive with its matching supporting
modules. Exact manuscript-result reproduction requires the actual data, splits,
preprocessing, checkpoints and final settings. Agreement with the manuscript's
methods and results has not been established by the synthetic checks.

## Availability and reuse

Repository: https://github.com/Jeongsoo-0815/HemoWick

No software license is specified in this package. Reuse permission should be
obtained from the rights holder; dependencies retain their own licenses.
Study-data access conditions should be specified in the manuscript's Data
Availability statement.
