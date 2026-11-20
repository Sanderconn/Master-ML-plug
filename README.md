# A Hybrid Machine Learning Approach for Health-State Monitoring of Pipes During Plug Formation

Code for the master's thesis *A Hybrid Machine Learning Approach for Health-State Monitoring of Pipes During Plug Formation* (Sander Conn Svendsby Ødegaard, Joint Master's Programme in Software Engineering, HVL and UiB, 2026).

The pipeline estimates the health state of a pipe from pressure measurements alone. An Isolation Forest trained only on normal operation produces an anomaly score. That score, together with four baseline-relative pressure features, is then fed to an LSTM that classifies each 60-second window into one of four health states:

| State | Meaning |
|-------|---------|
| 0 | Normal operation |
| 1 | Early deviation |
| 2 | Partial restriction |
| 3 | Critical restriction / plug |

The model is trained on ice-slurry plugging experiments from the HVL flow loop and evaluated without retraining on the public 3W dataset of undesirable events in offshore oil wells.

## Repository structure

```
.
├── data/
│   ├── raw/                  # Original experiment logs 
│   ├── split/                # Runs cut out of the raw logs
│   ├── labeled/              # Runs with derived features and health-state labels
│   ├── labeledAndAnomaly/    # Labeled runs with the Isolation Forest anomaly score added
│   ├── BigPlug/              # Long high-frequency plug run (1003), split into several CSV files
│   └── BigNoPlug/            # Long high-frequency no-plug run (1703), split into several CSV files
├── models/
│   ├── anomaly_isolation_forest.joblib   # Trained Isolation Forest
│   ├── anomaly_scaler.joblib             # Scaler used before the Isolation Forest
│   ├── lstm.pth                          # Final LSTM (5 features)
│   ├── lstm_scaler.joblib                # Scaler used before the LSTM
│   ├── no_a_lstm.pth                     # Ablation LSTM without the anomaly score (4 features)
│   └── no_a_lstm_scaler.joblib           # Scaler for the ablation LSTM
├── notebooks/
│   ├── split_raw.ipynb
│   ├── standardize_and_labeling.ipynb
│   ├── unsupervised_scoring.ipynb
│   ├── lstm.ipynb
│   ├── lstm_no_anom.ipynb
│   └── comparison.ipynb
├── Verification/
│   ├── 3w/
│   │   ├── data/             # 3W runs, one folder per class (0–8); not included
│   │   └── applying_model_to_3W.ipynb
│   └── outputs/
│       ├── 0/ … 8/           # Per-run results and plots, one folder per event class
│       ├── figures/          # Combined figures used in the thesis
│       ├── st_app.py         # Streamlit app for browsing the per-run plots
│       ├── summary.csv       # One row per run with all metrics
│       └── table_*.csv       # Summary tables for the Results chapter
└── Description of cases.docx   # Documentation of the experimental runs
```

## Pipeline

Run the notebooks in this order. Each step reads the output folder of the previous one.

### 1. `split_raw.ipynb` → `data/split/`

Reads the raw log files and checks them for basic problems: missing values, timestamps that are not monotonically increasing, and the runtime of each log. Each log is then split into individual experimental runs according to the experiment documentation (`Description of cases.docx`). The resulting runs are saved to `data/split/`.

### 2. `standardize_and_labeling.ipynb` → `data/labeled/`

- Gives every run the same column schema and resamples all runs to the same frequency (1 Hz).
- Computes the pressure-derived features: test-section differential pressure, pump-to-test-section differential pressure, total differential pressure and the pressure ratio.
- Converts these into baseline-relative features using the median of the first 60 seconds of each run:
  - `rel_tot_dp_change`
  - `rel_ts_dp_change`
  - `rel_p_ts_dp_change`
  - `Relative_Ratio_Dev` (RRD)
- Assigns the health-state labels (0–3) to each run.

The labeled runs are saved to `data/labeled/`.

### 3. `unsupervised_scoring.ipynb` → `data/labeledAndAnomaly/`, `models/`

Trains the Isolation Forest on the four baseline-relative features, using only the training portion (first 60 %) of the no-plug runs. It then adds an `anomaly_score` column to every run (positive = normal, negative = outlier) and saves the runs to `data/labeledAndAnomaly/`. The trained model and its scaler are saved as `models/anomaly_isolation_forest.joblib` and `models/anomaly_scaler.joblib`.

### 4. `lstm.ipynb` → `models/`

Trains the LSTM classifier on the four baseline-relative features plus the anomaly score:

- windows of 60 observations
- one unidirectional LSTM layer with 32 hidden units
- dropout before the output layer
- weighted cross-entropy loss

Each state section of every run is split chronologically into 60 % training, 20 % validation and 20 % test data. The model with the lowest validation loss is kept. It is saved as `models/lstm.pth` together with its scaler `models/lstm_scaler.joblib`.

### Additional notebooks

- **`lstm_no_anom.ipynb`**: the ablation experiment. Trains the same LSTM without the anomaly score (four features) and saves `models/no_a_lstm.pth` and `models/no_a_lstm_scaler.joblib`.
- **`comparison.ipynb`**: plots the RRD, the anomaly score and the health-state labels against the measured flow rate. Used for the label validation in the thesis.

## Using a trained model on new data

To apply the trained pipeline to a new run:

1. Compute the same four baseline-relative features. Use the median of the first 60 s of the run, which must be normal operation, as the baseline.
2. Transform them with `anomaly_scaler.joblib` and compute the anomaly score with `anomaly_isolation_forest.joblib` (`decision_function`).
3. Add the anomaly score as a fifth feature, transform with `lstm_scaler.joblib`, and classify sliding 60-sample windows with `lstm.pth`.

The thesis proposes a sustained prediction as the alert: State 2 or higher for at least 60 consecutive samples.

# External evaluation (3W dataset)

The 3W data is not included in this repository. This thesis uses the original version (1.0.0, 1,984 instances) of the 3W dataset, which can be downloaded from https://github.com/ricardovvargas/3w_dataset. Newer versions are maintained at https://github.com/petrobras/3W, but their instances and labels differ from the ones used here.

The trained pipeline is applied without retraining to 36 runs from the 3W dataset: four runs from each of the nine classes (normal operation and eight undesirable event types). Everything for this step is in `Verification/`.

### `Verification/3w/applying_model_to_3W.ipynb`

Run the notebook from inside `Verification/3w/`. All paths are relative to that folder: it reads the runs from `data/`, the models from `../../models/`, and writes to `../outputs/`. The runs used are listed in the `RUNS` dictionary at the top.

For each run, the notebook:

1. Maps the 3W pressure sensors onto the flow-loop features:
   - `P-PDG` → pump outlet pressure
   - `P-TPT` → test-section inlet pressure
   - `P-MON-CKP` → test-section outlet pressure

   It then computes the same four baseline-relative features, using the first 60 s of the run as the baseline. Scattered missing pressure values are filled forward and backward.
2. Computes the anomaly score with the saved Isolation Forest and predicts a health state for every sample with the saved LSTM.
3. Evaluates the predictions against the 3W labels (normal, transient, steady). A prediction counts as *sustained* if it stays at or above a state for 60 consecutive samples (`HOLD`). The event window starts 300 s before the first transient label (`TOL`). The settings are at the top of the notebook.

The second part of the notebook builds the combined figures used in the thesis (one per class) from the saved per-run series and writes them to `outputs/figures/` as PDF and PNG.

### Outputs

- `outputs/<class>/`: for each run, a subfolder with plots of the RRD, anomaly score, predicted state and 3W label.
- `outputs/summary.csv`: one row per run with all metrics, timings and diagnostics.
- `outputs/table_*.csv`: summary tables built from `summary.csv`:
  - `table_normal_only.csv`: results for the normal-operation runs
  - `table_events.csv`: response per event class and data source
  - `table_outcomes_events.csv` and `table_outcomes_normal_only.csv`: outcome counts
  - `table_excluded_not_normal_start.csv`: runs excluded because they do not start in normal operation
- `outputs/figures/`: the figures used in the thesis.

### Browsing the plots

`st_app.py` is a small Streamlit app for looking through the per-run plots in `outputs/0`–`outputs/8`:

```bash
cd Verification/outputs
streamlit run st_app.py
```


## Requirements

Python 3 with the packages in `requirements.txt`:

```bash
pip install -r requirements.txt
```
