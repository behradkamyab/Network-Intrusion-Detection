
##  EDA Report & Data Engineering Action Plan


### 1. Notes from EDA

* **Shape & Memory:**
  * Dataset has **225,745 rows** and **79 columns** (~136.1 MB in RAM).
  * 78 features are numerical (`int64` and `float64`); only the target column (`Label`) is categorical text.
* **Target Distribution:**
  * **DDoS:** 128,027 samples (~56.71%)
  * **BENIGN:** 97,718 samples (~43.29%)
  * *Takeaway:* this specific DDoS capture is well-balanced. Standard binary classification metrics (Accuracy, ROC-AUC, Binary Cross-Entropy) are okay to be used here, we will track Precision, Recall, and F1 to monitor false alarms.
* **Data Quality:**
  * **64 Infinite values (`Inf` / `-Inf`)** present in numerical columns (caused by zero-duration flows when computing rates like `Flow Bytes/s`).
  * **4 Missing values (`NaN`)** found in `Flow Bytes/s`.
  * *Takeaway:* Leaving these in will cause gradient updates to explode into `NaN` during backpropagation.
* **Multicollinearity:**
  * The correlation heatmap reveals strong blocks of collinear features:
    * `Total Fwd Packets` $\leftrightarrow$ `Subflow Fwd Packets` ($r \approx 1.0$)
    * `Total Length of Fwd Packets` $\leftrightarrow$ `Subflow Fwd Bytes` ($r \approx 1.0$)
  * Inter-arrival time (IAT) aggregates share heavy linear relationships.
  * Several bulk rate columns show zero variance (visible as blank/constant strips in the heatmap), which provide zero signal to a neural network.

---

### 2. Action Plan

1. **Strip Whitespace:** Standardize all column names using `.str.strip()`
2. **Fix INF and NaN:**
    * Convert all `+np.inf` and `-np.inf` entries directly to `np.nan`.
    * Drop rows containing `NaN` values. Dropping ~68 corrupted rows out of 225,745 preserves **99.97%** of our training distribution.
3. **Drop Zero-Variance & Leakage Columns:**
    * Remove zero-variance constant features (e.g., bulk rate fields) that contain no predictive variance.
4. **Target Encoding:**
    * Map `BENIGN` $\rightarrow$ `0` (Normal traffic).
    * Map `DDoS` $\rightarrow$ `1` (Malicious attack).
5. **Export Clean Artifact:**
    * Write the cleaned dataset to`data/processed/clean_ddos_traffic.csv`.
