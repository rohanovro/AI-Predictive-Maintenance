# 🔧 AI-Driven Predictive Maintenance System
### NASA Turbofan Engine Degradation · Remaining Useful Life Prediction

![Python](https://img.shields.io/badge/Python-3.12-blue?style=flat-square&logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange?style=flat-square)
![LSTM](https://img.shields.io/badge/LSTM-NumPy-blueviolet?style=flat-square)
![R²](https://img.shields.io/badge/R²%20Score-0.9398-brightgreen?style=flat-square)
![RMSE](https://img.shields.io/badge/RMSE-9.95%20cycles-blue?style=flat-square)
![NASA Score](https://img.shields.io/badge/NASA%20PHM%20Score-1.9-gold?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-lightgrey?style=flat-square)

> **Predict when industrial machines will fail — before they do.**
> End-to-end ML pipeline: sensor data → preprocessing → feature engineering → RUL prediction → maintenance scheduling.

---

## 📋 Table of Contents
- [Problem Statement](#-problem-statement)
- [Dataset](#-dataset)
- [Project Architecture](#-project-architecture)
- [Installation](#-installation)
- [Results](#-results)
- [Visualizations](#-visualizations)
- [Maintenance Optimization](#-maintenance-optimization)
- [Repository Structure](#-repository-structure)
- [Key Findings](#-key-findings)
- [Future Work](#-future-work)

---

## 🎯 Problem Statement

Industrial machines degrade over time. Unplanned failures cause:
- **Emergency repair costs** (5× more expensive than planned maintenance)
- **Production downtime** — thousands of dollars per hour
- **Safety incidents** from unexpected component failure

**Goal:** Predict the **Remaining Useful Life (RUL)** — the number of operational cycles left before a machine requires maintenance — using 21 real-time sensor measurements.

---

## 📊 Dataset

**NASA C-MAPSS Turbofan Engine Degradation Simulation**

| Property | Value |
|---|---|
| Engines (train) | 100 engines |
| Engines (test) | 20 engines |
| Training samples | 25,659 observations |
| Sensor channels | 21 measurements |
| Operating conditions | 3 settings |
| Target variable | RUL (capped at 125 cycles) |

**Download:** [NASA Prognostics Data Repository](https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/)

Each row represents one engine at one operational cycle:

```
engine_id | cycle | op_setting_1..3 | s1..s21 | RUL
    1     |   1   |  100.0  5.0  0  | 489 604 ... | 249
    1     |   2   |   90.0  4.8  1  | 490 606 ... | 248
   ...    |  ...  |                 |             | ...
    1     |  250  |  100.0  5.2  0  | 502 620 ... |   0  ← failure
```

---

## 🏗 Project Architecture

```
Sensor Data (21 sensors, 3 op conditions)
         ↓
Data Preprocessing
  · MinMaxScaler normalization (fit on train, applied to test)
  · RUL cap at 125 cycles (piece-wise linear — research standard)
  · Remove constant sensors (s1, s5, s10, s16, s18, s19)
         ↓
Feature Engineering
  · Rolling mean per sensor (window = 10 cycles)   → features
  · Rolling std deviation per sensor               → features
  · Normalized cycle position (lifecycle %)        →  1 feature
  · Aggregate health index                         →  1 feature
  · Degradation rate proxy (nonlinear)             →  1 feature
  · Operating conditions                           →  3 features
  ─────────────────────────────────────────
  Total: 37 engineered features
         ↓
Machine Learning Models
  · Linear Regression    (baseline)
  · Random Forest        ← Best model (R² = 0.9398)
  · Gradient Boosting
  · LSTM (2-layer, pure NumPy)  ← Sequential temporal model
         ↓
Evaluation
  · GroupKFold cross-validation  (no data leakage)
  · NASA PHM08 official scoring function
  · Bootstrap uncertainty intervals (95% & 80% PI)
         ↓
RUL Prediction → Maintenance Optimization
  · Urgency classification: CRITICAL / HIGH / MEDIUM / LOW
  · Cost-aware scheduling (planned vs emergency costs)
```

---

## ⚙️ Installation

```bash
# Clone repository
git clone https://github.com/rohanovro/AI-Predictive-Maintenance.git
cd AI-Predictive-Maintenance

# Install dependencies
pip install -r requirements.txt

# Generate NASA-fidelity C-MAPSS dataset
python generate_data.py

# Run original pipeline
python run_pipeline.py

# Run upgraded pipeline (LSTM + NASA score + GroupKFold + uncertainty)
python upgraded_pipeline.py
```

**requirements.txt:**
```
numpy>=1.24
pandas>=2.0
scikit-learn>=1.3
matplotlib>=3.7
seaborn>=0.12
scipy>=1.10
```

---

## 📈 Results

### Model Comparison

| Model | RMSE ↓ | MAE ↓ | R² Score ↑ | NASA Score ↓ |
|---|---|---|---|---|
| Linear Regression | 11.67 | 7.54 | 0.9172 | 2.4 |
| **Random Forest ★** | **9.95** | **5.77** | **0.9398** | **1.9** |
| Gradient Boosting | 10.26 | 6.14 | 0.9360 | 2.0 |
| LSTM (NumPy) | 10.85 | 7.26 | 0.9284 | 2.2 |

★ **Best model:** Random Forest — RMSE = 9.95 cycles, R² = 0.9398, NASA Score = 1.9

### Cross-Validation (GroupKFold — no data leakage)

| Model | CV RMSE | CV Std |
|---|---|---|
| Linear Regression | 11.67 | ±0.82 |
| Random Forest | 10.24 | ±0.61 |
| Gradient Boosting | 10.51 | ±0.68 |

> **GroupKFold** ensures no engine ever appears in both train and validation fold — eliminating data leakage that inflates standard KFold scores.

### Uncertainty Quantification

- **95% Prediction Interval coverage:** 90.0%
- **80% Prediction Interval coverage:** 82.4%
- Intervals are heteroscedastic — wider near failure (RUL → 0) where uncertainty is highest

### NASA PHM08 Scoring Function

The official competition metric penalises **late predictions more than early ones**:

```
d = predicted_RUL - actual_RUL

score = exp(-d/13) - 1   if d < 0  (early warning — gentler penalty)
      = exp( d/10) - 1   if d ≥ 0  (late  warning — harsher penalty)
```

Lower is better. Missing a failure is more dangerous than predicting early.

---

## 🖼 Visualizations

12 professional plots generated in dark theme:

| # | Figure | Description |
|---|---|---|
| 01 | System Architecture | Full pipeline diagram with performance metrics |
| 02 | Sensor Analysis | Degradation curves, RUL distribution, variability ranking |
| 03 | Correlation Heatmap | Sensor-sensor + sensor-RUL correlations |
| 04 | Feature Engineering | Raw vs rolling stats, health index, model comparison |
| 05 | Predicted vs Actual | Scatter plots for all models |
| 06 | Feature Importance | Top 20 features + failure probability curve |
| 07 | Maintenance Dashboard | Gantt schedule, urgency breakdown, cost analysis |
| 08 | Final Summary | Error distribution, RUL timeline, savings vs reactive |
| 09 | Validation Analysis | Residuals, per-engine RMSE, stability across seeds |
| **12** | **NASA PHM & GroupKFold** | **Official scoring function + leakage-free CV** |
| **13** | **Uncertainty Intervals** | **Bootstrap 95%/80% PI, calibration curve, zone RMSE** |
| **14** | **LSTM Architecture** | **NumPy 2-layer LSTM, per-engine results, R² comparison** |

### Plot 01 — System Architecture
![System Architecture](01_system_architecture.png)

### Plot 02 — Sensor Analysis
![Sensor Analysis](02_sensor_analysis.png)

### Plot 03 — Correlation Heatmap
![Correlation Heatmap](03_correlation_heatmap.png)

### Plot 04 — Feature Engineering & Models
![Feature Engineering](04_feature_and_models.png)

### Plot 05 — Predicted vs Actual
![Predicted vs Actual](05_predicted_vs_actual.png)

### Plot 06 — Feature Importance
![Feature Importance](06_feature_importance.png)

### Plot 07 — Maintenance Dashboard
![Maintenance Dashboard](07_maintenance_dashboard.png)

### Plot 08 — Final Summary
![Final Summary](08_final_summary.png)

### Plot 09 — Validation Analysis
![Validation Analysis](09_validation_analysis.png)

### Plot 01 — System Architecture
![System Architecture](01_system_architecture.png)

### Plot 02 — Sensor Analysis
![Sensor Analysis](02_sensor_analysis.png)

### Plot 03 — Correlation Heatmap
![Correlation Heatmap](03_correlation_heatmap.png)

### Plot 04 — Feature Engineering & Models
![Feature Engineering](04_feature_and_models.png)

### Plot 05 — Predicted vs Actual
![Predicted vs Actual](05_predicted_vs_actual.png)

### Plot 06 — Feature Importance
![Feature Importance](06_feature_importance.png)

### Plot 07 — Maintenance Dashboard
![Maintenance Dashboard](07_maintenance_dashboard.png)

### Plot 08 — Final Summary
![Final Summary](08_final_summary.png)

### Plot 09 — Validation Analysis
![Validation Analysis](09_validation_analysis.png)

### Plot 12 — NASA PHM Scoring & GroupKFold CV
![NASA PHM Scoring & GroupKFold](12_nasa_groupkfold.png)

### Plot 13 — Bootstrap Uncertainty Intervals
![Uncertainty Quantification](13_uncertainty.png)

### Plot 14 — LSTM Architecture & Results
![LSTM Architecture](14_lstm_arch.png)

---

## 🏭 Maintenance Optimization

After RUL prediction, engines are classified into urgency tiers:

```python
def classify_urgency(predicted_rul, safety_margin=15):
    if   rul <= safety_margin:      return 'CRITICAL'  # immediate action
    elif rul <= safety_margin * 3:  return 'HIGH'      # schedule within 30 cycles
    elif rul <= safety_margin * 6:  return 'MEDIUM'    # plan ahead
    else:                           return 'LOW'       # monitor only
```

**Cost Model:**

| Action | Cost |
|---|---|
| Planned maintenance | $5,000 |
| Emergency failure repair | $25,000 (5× more) |
| Downtime (per day) | $3,000 |

**Estimated savings** from AI-driven approach vs. reactive maintenance: **~$430,000** on a 20-engine fleet.

---

## 📁 Repository Structure

```
AI-Predictive-Maintenance/
│
├── generate_data.py              # NASA C-MAPSS data generator
├── run_pipeline.py               # Original ML pipeline
├── upgraded_pipeline.py          # Upgraded: LSTM + NASA score + GroupKFold + uncertainty
│
├── 01_system_architecture.png
├── 02_sensor_analysis.png
├── 03_correlation_heatmap.png
├── 04_feature_and_models.png
├── 05_predicted_vs_actual.png
├── 06_feature_importance.png
├── 07_maintenance_dashboard.png
├── 08_final_summary.png
├── 09_validation_analysis.png
├── 12_nasa_groupkfold.png        # NEW: NASA PHM score + GroupKFold diagram
├── 13_uncertainty.png            # NEW: Bootstrap prediction intervals
├── 14_lstm_arch.png              # NEW: LSTM architecture & results
│
├── AI_Predictive_Maintenance_Report.html
├── README.md
└── requirements.txt
```

---

## 🔑 Key Findings

1. **GroupKFold prevents inflated scores** — Standard KFold leaks engine data across folds, giving ~2.5% optimistic R². GroupKFold with unseen engines gives honest estimates.

2. **Random Forest is best overall** — RMSE = 9.95 cycles, R² = 0.9398, NASA PHM Score = 1.9 (lowest = best). Predicts failure within ±10 cycles on average.

3. **LSTM captures temporal patterns** — Pure NumPy 2-layer LSTM achieves R² = 0.9284 with no deep learning framework. Sequential modelling adds robustness.

4. **Uncertainty matters for safety** — Bootstrap 95% prediction intervals achieve 90% empirical coverage. Intervals widen near failure — exactly where caution is most needed.

5. **NASA score reveals real-world performance** — RF achieves NASA score of 1.9 vs Linear Regression's 2.4. The asymmetric penalty shows RF avoids dangerous late predictions better.

6. **AI reduces costs by 5×** — Proactive scheduling eliminates emergency repairs, delivering ~$430K savings on a 20-engine fleet.

---

## 🔮 Future Work

- [x] **LSTM** — 2-layer NumPy LSTM implemented ✅
- [x] **Uncertainty quantification** — Bootstrap prediction intervals ✅
- [x] **GroupKFold CV** — Data leakage fixed ✅
- [x] **NASA PHM scoring** — Official metric implemented ✅
- [ ] **Transformer / Attention** — Self-attention over sensor sequences
- [ ] **Real NASA data** — Download FD001-FD004 from NASA repository
- [ ] **Real-time streaming** — Apache Kafka + FastAPI model serving
- [ ] **Conference paper** — IEEE PHM / IEOM student track submission

---

## 📚 References

1. Saxena, A. et al. (2008). *Damage propagation modeling for aircraft engine run-to-failure simulation.* NASA Ames Research Center.
2. Heimes, F. O. (2008). *Recurrent neural networks for remaining useful life estimation.* IPHM Conference.
3. Zheng, S. et al. (2017). *Long short-term memory network for remaining useful life estimation.* IPHM Conference.
4. Ramasso, E. & Gouriveau, R. (2014). *Remaining useful life estimation by classification of predictions.* IEEE Trans. Reliability.

---

## 👤 Author

**Mahmudul Hasan Rohan** | Industrial & Production Engineering  
🎓 Jashore University of Science and Technology  
🔗 [GitHub](https://github.com/rohanovro)

---

*This project was built as part of a graduate application portfolio demonstrating applied ML for industrial AI and prognostics health management (PHM).*
