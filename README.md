# DengueGuard: AI for Urban Dengue Outbreak Forecasting

**EDU AI Engineering Hackathon 2026** — 24-Hour Remote Qualifier (Phase 1)  
**Challenge:** AI for Urban Dengue Outbreak  
**Target Cities:** Dhaka & Chattogram, Bangladesh  

---

## 1. Quickstart (< 3 Minutes)

Run the application locally on Windows, macOS, or Linux:

### Windows:
```cmd
# 1. Clone repository
git clone https://github.com/Tasfiq17/DengueGuard.git
cd DengueGuard

# 2. Create virtual environment and install requirements
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

# 3. Launch dashboard
streamlit run app.py
```

### macOS / Linux:
```bash
# 1. Clone repository
git clone https://github.com/Tasfiq17/DengueGuard.git
cd DengueGuard

# 2. Create virtual environment and install requirements
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 3. Launch dashboard
streamlit run app.py
```

Open your browser at **http://localhost:8501**.

> **Note on Python & Dependencies:** Python 3.10 to 3.13 is supported. `scikit-learn==1.6.1` is strictly pinned in `requirements.txt` to ensure exact binary compatibility with the serialized `.joblib` model artifacts.

---

## 2. Core Capabilities & Architecture

- **Multi-Week Weather Lag Pipeline:** Captures the biological vector breeding clock (7–14 days) and viral incubation delays through 1-to-4 week rainfall, humidity, and temperature lags, plus 2-, 4-, and 8-week cumulative rainfall sums.
- **Dynamic 2-to-4 Week Outbreak Warning:** Generates forward-looking single-week predictions for Horizon 2 (2 weeks), Horizon 3 (3 weeks), and Horizon 4 (4 weeks) with risk tier classifications: **Low**, **Moderate**, **High**, and **Critical**.
- **Interactive Scenario Forecasting (What-If Simulator):** Public health teams can customize rainfall spikes, humidity shifts, temperatures, and ward vulnerability indicators to simulate and predict future outbreak risk in real time.
- **Zero API Costs & High Efficiency:** Runs 100% locally with sub-50ms inference latency using pre-trained `HistGradientBoosting` pipelines; zero external paid API dependencies.
- **Strict Leakage Prevention:** Chronological target-date split (70% train / 15% validation / 15% held-out test).

---

## 3. Empirical Model Performance (Held-Out Test Set: Jul 2024 – Dec 2025)

The selected `HistGradientBoosting` models outperformed both historical baselines across all horizons:

| Horizon | Selected Model | Test MAE | Test R² | Persistence MAE | Seasonal Naive MAE | MAE Improvement |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **H2 (2 weeks)** | HistGradientBoosting | **1.25** | **0.71** | 2.03 | 1.96 | **−36.4%** |
| **H3 (3 weeks)** | HistGradientBoosting | **1.30** | **0.67** | 2.32 | 1.96 | **−33.8%** |
| **H4 (4 weeks)** | HistGradientBoosting | **1.48** | **0.58** | 2.58 | 1.96 | **−24.6%** |

*Peak Analysis:* In the top 10% highest outbreak weeks (≥9 cases/ward), the model accurately ranks relative ward risk while underpredicting peak amplitudes (~10 predicted vs 12.1 actual). Health officials should treat High and Critical warnings as high-priority intervention triggers.

---

## 4. Repository Structure

```
DengueGuard_Hackathon_Submission/
├── app.py                         # Streamlit Interactive Web Application
├── requirements.txt               # Pinned dependencies for reproducible setup
├── README.md                      # Setup instructions, performance & AI disclosure
├── ARCHITECTURE.md                # System data flow, split rationale & cost controls
├── .gitignore                     # Git ignore rules (virtual environment, cache)
├── data/
│   └── dataset_snapshot.csv       # Organizer historical dataset (2016 - 2025)
├── models/
│   ├── metadata.json              # Model configuration, thresholds, feature list
│   ├── model_h2.joblib            # Serialized pipeline & model for 2-week forecast
│   ├── model_h3.joblib            # Serialized pipeline & model for 3-week forecast
│   └── model_h4.joblib            # Serialized pipeline & model for 4-week forecast
├── test_metrics.csv               # Test partition MAE, RMSE, R²
├── test_baseline_comparison.csv   # ML models vs baselines comparison
├── validation_results.csv         # Validation tuning metrics across model families
├── feature_importance_h2.csv      # Permutation/model feature importance (H2)
├── feature_importance_h3.csv      # Permutation/model feature importance (H3)
├── feature_importance_h4.csv      # Permutation/model feature importance (H4)
└── peak_error_analysis.csv        # Top-decile peak error & underprediction rates
```

---

## 5. AI Disclosure

In compliance with the hackathon submission guidelines:
- **AI Assistants Used:** 
  - **Claude:** Used for structuring and giving proper headings/markdown documentation to every cell in the training and validation notebook.
  - **Google Antigravity:** Assisted in Streamlit UI structuring, rapid code exploration, and drafting system documentation.
- **External Paid APIs:** None ($0.00). All models run completely offline on local CPU using `.joblib` pipelines.
- **Data Integrity:** No synthetic or AI-generated training data was used; all training was conducted strictly on the provided hackathon dataset.

---

## 6. Team Details

- **Hackathon:** EDU AI Engineering Hackathon 2026 (East Delta University)
- **Track:** Challenge 1 — AI for Urban Dengue Outbreak
- **Repository:** https://github.com/Tasfiq17/DengueGuard
