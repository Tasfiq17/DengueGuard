# DengueGuard System Architecture

## 1. System Overview & Data Flow

DengueGuard is an end-to-end machine learning system designed to provide public health officials in Bangladesh (Dhaka and Chattogram) with a 2-to-4-week early warning window for localized dengue outbreaks. The system integrates epidemiological surveillance history with meteorological and environmental vulnerability indicators to forecast ward-level outbreak risk before hospital admissions spike.

```
+-----------------------------------------------------------------------------------+
|                           1. Raw Data Ingestion                                   |
|  - Weekly ward records: dengue cases, rainfall, temperature (avg/min/max),        |
|    relative humidity, water logging risk, sanitation score, population density     |
+-----------------------------------------------------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|                    2. Multi-Week Lag & Feature Pipeline                           |
|  - Grouped by ward_id, sorted chronologically                                     |
|  - Vector breeding clock lags: rainfall_mm_lag_1w to 4w, rainfall_sum_prev_2w/4w/8w|
|  - Temperature & humidity lags (1w-4w) and 4-week moving averages                 |
|  - Autoregressive dengue case lags (1w-4w) and 4-week moving average               |
|  - Cyclical seasonality encoding: week_sin, week_cos, month                       |
+-----------------------------------------------------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|                    3. Chronological Target-Date Split                             |
|  - Split: 70% Train (2016-2022) | 15% Validation (2023-mid 2024) | 15% Test      |
|  - Strictly partitioned by target date (t + h) to prevent temporal data leakage   |
+-----------------------------------------------------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|               4. Model Comparison & Regularization Tuning                         |
|  - Models: HistGradientBoosting, ExtraTrees, Random Forest                        |
|  - Baselines: Persistence (last observed count), Seasonal Naive (52-week lag)     |
|  - Hyperparameter regularization (leaf nodes, sample leaf, loss functions)        |
+-----------------------------------------------------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|                    5. Refit & Serialization (.joblib)                             |
|  - Best model (HistGradientBoosting) refit on Train + Validation                  |
|  - Bundled with Scikit-Learn ColumnTransformer pipeline (imputation + encoding)   |
|  - Evaluated strictly ONCE on held-out test set (Jul 2024 – Dec 2025)             |
+-----------------------------------------------------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|                    6. Streamlit Decision-Support Dashboard                        |
|  - Fast local cached inference (@st.cache_resource, @st.cache_data)              |
|  - Multi-horizon forecasts (H2, H3, H4) with dynamic risk tier classifications   |
|  - Interactive What-If Outbreak Scenario Simulator for health officers            |
|  - Feature driver breakdowns & held-out validation audit metrics                  |
+-----------------------------------------------------------------------------------+
```

---

## 2. Multi-Week Weather Lag Pipeline

Aedes aegypti mosquito breeding and viral transmission follow biological delays:
1. **Breeding Clock (7–14 days):** Stagnant water from heavy rainfall allows larvae to mature into adult vectors.
2. **Extrinsic Incubation Period (5–10 days):** Viral replication within the vector before it becomes infectious.
3. **Intrinsic Incubation & Symptoms (4–10 days):** Time between human infection and hospital admission.

To reflect this 2-to-4-week lag, the feature pipeline computes:
- **Direct weekly lags:** `rainfall_mm_lag_1w`, `rainfall_mm_lag_2w`, `rainfall_mm_lag_3w`, `rainfall_mm_lag_4w` computed per `ward_id`.
- **Cumulative rainfall sums:** `rainfall_sum_prev_2w`, `rainfall_sum_prev_4w`, `rainfall_sum_prev_8w` capturing prolonged water accumulation.
- **Microclimate conditions:** 4-week rolling means for `relative_humidity_pct` and `avg_temp_c`.
- **Ward vulnerability indices:** `water_logging_risk_index`, `sanitation_score`, and `population_density_sqkm`.

---

## 3. Chronological Validation Split (Leakage Prevention)

Standard random K-Fold cross-validation leaks future information into past predictions in time-series data. To guarantee real-world generalization, DengueGuard uses a strict **chronological target-date split**:

- **Training Period:** 2016-01-04 to ~mid-2022 (70% of chronological timeline)
- **Validation Period:** mid-2022 to 2024-07-01 (15% of timeline) — used exclusively for model architecture selection, tree-depth tuning, and regularization.
- **Held-Out Test Period:** 2024-07-08 to 2025-12-29 (15% of timeline, 4,680 ward-week observations) — left completely untouched until final verification.

All lag and rolling features are computed strictly on past observations ($t \le \text{issue date}$), ensuring zero leakage across partition boundaries.

---

## 4. Model Selection Rationale

Three diverse model families were systematically evaluated against two non-trivial domain baselines across all three forecast horizons (H2 = 2 weeks ahead, H3 = 3 weeks ahead, H4 = 4 weeks ahead):

| Horizon | Winning Model | Test MAE | Test R² | Persistence MAE | Seasonal Naive MAE | MAE Improvement |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **H2 (2 weeks)** | **HistGradientBoosting** | **1.25** | **0.71** | 2.03 | 1.96 | **36.4% lower MAE** |
| **H3 (3 weeks)** | **HistGradientBoosting** | **1.30** | **0.67** | 2.32 | 1.96 | **33.8% lower MAE** |
| **H4 (4 weeks)** | **HistGradientBoosting** | **1.48** | **0.58** | 2.58 | 1.96 | **24.6% lower MAE** |

### Key Rationale:
1. **HistGradientBoosting vs Random Forest / ExtraTrees:** HistGradientBoosting consistently delivered the lowest validation MAE with minimal generalization gap (val/train MAE ratio of 1.0–1.2 vs 6.5 in unregularized configurations).
2. **Poisson / Squared Error Loss:** Poisson loss natural handling of non-negative, zero-inflated count targets proved superior for longer horizons (H3, H4), preventing negative case predictions.
3. **Beating Baselines:** ML models outperformed seasonal naive and persistence across all horizons by 25% to 36%, proving substantial predictive value beyond simple historical averages.

---

## 5. Compute & API Cost Efficiency

Public health offices operate under strict resource constraints. DengueGuard is engineered for minimal compute footprint and zero recurring API costs:

1. **Zero External API Costs ($0.00):**
   - The entire prediction engine operates **100% locally** using serialized scikit-learn models (`.joblib`).
   - No external paid LLM or inference API calls are executed during dashboard operation or feature calculation.
2. **Sub-50ms Inference Latency:**
   - Pre-trained HistGradientBoosting models evaluate all 60 wards in under **35 milliseconds**.
   - Input preprocessing is bundled directly inside the serialized pipeline via `ColumnTransformer` with median/mode imputation, preventing runtime dependency on training matrices.
3. **In-Memory Caching Controls:**
   - Streamlit's `@st.cache_data` caches raw dataset ingestion and feature engineering computations.
   - `@st.cache_resource` caches the loaded `.joblib` model bundles, eliminating redundant disk I/O on user interaction.
4. **Hardware Footprint:**
   - Can run reliably on a single CPU core with under 512MB RAM, making it easily deployable on low-cost edge servers or municipal field laptops.
