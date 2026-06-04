# Pipeline Execution Report

# Overview

This document demonstrates the successful execution of the complete end-to-end MLOps pipeline.

Pipeline stages executed:

1. Data Ingestion
2. Feature Engineering
3. Model Training
4. SHAP Explainability
5. Backtesting
6. Regime Analysis
7. Baseline Simulation
8. Scheduled Retraining Simulation
9. Drift-Triggered Retraining Simulation
10. Drift Detection
11. Automatic Retraining
12. Testing and Validation

---

# 1. Data Ingestion

## Command

```bash
make fetch-data
```

## Purpose

Downloads historical stock data and market context indicators.

### Stocks

* AAPL
* MSFT
* GOOGL
* AMZN
* META
* NVDA
* TSLA
* AMD
* NFLX
* INTC

### Market Indicators

* SPY
* QQQ
* VIX

## Results

| Metric                 | Value       |
| ---------------------- | ----------- |
| Market Dataset Shape   | (2513, 7)   |
| Combined Dataset Shape | (25130, 13) |

Raw data stored in:

```text
data/raw/
```

## Screenshot

![Data Ingestion](/reports/pipeline/pipeline-fetch.png)

---

# 2. Feature Engineering

## Command

```bash
make build-features
```

## Purpose

Generates market-aware technical indicators.

Features created include:

* Moving Averages
* RSI
* MACD
* Bollinger Bands
* Momentum
* Volatility Features
* Relative SPY Strength
* Market Stress Indicators

## Results

| Metric              | Value       |
| ------------------- | ----------- |
| Final Dataset Shape | (24940, 42) |

## Screenshot

![Feature Engineering](/reports/pipeline/pipeline-features-train.png)

---

# 3. Model Training

## Command

```bash
make train
```

## Models Evaluated

| Model               | Average F1 |
| ------------------- | ---------- |
| Logistic Regression | 0.6111     |
| Random Forest       | 0.6337     |
| Extra Trees         | 0.6262     |
| XGBoost             | 0.6431     |

---

## Logistic Regression

Average F1 Score: **0.6111**

![Logistic Regression](/reports/pipeline/pipeline-train.png)

---

## Random Forest

Average F1 Score: **0.6337**

Top Features:

* QQQ_MA_10
* SPY_MA_5
* SPY_Volatility
* VIX_MA_5
* RSI

![Random Forest](/reports/pipeline/pipeline-train.png)

---

## Extra Trees

Average F1 Score: **0.6262**

![Extra Trees](/reports/pipeline/pipeline-train1.png)

---

## XGBoost

Average F1 Score: **0.6431**

Top Features:

* QQQ_MA_10
* SPY_MA_5
* VIX_Level
* VIX_MA_5
* SPY_Volatility

![XGBoost](/reports/pipeline/pipeline-train1.png)

---

## Best Model

**Selected Model:** XGBoost

**Average F1:** 0.6431

Model saved to:

```text
models/model_v1.pkl
```

Tracked using DVC.

![Best Model](/reports/pipeline/pipeline-train1.png)

---

# 4. SHAP Explainability

## Command

```bash
make shap
```

## Generated Artifacts

![Shap Summary](/reports/figures/shap_summary.png)

![Shap Bar](/reports/figures/shap_bar.png)

[View Best Model SHAP Feature Importance Data](/reports/figures/shap_feature_importance.csv)

## Top SHAP Features

1. SPY_Volatility
2. SPY_MA_5
3. QQQ_MA_10
4. VIX_Level
5. RSI
6. QQQ_Momentum

## Screenshot

![SHAP Analysis](/reports/pipeline/pipeline-train3-shap.png)

---

# 5. Backtesting

## Command

```bash
make backtest
```

## Results

| Metric          | Value   |
| --------------- | ------- |
| Strategy Return | 331.44% |
| Market Return   | 103.55% |
| Sharpe Ratio    | 3.3453  |
| Max Drawdown    | -18.49% |
| Win Rate        | 55.82%  |

## Screenshot

![Backtesting](/reports/pipeline/pipeline-backtest-simbaseline.png)

---

# 6. Regime Analysis

## Command

```bash
make regime
```

## Market Regimes

* Low Volatility
* Medium Volatility
* High Volatility

## Best Regime Performance

| Regime          | F1 Score |
| --------------- | -------- |
| High Volatility | 0.9208   |

## Screenshot

![Regime Analysis](/reports/pipeline/pipeline-backtest-simbaseline.png)

---

# 7. Baseline Simulation

## Command

```bash
make simulate-baseline
```

## Results

| Metric    | Value  |
| --------- | ------ |
| F1 Score  | 0.7250 |
| Precision | 0.8752 |
| Recall    | 0.6188 |

## Screenshot

![Baseline Simulation](/reports/pipeline/pipeline-simbaseline1.png)

---

# 8. Scheduled Retraining Simulation

## Command

```bash
make simulate-retraining
```

## Results

Final F1 Score:

**0.4114**

## Screenshot

![Scheduled Retraining](/reports/pipeline/pipeline-simbaseline2-drift.png)

---

# 9. Drift-Triggered Retraining Simulation

## Command

```bash
make simulate-drift
```

## Results

Detected Drift Events:

**8**

Drifted Features:

* Volatility
* MACD
* Relative_SPY_Strength

Final F1:

**0.4114**

## Screenshot

![Drift Triggered Retraining](/reports/pipeline/pipeline-simulatedrift.png)

---

# 10. Drift Detection

## Command

```bash
make drift
```

## Detected Drift

* Volatility
* MACD
* Relative_SPY_Strength

Generated Report:

[View Latest Drift Report File](/reports/monitoring/drift_report.json)

## Screenshot

![Drift Detection](/reports/pipeline/pipeline-drift-retrain.png)

---

# 11. Automatic Retraining

## Command

```bash
make retrain
```

## Results

* Previous model backed up
* New model trained
* New model version saved
* Monitoring report generated

Generated:

[View Latest Retraining Report File](/reports/monitoring/retraining_report.json)

## Screenshot

![Automatic Retraining](/reports/pipeline/pipeline-retrain.png)

---

# 12. Testing and Validation

## Unit Testing

### Command

```bash
make test
```

### Results

```text
2 tests passed
```

---

## Ruff Linting

### Command

```bash
make lint
```

### Results

```text
All checks passed
```

---

## Black Formatting

### Command

```bash
make format-check
```

### Results

```text
28 files would be left unchanged
```

## Screenshot

![Testing and Validation](/reports/pipeline/pipeline-test-format.png)

---

# Conclusion

The complete MLOps pipeline executed successfully from data ingestion through production monitoring and automated retraining.

## Key Accomplishments

* End-to-end pipeline execution completed
* Market-aware feature engineering implemented
* Multiple ML models trained and evaluated
* XGBoost selected as the best-performing model
* SHAP explainability integrated
* Trading strategy successfully backtested
* Regime-based evaluation completed
* Drift detection operational
* Automated retraining validated
* DVC used for model and dataset versioning
* Testing, linting, and formatting checks passed

This project demonstrates a production-oriented MLOps workflow covering data engineering, machine learning, explainability, monitoring, retraining, testing, and reproducibility.
