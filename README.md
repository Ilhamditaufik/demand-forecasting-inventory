# Nationwide Demand Forecasting and Inventory Optimization

**Capstone & Industry Project DS-02 — Nexvia Technologies**

A Data Science prototype for demand forecasting, hierarchical forecast reconciliation, and inventory optimization using the M5 Walmart dataset.

## 1. Project Overview

This project develops an analytical prototype that combines machine learning, time-series forecasting, and mathematical optimization to support inventory planning.

The system analyzes historical product-store sales, evaluates forecasting models, generates demand predictions, and uses OR-Tools to recommend replenishment quantities under simulated inventory constraints.

The M5 dataset represents Walmart sales across stores in California, Texas, and Wisconsin. The project title refers to the forecasting and inventory-planning objective; it does not imply that the prototype covers every Walmart location.

## 2. Objectives

- Analyze historical sales patterns and calendar effects.
- Compare Seasonal Naive, LightGBM, Prophet, and Temporal Fusion Transformer experiments.
- Evaluate point forecasts using WAPE, MAE, and RMSE.
- Examine prediction intervals and empirical coverage.
- Reconcile store, state, and national forecasts.
- Optimize simulated replenishment decisions under budget and storage constraints.
- Present findings through an interactive Plotly Dash dashboard.

## 3. Technology Stack

- **Python:** primary programming language.
- **Pandas and NumPy:** data processing and feature engineering.
- **Matplotlib and Plotly:** visualizations.
- **LightGBM:** product-store demand forecasting.
- **Prophet:** aggregate national time-series forecasting.
- **PyTorch Forecasting and Lightning:** experimental Temporal Fusion Transformer.
- **Darts:** hierarchical forecast reconciliation.
- **OR-Tools CP-SAT:** inventory optimization.
- **Dash:** interactive analytical dashboard.

## 4. Dataset

The project uses the M5 Forecasting dataset available through Kaggle.

Required files:

- `sales_train_validation.csv`
- `calendar.csv`
- `sell_prices.csv`

Dataset source: https://www.kaggle.com/competitions/m5-forecasting-accuracy/data

The raw dataset is not included in this repository. Users must obtain the files through the dataset source and place them in `data/raw/`.

## 5. System Workflow

1. Inspect and validate the input dataset.
2. Perform exploratory data analysis and aggregate sales by day and month.
3. Establish a Seasonal Naive baseline.
4. Train and evaluate LightGBM for product-store forecasting.
5. Evaluate Prophet on national aggregate sales.
6. Examine prediction-interval coverage.
7. Reconcile store, state, and national forecasts.
8. Optimize simulated replenishment quantities with OR-Tools.
9. Train a limited TFT experiment on selected product-store series.
10. Present analytical results in a Plotly Dash dashboard.
11. Run quality checks and prepare the final documentation.

## 6. Repository Structure

```text
demand-forecasting-inventory/
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
├── src/
├── tests/
├── outputs/
├── reports/
├── inspect_data.py
├── eda.py
├── baseline_validation.py
├── lightgbm_validation.py
├── prophet_validation.py
├── evaluate_intervals.py
├── hierarchical_reconciliation.py
├── inventory_optimization.py
├── tft_validation.py
├── validate_project.py
├── app.py
├── requirements.txt
├── .gitignore
└── README.md
```

Generated models, reports, and raw data should be managed according to repository size and data-distribution requirements.

## 7. Installation

Create a Python virtual environment in the project directory.

On Windows PowerShell:

```powershell
py -m venv .venv

.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

After placing the three required CSV files in `data/raw/`, run the scripts in the documented order. Some scripts train models and may require substantial memory and processing time.

## 8. Forecasting Results

### National aggregate evaluation

The following results use total national daily sales over the same 28-day evaluation period.

| Model | WAPE | MAE (units) | RMSE (units) |
|---|---:|---:|---:|
| Prophet | 5.97% | 2,523.85 | 3,255.08 |
| LightGBM | 6.12% | 2,587.46 | 3,137.86 |
| Seasonal Naive | 7.64% | 3,231.14 | 3,947.48 |

Prophet produced the lowest WAPE and MAE in this particular aggregate evaluation, while LightGBM produced the lowest RMSE.

### Product-store LightGBM evaluation

The initial 28-day evaluation across 30,490 product-store series produced:

| Metric | Seasonal Naive | LightGBM |
|---|---:|---:|
| WAPE | 88.89% | 78.44% |
| MAE | 1.2324 | 1.0875 |
| RMSE | 2.7595 | 2.3223 |

The LightGBM experiment improved all three metrics relative to the baseline in this evaluation.

### Temporal Fusion Transformer experiment

A limited experiment on 60 selected product-store series produced:

| Model | WAPE | MAE | RMSE |
|---|---:|---:|---:|
| TFT | 35.64% | 8.2687 | 13.5067 |
| Seasonal Naive | 38.96% | 9.0393 | 14.3397 |
| LightGBM | 50.10% | 11.6245 | 21.7037 |

These results apply only to the selected series and the specific experimental setup. They must not be compared directly with national aggregate metrics or the full product-store evaluation.

## 9. Prediction Intervals

The initial Prophet interval experiment used a nominal 80% interval.

- Evaluation period: 28 days.
- Empirical coverage: 71.43%.
- Days covered: 20 of 28.
- Mean interval width: approximately 7,235 units.

The observed coverage was below the nominal level. Further calibration and evaluation over a longer test period are needed before the intervals can be considered reliable for operational decisions.

## 10. Inventory Optimization Experiment

The OR-Tools prototype optimized a seven-day simulated planning scenario.

| Metric | Result |
|---|---:|
| Forecast demand | 309,314 units |
| Assumed initial stock | 48,744 units |
| Recommended order quantity | 195,427 units |
| Expected fulfilled demand | 244,171 units |
| Potential lost sales | 65,143 units |
| Potential lost-sales rate | 21.06% |
| Estimated purchase cost | $411,695.69 |

These are simulation outputs, not Walmart's actual operational results.

Initial inventory, procurement cost, holding cost, stockout penalty, purchasing budget, and storage capacity were configured as assumptions. The estimated selling price was used as a proxy for procurement cost, with a configurable discount factor.

The recommendations should therefore be interpreted as a demonstration of constrained optimization, not as real purchasing advice.

## 11. Dashboard

The Plotly Dash application provides:

- National forecasting and model-comparison charts.
- Forecasting evaluation metrics.
- State and store filters.
- Inventory summaries by store.
- Product-store replenishment recommendations.
- Estimated purchasing costs and potential lost sales.
- Disclosure of the simulation assumptions and evaluation limitations.

Start the local application with:

```powershell
.\.venv\Scripts\python.exe app.py
```

Open `http://127.0.0.1:8050/` in the browser.

## 12. Quality Assurance

The project quality check reported:

- 40 checks passed.
- 0 checks failed.
- 3 warnings.

The audit covered artifact availability, selected model metrics, hierarchical aggregation, inventory arithmetic, purchasing constraints, and prediction-interval properties.

Passing these checks does not establish that every model is production-ready. The three warnings and all modeling limitations must be reviewed and documented before final submission.

## 13. Limitations and Future Work

- Evaluate models through multiple rolling-origin backtests.
- Improve and validate prediction intervals.
- Expand TFT training beyond the limited sample and tune its hyperparameters.
- Compare forecasting models on identical populations and horizons.
- Test optimization under multiple budget, capacity, and stock scenarios.
- Replace simulated inventory and procurement assumptions with operational data when available.
- Add automated testing, reproducible environments, and application deployment.
- Extend product-level reconciliation and strengthen the validation of forecast coherence.

## 14. Project Status

This repository documents a working Data Science prototype with forecasting experiments, hierarchical aggregation, inventory optimization, and an interactive dashboard.

The reported results are experimental. Any production deployment would require additional validation, realistic business inputs, operational testing, and review of forecasting and optimization assumptions.
