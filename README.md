# QuickCart Warehouse Inventory Stockout Risk

## Project Overview

**QuickCart Warehouse Inventory Stockout Risk** is an end-to-end machine learning project designed to predict inventory stockout risk at the **Store × SKU × Day** level.

QuickCart operates **12 dark stores across 6 Indian metropolitan cities**, managing **60 SKUs across 8 categories and 15 suppliers**. The system classifies inventory positions into:

* 🟢 **Safe** — sufficient stock buffer before replenishment arrives
* 🟡 **At-Risk** — stock cover is getting close to replenishment wait
* 🔴 **Imminent** — inventory may run out before replenishment arrives

The goal is to provide an early-warning system that helps inventory teams balance **product availability** against **excess inventory**.

\---

## Business Problem

Stockouts in quick-commerce can cause lost sales, poor customer experience, reduced availability, and emergency replenishment. Excess inventory can simultaneously increase capital tied up in stock, storage requirements, wastage risk, and operating costs.

QuickCart therefore needs to identify **which Store × SKU combinations require attention before a stockout occurs**.

\---

## Project Objective

Build a multi-class classification system that predicts:

How risky is it that a product will run out before the next replenishment arrives?

The project uses:

```text
Inventory Buffer = Days of Stock Cover - Replenishment Wait
```

Risk categories:

|Inventory Buffer|Risk|
|-|-|
|`>= 3 days`|Safe|
|`0 to < 3 days`|At-Risk|
|`< 0 days`|Imminent|

\---

## Dataset

The project contains five interconnected datasets.

### `dim_stores.csv`

Store-level information including city, store size, square footage, opening year, and baseline daily orders.

**Rows:** 12

### `dim_skus.csv`

Product-level information including category, price, shelf life, perishability, festival relevance, popularity, and supplier.

**Rows:** 60

### `dim_suppliers.csv`

Supplier reliability and replenishment information including base lead time and lead-time variance.

**Rows:** 15

### `dim_events.csv`

Daily event and demand-multiplier information covering regular days, festival periods, and promotional events.

**Rows:** 30

### `fact_inventory_daily.csv`

Main analytical table containing one observation per:

```text
Store × SKU × Day
```

**Rows:** 21,600

Important fields include opening stock, closing stock, demand, sales, reorder point, expected lead time, sales velocity, days of cover, festival indicator, and stockout risk.

\---

## Data Quality \& Cleaning

The project contains realistic data-quality challenges.

### City Naming Inconsistency

Two records contained casing differences in `city_display`. The clean `city` field was retained as the primary join key and the display field was standardized.

### Supplier Reliability

`reliability_score` contained literal `N/A` values. These were converted to missing values using numeric coercion and imputed using the appropriate supplier-category median.

### Missing Actual Lead Time

`lead_time_days_actual` contains approximately **92.1% missing values**. This is expected because actual lead time is only available after replenishment has occurred, so it was not blindly imputed.

### Duplicate Validation

No duplicate records were found in the supplied datasets.

The inventory fact table contains exactly:

```text
12 stores × 60 SKUs × 30 days = 21,600 records
```

\---

## Target Distribution

|Stockout Risk|Share|
|-|-:|
|Safe|65.42%|
|At-Risk|24.01%|
|Imminent|10.57%|

Because of this class imbalance, **accuracy alone is not sufficient** for evaluation. Particular attention is given to **Imminent recall**, since missing a genuine stockout risk can have a direct business cost.

\---

## Feature Engineering

The model combines current inventory information, historical behavior, supplier characteristics, product characteristics, store characteristics, and event information.

### Inventory Features

* `opening_stock`
* `closing_stock`
* `units_demanded`
* `units_sold`
* `reorder_point`
* `reorder_gap`

### Historical Features

* Previous closing stock
* Previous units demanded
* Lagged 3-day sales velocity
* Lagged 7-day sales velocity
* Recent reorder count
* Previous-day imminent-risk indicator

### Supplier \& Replenishment Features

* Expected lead time
* Base lead time
* Lead-time variance
* Cleaned supplier reliability score

### Product Features

* Category
* Unit price
* Shelf life
* Perishable indicator
* Festival relevance
* Popularity tier

### Store Features

* Store size
* Store area
* Baseline daily orders
* Store demand share

### Temporal \& Event Features

* Day of month
* Day of week
* Weekend indicator
* Festival-week indicator
* Event type
* Festival demand multiplier
* Other demand multiplier

### Derived Features

```text
Reorder Gap = Reorder Point - Closing Stock
```

Historical cover pressure is also derived from lagged stock and demand information.

\---

## Leakage Prevention

A major consideration is **target leakage**.

The target is itself defined using stock-cover and replenishment information. Directly feeding variables that reproduce the target definition could make the model appear stronger than it really is.

Therefore, these direct/current variables were excluded:

* `stockout_risk`
* `days_of_cover`
* Current `sales_velocity_7d`
* `lead_time_days_actual`

Historical and lagged information is used where appropriate.

A model should use information that would actually be available at prediction time.

\---

## Machine Learning Approach

The project is formulated as a **three-class classification problem**.

### Baseline

A majority-class baseline predicts `Safe` for every observation. Since Safe represents approximately 65.42% of the data, this provides a useful benchmark.

### Models

1. **Logistic Regression**
2. **Random Forest**
3. **Gradient Boosting**

Categorical variables are transformed using **One-Hot Encoding** and numerical variables are standardized where appropriate.

Class balancing is applied to Logistic Regression and Random Forest.

\---

## Train/Test Strategy

A **time-based split** is used rather than a purely random split:

```text
Training Data
2026-10-01 → 2026-10-23

Testing Data
2026-10-24 → 2026-10-30
```

This better represents a real-world scenario in which historical data is used to evaluate performance on later observations.

\---

## Model Evaluation

Models are evaluated using:

* Accuracy
* Macro F1-score
* Precision
* Recall
* Classification report
* Confusion matrix
* **Imminent-class recall**

### Why Imminent Recall Matters

A particularly costly error can be:

```text
Actual: Imminent
Predicted: Safe
```

The system would fail to warn the business about a potential stockout. Therefore, overall accuracy is not treated as the only evaluation criterion.

\---

## Prediction Output

The final pipeline produces:

* Store ID
* SKU ID
* Date
* Actual risk
* Predicted risk
* Class probabilities

A separate high-risk inventory output is generated for records predicted as:

```text
Imminent
```

This connects ML predictions with practical inventory monitoring.

\---

## Business Workflow

```text
Daily Inventory Data
        ↓
Data Cleaning & Integration
        ↓
Feature Engineering
        ↓
Stockout Risk Model
        ↓
Risk Classification
        ↓
┌──────────┬───────────┬────────────┐
│   Safe   │  At-Risk  │  Imminent  │
└──────────┴───────────┴────────────┘
        ↓
Inventory Team Action
```

### Example Actions

**Safe**

* Continue normal replenishment planning
* No immediate intervention

**At-Risk**

* Monitor inventory closely
* Review upcoming demand
* Check supplier lead time
* Consider earlier replenishment

**Imminent**

* Prioritize replenishment
* Escalate to inventory operations
* Evaluate supplier alternatives
* Consider demand allocation or substitution

\---

## Technology Stack

|Technology|Purpose|
|-|-|
|Python|Core programming|
|Pandas|Data manipulation|
|NumPy|Numerical operations|
|Matplotlib|Visualization|
|Scikit-learn|Machine learning|
|Joblib|Model persistence|
|Google Colab|Development environment|
|CSV|Data storage|

\---

## Project Structure

```text
QuickCart-Warehouse-Stockout-Risk/
└── README.md
│
├── data/
│   ├── dim_events.csv
│   ├── dim_skus.csv
│   ├── dim_stores.csv
│   ├── dim_suppliers.csv
│   └── fact_inventory_daily.csv
│
├── notebook/
│   └── QuickCart_Stockout_Risk.ipynb

```

\---

## How to Run

### 1\. Clone the repository

```bash
 QuickCart-Warehouse
```

### 2\. Install dependencies

```bash
pip install pandas numpy matplotlib scikit-learn joblib
```

### 3\. Open the notebook


```text
https://colab.research.google.com/drive/1kWV-ErfkEGRDIkOBBx0rp1mTZPflUWln
```

using Google Colab or Jupyter Notebook.

### 4\. Run the pipeline

```text
Data Loading
→ Data Validation
→ Data Cleaning
→ Data Integration
→ Feature Engineering
→ Train/Test Split
→ Model Training
→ Model Evaluation
→ Risk Prediction
→ High-Risk Export
```

\---

## Key Project Learnings

This project demonstrates practical understanding of:

* Multi-table data integration
* Data-quality handling
* Feature engineering
* Time-aware train/test splitting
* Multi-class classification
* Class imbalance
* Model comparison
* Confusion-matrix analysis
* Feature importance
* Data leakage prevention
* Business-focused ML evaluation
* Translating predictions into operational actions

\---

## Future Improvements

A more production-oriented version could move the target into the future:

```text
Today's Information
        ↓
Predict Tomorrow / Next Few Days
        ↓
Future Stockout Risk
```

Potential extensions:

* Future-horizon stockout prediction
* XGBoost / LightGBM comparison
* Hyperparameter optimization
* SHAP-based explainability
* Cost-sensitive classification
* Supplier-specific risk analysis
* Automated daily scoring
* Power BI inventory-risk dashboard
* Real-time alerts for Imminent SKUs
* Store-SKU replenishment recommendations

\---

## Business Impact

The project transforms raw operational data into an actionable inventory-risk signal.

Instead of only asking:

“Which products are already out of stock?”

the system moves toward:

“Which products are likely to become a stockout problem, and where should the inventory team investigate?”

This represents a shift from reactive inventory monitoring toward predictive inventory management.

\---

## Disclaimer

This project uses a simulated QuickCart-style dataset created for educational and portfolio purposes. The business scenario, entities, and operational data do not represent actual QuickCart internal data.

\---

## Author

**Abhay Pratap Singh**

B.Tech — Artificial Intelligence \& Data Science  
Aspiring Data Analyst | Data Analytics | Business Intelligence

\---

### Project Focus

**Data Analytics • Machine Learning • Inventory Analytics • Supply Chain Analytics • Predictive Modeling**

