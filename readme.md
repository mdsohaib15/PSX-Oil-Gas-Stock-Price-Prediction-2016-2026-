# PSX Oil & Gas Stock Price Prediction (2016–2026)

A full-scale **data mining, machine learning, and deployment project** designed to analyze and forecast stock prices of the Pakistan Stock Exchange (PSX) Oil & Gas sector using historical time-series data.

## Problem Statement

Stock price prediction in emerging markets like Pakistan is  **noisy, volatile, and non-linear** . Traditional analysis methods fail to capture:

* Temporal dependencies
* Non-linear interactions between features
* Company-specific behavior

This project builds a **data-driven forecasting system** that:

* Learns from **historical price momentum**
* Captures **short-term trends**
* Generalizes across **multiple companies in a single model**

## Objectives

* Build a **clean, unified dataset** for 17 PSX Oil & Gas companies
* Engineer **financially meaningful features**
* Benchmark **multiple ML algorithms**
* Optimize and deploy the **best-performing model**
* Provide a **real-time prediction dashboard**

## Dataset Details

### Coverage

* **Sector:** Oil & Gas (PSX)
* **Time Period:** 2016–2026
* **Granularity:** Daily trading data
* **Records:** Tens of thousands

### Companies Included

**Exploration & Production**

* MARI, OGDC, POL, PPL

**Refining**

* ATRL, NRL, PRL, CNERGY

**Marketing & Distribution**

* APL, HASCOL, PSO, HTL, WAFI, OBOY

**Gas Utilities**

* SSGC, SNGP, BLPL

## Features

### Raw Features

* Open
* High
* Low
* Close Price
* Volume
* Change %

### Engineered Features

#### 1. Lag Features

* `Price_Yesterday`
* `Price_5_Days_Ago`

 Captures **short-term memory of the market**

#### 2. Moving Averages

* `MA7` (short-term trend)
* `MA30` (medium-term trend)

Smooths volatility and highlights trends

#### 3. Temporal Features

* Day of Week
* Month

 Captures **seasonality and trading patterns**

## Data Preprocessing Pipeline

* Converted **Volume (K/M format → numeric)**
* Removed **missing values**
* Fixed **data types**
* Eliminated **duplicates**
* Applied **Standard Scaling**
* Applied **One-Hot Encoding (17 companies)**

## Model Development

### Algorithms Tested (13 Total)

**Linear Models**

* Linear Regression
* Ridge
* Lasso
* Elastic Net

**Tree-Based Models**

* Decision Tree

**Ensemble Models**

* Random Forest
* AdaBoost
* Gradient Boosting
* XGBoost
* LightGBM
* CatBoost

**Distance / Kernel Models**

* KNN
* SVR

## Training Strategy

* **Train-Test Split:** 80/20
* **Chronological Split:** No shuffling
* Prevents **data leakage** (critical in time series)

## Hyperparameter Tuning

* **GridSearchCV** → Smaller models
* **RandomizedSearchCV** → Large models (Random Forest)
* **5-Fold Time-Series Cross Validation**

## Final Model

### Tuned Random Forest Regressor

**Why this model?**

* Handles **non-linearity**
* Resistant to **overfitting**
* Works well with **tabular structured data**
* Provides **feature importance**

## Performance Metrics

| Metric          | Value   |
| --------------- | ------- |
| R² Score       | 99.84%  |
| MAPE            | 2.21%   |
| Overfitting Gap | ~0.0003 |

## Feature Importance Insight

Top contributing features:

* Current Price
* Lag Features
* Moving Averages

Low contribution:

* Volume
* Day of Week

**Price momentum dominates everything.**

## Streamlit Web Application

The project is deployed as an  **interactive dashboard** .

## Live Demo

**Streamlit App Demo:** [App Link](https://psx-oil-gas-stock-price-prediction.streamlit.app/)

## Dashboard Modules

### 1. Overview

* Project summary
* Key metrics

### 2. Data Explorer

* Filter by company, date, price
* Download dataset

### 3. Exploratory Data Analysis

* Candlestick charts
* Volume analysis
* Moving averages

### 4. Model Insights

* Feature importance
* Correlation heatmap
* Actual vs Predicted

### 5. Forecasting

* 15-day autoregressive predictions
* Trend indicators

### 6. Manual Prediction Tool

* User inputs custom OHLCV values
* Predict next-day price instantly

## Forecasting Approach

Uses  **autoregressive rolling prediction** :

1. Predict next day
2. Feed prediction back as input
3. Repeat for 15 days

## Future Improvements

* Add **LSTM / GRU models**
* Integrate **real-time APIs**
* Include **macroeconomic indicators**
* Add **news sentiment analysis**
* Portfolio optimization system

## Key Takeaways

* Stock prices are highly **auto-correlated**
* Simple features (lags + averages) are extremely powerful
* Ensemble models outperform linear models in this domain
* Deployment matters as much as modeling
