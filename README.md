# Team Sentinels - Kaggle ML Datathon Nov 2025 . 

## 🏎️ Racing Lap Time Prediction - Advanced Stacking Ensemble

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Machine Learning](https://img.shields.io/badge/ML-XGBoost%20%7C%20LightGBM%20%7C%20CatBoost-orange.svg)](https://github.com)
[![RMSE](https://img.shields.io/badge/RMSE-0.15087-success.svg)](https://github.com)

> **Advanced stacking ensemble model for predicting Formula racing lap times with state-of-the-art accuracy**

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [Performance Results](#-performance-results)
- [Architecture](#-architecture)
- [Feature Engineering](#-feature-engineering)
- [Installation](#-installation)
- [Usage](#-usage)
- [Model Details](#-model-details)
- [File Structure](#-file-structure)
- [Requirements](#-requirements)
- [Acknowledgements](#-acknowledgements)
- [Learnings & Scope of Improvements](#-learnings&scope-of-improvements)

---

## 🎯 Overview

Though during this competition of predicting lap times based on given dataset and features we experimented with various different models and tried multiple techniques , In this ReadMe file I am going to write about the model which gave us the best RMSE . This project implements a **sophisticated stacking ensemble** for predicting Formula racing lap times using machine learning. By combining three powerful gradient boosting algorithms (XGBoost, LightGBM, and CatBoost) with a Ridge meta-learner, we achieve exceptional predictive accuracy with RMSE of 0.15087.

**Final Result:** 🏆 **RMSE: 0.15087 seconds**

---

## ✨ Key Features

- **🤖 Triple Gradient Boosting Ensemble**
  - XGBoost with 30,000 estimators
  - LightGBM with 12,500 estimators  
  - CatBoost with 12,500 estimators

- **🧠 Intelligent Meta-Learning**
  - Ridge Regression meta-learner for optimal stacking
  - Combines base model predictions for superior accuracy

- **📊 Advanced Feature Engineering**
  - **67 total features**: 29 original + 38 engineered
  - 23 core engineered features
  - 15 innovative new features

- **🔧 Robust Preprocessing**
  - SafeLabelEncoder for handling unseen categories
  - StandardScaler for feature normalization
  - Comprehensive missing value handling

- **💾 Google Drive Integration**
  - Seamless data loading from Drive
  - Automated prediction saving
  - Individual model outputs + final stacked predictions

---

## 📈 Performance Results

| Model | Training RMSE | Notes |
|-------|--------------|-------|
| **XGBoost** | ~0.18-0.20 | Strong baseline performance |
| **LightGBM** | ~0.17-0.19 | Fast and efficient |
| **CatBoost** | ~0.17-0.19 | Excellent categorical handling |
| **🏆 Stacked Ensemble** | **0.15087** | **Best performance - 25-30% improvement** |

### Key Achievements
- ✅ **0.15087 seconds RMSE** on test predictions
- ✅ **25-30% improvement** over individual models
- ✅ **67 engineered features** for comprehensive signal capture
- ✅ **Robust handling** of unseen categories and missing data

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────┐
│                   INPUT DATA                        │
│  (train.csv: Racing telemetry + driver statistics)  │
└────────────────────┬────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────┐
│              PREPROCESSING PIPELINE                 │
│  • Handle missing values (median/mode imputation)   │
│  • SafeLabelEncoder (10 categorical features)       │
│  • Feature engineering (38 new features)            │
│  • StandardScaler normalization                     │
└────────────────────┬────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────┐
│                BASE MODELS (Layer 1)                │
│  ┌──────────────┐  ┌──────────────┐  ┌───────────┐  │
│  │   XGBoost    │  │  LightGBM    │  │ CatBoost  │  │
│  │ 30K trees    │  │ 12.5K trees  │  │12.5K trees│  │
│  │ depth=20     │  │ depth=15     │  │ depth=12  │  │
│  │ lr=0.1       │  │ lr=0.08      │  │ lr=0.08   │  │
│  └──────┬───────┘  └──────┬───────┘  └─────┬─────┘  │
│         │                 │                │        │
│         └─────────────────┴────────────────┘        │
└────────────────────────────┬────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────┐
│            META-LEARNER (Layer 2)                   │
│         Ridge Regression (alpha=1.0)                │
│    Optimally combines base predictions              │
└────────────────────────────┬────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────┐
│              FINAL PREDICTIONS                      │
│         RMSE: 0.15087 seconds                       │
└─────────────────────────────────────────────────────┘
```

---

## 🔬 Feature Engineering

### Original Features (29)
- **Circuit Characteristics**: Length, corners, laps, average speed
- **Environmental**: Temperature (ambient/track), humidity, weather
- **Driver Statistics**: Starts, wins, podiums, points, finishes
- **Race Conditions**: Tire compound, degradation, pit stop duration
- **Positions**: Start position, final position

### Engineered Features (38)

#### Core Features (23)
1. **Ratio Features**: Speed-to-circuit ratio, position rates
2. **Performance Metrics**: Win rate, podium rate, DNF rate, success rate
3. **Interaction Terms**: Speed×Corners, Temp×Humidity, Degradation×Distance
4. **Polynomial Features**: Speed², Corners², Temperature²
5. **Complexity Indices**: Circuit complexity, average speed per corner
6. **Experience**: Log-scaled starts, points per race

#### New Advanced Features (15)
7. **Lap Calculations**: Seconds per lap, pit impact per lap, time lost in pits
8. **Position Analysis**: Starting advantage, position change, final position impact
9. **Technical Metrics**: Technical difficulty, speed degradation, corner-speed ratio
10. **Performance Synthesis**: Experience-success ratio, consistency score
11. **Environmental Interactions**: Weather-temp combined, tire-temp interaction
12. **Efficiency Metrics**: Points per podium, win efficiency

---

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- Google Colab (recommended) or Jupyter Notebook
- Google Drive account (for data storage)

### Required Libraries

```bash
pip install xgboost lightgbm catboost pandas numpy scikit-learn
```

### Quick Start

1. **Clone or download** `My_Best_Code.ipynb`
2. **Upload to Google Colab**
3. **Prepare your data**:
   - Place `train(1).csv` in Google Drive
   - Place `test.csv` in Google Drive
4. **Run all cells** in the notebook

---

## 💻 Usage

### Basic Workflow

```python
# 1. Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# 2. Load data
train_df = pd.read_csv('/content/drive/MyDrive/train(1).csv')
test_df = pd.read_csv('/content/drive/MyDrive/test.csv')

# 3. Initialize and train
ensemble = StackingEnsemblePredictor()
train_rmse = ensemble.train(train_df, OUTPUT_DIR)

# 4. Generate predictions
stacked_preds, individual_preds = ensemble.predict(test_df, OUTPUT_DIR)

# 5. Save results
# Automatically saved to Google Drive:
# - predictions_xgboost.csv
# - predictions_lightgbm.csv
# - predictions_catboost.csv
# - predictions_STACKED_ENSEMBLE.csv ⭐
```

### Expected Runtime
- **Training Time**: ~90-120 minutes (Google Colab with GPU)
- **Prediction Time**: ~2-5 minutes

---

## 🤖 Model Details

### XGBoost Configuration
```python
n_estimators=30000
max_depth=20
learning_rate=0.1
subsample=0.8
colsample_bytree=0.8
min_child_weight=3
gamma=0.1
reg_alpha=0.1
reg_lambda=1.0
```

### LightGBM Configuration
```python
n_estimators=12500
max_depth=15
learning_rate=0.08
num_leaves=63
subsample=0.8
colsample_bytree=0.8
min_child_samples=20
reg_alpha=0.1
reg_lambda=1.0
```

### CatBoost Configuration
```python
iterations=12500
depth=12
learning_rate=0.08
l2_leaf_reg=3
```

### Ridge Meta-Learner
```python
alpha=1.0  # L2 regularization strength
```

---

## 📁 File Structure

```
racing-lap-time-prediction/
│
├── My_Best_Code.ipynb           # Main notebook (RMSE: 0.15087)
├── README.md                     # This file
│
├── data/
│   ├── train(1).csv             # Training dataset
│   └── test.csv                 # Test dataset
│
└── outputs/
    ├── predictions_xgboost.csv
    ├── predictions_lightgbm.csv
    ├── predictions_catboost.csv
    └── predictions_STACKED_ENSEMBLE.csv  ⭐ BEST
```

---

## 📦 Requirements

```
python>=3.8
xgboost>=1.5.0
lightgbm>=3.3.0
catboost>=1.0.0
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
```

---

## 🎓 Key Innovations

1. **SafeLabelEncoder Class**
   - Gracefully handles unseen categorical values
   - Prevents prediction failures on new data

2. **Comprehensive Feature Engineering**
   - 38 carefully designed features
   - Captures non-linear relationships
   - Domain-specific racing insights

3. **Optimized Stacking Strategy**
   - Multiple diverse base models
   - Ridge regression for stable meta-learning
   - Reduces overfitting while maximizing accuracy

4. **Production-Ready Pipeline**
   - Robust error handling
   - Automatic file management
   - Clear progress tracking

---

## 📊 Results Breakdown

| Metric | Value |
|--------|-------|
| **Final RMSE** | **0.15087 seconds** |
| Training Samples | ~300,000+ |
| Test Samples | ~700,000+ |
| Total Features | 67 |
| Model Training Time | ~100 minutes |
| Prediction Time | ~3 minutes |


---

## 🙏 Acknowledgements

- Teammate - Jaimin Koriya 
- GDG SVNIT for hosting this Competition 

---


## 🤝 Learnings and Scope of Improvements 

During this competition we did indeed do optuna tuning for hyperparameters but did not implement it , thus making a crucial mistake of just focusing on 1 hyperparameter which is n_estimators to get performance improvement . this can also be observed by a look on our hyperparameters which speaks by itself , (30k,20k,12.5k) n_estimators . over many models uploaded in the repositories not only the one we have talked about in this ReadMe file. 

 Areas for improvement Identified :
- [ ] Hyperparameter tuning with Optuna
- [ ] Additional feature engineering
- [ ] Cross-validation implementation
- [ ] Feature importance analysis
- [ ] Ensemble weight optimization
