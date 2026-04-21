# Model Training — Random Forest Classifier

This document covers how the NHL game outcome predictor is trained: the dataset construction, cross-validation strategy, hyperparameter tuning process, evaluation methodology, and how to interpret the results. The full implementation lives in [ts_predict.ipynb](ts_predict.ipynb).

---

## Problem Framing

The prediction task is a **binary classification** problem. Each game is an observation, and the label is:

```
target = 1  →  home team wins
target = 0  →  away team wins
```

Overtime and shootout losses are included as losses for the losing team. There is no draw in NHL regular season play.

**Why binary classification instead of regression (goal differential)?**
Predicting exact scores adds model complexity and introduces regression-to-mean artifacts. The binary outcome is directly usable: predict the winner. A probability output from `predict_proba` gives a confidence level on top of that.

---

## Dataset

**Source file:** `scripts/generated/data/nhl_data.csv`

**Total games:** ~3,877 across 3 seasons (Seasons 1–3)

| Season | Games | Role |
|---|---|---|
| Season 1 | ~1,293 | Training (Fold 1) |
| Season 2 | ~1,292 | Training (Fold 1+2); Validation (Fold 1) |
| Season 3 | ~1,292 | Test set (never seen during tuning) |

**Target distribution:** 54.19% home wins, 45.81% away wins. The home-ice advantage is a real and well-documented phenomenon in hockey. The model must learn to beat this baseline, not just predict home wins all the time.

**Features used:** 71 engineered features (no raw in-game stats). Full feature list and construction details are in [ml_dev/README.md](../README.md).

**NaN handling:** Rows where L5 features are NaN (first 1–4 games of any season) are dropped before training. This affects only a small number of rows per season and is preferable to imputation, which could introduce spurious signal in the early-season context.

---

## Why a Random Forest?

Random Forests are well-suited to this problem for several reasons:

- **Tabular data:** The feature set is a flat table of 71 numeric columns. Tree-based models generally outperform linear models on tabular data without requiring extensive scaling or normalization.
- **Non-linearity:** Interactions between features (e.g., goalie save percentage combined with opponent scoring EWM) are not easily captured by linear models. Trees handle these naturally.
- **Robustness to outliers:** A single outlier game (40+ shots, goalie injury) affects only the specific trees and splits it falls into, not the entire model.
- **No feature scaling required:** EWM values, percentages, and rest day counts live on very different scales. Random Forests are invariant to monotonic feature transformations.
- **Interpretability via feature importance:** Gini importance from the ensemble gives a rough but useful picture of which features are most predictive.

The project also has XGBoost installed as a dependency for future experimentation. The current production model is a Random Forest.

---

## Cross-Validation Strategy

Standard k-fold cross-validation shuffles data randomly, which is fundamentally wrong for time-series data. Using data from January to evaluate October predictions means your validation set precedes your training set in calendar time — a form of look-ahead bias.

**Solution: Season-level expanding window cross-validation.**

This is a walk-forward approach where each fold expands the training window by one season and evaluates on the next:

```
Fold 1:  Train = [Season 1]              →  Validate = Season 2
Fold 2:  Train = [Season 1 + Season 2]   →  Validate = Season 3*
```

*Season 3 serves as the final test set and is held out entirely during hyperparameter search. Fold 2's validation role during grid search uses an internal `TimeSeriesSplit` on the training portion only.

**Why this matters:**
- It simulates exactly how the model will be used in production: train on historical seasons, predict the next one.
- It reveals whether the model generalizes across seasons or just memorizes patterns specific to certain teams or years.
- The expanding window (adding Season 1+2 for Fold 2) reflects that more data is better for training — you don't discard Season 1 just because it's older.

**Baseline for each fold:** Since the target is binary, a trivial baseline is to always predict the home team wins. This achieves ~54.19% accuracy (the home win rate in the data).

---

## Hyperparameter Tuning

### Grid Search Setup

```python
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
}
```

Total combinations: 36

The grid search is run with `TimeSeriesSplit(n_splits=3)` on the training set (Seasons 1+2). This means within-training cross-validation also respects temporal ordering: early games train, later games validate.

**Scoring metric:** Accuracy. For a balanced binary classification problem (54/46 split is close enough to balanced), accuracy is a reasonable optimization target. F1 or log-loss would also be valid alternatives.

### Best Parameters Found

| Parameter | Value | Explanation |
|---|---|---|
| `n_estimators` | 300 | More trees reduce variance; diminishing returns beyond ~200-300 |
| `max_depth` | 10 | Key regularization parameter — prevents overfitting to training seasons |
| `min_samples_split` | 5 | Require at least 5 samples to justify a split; smooths decision boundaries |
| `min_samples_leaf` | 2 | Leaf nodes must have ≥ 2 samples; prevents highly specific leaves |

**The most impactful parameter:** `max_depth`. Setting it to `None` (unlimited) leads to trees that can memorize individual training games, producing high training accuracy (~75%+) but poor cross-season generalization. `max_depth=10` substantially closes the train/test gap.

**Best CV score (on training folds):** 67.91%

---

## Model Evaluation

### Final Test Results (Season 3 — Fully Held Out)

The model is retrained on the full training set (Seasons 1+2) with the best hyperparameters, then evaluated once on Season 3. This score is the honest estimate of generalization performance.

| Metric | Value |
|---|---|
| **Accuracy** | **67.18%** |
| Naive baseline (always home) | 54.19% |
| Precision — home win (class 1) | 65% |
| Recall — home win (class 1) | 80% |
| F1-score — home win (class 1) | 0.72 |
| Precision — away win (class 0) | 73% |
| Recall — away win (class 0) | 53% |
| F1-score — away win (class 0) | 0.61 |

### Confusion Matrix (Season 3 Test Set, 1,292 games)

```
                     Predicted Home Win    Predicted Away Win
Actual Home Win           544 (TP)              132 (FN)
Actual Away Win           292 (FP)              324 (TN)
```

**Interpretation:**

The model has higher recall for home wins (80%) than for away wins (53%). It is good at catching home wins but misses a fair number of away wins. This is partly a product of the data distribution (more home wins exist to train on) and partly reflects genuine difficulty — away wins are more upset-like and harder to anticipate.

The 73% precision on away wins is actually better than home win precision (65%), meaning when the model does predict an away win, it is more often right. Away win predictions are rarer and more confident.

### Cross-Season Validation Summary

| Fold | Train | Test | Accuracy |
|---|---|---|---|
| Fold 1 | Season 1 | Season 2 | 69.04% |
| Fold 2 | Seasons 1+2 | Season 3 | 66.80% |
| Mean | — | — | 67.92% |
| **Final test** | **Seasons 1+2** | **Season 3 (held out)** | **67.18%** |

The slight drop from Fold 1 to Fold 2 is expected and healthy — it reflects the model encountering genuine year-to-year variance rather than overfitting.

---

## Feature Importance

Feature importance in a Random Forest is measured by the mean decrease in Gini impurity across all splits that use a given feature, averaged over all trees (also called Gini importance or MDI).

**Top contributors to the model (approximate ranking):**

1. **Goalie EWM features** — save percentage EWM and even-strength goals against EWM are consistently among the highest-importance features. Goalie performance is the single largest game-to-game variance factor in hockey.
2. **Season standings (pointPctg, win percentage)** — these encode overall team quality in a way that EWMs cannot: a team 30 games into a dominant season shows it clearly in their point percentage.
3. **Team EWM differentials** — `home_goal_diff_ewm` and `home_ga_diff_ewm` summarize relative offensive/defensive strength in a single number.
4. **Team GF/GA EWMs** — the individual team offensive and defensive EWM values provide granularity that differentials compress away.
5. **Recent momentum (win_pct_l5)** — short-term form adds signal beyond the slower-moving EWMs.
6. **Rest days** — back-to-back scheduling effects are real, especially for goalies.
7. **Head-to-head features** — lower importance overall but useful for matchup-specific dynamics within a season.
8. **Faceoff, hits, PIM EWMs** — lower individual importance but collectively useful; they describe style of play differences.

**Caveat on Gini importance:** MDI-based importance can be biased toward high-cardinality continuous features and toward correlated features (splitting importance among correlated columns). It is directionally informative but not a precise causal ranking.

---

## Model Artifacts

The trained model is serialized and stored for use by the prediction engine:

| File | Description |
|---|---|
| `models/nhl_rf_model.pkl` | Trained `RandomForestClassifier` — 300 trees, max_depth=10 |
| `models/feature_names.pkl` | Python list of 71 feature names in the exact order expected by the model |

`feature_names.pkl` is critical for inference. The prediction engine uses it to assemble the feature vector in exactly the same column order as training. A mismatch in feature order would silently produce incorrect predictions.

**Model size:** ~8.2 MB. Inference on a single game is effectively instant; 15–30 games per day runs in milliseconds.

---

## Verification

[verify_calculations.ipynb](verify_calculations.ipynb) contains manual spot-checks of the engineered features:

- L5 win counts checked against manually counted game-by-game results
- EWM values verified against expected recursive formula outputs
- Goal differentials verified as home − away
- Season win percentages cross-checked against official standings
- H2H cumulative win counts verified row-by-row for sample matchups

These tests exist because silent feature engineering bugs are the most dangerous class of ML error — they don't cause exceptions, they just make the model train on subtly wrong data.

---

## Potential Improvements

| Improvement | Expected Impact | Notes |
|---|---|---|
| XGBoost or LightGBM | Moderate | Generally outperform RF on tabular data; worth a direct comparison |
| Probability calibration (Platt scaling / isotonic regression) | Low–Moderate | RF probabilities can be poorly calibrated; calibration improves log-loss |
| SHAP values | Explainability only | Per-prediction feature attribution for inspecting individual games |
| Vegas line as feature | High | Betting lines aggregate enormous information; may outweigh many engineered features |
| Expected goals (xG) | Moderate | More predictive than shot count alone; requires a separate model or data source |
| Player-level injury flags | Moderate–High | Missing star players is high-signal; currently not captured |
| Larger hyperparameter grid | Low | Current grid is shallow; Bayesian optimization could find better configurations |
| Ensemble (RF + XGB + LR) | Low–Moderate | Stacking can squeeze out a few percentage points |
