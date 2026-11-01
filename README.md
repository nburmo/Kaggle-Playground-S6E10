# Kaggle Playground Series S6E10 – Airline Passenger Satisfaction

**Rank 964 / 3,748 (top 26%) · ROC-AUC 0.96218**

My final solution is a stacked ensemble of LightGBM, XGBoost, CatBoost, RealMLP and TabPFN3.5, built on features from a "teacher" model trained on the original dataset, plus out-of-fold target encoding of flight distance and passenger segments.

## Competition overview

- **Competition:** [Playground Series S6E10](https://www.kaggle.com/competitions/playground-series-s6e10)
- **Task:** predict whether a passenger is satisfied (binary
- **Metric:** area under the ROC curve
- **Data:** 699,635 training rows and 299,844 test rows. Like all Playground competitions, the data is synthetic, generated from a real dataset. I also used that [original dataset](https://www.kaggle.com/datasets/arseniyshutko/binary-aviation-satisfaction-129k) (~129k rows).

Features include passenger details (age, gender, loyalty, travel type, class), flight distance, delays, and 13 service ratings on a 0–5 scale.

## Approach

The pipeline has three stages:

```
feature_engineering  →  tune_with_optuna  →  train_and_submit
 (features CSVs)         (best_params.json)    (submission.csv)
```

### Cleaning

- **Ratings of 0 mean "not applicable"**, not "terrible". I replaced them with 4.5, as satisfaction rates, pivoted on the 0 ratings, were better than 4, yet worse than 5.
  This didn't really matter for decision trees, but helped out RealMLP alot.
- Mapped the text columns (Class, Gender, Customer Type, Type of Travel) to integers. Class is ordinal: Eco < Eco Plus < Business.

### Feature engineering

**Teacher model from the original data.** I trained an XGBoost model on the original dataset and used its prediction (as log-odds) as a feature for the competition data. Since the synthetic data was generated from the original, this passes along the "true" relationships the generator may have blurred.

**Frequency features.** How often each exact `Flight Distance` and `Age` value appears across train and test.

**Per-distance profiles.** For each exact flight distance, the average Class, Type of Travel, Customer Type, Age, Online boarding and Inflight wifi rating. Exact distances behave a bit like route identifiers, so this describes the typical passenger on that "route".

**Target encoding, done inside each fold.** Using scikit-learn's `TargetEncoder` (smoothing 10, internal 5-fold) on:
- Flight Distance
- Flight Distance × Class
- Flight Distance × Type of Travel
- Age × Class × Type of Travel

The encoder is fit only on each fold's training rows, so the validation rows never see their own target.

### Ensembling

- **Stacking:** a logistic regression on the out-of-fold log-odds of the four models. The stack's own score is measured with a further 5-fold cross-validation so it isn't scored on rows it was trained on.

- **Final blend:** a 50/50 rank average of my stack with a [public TabPFN submission](https://www.kaggle.com/datasets/nburmo/tabpfn-csv). Rank averaging puts both on the same scale, which matters because AUC only cares about ordering.
