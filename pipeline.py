import json
import os
import warnings
import numpy as np
import pandas as pd
import optuna
import torch
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
from pytabkit import RealMLP_TD_Classifier
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.preprocessing import TargetEncoder

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

# The 13 service ratings (0 means 'not applicable')
rating_cols = [
    "Inflight wifi service", "Departure/Arrival time convenient",
    "Ease of Online booking", "Gate location", "Food and drink",
    "Online boarding", "Seat comfort", "Inflight entertainment",
    "On-board service", "Leg room service", "Baggage handling",
    "Checkin service", "Cleanliness",
]


# Probability to log-odds
def logit(p):
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return np.log(p / (1 - p))

# Feature Engineering

# Competition data plus the original dataset it was generated from
train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)
orig = pd.read_csv(ORIG_PATH)

test_ids = test["id"].copy()
X = train.drop(columns=["satisfaction", "id"])
X_test = test.drop(columns=["id"])
y = train["satisfaction"].astype(int).values


# Ratings of 0 become 4.5, text columns become numbers
def clean_data(df):
    df = df.copy()
    df[rating_cols] = df[rating_cols].replace(0, np.nan).fillna(4.5)
    df["Class"] = df["Class"].map({"Eco": 0, "Eco Plus": 1, "Business": 2})
    df["Gender"] = df["Gender"].map({"Male": 0, "Female": 1})
    df["Customer Type"] = df["Customer Type"].map({"Loyal Customer": 0, "disloyal Customer": 1})
    df["Type of Travel"] = df["Type of Travel"].map({"Personal Travel": 0, "Business travel": 1})
    return df


X = clean_data(X)
X_test = clean_data(X_test)
orig = clean_data(orig)


# How often each flight distance and age appears
freq = pd.concat([X["Flight Distance"], X_test["Flight Distance"]]).value_counts()
X["dist_freq"] = X["Flight Distance"].map(freq)
X_test["dist_freq"] = X_test["Flight Distance"].map(freq)

age_freq = pd.concat([X["Age"], X_test["Age"]]).value_counts()
X["age_freq"] = X["Age"].map(age_freq)
X_test["age_freq"] = X_test["Age"].map(age_freq)

# Average passenger profile for each exact flight distance
group_cols = ["Class", "Type of Travel", "Customer Type", "Age",
              "Online boarding", "Inflight wifi service"]
allx = pd.concat([X, X_test], ignore_index=True)

for c in group_cols:
    m = allx.groupby("Flight Distance")[c].mean()
    X[f"fd_mean_{c}"] = X["Flight Distance"].map(m)
    X_test[f"fd_mean_{c}"] = X_test["Flight Distance"].map(m)


# Teacher: a model trained on the original data, its prediction becomes a feature
orig_target = (
    orig["satisfaction"].astype(str).str.lower()
    .map({"true": 1, "false": 0, "satisfied": 1, "neutral or dissatisfied": 0})
    .astype(int)
)
orig_feats = [c for c in X.columns if c in orig.columns]

teacher_model = xgb.XGBClassifier(
    n_estimators=600, learning_rate=0.05, max_depth=8, subsample=0.8,
    colsample_bytree=0.5, tree_method="hist", device="cuda", random_state=0,
)
teacher_model.fit(orig[orig_feats], orig_target)

X["teacher"] = logit(teacher_model.predict_proba(X[orig_feats])[:, 1]).astype(np.float32)
X_test["teacher"] = logit(teacher_model.predict_proba(X_test[orig_feats])[:, 1]).astype(np.float32)

# Optuna parameter optimisation

# Reload the saved features
X = pd.read_csv("train_features.csv")
X_test = pd.read_csv("test_features.csv")

y = X.pop("satisfaction").values
test_ids = X_test.pop("id")


# Keys for target encoding (encoded inside each split so the target never leaks)
for df in [X, X_test]:
    df["te_fd"] = df["Flight Distance"].astype(str)
    df["te_fd_class"] = df["Flight Distance"].astype(str) + "_" + df["Class"].astype(str)
    df["te_fd_travel"] = df["Flight Distance"].astype(str) + "_" + df["Type of Travel"].astype(str)
    df["te_seg"] = df["Age"].astype(str) + "_" + df["Class"].astype(str) + "_" + df["Type of Travel"].astype(str)

te_cols = ["te_fd", "te_fd_class", "te_fd_travel", "te_seg"]


# Swap the string keys for their encoded numbers
def put_te(df, arr):
    df = df.drop(columns=te_cols)
    df[te_cols] = arr
    return df


# CatBoost treats these as categories, so every exact value gets its own stats
catboost_cat_cols = rating_cols + [
    "Age", "Flight Distance", "Class", "Gender", "Customer Type", "Type of Travel",
]


def to_catboost(df):
    df = df.copy()
    df[catboost_cat_cols] = df[catboost_cat_cols].astype(str)
    return df

# Split: train / early stopping / validation
tr_idx, va_idx = train_test_split(sample_idx, test_size=0.2, stratify=y[sample_idx], random_state=42)
in_tr_idx, in_es_idx = train_test_split(tr_idx, test_size=0.1, stratify=y[tr_idx], random_state=42)

Xtr = X.iloc[in_tr_idx].copy()
Xes = X.iloc[in_es_idx].copy()
Xva = X.iloc[va_idx].copy()

# Fit the encoder on the training rows only
te = TargetEncoder(target_type="binary", smooth=10, cv=5, random_state=42)
Xtr = put_te(Xtr, te.fit_transform(Xtr[te_cols], y[in_tr_idx]))
Xes = put_te(Xes, te.transform(Xes[te_cols]))
Xva = put_te(Xva, te.transform(Xva[te_cols]))

ytr = y[in_tr_idx]
yes = y[in_es_idx]
yva = y[va_idx]

# RealMLP can't take NaN, so fill any left with medians
realmlp_medians = Xtr.median()
Xtr_mlp = Xtr.fillna(realmlp_medians)
Xva_mlp = Xva.fillna(realmlp_medians)

# CatBoost copies with category columns as strings
Xtr_cat = to_catboost(Xtr)
Xes_cat = to_catboost(Xes)
Xva_cat = to_catboost(Xva)

# Each objective trains one model and returns its validation AUC
def objective_lightgbm(trial):
    params = dict(
        learning_rate=trial.suggest_float("learning_rate", 0.02, 0.1, log=True),
        num_leaves=trial.suggest_int("num_leaves", 31, 255),
        min_child_samples=trial.suggest_int("min_child_samples", 5, 100),
        subsample=trial.suggest_float("subsample", 0.6, 1.0),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.4, 1.0),
        reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 20.0, log=True),
        reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 20.0, log=True),
    )
    model = lgb.LGBMClassifier(
        objective="binary", metric="auc", n_estimators=5000, subsample_freq=1,
        max_bin=4095, random_state=42, verbose=-1, **params,
    )
    model.fit(Xtr, ytr, eval_set=[(Xes, yes)],
              callbacks=[lgb.early_stopping(150, verbose=False)])
    return roc_auc_score(yva, model.predict_proba(Xva)[:, 1])

def objective_xgboost(trial):
    params = dict(
        learning_rate=trial.suggest_float("learning_rate", 0.02, 0.1, log=True),
        max_depth=trial.suggest_int("max_depth", 4, 10),
        min_child_weight=trial.suggest_float("min_child_weight", 1.0, 30.0, log=True),
        subsample=trial.suggest_float("subsample", 0.6, 1.0),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.4, 1.0),
        reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 20.0, log=True),
        reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 20.0, log=True),
        gamma=trial.suggest_float("gamma", 0.0, 5.0),
    )
    model = xgb.XGBClassifier(
        objective="binary:logistic", eval_metric="auc", n_estimators=5000,
        tree_method="hist", device="cuda", max_bin=4096, early_stopping_rounds=150,
        random_state=42, **params,
    )
    model.fit(Xtr, ytr, eval_set=[(Xes, yes)], verbose=False)
    return roc_auc_score(yva, model.predict_proba(Xva)[:, 1])

def objective_catboost(trial):
    params = dict(
        learning_rate=trial.suggest_float("learning_rate", 0.03, 0.15, log=True),
        depth=trial.suggest_int("depth", 6, 8),
        l2_leaf_reg=trial.suggest_float("l2_leaf_reg", 1.0, 30.0, log=True),
        subsample=trial.suggest_float("subsample", 0.6, 1.0),
        random_strength=trial.suggest_float("random_strength", 0.1, 10.0, log=True),
        border_count=trial.suggest_int("border_count", 64, 254),
    )
    model = CatBoostClassifier(
        loss_function="Logloss", eval_metric="AUC", iterations=5000,
        bootstrap_type="Bernoulli", task_type="GPU", cat_features=catboost_cat_cols,
        max_ctr_complexity=4, one_hot_max_size=2, random_seed=42, verbose=0, **params,
    )
    model.fit(Xtr_cat, ytr, eval_set=(Xes_cat, yes), early_stopping_rounds=150)
    return roc_auc_score(yva, model.predict_proba(Xva_cat)[:, 1])

def objective_realmlp(trial):
    params = dict(
        hidden_sizes="rectangular",
        n_hidden_layers=trial.suggest_int("n_hidden_layers", 2, 4),
        hidden_width=trial.suggest_categorical("hidden_width", [128, 256, 512]),
        lr=trial.suggest_float("lr", 0.01, 0.15, log=True),
        p_drop=trial.suggest_float("p_drop", 0.0, 0.4),
        wd=trial.suggest_float("wd", 1e-3, 0.1, log=True),
        plr_sigma=trial.suggest_float("plr_sigma", 0.03, 1.0, log=True),
        batch_size=trial.suggest_categorical("batch_size", [512, 1024, 2048]),
        n_epochs=trial.suggest_int("n_epochs", 32, 128, step=16),
    )
    model = RealMLP_TD_Classifier(
        device="cuda", random_state=42, n_cv=1, n_refit=0,
        val_fraction=0.1, verbosity=0, **params,
    )
    model.fit(Xtr_mlp, ytr)
    return roc_auc_score(yva, model.predict_proba(Xva_mlp)[:, 1])

# Run the searches; LightGBM starts from the old hand-tuned params
best_params = {}

study_lgb = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
study_lgb.enqueue_trial(dict(
    learning_rate=0.1, num_leaves=72, min_child_samples=17, subsample=0.9848,
    colsample_bytree=0.8426, reg_alpha=5.7279, reg_lambda=3.7959,
))
study_lgb.optimize(objective_lightgbm, n_trials=40, show_progress_bar=True)
best_params["lightgbm"] = study_lgb.best_params
print("LightGBM best AUC:", round(study_lgb.best_value, 5))

study_xgb = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
study_xgb.optimize(objective_xgboost, n_trials=40, show_progress_bar=True)
best_params["xgboost"] = study_xgb.best_params
print("XGBoost best AUC:", round(study_xgb.best_value, 5))

study_cat = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
study_cat.optimize(objective_catboost, n_trials=30, show_progress_bar=True)
best_params["catboost"] = study_cat.best_params
print("CatBoost best AUC:", round(study_cat.best_value, 5))

study_mlp = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
study_mlp.optimize(objective_realmlp, n_trials=15, show_progress_bar=True)
best_params["realmlp"] = {"hidden_sizes": "rectangular", **study_mlp.best_params}
print("RealMLP best AUC:", round(study_mlp.best_value, 5))

# Save the winners for the training stage
with open("best_params.json", "w") as f:
    json.dump(best_params, f, indent=2)
print("best_params.json written")


# Final training and Submission

with open("best_params.json") as f:
    best_params = json.load(f)

# Each model is averaged over these seeds
seeds = [0,1,2]
model_names = ["lightgbm", "xgboost", "catboost", "realmlp"]

oof = {name: np.zeros(len(X)) for name in model_names}
test_preds = {name: np.zeros(len(X_test)) for name in model_names}

# 10 folds, same folds for every model
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
n_runs = skf.n_splits * len(seeds)

for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y), start=1):
    print(f" Fold {fold}/10")

    # Hold back 10% of the fold's training rows for early stopping
    in_tr_idx, in_es_idx = train_test_split(
        tr_idx, test_size=0.1, stratify=y[tr_idx], random_state=42 + fold
    )

    Xtr = X.iloc[in_tr_idx].copy()
    Xes = X.iloc[in_es_idx].copy()
    Xva = X.iloc[va_idx].copy()
    Xte = X_test.copy()

    ytr = y[in_tr_idx]
    yes = y[in_es_idx]
    yva = y[va_idx]

    # Fit the encoder on the training rows only
    te = TargetEncoder(target_type="binary", smooth=10, cv=5, random_state=42 + fold)
    Xtr = put_te(Xtr, te.fit_transform(Xtr[te_cols], ytr))
    Xes = put_te(Xes, te.transform(Xes[te_cols]))
    Xva = put_te(Xva, te.transform(Xva[te_cols]))
    Xte = put_te(Xte, te.transform(Xte[te_cols]))

    # Model-specific copies of the data
    Xtr_cat = to_catboost(Xtr)
    Xes_cat = to_catboost(Xes)
    Xva_cat = to_catboost(Xva)
    Xte_cat = to_catboost(Xte)

    mlp_medians = Xtr.median()
    Xtr_mlp = Xtr.fillna(mlp_medians)
    Xva_mlp = Xva.fillna(mlp_medians)
    Xte_mlp = Xte.fillna(mlp_medians)

    for seed in seeds:

        # LightGBM (CPU), fine bins so each exact distance is kept
        model_lgb = lgb.LGBMClassifier(
            objective="binary", metric="auc", n_estimators=5000, subsample_freq=1,
            max_bin=4095, random_state=seed, verbose=-1, **best_params["lightgbm"],
        )
        model_lgb.fit(Xtr, ytr, eval_set=[(Xes, yes)],
                      callbacks=[lgb.early_stopping(150, verbose=False)])
        oof["lightgbm"][va_idx] += model_lgb.predict_proba(Xva)[:, 1] / len(seeds)
        test_preds["lightgbm"] += model_lgb.predict_proba(Xte)[:, 1] / n_runs

        # XGBoost (GPU)
        model_xgb = xgb.XGBClassifier(
            objective="binary:logistic", eval_metric="auc", n_estimators=5000,
            tree_method="hist", device="cuda", max_bin=4096, early_stopping_rounds=150,
            random_state=seed, **best_params["xgboost"],
        )
        model_xgb.fit(Xtr, ytr, eval_set=[(Xes, yes)], verbose=False)
        oof["xgboost"][va_idx] += model_xgb.predict_proba(Xva)[:, 1] / len(seeds)
        test_preds["xgboost"] += model_xgb.predict_proba(Xte)[:, 1] / n_runs

        # CatBoost (GPU) with the category columns
        model_cat = CatBoostClassifier(
            loss_function="Logloss", eval_metric="AUC", iterations=5000,
            bootstrap_type="Bernoulli", task_type="GPU", cat_features=catboost_cat_cols,
            max_ctr_complexity=4, one_hot_max_size=2, random_seed=seed, verbose=0,
            **best_params["catboost"],
        )
        model_cat.fit(Xtr_cat, ytr, eval_set=(Xes_cat, yes), early_stopping_rounds=150)
        oof["catboost"][va_idx] += model_cat.predict_proba(Xva_cat)[:, 1] / len(seeds)
        test_preds["catboost"] += model_cat.predict_proba(Xte_cat)[:, 1] / n_runs

        # RealMLP (GPU)
        model_mlp = RealMLP_TD_Classifier(
            device="cuda", random_state=seed, n_cv=1, n_refit=0,
            val_fraction=0.1, verbosity=0, **best_params["realmlp"],
        )
        model_mlp.fit(Xtr_mlp, ytr)
        oof["realmlp"][va_idx] += model_mlp.predict_proba(Xva_mlp)[:, 1] / len(seeds)
        test_preds["realmlp"] += model_mlp.predict_proba(Xte_mlp)[:, 1] / n_runs

    for name in model_names:
        print(f"  {name:9s} fold AUC: {roc_auc_score(yva, oof[name][va_idx]):.5f}")


# Score each model on its out-of-fold predictions
print("\nIndividual Model OOF AUCs")
for name in model_names:
    print(f"{name:9s} {roc_auc_score(y, oof[name]):.6f}")


# Stack: logistic regression on each model's log-odds
X_oof_logits = np.column_stack([logit(oof[name]) for name in model_names])
X_test_logits = np.column_stack([logit(test_preds[name]) for name in model_names])

meta_model = LogisticRegression(fit_intercept=True, random_state=42)

# Honest score for the stack, scored on rows the meta model didn't see
stacked_oof = cross_val_predict(
    meta_model, X_oof_logits, y,
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    method="predict_proba",
)[:, 1]
print(f"\nStacked ensemble OOF AUC (cross-validated meta model): {roc_auc_score(y, stacked_oof):.6f}")

meta_model.fit(X_oof_logits, y)
print("Stacking weights:", dict(zip(model_names, meta_model.coef_[0].round(4))))
print("Intercept:", meta_model.intercept_[0])

final_test_preds = meta_model.predict_proba(X_test_logits)[:, 1]


# Rank-blend with a public TabPFN submission 
other_submission = pd.read_csv(TABPFN_PATH)
other_preds = other_submission.set_index("id").loc[test_ids, "satisfaction"].values

our_ranks = rankdata(final_test_preds) / len(final_test_preds)
other_ranks = rankdata(other_preds) / len(other_preds)

print(f"\nCorrelation with other submission: {np.corrcoef(our_ranks, other_ranks)[0, 1]:.4f}")

our_weight = 0.5
final_test_preds = our_weight * our_ranks + (1 - our_weight) * other_ranks

# Write the submission
submission = pd.DataFrame({"id": test_ids, "satisfaction": final_test_preds})
submission.to_csv("submission.csv", index=False)
print("submission.csv written")
