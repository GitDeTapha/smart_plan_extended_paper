# utils_ml.py
# Routines of the two prediction notebooks: 02_no_show_prediction.ipynb and
# 03_service_time_groups.ipynb (same pipeline, same metrics).
# Taken from utils_ns.py of the conference paper: the metrics are computed
# exactly as there (sklearn, average='macro' for recall, precision and F1).
# Every choice (grids, threshold candidates, SMOTE ratio, encoder) is passed as
# an argument from the "Paramètres" cell of the notebook.

import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (accuracy_score, auc, confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, TargetEncoder


############################################################
#                                                          #
#        ------  Service-time groups    -------            #
#                                                          #
############################################################

def kmedians_two_groups(values):
    """
    Exact K-medians with K = 2 on one variable: the cutoff c that splits the
    values into a short group (<= c) and a long group (> c) minimising
        sum_short |x - median(short)| + sum_long |x - median(long)|.
    With one variable the groups of an optimal solution are two intervals, so
    trying every cutoff between two distinct sorted values gives the exact
    optimum (no random start, unlike Lloyd's algorithm). For each candidate
    the sum of absolute deviations of a sorted segment is read from
    cumulative sums.

    Args:
        values (array): The values to split (e.g. service_time of the training consultations).

    Returns:
        dict: cutoff, cost, median of each group, share of the long group.
    """
    x = np.sort(np.asarray(values, dtype=float))
    n = len(x)
    prefix = np.concatenate([[0.0], np.cumsum(x)])

    def segment_cost(lo, hi):
        """Sum of |x - median| on the sorted segment x[lo:hi], for arrays of lo and hi."""
        m = (lo + hi - 1) // 2
        return (x[m] * (m - lo) - (prefix[m] - prefix[lo])) + ((prefix[hi] - prefix[m + 1]) - x[m] * (hi - m - 1))

    cuts = np.flatnonzero(x[1:] > x[:-1]) + 1            # first index of the long group, between distinct values
    costs = segment_cost(np.zeros_like(cuts), cuts) + segment_cost(cuts, np.full_like(cuts, n))
    k = cuts[np.argmin(costs)]
    return {"cutoff": x[k - 1], "cost": costs.min(), "median_short": np.median(x[:k]),
            "median_long": np.median(x[k:]), "share_long": (n - k) / n}


############################################################
#                                                          #
#        ------      Training           -------            #
#                                                          #
############################################################

def make_pipeline(model, categorical, target_encoder, smote_ratio, random_state):
    """
    Pipeline applied to the training data of every fold, then to validation
    and test (where SMOTE does nothing: imblearn only resamples during fit):
      1. target encoding of the categorical columns (output first, then the others);
      2. SMOTE up to `smote_ratio` minority / majority (skipped if None);
      3. StandardScaler (the paper used MinMaxScaler: with it the MLP only
         predicted the majority class, see notebook 02; trees are not affected);
      4. the model.

    Args:
        model: Unfitted classifier.
        categorical (list): Categorical columns, target-encoded.
        target_encoder (dict): Arguments of sklearn's TargetEncoder.
        smote_ratio (float or None): sampling_strategy of SMOTE.
        random_state (int): Seed of SMOTE.

    Returns:
        imblearn Pipeline.
    """
    encoder = ColumnTransformer([("target", TargetEncoder(**target_encoder), categorical)], remainder="passthrough")
    steps = [("encoder", encoder)]
    if smote_ratio is not None:
        steps.append(("smote", SMOTE(sampling_strategy=smote_ratio, random_state=random_state)))
    steps += [("scaler", StandardScaler()), ("model", model)]
    return Pipeline(steps)


def train_model(pipeline, grid, X_train, y_train, scoring, cv_folds, random_state, model_path, retrain):
    """
    Grid search with cross-validation on the training set (whole pipeline in
    each fold), then refit of the best pipeline on the whole training set. A
    pipeline without grid (None) is fitted once.

    The fitted pipeline is saved to `model_path` (joblib) with its
    cross-validation results and its settings. When the notebook is run again,
    the saved pipeline is loaded instead of trained, unless `retrain` is True.
    If the settings changed since it was saved (grid, model parameters, size of
    the training set), an error is raised: a stale model is never reused.

    Args:
        pipeline: Output of make_pipeline.
        grid (dict or None): Hyperparameters of the model, e.g. {"max_depth": [10, 20]}.
        X_train, y_train: Training set.
        scoring (str): Score of the grid search, e.g. "f1_macro".
        cv_folds (int): Number of folds.
        random_state (int): Seed of the folds.
        model_path (str): File of the saved pipeline.
        retrain (bool): Train again even if a saved pipeline exists.

    Returns:
        tuple: (fitted pipeline, DataFrame of the cross-validation results or None,
        training time in minutes, True if loaded from model_path).
    """
    settings = {"grid": grid, "model": str(pipeline.named_steps["model"].get_params()),
                "n_train": len(y_train), "scoring": scoring, "cv_folds": cv_folds}
    path = Path(model_path)
    if path.exists() and not retrain:
        saved = joblib.load(path)
        if saved["settings"] != settings:
            raise ValueError(f"{path} was trained with other settings: set RETRAIN = True to train it again")
        return saved["pipeline"], saved["cv"], saved["minutes"], True

    start = time.time()
    if not grid:
        fitted, cv = pipeline.fit(X_train, y_train), None
    else:
        search = GridSearchCV(pipeline, {f"model__{k}": v for k, v in grid.items()}, scoring=scoring,
                              cv=StratifiedKFold(cv_folds, shuffle=True, random_state=random_state),
                              n_jobs=1, refit=True)
        search.fit(X_train, y_train)
        fitted = search.best_estimator_
        cv = pd.DataFrame(search.cv_results_)
        cv["params"] = cv["params"].map(lambda p: {k.replace("model__", ""): v for k, v in p.items()})
        cv = cv[["params", "mean_test_score", "std_test_score", "mean_fit_time"]].sort_values(
            "mean_test_score", ascending=False)
    minutes = (time.time() - start) / 60
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": fitted, "cv": cv, "minutes": minutes, "settings": settings}, path)
    return fitted, cv, minutes, False


############################################################
#                                                          #
#        ------          Evaluation     -------            #
#                                                          #
############################################################

def evaluate_model(y_train, y_train_proba, y_test, y_test_proba, threshold, title):
    """
    Evaluate a model by calculating metrics and displaying confusion matrices.
    Same metrics and same averaging as evaluate_tree_ensemble_model of the
    paper; the only difference is that the class is predicted with
    `threshold` (chosen on the validation set) instead of model.predict (0.5).

    Args:
        y_train, y_train_proba: Training labels and predicted probabilities.
        y_test, y_test_proba: Testing labels and predicted probabilities.
        threshold (float): A consultation is predicted no-show if its probability is >= threshold.
        title (str): Name of the model.

    Returns:
        DataFrame: Metrics (rows) for the train and test sets (columns).
    """
    y_train_pred = (y_train_proba >= threshold).astype(int)
    y_test_pred = (y_test_proba >= threshold).astype(int)

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    for axis, y, y_pred, name in [(ax[0], y_train, y_train_pred, "Train"), (ax[1], y_test, y_test_pred, "Test")]:
        sns.heatmap(confusion_matrix(y, y_pred), annot=True, fmt="d", ax=axis, cmap="Blues")
        axis.set_title(f"Confusion Matrix ({name}) - {title}")
        axis.set_xlabel("Predicted Labels")
        axis.set_ylabel("True Labels")
    plt.tight_layout()
    plt.show()

    metrics = {
        'Metric': ['Recall', 'AUC', 'Accuracy', 'Precision', 'F1 Score'],
        'Train': [
            recall_score(y_train, y_train_pred, average='macro'),
            roc_auc_score(y_train, y_train_proba),
            accuracy_score(y_train, y_train_pred),
            precision_score(y_train, y_train_pred, average='macro'),
            f1_score(y_train, y_train_pred, average='macro')
        ],
        'Test': [
            recall_score(y_test, y_test_pred, average='macro'),
            roc_auc_score(y_test, y_test_proba),
            accuracy_score(y_test, y_test_pred),
            precision_score(y_test, y_test_pred, average='macro'),
            f1_score(y_test, y_test_pred, average='macro')
        ]
    }
    return pd.DataFrame(metrics).set_index('Metric')


def evaluate_classifier(name, fitted, probas, y_train, y_val, y_test, X_test, thresholds,
                        feature_names, permutation_rows, n_bootstrap, random_state):
    """
    Everything shown for one model:
      1. the decision threshold maximising the F1 score on the validation set (plot);
      2. confusion matrices and the five metrics of the paper on train and test;
      3. ROC curve, errors around the threshold and feature importances, on test.

    Args:
        name (str): Name of the model.
        fitted: Fitted pipeline (output of train_model).
        probas (dict): Predicted probabilities of class 1 on "train", "val" and "test".
        y_train, y_val, y_test: Labels.
        X_test: Test set (for the permutation importance).
        thresholds (array): Candidate thresholds.
        feature_names (list): Names of the columns after the encoder (categorical first).
        permutation_rows (int): Rows of each bootstrap sample of the permutation importance.
        n_bootstrap (int): Number of bootstrap samples of the permutation importance.
        random_state (int): Seed of the permutation importance.

    Returns:
        tuple: (threshold, DataFrame of the metrics on train and test).
    """
    threshold, f1_by_threshold = best_f1_threshold(y_val, probas["val"], thresholds)
    f1_by_threshold.plot(figsize=(5, 3), title=f"{name}: F1 (macro) on the validation set")
    plt.axvline(threshold, color="black", ls="--", lw=1)
    plt.xlabel("decision threshold")
    plt.show()
    print(f"Threshold chosen on the validation set: {threshold}")
    results = evaluate_model(y_train, probas["train"], y_test, probas["test"], threshold, name)
    plot_roc_curve(y_test, probas["test"], name)
    plot_confidence_histogram(y_test, probas["test"], f"{name}: errors on the test set", threshold)
    plot_feature_importances(fitted.named_steps["model"], fitted[:-1].transform(X_test), y_test,
                             feature_names, f"{name}: feature importances", threshold,
                             permutation_rows, n_bootstrap, random_state)
    return threshold, results


def best_f1_threshold(y_val, y_val_proba, thresholds):
    """
    Threshold that maximises the F1 score (macro, as in the paper) on the
    validation set. The test set is never used to choose it.

    Args:
        y_val, y_val_proba: Validation labels and predicted probabilities.
        thresholds (array): Candidate thresholds.

    Returns:
        tuple: (best threshold, Series of the F1 score for each candidate).
    """
    scores = pd.Series([f1_score(y_val, (y_val_proba >= t).astype(int), average='macro') for t in thresholds],
                       index=thresholds, name='F1 (macro)')
    return scores.idxmax(), scores


def plot_roc_curve(y_true, y_pred_proba, model_name="Model"):
    """
    Plot ROC curve for model's predicted probabilities.

    Args:
        y_true (array): True labels.
        y_pred_proba (array): Predicted probabilities for the positive class.
        model_name (str): Name of the model for the plot title.
    """
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(5, 4))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC curve - {model_name} (test)')
    plt.legend(loc="lower right")
    plt.show()


def plot_feature_importances(model, X, y, feature_names, title, threshold, n_rows, n_bootstrap, random_state):
    """
    Permutation importance with a bootstrap, the same method for the four models.
    The trained model is kept as it is (no retraining). For each of `n_bootstrap`
    samples of `n_rows` rows drawn with replacement from (X, y):
      - F1 score (macro) of the model at its decision threshold;
      - for each feature, F1 score once that column is shuffled;
      - importance of the feature = drop of the F1 score.
    The bar is the mean over the samples, the error bar their standard deviation
    (as plot_feature_importances_bootstrap in the code of the paper).

    Args:
        model: Trained model (last step of the pipeline).
        X, y: Test set, already encoded and scaled by the pipeline.
        feature_names (list): Names of the features (columns of X).
        title (str): Title of the plot.
        threshold (float): Decision threshold of the model.
        n_rows (int): Rows of each bootstrap sample.
        n_bootstrap (int): Number of bootstrap samples.
        random_state (int): Seed of the samples and of the shuffles.

    Returns:
        DataFrame: mean and standard deviation of the importance of each feature, sorted.
    """
    rng = np.random.default_rng(random_state)
    X, y = np.asarray(X), np.asarray(y)
    drops = np.zeros((n_bootstrap, X.shape[1]))
    for b in range(n_bootstrap):
        rows = rng.choice(len(y), size=min(n_rows, len(y)), replace=True)
        X_b, y_b = X[rows], y[rows]
        base = f1_score(y_b, (model.predict_proba(X_b)[:, 1] >= threshold).astype(int), average='macro')
        for j in range(X.shape[1]):
            X_perm = X_b.copy()
            X_perm[:, j] = rng.permutation(X_perm[:, j])
            score = f1_score(y_b, (model.predict_proba(X_perm)[:, 1] >= threshold).astype(int), average='macro')
            drops[b, j] = base - score
    importances = pd.DataFrame({'mean': drops.mean(axis=0), 'std': drops.std(axis=0)},
                               index=feature_names).sort_values('mean', ascending=False)
    plt.figure(figsize=(7, 4))
    plt.barh(importances.index[::-1], importances['mean'][::-1], xerr=importances['std'][::-1],
             color='steelblue', ecolor='black', capsize=3)
    plt.title(title)
    plt.xlabel(f'Drop of the F1 score (macro) when the feature is shuffled\n'
               f'mean and standard deviation over {n_bootstrap} bootstrap samples of {min(n_rows, len(y)):,} test rows')
    plt.ylabel('Features')
    plt.show()
    return importances


def plot_confidence_histogram(y_test, y_test_prob, title, threshold):
    """
    Plot histogram of predicted probabilities for false positives and false negatives.

    Args:
        y_test (Series or ndarray): True labels.
        y_test_prob (ndarray): Predicted probabilities.
        title (str): Title of the plot.
        threshold (float): Threshold for classification.
    """
    y_test = np.asarray(y_test)
    fp_probs = y_test_prob[(y_test_prob >= threshold) & (y_test == 0)]
    fn_probs = y_test_prob[(y_test_prob < threshold) & (y_test == 1)]
    plt.figure(figsize=(7, 3.5))
    plt.hist(fp_probs, bins=30, alpha=0.5, label='False Positives', color='red')
    plt.hist(fn_probs, bins=30, alpha=0.5, label='False Negatives', color='blue')
    plt.axvline(threshold, color='black', lw=1, ls='--', label=f'threshold = {threshold:.2f}')
    plt.xlabel('Predicted Probability')
    plt.ylabel('Number of Observations')
    plt.title(title)
    plt.xlim(0, 1)
    plt.legend()
    plt.show()


############################################################
#                                                          #
#        ------  Probabilities for the ILP  -------        #
#                                                          #
############################################################

def platt_scaling(p_val, y_val):
    """
    Platt scaling (Platt, 1999): a logistic regression with two parameters,
        p' = 1 / (1 + exp(-(A * logit(p) + B))),
    fitted on the validation set, that turns the scores of a model trained on
    resampled and reweighted data into probabilities matching the real rates.
    It is monotone: the ranking of the consultations (hence the AUC) is unchanged.

    Args:
        p_val (array): Predicted probabilities on the validation set.
        y_val (array): Labels of the validation set.

    Returns:
        function: maps predicted probabilities to calibrated probabilities.
    """
    def logit(p):
        p = np.clip(p, 1e-6, 1 - 1e-6)
        return np.log(p / (1 - p))[:, None]
    model = LogisticRegression().fit(logit(np.asarray(p_val)), y_val)
    return lambda p: model.predict_proba(logit(np.asarray(p)))[:, 1]


def plot_reliability(y_true, probas, n_bins, title):
    """
    Reliability diagram: mean predicted probability against observed no-show
    rate, by bins of equal size (quantiles). A well-calibrated model is on
    the diagonal.

    Args:
        y_true (array): True labels.
        probas (dict): {label: predicted probabilities}.
        n_bins (int): Number of bins.
        title (str): Title of the plot.
    """
    plt.figure(figsize=(5, 4.5))
    top = 0
    for label, p in probas.items():
        bins = pd.qcut(p, q=n_bins, duplicates='drop')
        table = pd.DataFrame({'p': p, 'y': np.asarray(y_true)}).groupby(bins, observed=True).mean()
        plt.plot(table['p'], table['y'], marker='o', markersize=3, label=label)
        top = max(top, table['p'].max(), table['y'].max())
    plt.plot([0, top], [0, top], color='grey', ls='--', lw=1, label='perfect calibration')
    plt.xlabel('Mean predicted no-show probability')
    plt.ylabel('Observed no-show rate')
    plt.title(title)
    plt.legend()
    plt.show()
