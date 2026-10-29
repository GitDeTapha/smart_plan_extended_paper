"""Audit 1.3, 1.4, 1.6: halving search on fit / valid, imbalance, calibration on calib."""
import numpy as np
import pytest

from data_driven_plan import (BalancedFit, TrainingConfig, calibrate_and_select,
                              classification_metrics, make_estimator, prior_correct,
                              share_matching_threshold, train_binary_classifiers)


def data(n=6000, prior=0.05, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 6)).astype(np.float32)
    logit = 1.5 * X[:, 0] - X[:, 1] + np.log(prior / (1 - prior))
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    return X, y


SMALL = dict(n_candidates=3, halving_factor=2, min_resources=1000, adaboost_max_estimators=15,
             n_jobs=1, verbose=0,
             param_grids={"RandomForest": {"n_estimators": [20], "max_depth": [3, 5]},
                          "XGBoost": {"n_estimators": [20], "max_depth": [2, 3]},
                          "ANN": {"hidden_layer_sizes": [(8,)], "batch_size": [256], "alpha": [1e-4]}},
             base_params={"ANN": {"max_iter": 30}})


def test_smote_only_below_the_ratio_and_weights_balance():
    cfg = TrainingConfig(**SMALL)
    assert "smote" in make_estimator("XGBoost", cfg, minority_ratio=0.03).named_steps
    assert "smote" not in make_estimator("XGBoost", cfg, minority_ratio=0.5).named_steps
    X, y = data(prior=0.3)
    m = BalancedFit(make_estimator("XGBoost", cfg, 0.5).named_steps["clf"].estimator,
                    mode="scale_pos_weight").fit(X, y)
    assert m.estimator_.get_params()["scale_pos_weight"] == pytest.approx((y == 0).sum() / (y == 1).sum())
    with pytest.raises(ValueError):
        TrainingConfig(smote_sampling_strategy=0.3)


def test_prior_correction_inverts_balanced_training():
    assert prior_correct(np.array([0.5]), prior=0.03)[0] == pytest.approx(0.03)


def test_train_calibrate_and_score_without_touching_test():
    X, y = data()
    n_fit = 4000
    models = train_binary_classifiers(X[:4800], y[:4800], n_fit, TrainingConfig(**SMALL))
    assert set(models) == {"RandomForest", "XGBoost", "AdaBoost", "ANN"}
    assert models.prior == pytest.approx(y[:n_fit].mean())
    assert "n_estimators" in models.best_params["AdaBoost"]
    cal = calibrate_and_select(models, X[4800:5400], y[4800:5400])
    assert cal.best_name in models and set(cal.thresholds) == set(models)
    p = cal.models[cal.best_name].predict_proba(X[5400:])[:, 1]
    met = classification_metrics(y[5400:], p, cal.thresholds[cal.best_name], base_rate=models.prior)
    for key in ("AUC-ROC", "AUC-PR", "Log-loss", "Brier", "BSS", "F1 (macro)", "F1 (positive)",
                "Accuracy (majority class)"):
        assert np.isfinite(met[key])
    assert met["AUC-ROC"] > 0.7


def test_share_matching_threshold():
    p = np.linspace(0, 1, 1001)
    t = share_matching_threshold(p, 0.35)
    assert np.mean(p >= t) == pytest.approx(0.35, abs=2e-3)
