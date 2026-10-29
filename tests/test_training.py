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


SMALL = dict(halving_iterations=2, last_candidates=2, min_resources=500, adaboost_max_estimators=15,
             n_jobs=1, verbose=0,
             param_grids={"RandomForest": {"n_estimators": [20], "max_depth": [3, 5]},
                          "XGBoost": {"n_estimators": [10, 20], "max_depth": [2, 3, 4],
                                      "learning_rate": [0.1, 0.3]},     # 12 combinations: halving path
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


def test_predict_patient_and_duration_modes():
    import pandas as pd
    from data_driven_plan import build_instance_from_patients, predict_patient

    class Const:
        def __init__(self, p):
            self.p = np.asarray(p)

        def predict_proba(self, X):
            return np.c_[1 - self.p, self.p]

    X = np.zeros((3, 2))
    out = predict_patient(X, Const([0.1, 0.0, 0.2]), Const([0.9, 0.3, 0.5]), threshold_A=0.5,
                          s_A=36.0, s_B=7.0)
    assert list(out["group"]) == ["A", "B", "A"]
    assert out["s_expected"].tolist() == pytest.approx([0.9 * 36 + 0.1 * 7, 0.3 * 36 + 0.7 * 7, 21.5])
    kw = dict(T=[2], L=[[30.0, 60.0]], s_A=36.0, s_B=7.0, c_I=1.0, c_W=0.5, c_O=0.5)
    cls = build_instance_from_patients(out, duration_mode="class", **kw)
    exp = build_instance_from_patients(out, duration_mode="expected", **kw)
    assert cls.s.tolist() == [36.0, 7.0, 36.0] and exp.s.tolist() == pytest.approx(out["s_expected"].tolist())
    assert cls.q.max() == pytest.approx(0.999)          # q = 1.0 clipped explicitly
    with pytest.raises(ValueError):
        build_instance_from_patients(out, duration_mode="mean", **kw)


def test_adaboost_tuned_in_a_separate_process(tmp_path):
    from data_driven_plan import tune_binary_classifiers
    X, y = data(3000)
    np.save(tmp_path / "XA.npy", X)
    np.save(tmp_path / "y.npy", y)
    cfg = TrainingConfig(**{**SMALL, "models": ["AdaBoost", "XGBoost"]})
    sub = tune_binary_classifiers(X[:2500], y[:2500], 2000, cfg,
                                  adaboost_files=(str(tmp_path / "XA.npy"), str(tmp_path / "y.npy")))
    same = tune_binary_classifiers(X[:2500], y[:2500], 2000, cfg)
    assert sub["best"]["AdaBoost"] == same["best"]["AdaBoost"]
    assert sub["criterion"]["AdaBoost"] == pytest.approx(same["criterion"]["AdaBoost"])


def test_smote_cache_leaves_the_search_unchanged(tmp_path):
    from data_driven_plan import tune_binary_classifiers
    X, y = data(3000, prior=0.01)   # positive share about 3 %: SMOTE applies
    cfg = TrainingConfig(**{**SMALL, "models": ["XGBoost", "AdaBoost"]})
    plain = tune_binary_classifiers(X[:2500], y[:2500], 2000, cfg)
    cached = tune_binary_classifiers(X[:2500], y[:2500], 2000, cfg.replace(smote_cache_dir=str(tmp_path)))
    assert plain["best"] == cached["best"]
    assert plain["criterion"] == pytest.approx(cached["criterion"])
    assert list(tmp_path.glob("smote_*.npz"))
