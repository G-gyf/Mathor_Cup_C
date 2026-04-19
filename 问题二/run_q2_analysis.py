from __future__ import annotations

import json
import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.inspection import PartialDependenceDisplay
from sklearn.metrics import (
    brier_score_loss,
    log_loss,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, _tree


warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "附件1：样例数据.xlsx"
OUTPUT_ROOT = ROOT / "q2_diagnostics"
SEED = 42


class IQRWinsorizer(BaseEstimator, TransformerMixin):
    def __init__(self, columns: list[str]):
        self.columns = columns
        self.bounds_: dict[str, tuple[float, float]] = {}

    def fit(self, X: pd.DataFrame, y: Any = None) -> "IQRWinsorizer":
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.columns)
        self.bounds_ = {}
        for col in self.columns:
            series = pd.to_numeric(X[col], errors="coerce")
            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            self.bounds_[col] = (lower, upper)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.columns)
        X_out = X.copy()
        for col, (lower, upper) in self.bounds_.items():
            X_out[col] = pd.to_numeric(X_out[col], errors="coerce").clip(lower=lower, upper=upper)
        return X_out


@dataclass
class PathConfig:
    name: str
    slug: str
    feature_columns: list[str]
    continuous_columns: list[str]


PATH_CONFIGS = [
    PathConfig(
        name="Path1_九种体质全量路径",
        slug="path1_rf_calibrated",
        feature_columns=[
            "平和质",
            "气虚质",
            "阳虚质",
            "阴虚质",
            "痰湿质",
            "湿热质",
            "血瘀质",
            "气郁质",
            "特禀质",
            "ADL总分",
            "IADL总分",
            "TG（甘油三酯）",
            "TC（总胆固醇）",
            "LDL-C（低密度脂蛋白）",
            "HDL-C（高密度脂蛋白）",
            "血尿酸",
            "年龄组",
            "性别",
            "吸烟史",
        ],
        continuous_columns=[
            "平和质",
            "气虚质",
            "阳虚质",
            "阴虚质",
            "痰湿质",
            "湿热质",
            "血瘀质",
            "气郁质",
            "特禀质",
            "ADL总分",
            "IADL总分",
            "TG（甘油三酯）",
            "TC（总胆固醇）",
            "LDL-C（低密度脂蛋白）",
            "HDL-C（高密度脂蛋白）",
            "血尿酸",
        ],
    ),
    PathConfig(
        name="Path2_痰湿质聚焦路径",
        slug="path2_rf_calibrated",
        feature_columns=[
            "痰湿质",
            "ADL总分",
            "TG（甘油三酯）",
            "TC（总胆固醇）",
            "LDL-C（低密度脂蛋白）",
            "HDL-C（高密度脂蛋白）",
            "血尿酸",
            "年龄组",
            "性别",
            "吸烟史",
        ],
        continuous_columns=[
            "痰湿质",
            "ADL总分",
            "TG（甘油三酯）",
            "TC（总胆固醇）",
            "LDL-C（低密度脂蛋白）",
            "HDL-C（高密度脂蛋白）",
            "血尿酸",
        ],
    ),
    PathConfig(
        name="Path1_九种体质全量路径_去直接血脂指标",
        slug="path1_rf_no_direct_lipids",
        feature_columns=[
            "平和质",
            "气虚质",
            "阳虚质",
            "阴虚质",
            "痰湿质",
            "湿热质",
            "血瘀质",
            "气郁质",
            "特禀质",
            "ADL总分",
            "IADL总分",
            "血尿酸",
            "年龄组",
            "性别",
            "吸烟史",
        ],
        continuous_columns=[
            "平和质",
            "气虚质",
            "阳虚质",
            "阴虚质",
            "痰湿质",
            "湿热质",
            "血瘀质",
            "气郁质",
            "特禀质",
            "ADL总分",
            "IADL总分",
            "血尿酸",
        ],
    ),
    PathConfig(
        name="Path2_痰湿质聚焦路径_去直接血脂指标",
        slug="path2_rf_no_direct_lipids",
        feature_columns=[
            "痰湿质",
            "ADL总分",
            "血尿酸",
            "年龄组",
            "性别",
            "吸烟史",
        ],
        continuous_columns=[
            "痰湿质",
            "ADL总分",
            "血尿酸",
        ],
    ),
]


def load_source_data() -> pd.DataFrame:
    df = pd.read_excel(DATA_PATH)
    required = {
        "样本ID",
        "高血脂症二分类标签",
        "年龄组",
        "性别",
        "吸烟史",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"源数据缺少必要列: {sorted(missing)}")
    df["高血脂症二分类标签"] = df["高血脂症二分类标签"].astype(int)
    return df


def ensure_output_dirs(base_dir: Path) -> dict[str, Path]:
    paths = {
        "base": base_dir,
        "tables": base_dir / "tables",
        "figures": base_dir / "figures",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def build_base_pipeline(config: PathConfig, rf_params: dict[str, Any] | None = None, rf_n_jobs: int = 1) -> Pipeline:
    rf_defaults = {
        "random_state": SEED,
        "n_jobs": rf_n_jobs,
    }
    if rf_params:
        rf_defaults.update(rf_params)
    pipeline = Pipeline(
        steps=[
            ("winsorizer", IQRWinsorizer(columns=config.continuous_columns)),
            ("imputer", SimpleImputer(strategy="median")),
            ("rf", RandomForestClassifier(**rf_defaults)),
        ]
    )
    return pipeline


def tune_hyperparameters(X_train: pd.DataFrame, y_train: pd.Series, config: PathConfig) -> tuple[dict[str, Any], pd.DataFrame]:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    base_pipeline = build_base_pipeline(config=config, rf_n_jobs=1)
    param_grid = {
        "rf__n_estimators": [300, 500, 800],
        "rf__max_depth": [None, 6, 10, 14],
        "rf__min_samples_split": [2, 5, 10],
        "rf__min_samples_leaf": [1, 2, 5],
        "rf__max_features": ["sqrt", 0.5],
        "rf__class_weight": [None, "balanced_subsample"],
    }
    search = GridSearchCV(
        estimator=base_pipeline,
        param_grid=param_grid,
        scoring="roc_auc",
        cv=cv,
        n_jobs=-1,
        refit=True,
        verbose=0,
        return_train_score=False,
    )
    search.fit(X_train, y_train)
    results = pd.DataFrame(search.cv_results_).sort_values(by="rank_test_score").reset_index(drop=True)
    best_params = {
        key.replace("rf__", ""): value
        for key, value in search.best_params_.items()
        if key.startswith("rf__")
    }
    return best_params, results


def evaluate_oof_probabilities(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    config: PathConfig,
    best_params: dict[str, Any],
) -> tuple[str, pd.DataFrame, dict[str, np.ndarray]]:
    cv_outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    raw_pipeline = build_base_pipeline(config=config, rf_params=best_params, rf_n_jobs=1)
    raw_oof = cross_val_predict(
        raw_pipeline,
        X_train,
        y_train,
        cv=cv_outer,
        method="predict_proba",
        n_jobs=-1,
    )[:, 1]
    records = []
    prob_map: dict[str, np.ndarray] = {"raw": raw_oof}
    for method in ["sigmoid", "isotonic"]:
        calibrated = CalibratedClassifierCV(
            estimator=build_base_pipeline(config=config, rf_params=best_params, rf_n_jobs=1),
            method=method,
            cv=5,
        )
        probs = cross_val_predict(
            calibrated,
            X_train,
            y_train,
            cv=cv_outer,
            method="predict_proba",
            n_jobs=-1,
        )[:, 1]
        prob_map[method] = probs
    for method, probs in prob_map.items():
        clipped = np.clip(probs, 1e-6, 1 - 1e-6)
        records.append(
            {
                "method": method,
                "brier_score": brier_score_loss(y_train, probs),
                "log_loss": log_loss(y_train, clipped),
                "roc_auc": roc_auc_score(y_train, probs),
            }
        )
    comparison = pd.DataFrame(records).sort_values(
        by=["brier_score", "log_loss", "roc_auc"],
        ascending=[True, True, False],
    ).reset_index(drop=True)
    calibrated_only = comparison[comparison["method"].isin(["sigmoid", "isotonic"])].reset_index(drop=True)
    best_method = calibrated_only.iloc[0]["method"]
    return best_method, comparison, prob_map


def fit_final_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    config: PathConfig,
    best_params: dict[str, Any],
    calibration_method: str,
) -> tuple[Pipeline, Any]:
    base_model = build_base_pipeline(config=config, rf_params=best_params, rf_n_jobs=-1)
    base_model.fit(X_train, y_train)
    calibrated_model = CalibratedClassifierCV(
        estimator=build_base_pipeline(config=config, rf_params=best_params, rf_n_jobs=-1),
        method=calibration_method,
        cv=5,
    )
    calibrated_model.fit(X_train, y_train)
    return base_model, calibrated_model


def determine_thresholds(y_train: pd.Series, oof_probs: np.ndarray) -> dict[str, float]:
    fpr, tpr, thresholds = roc_curve(y_train, oof_probs)
    valid_mask = np.isfinite(thresholds)
    valid_thresholds = thresholds[valid_mask]
    youden = tpr[valid_mask] - fpr[valid_mask]
    tlow = float(valid_thresholds[np.argmax(youden)])
    q90 = float(np.quantile(oof_probs, 0.90))
    q95 = float(np.quantile(oof_probs, 0.95))
    if q90 > tlow:
        thigh = q90
        high_quantile_used = 0.90
    elif q95 > tlow:
        thigh = q95
        high_quantile_used = 0.95
    else:
        above = np.sort(oof_probs[oof_probs > tlow])
        if len(above) == 0:
            thigh = float(min(tlow + 1e-6, 0.999999))
        else:
            thigh = float(above[0])
        high_quantile_used = math.nan
    return {
        "tlow": tlow,
        "thigh": thigh,
        "high_quantile_used": high_quantile_used,
    }


def assign_risk_group(probabilities: np.ndarray, tlow: float, thigh: float) -> pd.Categorical:
    labels = np.where(
        probabilities < tlow,
        "低风险",
        np.where(probabilities < thigh, "中风险", "高风险"),
    )
    return pd.Categorical(labels, categories=["低风险", "中风险", "高风险"], ordered=True)


def summarize_split(y_train: pd.Series, y_test: pd.Series) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "split": "train",
                "n": len(y_train),
                "positive_n": int(y_train.sum()),
                "negative_n": int((1 - y_train).sum()),
                "positive_rate": float(y_train.mean()),
            },
            {
                "split": "test",
                "n": len(y_test),
                "positive_n": int(y_test.sum()),
                "negative_n": int((1 - y_test).sum()),
                "positive_rate": float(y_test.mean()),
            },
        ]
    )


def summarize_risk_groups(df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df.groupby("风险等级", observed=False)
        .agg(
            样本量=("真实标签", "size"),
            阳性数=("真实标签", "sum"),
            平均预测概率=("预测概率", "mean"),
        )
        .reset_index()
    )
    summary["样本占比"] = summary["样本量"] / summary["样本量"].sum()
    summary["事件率"] = summary["阳性数"] / summary["样本量"]
    summary["风险等级"] = pd.Categorical(summary["风险等级"], ["低风险", "中风险", "高风险"], ordered=True)
    summary = summary.sort_values("风险等级").reset_index(drop=True)
    return summary


def monotonicity_flag(summary: pd.DataFrame) -> bool:
    rates = summary["事件率"].to_numpy(dtype=float)
    return bool(np.all(np.diff(rates) >= -1e-9))


def plot_roc_curve(y_test: pd.Series, probs: np.ndarray, output_path: Path) -> float:
    auc = roc_auc_score(y_test, probs)
    fpr, tpr, _ = roc_curve(y_test, probs)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"AUC = {auc:.4f}", color="#1f77b4", linewidth=2)
    plt.plot([0, 1], [0, 1], linestyle="--", color="#999999")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    return auc


def plot_calibration_curve(
    y_test: pd.Series,
    raw_probs: np.ndarray,
    calibrated_probs: np.ndarray,
    output_path: Path,
) -> tuple[float, float]:
    frac_raw, mean_raw = calibration_curve(y_test, raw_probs, n_bins=10, strategy="quantile")
    frac_cal, mean_cal = calibration_curve(y_test, calibrated_probs, n_bins=10, strategy="quantile")
    plt.figure(figsize=(6, 5))
    plt.plot([0, 1], [0, 1], linestyle="--", color="#999999", label="Perfect")
    plt.plot(mean_raw, frac_raw, marker="o", label="Raw RF")
    plt.plot(mean_cal, frac_cal, marker="o", label="Calibrated RF")
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Observed Positive Rate")
    plt.title("Calibration Curve")
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    return brier_score_loss(y_test, raw_probs), brier_score_loss(y_test, calibrated_probs)


def plot_probability_histogram(
    probabilities: np.ndarray,
    y_true: pd.Series,
    tlow: float,
    thigh: float,
    output_path: Path,
) -> None:
    plot_df = pd.DataFrame({"probability": probabilities, "label": y_true.map({0: "未确诊", 1: "确诊"})})
    plt.figure(figsize=(7, 5))
    sns.histplot(data=plot_df, x="probability", hue="label", bins=25, stat="density", common_norm=False, alpha=0.45)
    plt.axvline(tlow, color="#ff7f0e", linestyle="--", linewidth=2, label=f"tlow={tlow:.3f}")
    plt.axvline(thigh, color="#d62728", linestyle="--", linewidth=2, label=f"thigh={thigh:.3f}")
    plt.xlabel("Calibrated Probability")
    plt.ylabel("Density")
    plt.title("Test Probability Distribution")
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend(handles=handles, labels=labels, loc="upper center")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_event_rate(summary: pd.DataFrame, output_path: Path) -> None:
    plt.figure(figsize=(6, 4))
    sns.barplot(data=summary, x="风险等级", y="事件率", palette=["#8dd3c7", "#fb8072", "#bebada"])
    plt.ylim(0, 1)
    plt.ylabel("Event Rate")
    plt.title("Risk Group Event Rate")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def extract_rf_for_shap(base_model: Pipeline, feature_names: list[str], X_train: pd.DataFrame) -> tuple[np.ndarray, RandomForestClassifier]:
    X_wins = base_model.named_steps["winsorizer"].transform(X_train[feature_names])
    X_imp = base_model.named_steps["imputer"].transform(X_wins)
    rf_model: RandomForestClassifier = base_model.named_steps["rf"]
    return X_imp, rf_model


def resolve_shap_values(explainer: shap.TreeExplainer, X_values: np.ndarray) -> np.ndarray:
    shap_values = explainer.shap_values(X_values)
    if isinstance(shap_values, list):
        return np.asarray(shap_values[1])
    shap_values = np.asarray(shap_values)
    if shap_values.ndim == 3:
        return shap_values[:, :, 1]
    return shap_values


def compute_shap_outputs(
    base_model: Pipeline,
    X_train: pd.DataFrame,
    feature_names: list[str],
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    X_imp, rf_model = extract_rf_for_shap(base_model, feature_names, X_train)
    explainer = shap.TreeExplainer(rf_model)
    shap_values = resolve_shap_values(explainer, X_imp)
    feature_values = np.asarray(X_imp)
    mean_abs = np.abs(shap_values).mean(axis=0)
    importance = pd.DataFrame(
        {"feature": feature_names, "mean_abs_shap": mean_abs}
    ).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    return importance, feature_values, shap_values


def plot_shap_summary(
    shap_values: np.ndarray,
    feature_values: np.ndarray,
    feature_names: list[str],
    output_path: Path,
) -> None:
    plt.figure()
    shap.summary_plot(
        shap_values,
        features=feature_values,
        feature_names=feature_names,
        show=False,
        max_display=min(12, len(feature_names)),
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()


def extract_shap_threshold(feature_values: np.ndarray, shap_values: np.ndarray, feature_name: str) -> dict[str, Any]:
    series = pd.Series(feature_values, name=feature_name)
    shap_series = pd.Series(shap_values, name="shap_value")
    unique_count = series.nunique(dropna=True)
    if unique_count <= 10:
        grouped = (
            pd.DataFrame({"value": series, "shap_value": shap_series})
            .groupby("value", observed=True)["shap_value"]
            .median()
            .reset_index()
            .sort_values("value")
        )
        levels = "; ".join([f"{row['value']}: {row['shap_value']:.4f}" for _, row in grouped.iterrows()])
        direction = "升高风险" if grouped["shap_value"].max() > 0 else "未见明显升高"
        return {
            "feature": feature_name,
            "feature_type": "离散/有序",
            "threshold": np.nan,
            "threshold_rule": levels,
            "direction": direction,
            "notes": "离散变量按各水平SHAP中位数汇总",
        }
    temp = pd.DataFrame({"value": series, "shap_value": shap_series}).dropna()
    temp["bin"] = pd.qcut(temp["value"], q=min(10, temp["value"].nunique()), duplicates="drop")
    grouped = (
        temp.groupby("bin", observed=True)
        .agg(value_median=("value", "median"), shap_median=("shap_value", "median"))
        .reset_index()
        .sort_values("value_median")
    )
    threshold = np.nan
    note = "未出现负转正，使用最大增幅拐点"
    shap_medians = grouped["shap_median"].to_numpy()
    value_medians = grouped["value_median"].to_numpy()
    for idx in range(1, len(grouped)):
        if shap_medians[idx - 1] < 0 <= shap_medians[idx]:
            threshold = float(value_medians[idx])
            note = "SHAP中位数由负转正"
            break
    if math.isnan(threshold):
        diffs = np.diff(shap_medians)
        if len(diffs) > 0:
            jump_idx = int(np.argmax(np.abs(diffs))) + 1
            threshold = float(value_medians[jump_idx])
        else:
            threshold = float(value_medians[0])
    direction = "升高风险" if grouped["shap_median"].iloc[-1] > grouped["shap_median"].iloc[0] else "降低风险"
    return {
        "feature": feature_name,
        "feature_type": "连续",
        "threshold": threshold,
        "threshold_rule": f"约在 {threshold:.4f} 附近出现风险贡献变化",
        "direction": direction,
        "notes": note,
    }


def plot_shap_dependence(
    feature_values: np.ndarray,
    shap_values: np.ndarray,
    feature_name: str,
    output_path: Path,
) -> None:
    temp = pd.DataFrame({"feature_value": feature_values, "shap_value": shap_values}).dropna()
    temp["bin"] = pd.qcut(temp["feature_value"], q=min(10, temp["feature_value"].nunique()), duplicates="drop")
    grouped = (
        temp.groupby("bin", observed=True)
        .agg(feature_median=("feature_value", "median"), shap_median=("shap_value", "median"))
        .reset_index()
        .sort_values("feature_median")
    )
    plt.figure(figsize=(6.5, 4.5))
    plt.scatter(temp["feature_value"], temp["shap_value"], alpha=0.25, s=16, color="#4c78a8")
    plt.plot(grouped["feature_median"], grouped["shap_median"], color="#e45756", linewidth=2)
    plt.axhline(0, color="#999999", linestyle="--", linewidth=1)
    plt.xlabel(feature_name)
    plt.ylabel("SHAP value")
    plt.title(f"SHAP Dependence: {feature_name}")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def tree_leaf_rules(tree_model: DecisionTreeClassifier, feature_names: list[str]) -> pd.DataFrame:
    tree_ = tree_model.tree_
    paths: list[dict[str, Any]] = []

    def recurse(node: int, conditions: list[str]) -> None:
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_names[tree_.feature[node]]
            threshold = tree_.threshold[node]
            recurse(tree_.children_left[node], conditions + [f"{name} <= {threshold:.4f}"])
            recurse(tree_.children_right[node], conditions + [f"{name} > {threshold:.4f}"])
            return
        values = tree_.value[node][0]
        total = int(tree_.n_node_samples[node])
        if values.sum() > 0 and abs(values.sum() - total) > 1e-9:
            counts = values / values.sum() * total
        else:
            counts = values
        dominant = int(np.argmax(counts))
        risk_name = ["低风险", "中风险", "高风险"][dominant]
        paths.append(
            {
                "leaf_node": node,
                "rule": " and ".join(conditions) if conditions else "ALL",
                "sample_count": total,
                "low_count": int(round(counts[0])),
                "mid_count": int(round(counts[1])),
                "high_count": int(round(counts[2])),
                "dominant_risk": risk_name,
                "dominant_purity": float(counts[dominant] / total) if total else 0.0,
            }
        )

    recurse(0, [])
    return pd.DataFrame(paths).sort_values(by=["dominant_risk", "dominant_purity"], ascending=[True, False]).reset_index(drop=True)


def save_markdown_summary(
    base_dir: Path,
    config: PathConfig,
    best_params: dict[str, Any],
    calibration_method: str,
    threshold_info: dict[str, float],
    metrics: dict[str, float],
    group_summary: pd.DataFrame,
    monotonic_ok: bool,
    shap_importance: pd.DataFrame,
) -> None:
    summary_path = base_dir / "result_summary.md"
    top_features = shap_importance.head(6)
    group_table = group_summary.to_csv(index=False)
    shap_table = top_features.to_csv(index=False)
    lines = [
        f"# {config.name} 结果汇总",
        "",
        "## 模型训练与校准",
        f"- 特征数：`{len(config.feature_columns)}`",
        f"- 最优校准方法：`{calibration_method}`",
        f"- 最优超参数：`{json.dumps(best_params, ensure_ascii=False)}`",
        f"- 测试集AUC：`{metrics['auc']:.4f}`",
        f"- 测试集Brier：`{metrics['brier_calibrated']:.4f}`",
        f"- 测试集LogLoss：`{metrics['log_loss']:.4f}`",
        "",
        "## 概率阈值",
        f"- `tlow = {threshold_info['tlow']:.4f}`",
        f"- `thigh = {threshold_info['thigh']:.4f}`",
        f"- 高风险分位点：`{threshold_info['high_quantile_used']}`",
        f"- 测试集风险单调性：`{'通过' if monotonic_ok else '未通过'}`",
        "",
        "## 测试集三级风险概览",
        "```csv",
        group_table.strip(),
        "```",
        "",
        "## Top 6 SHAP 特征",
        "```csv",
        shap_table.strip(),
        "```",
        "",
        "## 说明",
        "- 概率阈值仅基于训练集OOF校准概率确定。",
        "- 近似规则树和SHAP阈值回溯结果详见 `tables/` 与 `figures/`。",
    ]
    summary_path.write_text("\n".join(lines), encoding="utf-8")


def run_single_path(df: pd.DataFrame, config: PathConfig) -> None:
    print(f"[INFO] Running {config.name}")
    base_dir = OUTPUT_ROOT / config.slug
    dirs = ensure_output_dirs(base_dir)

    columns = ["样本ID", "高血脂症二分类标签"] + config.feature_columns
    work_df = df[columns].copy()
    X = work_df[config.feature_columns]
    y = work_df["高血脂症二分类标签"].astype(int)
    sample_ids = work_df["样本ID"]

    split = train_test_split(
        X,
        y,
        sample_ids,
        test_size=0.30,
        stratify=y,
        random_state=SEED,
    )
    X_train, X_test, y_train, y_test, id_train, id_test = split

    split_summary = summarize_split(y_train, y_test)
    split_summary.to_csv(dirs["tables"] / "train_test_split_summary.csv", index=False, encoding="utf-8-sig")

    feature_def = pd.DataFrame(
        {
            "path_name": config.name,
            "feature_name": config.feature_columns,
            "feature_role": [
                "连续变量" if col in config.continuous_columns else "离散变量"
                for col in config.feature_columns
            ],
        }
    )
    feature_def.to_csv(dirs["tables"] / "feature_definition.csv", index=False, encoding="utf-8-sig")

    best_params, tuning_results = tune_hyperparameters(X_train, y_train, config)
    tuning_results.head(20).to_csv(dirs["tables"] / "grid_search_top20.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([best_params]).to_csv(dirs["tables"] / "best_hyperparameters.csv", index=False, encoding="utf-8-sig")

    best_method, calibration_comparison, prob_map = evaluate_oof_probabilities(X_train, y_train, config, best_params)
    calibration_comparison.to_csv(dirs["tables"] / "calibration_comparison.csv", index=False, encoding="utf-8-sig")

    threshold_info = determine_thresholds(y_train, prob_map[best_method])
    pd.DataFrame([threshold_info]).to_csv(dirs["tables"] / "probability_thresholds.csv", index=False, encoding="utf-8-sig")

    base_model, calibrated_model = fit_final_models(X_train, y_train, config, best_params, best_method)
    raw_test_probs = base_model.predict_proba(X_test)[:, 1]
    cal_test_probs = calibrated_model.predict_proba(X_test)[:, 1]

    train_pred_df = pd.DataFrame(
        {
            "样本ID": id_train.to_numpy(),
            "split": "train_oof",
            "真实标签": y_train.to_numpy(),
            "预测概率": prob_map[best_method],
            "风险等级": assign_risk_group(prob_map[best_method], threshold_info["tlow"], threshold_info["thigh"]),
        }
    )
    test_pred_df = pd.DataFrame(
        {
            "样本ID": id_test.to_numpy(),
            "split": "test",
            "真实标签": y_test.to_numpy(),
            "预测概率": cal_test_probs,
            "风险等级": assign_risk_group(cal_test_probs, threshold_info["tlow"], threshold_info["thigh"]),
        }
    )
    predictions = pd.concat([train_pred_df, test_pred_df], ignore_index=True)
    predictions.to_csv(dirs["tables"] / "sample_predictions.csv", index=False, encoding="utf-8-sig")

    test_group_summary = summarize_risk_groups(test_pred_df)
    mono_ok = monotonicity_flag(test_group_summary)
    test_group_summary["事件率单调递增"] = mono_ok
    test_group_summary.to_csv(dirs["tables"] / "test_risk_group_summary.csv", index=False, encoding="utf-8-sig")

    auc = plot_roc_curve(y_test, cal_test_probs, dirs["figures"] / "roc_curve.png")
    brier_raw, brier_cal = plot_calibration_curve(
        y_test,
        raw_test_probs,
        cal_test_probs,
        dirs["figures"] / "calibration_curve.png",
    )
    plot_probability_histogram(
        cal_test_probs,
        y_test,
        threshold_info["tlow"],
        threshold_info["thigh"],
        dirs["figures"] / "probability_histogram.png",
    )
    plot_event_rate(test_group_summary, dirs["figures"] / "risk_group_event_rate.png")

    shap_importance, feature_values, shap_values = compute_shap_outputs(base_model, X_train, config.feature_columns)
    shap_importance.to_csv(dirs["tables"] / "shap_importance.csv", index=False, encoding="utf-8-sig")
    plot_shap_summary(shap_values, feature_values, config.feature_columns, dirs["figures"] / "shap_summary.png")

    top_features = shap_importance["feature"].head(6).tolist()
    threshold_records = []
    for feature in top_features:
        idx = config.feature_columns.index(feature)
        values = feature_values[:, idx]
        contributions = shap_values[:, idx]
        threshold_records.append(extract_shap_threshold(values, contributions, feature))
        safe_name = feature.replace("/", "_").replace(" ", "_")
        plot_shap_dependence(values, contributions, feature, dirs["figures"] / f"shap_dependence_{safe_name}.png")
    pd.DataFrame(threshold_records).to_csv(dirs["tables"] / "shap_threshold_summary.csv", index=False, encoding="utf-8-sig")

    train_risk_numeric = pd.Categorical(
        train_pred_df["风险等级"],
        categories=["低风险", "中风险", "高风险"],
        ordered=True,
    ).codes
    surrogate_features = top_features
    surrogate_X = X_train[surrogate_features].copy()
    surrogate_X = IQRWinsorizer([c for c in surrogate_features if c in config.continuous_columns]).fit_transform(surrogate_X)
    surrogate_X = pd.DataFrame(SimpleImputer(strategy="median").fit_transform(surrogate_X), columns=surrogate_features)
    min_leaf = max(10, int(len(surrogate_X) * 0.05))
    surrogate_tree = DecisionTreeClassifier(max_depth=3, min_samples_leaf=min_leaf, random_state=SEED)
    surrogate_tree.fit(surrogate_X, train_risk_numeric)
    rule_df = tree_leaf_rules(surrogate_tree, surrogate_features)
    rule_df.to_csv(dirs["tables"] / "surrogate_tree_rules.csv", index=False, encoding="utf-8-sig")

    metrics = {
        "auc": auc,
        "brier_raw": brier_raw,
        "brier_calibrated": brier_cal,
        "log_loss": log_loss(y_test, np.clip(cal_test_probs, 1e-6, 1 - 1e-6)),
    }
    pd.DataFrame([metrics]).to_csv(dirs["tables"] / "test_metrics.csv", index=False, encoding="utf-8-sig")

    save_markdown_summary(
        base_dir=base_dir,
        config=config,
        best_params=best_params,
        calibration_method=best_method,
        threshold_info=threshold_info,
        metrics=metrics,
        group_summary=test_group_summary,
        monotonic_ok=mono_ok,
        shap_importance=shap_importance,
    )


def main() -> None:
    sns.set_theme(style="whitegrid")
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    df = load_source_data()
    for config in PATH_CONFIGS:
        run_single_path(df, config)
    print("[INFO] Question 2 analysis completed.")


if __name__ == "__main__":
    main()
