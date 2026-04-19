from __future__ import annotations

import math
import warnings
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier, _tree, plot_tree

from run_q2_analysis import (
    DATA_PATH,
    OUTPUT_ROOT,
    PATH_CONFIGS,
    ROOT,
    SEED,
    IQRWinsorizer,
    build_base_pipeline,
    extract_rf_for_shap,
    resolve_shap_values,
)


warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


CONSTITUTION_OTHER = {
    "平和质",
    "气虚质",
    "阳虚质",
    "阴虚质",
    "湿热质",
    "血瘀质",
    "气郁质",
    "特禀质",
}


def safe_name(text: str) -> str:
    invalid = '\\/:*?"<>|'
    result = text
    for ch in invalid:
        result = result.replace(ch, "_")
    return result.replace(" ", "_")


def load_subgroup_data() -> pd.DataFrame:
    df = pd.read_excel(DATA_PATH)
    subgroup = df[df["体质标签"] == 5].copy().reset_index(drop=True)
    return subgroup


def make_dirs(base_dir: Path) -> dict[str, Path]:
    subgroup_dir = base_dir / "subgroup_t5"
    tables_dir = subgroup_dir / "tables"
    figures_dir = subgroup_dir / "figures"
    subgroup_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    return {"base": subgroup_dir, "tables": tables_dir, "figures": figures_dir}


def load_best_params(path_dir: Path) -> dict[str, Any]:
    df = pd.read_csv(path_dir / "tables" / "best_hyperparameters.csv")
    row = df.iloc[0].to_dict()
    params: dict[str, Any] = {}
    for key, value in row.items():
        if pd.isna(value):
            params[key] = None
        elif key in {"n_estimators", "max_depth", "min_samples_split", "min_samples_leaf"}:
            params[key] = int(value)
        else:
            params[key] = value
    return params


def preprocess_features(X: pd.DataFrame, continuous_columns: list[str]) -> tuple[pd.DataFrame, IQRWinsorizer, SimpleImputer]:
    winsorizer = IQRWinsorizer(columns=[col for col in continuous_columns if col in X.columns])
    X_wins = winsorizer.fit_transform(X)
    imputer = SimpleImputer(strategy="median")
    X_imp = pd.DataFrame(imputer.fit_transform(X_wins), columns=X.columns, index=X.index)
    return X_imp, winsorizer, imputer


def prune_alphas(X: pd.DataFrame, y: pd.Series) -> np.ndarray:
    base_tree = DecisionTreeClassifier(
        criterion="gini",
        class_weight="balanced",
        max_depth=4,
        min_samples_leaf=max(15, math.ceil(0.08 * len(X))),
        random_state=SEED,
    )
    path = base_tree.cost_complexity_pruning_path(X, y)
    alphas = np.unique(np.clip(path.ccp_alphas, 0, None))
    if len(alphas) == 0:
        return np.array([0.0])
    if len(alphas) > 20:
        idx = np.unique(np.linspace(0, len(alphas) - 1, 20).astype(int))
        alphas = alphas[idx]
    return alphas


def select_cart_alpha(X: pd.DataFrame, y: pd.Series) -> tuple[float, pd.DataFrame]:
    candidates = prune_alphas(X, y)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    rows = []
    for alpha in candidates:
        scores = []
        leaves = []
        depths = []
        for train_idx, val_idx in cv.split(X, y):
            X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
            tree = DecisionTreeClassifier(
                criterion="gini",
                class_weight="balanced",
                max_depth=4,
                min_samples_leaf=max(15, math.ceil(0.08 * len(X))),
                ccp_alpha=float(alpha),
                random_state=SEED,
            )
            tree.fit(X_tr, y_tr)
            pred = tree.predict(X_val)
            scores.append(balanced_accuracy_score(y_val, pred))
            leaves.append(tree.get_n_leaves())
            depths.append(tree.get_depth())
        rows.append(
            {
                "ccp_alpha": float(alpha),
                "mean_balanced_accuracy": float(np.mean(scores)),
                "std_balanced_accuracy": float(np.std(scores)),
                "mean_n_leaves": float(np.mean(leaves)),
                "mean_depth": float(np.mean(depths)),
            }
        )
    result = pd.DataFrame(rows).sort_values(
        by=["mean_balanced_accuracy", "ccp_alpha"],
        ascending=[False, False],
    ).reset_index(drop=True)
    return float(result.iloc[0]["ccp_alpha"]), result


def cart_leaf_rules_binary(tree_model: DecisionTreeClassifier, feature_names: list[str]) -> pd.DataFrame:
    tree_ = tree_model.tree_
    rows: list[dict[str, Any]] = []

    def recurse(node: int, conditions: list[str]) -> None:
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            feat_name = feature_names[tree_.feature[node]]
            threshold = tree_.threshold[node]
            recurse(tree_.children_left[node], conditions + [f"{feat_name} <= {threshold:.4f}"])
            recurse(tree_.children_right[node], conditions + [f"{feat_name} > {threshold:.4f}"])
            return
        values = tree_.value[node][0]
        total = int(tree_.n_node_samples[node])
        if values.sum() > 0 and abs(values.sum() - total) > 1e-9:
            counts = values / values.sum() * total
        else:
            counts = values
        negative_count = int(round(counts[0]))
        positive_count = int(round(counts[1]))
        positive_rate = float(positive_count / total) if total else 0.0
        rows.append(
            {
                "leaf_node": node,
                "rule": " and ".join(conditions) if conditions else "ALL",
                "sample_count": total,
                "negative_count": negative_count,
                "positive_count": positive_count,
                "positive_rate": positive_rate,
            }
        )

    recurse(0, [])
    return pd.DataFrame(rows).sort_values(
        by=["positive_rate", "sample_count"],
        ascending=[False, False],
    ).reset_index(drop=True)


def plot_cart_tree(tree_model: DecisionTreeClassifier, feature_names: list[str], output_path: Path) -> None:
    plt.figure(figsize=(20, 10))
    plot_tree(
        tree_model,
        feature_names=feature_names,
        class_names=["阴性", "阳性"],
        filled=True,
        rounded=True,
        impurity=False,
        proportion=False,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()


def label_age_group(value: Any) -> str:
    mapping = {1: "40-49", 2: "50-59", 3: "60-69", 4: "70-79", 5: "80-89"}
    return mapping.get(int(value), f"未知{value}")


def discretize_item(feature: str, value: Any, row: pd.Series) -> str:
    if feature == "痰湿质":
        if value <= 58:
            return "痰湿质_<=58"
        if value <= 61:
            return "痰湿质_59-61"
        return "痰湿质_>=62"
    if feature in CONSTITUTION_OTHER:
        if value < 20:
            return f"{feature}_<20"
        if value < 35:
            return f"{feature}_20-34"
        return f"{feature}_>=35"
    if feature == "ADL总分":
        if value < 20:
            return "ADL总分_<20"
        if value < 30:
            return "ADL总分_20-29"
        return "ADL总分_>=30"
    if feature == "IADL总分":
        if value < 20:
            return "IADL总分_<20"
        if value < 30:
            return "IADL总分_20-29"
        return "IADL总分_>=30"
    if feature == "TG（甘油三酯）":
        return "TG_>1.7" if value > 1.7 else "TG_<=1.7"
    if feature == "TC（总胆固醇）":
        return "TC_>6.2" if value > 6.2 else "TC_<=6.2"
    if feature == "LDL-C（低密度脂蛋白）":
        return "LDL-C_>3.1" if value > 3.1 else "LDL-C_<=3.1"
    if feature == "HDL-C（高密度脂蛋白）":
        if value < 1.04:
            return "HDL-C_<1.04"
        if value <= 1.55:
            return "HDL-C_1.04-1.55"
        return "HDL-C_>1.55"
    if feature == "血尿酸":
        threshold = 428 if int(row["性别"]) == 1 else 357
        return "高尿酸_是" if value > threshold else "高尿酸_否"
    if feature == "年龄组":
        return f"年龄组_{label_age_group(value)}"
    if feature == "性别":
        return "性别_男" if int(value) == 1 else "性别_女"
    if feature == "吸烟史":
        return "吸烟史_有" if int(value) == 1 else "吸烟史_无"
    return f"{feature}_{value}"


def build_transactions(subgroup: pd.DataFrame, feature_columns: list[str]) -> tuple[list[list[str]], pd.DataFrame]:
    transactions: list[list[str]] = []
    preview_rows: list[dict[str, Any]] = []
    for _, row in subgroup.iterrows():
        items = []
        for feature in feature_columns:
            items.append(discretize_item(feature, row[feature], row))
        items.append("高血脂阳性" if int(row["高血脂症二分类标签"]) == 1 else "高血脂阴性")
        transactions.append(items)
        preview_rows.append(
            {
                "样本ID": row["样本ID"],
                "高血脂症状态": "阳性" if int(row["高血脂症二分类标签"]) == 1 else "阴性",
                "items": " | ".join(items),
            }
        )
    return transactions, pd.DataFrame(preview_rows)


def rules_to_frame(rules: pd.DataFrame) -> pd.DataFrame:
    if rules.empty:
        return pd.DataFrame(
            columns=[
                "antecedent",
                "consequent",
                "antecedent_len",
                "support",
                "confidence",
                "lift",
                "leverage",
                "conviction",
            ]
        )
    out = rules.copy()
    out["antecedent"] = out["antecedents"].apply(lambda x: " + ".join(sorted(list(x))))
    out["consequent"] = out["consequents"].apply(lambda x: " + ".join(sorted(list(x))))
    out["antecedent_len"] = out["antecedents"].apply(len)
    keep = [
        "antecedent",
        "consequent",
        "antecedent_len",
        "support",
        "confidence",
        "lift",
        "leverage",
        "conviction",
    ]
    return out[keep].sort_values(by=["lift", "confidence", "support"], ascending=[False, False, False]).reset_index(drop=True)


def run_apriori_search(transaction_df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    attempts = [
        {"min_support": 0.08, "min_confidence": 0.85, "min_lift": 1.08},
        {"min_support": 0.06, "min_confidence": 0.85, "min_lift": 1.08},
        {"min_support": 0.06, "min_confidence": 0.82, "min_lift": 1.08},
        {"min_support": 0.06, "min_confidence": 0.82, "min_lift": 1.05},
    ]
    final_rules = pd.DataFrame()
    final_attempt = attempts[-1]
    for attempt in attempts:
        itemsets = apriori(transaction_df, min_support=attempt["min_support"], use_colnames=True, max_len=5)
        if itemsets.empty:
            continue
        rules = association_rules(itemsets, metric="confidence", min_threshold=attempt["min_confidence"])
        if rules.empty:
            continue
        rules = rules[
            rules["consequents"].apply(lambda x: x == frozenset({"高血脂阳性"}))
            & rules["antecedents"].apply(lambda x: 2 <= len(x) <= 4)
            & (rules["confidence"] >= attempt["min_confidence"])
            & (rules["lift"] >= attempt["min_lift"])
        ].copy()
        if not rules.empty:
            final_rules = rules
            final_attempt = attempt
            break
    return rules_to_frame(final_rules), final_attempt


def resolve_shap_interactions(explainer: shap.TreeExplainer, X_values: np.ndarray) -> np.ndarray:
    interaction_values = explainer.shap_interaction_values(X_values)
    if isinstance(interaction_values, list):
        return np.asarray(interaction_values[1])
    interaction_values = np.asarray(interaction_values)
    if interaction_values.ndim == 4:
        return interaction_values[:, :, :, 1]
    return interaction_values


def interaction_pairs(feature_columns: list[str]) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    primary = "痰湿质"
    if primary not in feature_columns:
        return pairs
    for feature in ["ADL总分", "IADL总分", "TG（甘油三酯）", "TC（总胆固醇）", "LDL-C（低密度脂蛋白）", "HDL-C（高密度脂蛋白）", "血尿酸"]:
        if feature in feature_columns:
            pairs.append((primary, feature))
    return pairs


def plot_interaction_heatmap(matrix: pd.DataFrame, output_path: Path) -> None:
    plt.figure(figsize=(max(8, matrix.shape[1] * 0.6), max(6, matrix.shape[0] * 0.6)))
    sns.heatmap(matrix, cmap="YlOrRd", square=True)
    plt.title("Mean Absolute SHAP Interaction")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_pair_interaction(
    feature_values: np.ndarray,
    interaction_values: np.ndarray,
    feature_names: list[str],
    feature_a: str,
    feature_b: str,
    output_path: Path,
) -> None:
    idx_a = feature_names.index(feature_a)
    idx_b = feature_names.index(feature_b)
    x = feature_values[:, idx_a]
    interaction = interaction_values[:, idx_a, idx_b]
    color = feature_values[:, idx_b]

    plt.figure(figsize=(6.5, 4.8))
    scatter = plt.scatter(x, interaction, c=color, cmap="viridis", s=24, alpha=0.7)
    plt.axhline(0, linestyle="--", color="#999999", linewidth=1)
    plt.xlabel(feature_a)
    plt.ylabel(f"Interaction: {feature_a} x {feature_b}")
    plt.title(f"SHAP Interaction: {feature_a} x {feature_b}")
    cbar = plt.colorbar(scatter)
    cbar.set_label(feature_b)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()


def build_subgroup_summary_markdown(
    output_path: Path,
    path_name: str,
    sample_summary: pd.DataFrame,
    cart_rules: pd.DataFrame,
    apriori_rules: pd.DataFrame,
    interaction_summary: pd.DataFrame,
) -> None:
    lines = [
        f"# {path_name} 痰湿质亚群结果汇总",
        "",
        "## 亚群样本概况",
        "```csv",
        sample_summary.to_csv(index=False).strip(),
        "```",
        "",
        "## CART 高风险规则",
        "```csv",
        cart_rules.head(10).to_csv(index=False).strip(),
        "```",
        "",
        "## Apriori 规则",
        "```csv",
        apriori_rules.head(10).to_csv(index=False).strip(),
        "```",
        "",
        "## SHAP 交互强度",
        "```csv",
        interaction_summary.head(10).to_csv(index=False).strip(),
        "```",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")


def run_subgroup_for_path(config: Any) -> dict[str, Any]:
    print(f"[INFO] Running subgroup_t5 for {config.slug}")
    path_dir = OUTPUT_ROOT / config.slug
    dirs = make_dirs(path_dir)
    best_params = load_best_params(path_dir)

    subgroup = load_subgroup_data()
    use_cols = ["样本ID", "体质标签", "高血脂症二分类标签", "性别"] + config.feature_columns
    subgroup = subgroup[use_cols].copy()
    subgroup = subgroup.loc[:, ~subgroup.columns.duplicated()].copy()

    sample_summary = pd.DataFrame(
        [
            {
                "path_slug": config.slug,
                "subgroup_n": len(subgroup),
                "positive_n": int(subgroup["高血脂症二分类标签"].sum()),
                "negative_n": int((1 - subgroup["高血脂症二分类标签"]).sum()),
                "positive_rate": float(subgroup["高血脂症二分类标签"].mean()),
            }
        ]
    )
    sample_summary.to_csv(dirs["tables"] / "subgroup_sample_summary.csv", index=False, encoding="utf-8-sig")

    X_raw = subgroup[config.feature_columns].copy()
    y = subgroup["高血脂症二分类标签"].astype(int).copy()
    X_imp, _, _ = preprocess_features(X_raw, config.continuous_columns)

    best_alpha, alpha_df = select_cart_alpha(X_imp, y)
    alpha_df.to_csv(dirs["tables"] / "cart_alpha_cv_results.csv", index=False, encoding="utf-8-sig")
    cart_model = DecisionTreeClassifier(
        criterion="gini",
        class_weight="balanced",
        max_depth=4,
        min_samples_leaf=max(15, math.ceil(0.08 * len(subgroup))),
        ccp_alpha=best_alpha,
        random_state=SEED,
    )
    cart_model.fit(X_imp, y)
    cart_rules_all = cart_leaf_rules_binary(cart_model, config.feature_columns)
    cart_rules_all.to_csv(dirs["tables"] / "cart_rules_all.csv", index=False, encoding="utf-8-sig")
    cart_high = cart_rules_all[
        (cart_rules_all["positive_rate"] >= 0.85) & (cart_rules_all["sample_count"] >= 15)
    ].reset_index(drop=True)
    cart_high.to_csv(dirs["tables"] / "cart_high_risk_rules.csv", index=False, encoding="utf-8-sig")
    plot_cart_tree(cart_model, config.feature_columns, dirs["figures"] / "cart_tree.png")

    transactions, preview = build_transactions(subgroup, config.feature_columns)
    preview.to_csv(dirs["tables"] / "apriori_transactions_preview.csv", index=False, encoding="utf-8-sig")
    te = TransactionEncoder()
    te_array = te.fit(transactions).transform(transactions)
    transaction_df = pd.DataFrame(te_array, columns=te.columns_)
    apriori_rules_all, apriori_attempt = run_apriori_search(transaction_df)
    apriori_rules_all.to_csv(dirs["tables"] / "apriori_rules_all.csv", index=False, encoding="utf-8-sig")
    apriori_filtered = apriori_rules_all.head(15).copy()
    apriori_filtered["used_min_support"] = apriori_attempt["min_support"]
    apriori_filtered["used_min_confidence"] = apriori_attempt["min_confidence"]
    apriori_filtered["used_min_lift"] = apriori_attempt["min_lift"]
    apriori_filtered.to_csv(dirs["tables"] / "apriori_rules_filtered.csv", index=False, encoding="utf-8-sig")

    subgroup_model = build_base_pipeline(config=config, rf_params=best_params, rf_n_jobs=-1)
    subgroup_model.fit(X_raw, y)
    X_values, rf_model = extract_rf_for_shap(subgroup_model, config.feature_columns, X_raw)
    explainer = shap.TreeExplainer(rf_model)
    shap_values = resolve_shap_values(explainer, X_values)
    interaction_values = resolve_shap_interactions(explainer, X_values)
    mean_abs_interactions = np.abs(interaction_values).mean(axis=0)

    pair_rows = []
    pairs = interaction_pairs(config.feature_columns)
    for feat_a, feat_b in pairs:
        idx_a = config.feature_columns.index(feat_a)
        idx_b = config.feature_columns.index(feat_b)
        pair_rows.append(
            {
                "feature_a": feat_a,
                "feature_b": feat_b,
                "mean_abs_interaction": float(mean_abs_interactions[idx_a, idx_b]),
            }
        )
    interaction_summary = pd.DataFrame(pair_rows).sort_values(
        by="mean_abs_interaction", ascending=False
    ).reset_index(drop=True)
    interaction_summary.to_csv(dirs["tables"] / "shap_interaction_summary.csv", index=False, encoding="utf-8-sig")

    selected_features = list(dict.fromkeys([item for pair in pairs for item in pair]))
    if selected_features:
        idx = [config.feature_columns.index(name) for name in selected_features]
        matrix = pd.DataFrame(
            mean_abs_interactions[np.ix_(idx, idx)],
            index=selected_features,
            columns=selected_features,
        )
        plot_interaction_heatmap(matrix, dirs["figures"] / "shap_interaction_heatmap.png")

    for feat_a, feat_b in pairs:
        fname = f"shap_interaction_{safe_name(feat_a)}__{safe_name(feat_b)}.png"
        plot_pair_interaction(
            X_values,
            interaction_values,
            config.feature_columns,
            feat_a,
            feat_b,
            dirs["figures"] / fname,
        )

    build_subgroup_summary_markdown(
        dirs["base"] / "subgroup_result_summary.md",
        config.name,
        sample_summary,
        cart_high if not cart_high.empty else cart_rules_all.head(0),
        apriori_filtered,
        interaction_summary,
    )

    top_pairs = interaction_summary.head(3)["feature_a"].str.cat(interaction_summary.head(3)["feature_b"], sep=" x ").tolist()
    while len(top_pairs) < 3:
        top_pairs.append("")
    return {
        "path_slug": config.slug,
        "subgroup_n": int(sample_summary.iloc[0]["subgroup_n"]),
        "positive_n": int(sample_summary.iloc[0]["positive_n"]),
        "positive_rate": float(sample_summary.iloc[0]["positive_rate"]),
        "cart_high_risk_rule_count": int(len(cart_high)),
        "apriori_rule_count": int(len(apriori_rules_all)),
        "top1_interaction_pair": top_pairs[0],
        "top2_interaction_pair": top_pairs[1],
        "top3_interaction_pair": top_pairs[2],
    }


def main() -> None:
    sns.set_theme(style="whitegrid")
    comparison_rows = []
    for config in PATH_CONFIGS:
        comparison_rows.append(run_subgroup_for_path(config))
    comparison_df = pd.DataFrame(comparison_rows)
    comparison_df.to_csv(OUTPUT_ROOT / "subgroup_t5_path_comparison.csv", index=False, encoding="utf-8-sig")
    print("[INFO] subgroup_t5 analysis completed.")


if __name__ == "__main__":
    main()
