from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.diagnostic import linear_reset
from statsmodels.stats.outliers_influence import variance_inflation_factor


ROOT = Path(__file__).resolve().parent
D1_PATH = ROOT / "Q1_D1_异常值标记数据.csv"
D2_PATH = ROOT / "Q1_D2_缩尾数据.csv"
D2_PREPROCESSED_PATH = ROOT / "Q1_D2_预处理结果.csv"
OUTPUT_DIR = ROOT / "q1_diagnostics"
SCATTER_DIR = OUTPUT_DIR / "figures" / "scatter_lowess"
BOXPLOT_DIR = OUTPUT_DIR / "figures" / "boxplots"
HEATMAP_DIR = OUTPUT_DIR / "figures" / "heatmaps"
TABLE_DIR = OUTPUT_DIR / "tables"
RCS_DIR = OUTPUT_DIR / "figures" / "rcs_effects"
RCS_PREPROCESSED_DIR = OUTPUT_DIR / "figures" / "rcs_effects_preprocessed"
SEVERITY_SUMMARY_PATH = OUTPUT_DIR / "Q1_体质严重度表征模型结果汇总.md"
SEVERITY_PREPROCESSED_SUMMARY_PATH = OUTPUT_DIR / "Q1_体质严重度表征模型结果汇总_预处理版.md"

Y_COL = "痰湿质"
CONTINUOUS_VARS = [
    "HDL-C（高密度脂蛋白）",
    "LDL-C（低密度脂蛋白）",
    "TG（甘油三酯）",
    "TC（总胆固醇）",
    "空腹血糖",
    "血尿酸",
    "BMI",
]
ACTIVITY_VARS = ["ADL总分", "IADL总分"]
ALL_VARS = CONTINUOUS_VARS + ACTIVITY_VARS

RESET_NAME_MAP = {
    "HDL-C（高密度脂蛋白）": "hdl_c",
    "LDL-C（低密度脂蛋白）": "ldl_c",
    "TG（甘油三酯）": "tg",
    "TC（总胆固醇）": "tc",
    "空腹血糖": "fbg",
    "血尿酸": "uric_acid",
    "BMI": "bmi",
    "ADL总分": "adl_total",
    "IADL总分": "iadl_total",
    "痰湿质": "phlegm_dampness",
}


@dataclass
class ShapeAssessment:
    variable: str
    source: str
    ols_slope: float
    ols_intercept: float
    lowess_deviation_ratio: float
    boxplot_groups: int
    boxplot_median_trend: str
    shape_judgment: str


def configure_plotting() -> None:
    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False
    sns.set_theme(style="whitegrid", font="Microsoft YaHei")


def ensure_output_dirs() -> None:
    for path in [SCATTER_DIR, BOXPLOT_DIR, HEATMAP_DIR, TABLE_DIR, RCS_DIR, RCS_PREPROCESSED_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig")


def df_to_markdown(df: pd.DataFrame) -> str:
    df_fmt = df.copy()
    for column in df_fmt.columns:
        if pd.api.types.is_float_dtype(df_fmt[column]):
            df_fmt[column] = df_fmt[column].map(lambda value: f"{value:.4f}")
    header = "| " + " | ".join(map(str, df_fmt.columns)) + " |"
    divider = "| " + " | ".join(["---"] * len(df_fmt.columns)) + " |"
    rows = [
        "| " + " | ".join(map(str, row)) + " |"
        for row in df_fmt.astype(str).itertuples(index=False, name=None)
    ]
    return "\n".join([header, divider] + rows)


def validate_inputs(d1: pd.DataFrame, d2: pd.DataFrame) -> pd.DataFrame:
    checks = []
    for name, df in [("D1", d1), ("D2", d2)]:
        checks.append(
            {
                "dataset": name,
                "rows": int(df.shape[0]),
                "columns": int(df.shape[1]),
                "sample_id_unique": bool(df["样本ID"].is_unique),
                "missing_analysis_values": int(df[[Y_COL] + ALL_VARS].isna().sum().sum()),
            }
        )
    result = pd.DataFrame(checks)
    result.to_csv(TABLE_DIR / "input_validation.csv", index=False, encoding="utf-8-sig")
    return result


def strength_label(value: float) -> str:
    abs_value = abs(value)
    if abs_value < 0.1:
        return "很弱"
    if abs_value < 0.3:
        return "弱"
    if abs_value < 0.5:
        return "中等"
    return "较强"


def direction_label(value: float) -> str:
    if value > 0:
        return "正相关"
    if value < 0:
        return "负相关"
    return "无明显方向"


def sanitize_filename(text: str) -> str:
    replacements = {
        "（": "_",
        "）": "",
        "(": "_",
        ")": "",
        "-": "_",
        "/": "_",
        " ": "_",
    }
    cleaned = text
    for old, new in replacements.items():
        cleaned = cleaned.replace(old, new)
    return cleaned


def compute_ols_line(x: pd.Series, y: pd.Series) -> tuple[float, float]:
    slope, intercept = np.polyfit(x.to_numpy(dtype=float), y.to_numpy(dtype=float), 1)
    return float(slope), float(intercept)


def compute_lowess(x: pd.Series, y: pd.Series, frac: float = 0.66) -> np.ndarray:
    return sm.nonparametric.lowess(
        endog=y.to_numpy(dtype=float),
        exog=x.to_numpy(dtype=float),
        frac=frac,
        return_sorted=True,
    )


def classify_boxplot_trend(medians: pd.Series) -> str:
    if medians.size < 2:
        return "分组不足"
    if medians.is_monotonic_increasing:
        return "单调上升"
    if medians.is_monotonic_decreasing:
        return "单调下降"
    return "非单调"


def classify_shape(lowess_sorted: np.ndarray, x: pd.Series, y: pd.Series, group_trend: str) -> tuple[float, float, float, str]:
    slope, intercept = compute_ols_line(x, y)
    lowess_x = lowess_sorted[:, 0]
    lowess_y = lowess_sorted[:, 1]
    ols_pred = slope * lowess_x + intercept
    deviation_ratio = float(np.mean(np.abs(lowess_y - ols_pred)) / max(y.std(ddof=0), 1e-8))
    corr = float(np.corrcoef(x.to_numpy(dtype=float), y.to_numpy(dtype=float))[0, 1])

    if abs(corr) < 0.1 and deviation_ratio < 0.08:
        judgment = "关系较弱"
    elif deviation_ratio < 0.08:
        judgment = "近似线性"
    elif group_trend.startswith("单调"):
        judgment = "单调但非线性"
    else:
        judgment = "疑似弯曲/拐点"
    return slope, intercept, deviation_ratio, judgment


def anomaly_flag_column(variable: str, columns: Iterable[str]) -> str | None:
    expected = f"{variable}_异常值标记"
    return expected if expected in columns else None


def scatter_and_lowess_plots(d1: pd.DataFrame) -> list[ShapeAssessment]:
    results: list[ShapeAssessment] = []
    rng = np.random.default_rng(20260419)

    for variable in ALL_VARS:
        x = d1[variable]
        y = d1[Y_COL]
        flag_col = anomaly_flag_column(variable, d1.columns)
        lowess_sorted = compute_lowess(x, y)

        quantile_groups = pd.qcut(x, q=4, duplicates="drop")
        medians = y.groupby(quantile_groups, observed=False).median()
        group_trend = classify_boxplot_trend(medians)
        slope, intercept, deviation_ratio, judgment = classify_shape(lowess_sorted, x, y, group_trend)

        fig, ax = plt.subplots(figsize=(9, 6))
        if variable in ACTIVITY_VARS:
            x_for_scatter = x + rng.normal(0, 0.12, size=len(x))
        else:
            x_for_scatter = x

        if flag_col:
            normal_mask = d1[flag_col].eq(0)
            anomaly_mask = d1[flag_col].eq(1)
            ax.scatter(
                x_for_scatter[normal_mask],
                y[normal_mask],
                s=22,
                alpha=0.55,
                color="#4C78A8",
                edgecolors="none",
                label="正常点",
            )
            if anomaly_mask.any():
                ax.scatter(
                    x_for_scatter[anomaly_mask],
                    y[anomaly_mask],
                    s=34,
                    alpha=0.85,
                    color="#E45756",
                    edgecolors="none",
                    label="异常值点",
                )
        else:
            ax.scatter(
                x_for_scatter,
                y,
                s=22,
                alpha=0.6,
                color="#4C78A8",
                edgecolors="none",
                label="样本点",
            )

        sorted_x = np.sort(x.to_numpy(dtype=float))
        ax.plot(sorted_x, slope * sorted_x + intercept, color="#F58518", linewidth=2.2, label="OLS拟合线")
        ax.plot(lowess_sorted[:, 0], lowess_sorted[:, 1], color="#54A24B", linewidth=2.4, label="LOWESS")
        ax.set_xlabel(variable)
        ax.set_ylabel(Y_COL)
        ax.set_title(f"{variable} 与 {Y_COL}：散点图 + OLS + LOWESS")
        ax.legend(frameon=True)
        ax.text(
            0.02,
            0.98,
            f"形态判断：{judgment}\nLOWESS偏离比：{deviation_ratio:.3f}",
            transform=ax.transAxes,
            va="top",
            ha="left",
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.9},
        )
        fig.tight_layout()
        fig.savefig(
            SCATTER_DIR / f"scatter_lowess_{sanitize_filename(variable)}.png",
            dpi=220,
            bbox_inches="tight",
        )
        plt.close(fig)

        results.append(
            ShapeAssessment(
                variable=variable,
                source="Q1_D1_异常值标记数据.csv",
                ols_slope=slope,
                ols_intercept=intercept,
                lowess_deviation_ratio=deviation_ratio,
                boxplot_groups=int(medians.size),
                boxplot_median_trend=group_trend,
                shape_judgment=judgment,
            )
        )
    return results


def make_group_labels(categories: pd.IntervalIndex) -> list[str]:
    labels = []
    for idx, interval in enumerate(categories, start=1):
        left = interval.left
        right = interval.right
        labels.append(f"Q{idx}\n({left:.2f}, {right:.2f}]")
    return labels


def boxplots(d1: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for variable in ALL_VARS:
        grouped = pd.qcut(d1[variable], q=4, duplicates="drop")
        categories = grouped.cat.categories
        label_map = dict(zip(categories, make_group_labels(categories)))
        plot_df = pd.DataFrame(
            {
                "group": grouped.map(label_map),
                Y_COL: d1[Y_COL],
            }
        )

        medians = (
            d1.groupby(grouped, observed=False)[Y_COL]
            .median()
            .rename(index=label_map)
        )
        trend = classify_boxplot_trend(medians)

        fig, ax = plt.subplots(figsize=(9, 6))
        sns.boxplot(data=plot_df, x="group", y=Y_COL, color="#A0CBE8", ax=ax)
        ax.set_xlabel(f"{variable} 分箱")
        ax.set_ylabel(Y_COL)
        ax.set_title(f"{variable} 分组后 {Y_COL} 箱线图")
        note = f"分组数：{len(categories)}；中位数趋势：{trend}"
        if len(categories) < 4:
            note += "；因重复分位点自动合并分组"
        ax.text(
            0.02,
            0.98,
            note,
            transform=ax.transAxes,
            va="top",
            ha="left",
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.9},
        )
        fig.tight_layout()
        fig.savefig(
            BOXPLOT_DIR / f"boxplot_{sanitize_filename(variable)}.png",
            dpi=220,
            bbox_inches="tight",
        )
        plt.close(fig)

        rows.append(
            {
                "变量": variable,
                "数据源": "Q1_D1_异常值标记数据.csv",
                "分组数": int(len(categories)),
                "中位数趋势": trend,
                "是否自动合并分组": "是" if len(categories) < 4 else "否",
            }
        )
    return pd.DataFrame(rows)


def pearson_analysis(d1: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for variable in CONTINUOUS_VARS:
        coefficient, p_value = stats.pearsonr(d1[variable], d1[Y_COL])
        rows.append(
            {
                "变量名": variable,
                "n": int(d1.shape[0]),
                "Pearson_r": float(coefficient),
                "p值": float(p_value),
                "方向": direction_label(float(coefficient)),
                "强度分级": strength_label(float(coefficient)),
            }
        )
    table = pd.DataFrame(rows).sort_values(by="Pearson_r", ascending=False)
    table.to_csv(TABLE_DIR / "pearson_results.csv", index=False, encoding="utf-8-sig")

    corr = d1[[Y_COL] + CONTINUOUS_VARS].corr(method="pearson")
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap="RdBu_r", center=0, fmt=".2f", square=True, ax=ax)
    ax.set_title("Pearson 相关矩阵热力图")
    fig.tight_layout()
    fig.savefig(HEATMAP_DIR / "pearson_heatmap.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    corr.to_csv(TABLE_DIR / "pearson_correlation_matrix.csv", encoding="utf-8-sig")
    return table


def spearman_analysis(d1: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for variable in ACTIVITY_VARS:
        coefficient, p_value = stats.spearmanr(d1[variable], d1[Y_COL])
        rows.append(
            {
                "变量名": variable,
                "n": int(d1.shape[0]),
                "Spearman_rho": float(coefficient),
                "p值": float(p_value),
                "方向": direction_label(float(coefficient)),
                "强度分级": strength_label(float(coefficient)),
            }
        )
    table = pd.DataFrame(rows).sort_values(by="Spearman_rho", ascending=False)
    table.to_csv(TABLE_DIR / "spearman_results.csv", index=False, encoding="utf-8-sig")

    corr = d1[[Y_COL] + ACTIVITY_VARS].corr(method="spearman")
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(corr, annot=True, cmap="RdBu_r", center=0, fmt=".2f", square=True, ax=ax)
    ax.set_title("Spearman 相关矩阵热力图")
    fig.tight_layout()
    fig.savefig(HEATMAP_DIR / "spearman_heatmap.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    corr.to_csv(TABLE_DIR / "spearman_correlation_matrix.csv", encoding="utf-8-sig")
    return table


def reset_analysis(d2: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    reset_df = d2[[Y_COL] + ALL_VARS].rename(columns=RESET_NAME_MAP)
    y = reset_df[RESET_NAME_MAP[Y_COL]]
    x = sm.add_constant(reset_df[[RESET_NAME_MAP[col] for col in ALL_VARS]])
    model = sm.OLS(y, x).fit()
    reset_result = linear_reset(model, power=3, use_f=True)

    coef_table = (
        pd.DataFrame(
            {
                "变量": model.params.index,
                "系数": model.params.values,
                "标准误": model.bse.values,
                "t值": model.tvalues.values,
                "p值": model.pvalues.values,
            }
        )
        .replace({"变量": {"const": "截距"}})
    )
    coef_table.to_csv(TABLE_DIR / "reset_model_coefficients.csv", index=False, encoding="utf-8-sig")

    reset_table = pd.DataFrame(
        [
            {
                "数据源": "Q1_D2_缩尾数据.csv",
                "n": int(d2.shape[0]),
                "R2": float(model.rsquared),
                "调整R2": float(model.rsquared_adj),
                "RESET_F统计量": float(reset_result.fvalue),
                "p值": float(reset_result.pvalue),
                "是否拒绝H0": "是" if reset_result.pvalue < 0.05 else "否",
                "结论": (
                    "提示线性函数形式可能不足，后续应考虑非线性项或分段设定"
                    if reset_result.pvalue < 0.05
                    else "未发现显著函数形式设定错误，当前线性形式暂可接受"
                ),
            }
        ]
    )
    reset_table.to_csv(TABLE_DIR / "reset_results.csv", index=False, encoding="utf-8-sig")

    with (TABLE_DIR / "reset_model_summary.txt").open("w", encoding="utf-8") as handle:
        handle.write(model.summary().as_text())

    return reset_table, coef_table


def vif_risk_label(vif: float, tolerance: float) -> tuple[str, str]:
    if vif >= 10 or tolerance <= 0.1:
        return "是", "存在严重多重共线性风险"
    if vif >= 5 or tolerance <= 0.2:
        return "是", "存在中度多重共线性风险"
    return "否", "未发现明显多重共线性"


def multicollinearity_analysis(d2: pd.DataFrame) -> pd.DataFrame:
    x = sm.add_constant(d2[ALL_VARS].astype(float))
    rows = []
    for index, column in enumerate(x.columns):
        if column == "const":
            continue
        vif = float(variance_inflation_factor(x.values, index))
        tolerance = 1.0 / vif if vif != 0 else np.nan
        exceeds, conclusion = vif_risk_label(vif, tolerance)
        rows.append(
            {
                "变量名": column,
                "VIF": vif,
                "Tolerance": tolerance,
                "是否超过阈值": exceeds,
                "结论": conclusion,
            }
        )
    vif_df = pd.DataFrame(rows).sort_values(by="VIF", ascending=False)
    vif_df.to_csv(TABLE_DIR / "multicollinearity_vif_results.csv", index=False, encoding="utf-8-sig")
    return vif_df


def prepare_modeling_data(d2: pd.DataFrame) -> pd.DataFrame:
    return d2[[Y_COL] + ALL_VARS].rename(columns=RESET_NAME_MAP).copy()


def compute_rmse(model: sm.regression.linear_model.RegressionResultsWrapper) -> float:
    return float(np.sqrt(np.mean(np.square(model.resid))))


def compute_rcs_knots(model_input: pd.DataFrame, output_path: Path) -> pd.DataFrame:
    quantiles = [0.05, 0.35, 0.65, 0.95]
    rows = []
    for variable in ALL_VARS:
        values = model_input[variable].quantile(quantiles)
        rows.append(
            {
                "变量名": variable,
                "别名": RESET_NAME_MAP[variable],
                "knot_5%": float(values.loc[0.05]),
                "knot_35%": float(values.loc[0.35]),
                "knot_65%": float(values.loc[0.65]),
                "knot_95%": float(values.loc[0.95]),
            }
        )
    knots_df = pd.DataFrame(rows)
    knots_df.to_csv(output_path, index=False, encoding="utf-8-sig")
    return knots_df


def format_float(value: float) -> str:
    return f"{float(value):.12g}"


def build_rcs_term(alias: str, knot_row: pd.Series) -> str:
    return (
        f"cr({alias}, "
        f"knots=({format_float(knot_row['knot_35%'])}, {format_float(knot_row['knot_65%'])}), "
        f"lower_bound={format_float(knot_row['knot_5%'])}, "
        f"upper_bound={format_float(knot_row['knot_95%'])})"
    )


def build_rcs_formula(knots_df: pd.DataFrame, linear_overrides: set[str] | None = None, drop_alias: str | None = None) -> str:
    linear_overrides = linear_overrides or set()
    knot_map = knots_df.set_index("别名")
    terms = []
    for variable in ALL_VARS:
        alias = RESET_NAME_MAP[variable]
        if alias == drop_alias:
            continue
        if alias in linear_overrides:
            terms.append(alias)
        else:
            terms.append(build_rcs_term(alias, knot_map.loc[alias]))
    return f"{RESET_NAME_MAP[Y_COL]} ~ " + " + ".join(terms)


def model_fit_row(name: str, data_source: str, model: sm.regression.linear_model.RegressionResultsWrapper) -> dict[str, float | str]:
    return {
        "模型": name,
        "数据源": data_source,
        "n": int(model.nobs),
        "R2": float(model.rsquared),
        "调整R2": float(model.rsquared_adj),
        "AIC": float(model.aic),
        "BIC": float(model.bic),
        "RMSE": compute_rmse(model),
    }


def fit_severity_models(
    model_input: pd.DataFrame,
    data_source: str,
    suffix: str = "",
) -> tuple[
    sm.regression.linear_model.RegressionResultsWrapper,
    sm.regression.linear_model.RegressionResultsWrapper,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    model_df = prepare_modeling_data(model_input)
    knots_df = compute_rcs_knots(model_input, TABLE_DIR / f"rcs_knots_table{suffix}.csv")

    ols_formula = f"{RESET_NAME_MAP[Y_COL]} ~ " + " + ".join(RESET_NAME_MAP[var] for var in ALL_VARS)
    ols_model = smf.ols(ols_formula, data=model_df).fit()
    rcs_formula = build_rcs_formula(knots_df)
    rcs_model = smf.ols(rcs_formula, data=model_df).fit()

    fit_table = pd.DataFrame(
        [
            model_fit_row("OLS基线模型", data_source, ols_model),
            model_fit_row("RCS全变量模型", data_source, rcs_model),
        ]
    )
    fit_table.to_csv(TABLE_DIR / f"rcs_model_fit_comparison{suffix}.csv", index=False, encoding="utf-8-sig")

    coef_table = (
        pd.DataFrame(
            {
                "变量名": [Y_COL if name == "Intercept" else next((cn for cn, alias in RESET_NAME_MAP.items() if alias == name), name) for name in ols_model.params.index],
                "系数": ols_model.params.values,
                "标准误": ols_model.bse.values,
                "t值": ols_model.tvalues.values,
                "p值": ols_model.pvalues.values,
                "下限95CI": ols_model.conf_int()[0].values,
                "上限95CI": ols_model.conf_int()[1].values,
            }
        )
        .replace({"变量名": {Y_COL: "截距"}})
    )
    coef_table.to_csv(TABLE_DIR / f"ols_severity_coefficients{suffix}.csv", index=False, encoding="utf-8-sig")

    ols_summary = pd.DataFrame([model_fit_row("OLS基线模型", data_source, ols_model)])
    ols_summary.to_csv(TABLE_DIR / f"ols_severity_model_summary{suffix}.csv", index=False, encoding="utf-8-sig")

    with (TABLE_DIR / f"rcs_model_summary{suffix}.txt").open("w", encoding="utf-8") as handle:
        handle.write(rcs_model.summary().as_text())
    with (TABLE_DIR / f"ols_severity_model_summary{suffix}.txt").open("w", encoding="utf-8") as handle:
        handle.write(ols_model.summary().as_text())

    return ols_model, rcs_model, fit_table, knots_df, model_df


def safe_compare_f_test(
    full_model: sm.regression.linear_model.RegressionResultsWrapper,
    reduced_model: sm.regression.linear_model.RegressionResultsWrapper,
) -> tuple[float, float, float]:
    f_stat, p_value, df_diff = full_model.compare_f_test(reduced_model)
    return float(f_stat), float(p_value), float(df_diff)


def classify_rcs_shape(x_values: np.ndarray, y_values: np.ndarray) -> str:
    if len(x_values) < 5:
        return "近似线性"

    curve_range = float(np.max(y_values) - np.min(y_values))
    if curve_range < 1e-6:
        return "平台"

    straight_line = np.linspace(y_values[0], y_values[-1], len(y_values))
    deviation_ratio = float(np.mean(np.abs(y_values - straight_line)) / max(curve_range, 1e-8))
    first_diff = np.diff(y_values)
    tol = max(np.quantile(np.abs(first_diff), 0.2), curve_range * 0.01, 1e-6)
    pos = first_diff > tol
    neg = first_diff < -tol

    if not neg.any() or not pos.any():
        edge = max(int(len(first_diff) * 0.2), 1)
        start_mean = float(np.mean(np.abs(first_diff[:edge])))
        mid_left = max(int(len(first_diff) * 0.4), 1)
        mid_right = min(int(len(first_diff) * 0.6), len(first_diff))
        mid_mean = float(np.mean(np.abs(first_diff[mid_left:mid_right])))
        end_mean = float(np.mean(np.abs(first_diff[-edge:])))

        if deviation_ratio < 0.05:
            return "近似线性"
        if mid_mean < 0.35 * max(start_mean, end_mean, 1e-6):
            return "存在平台"
        if max(start_mean, end_mean) > 3 * min(max(start_mean, 1e-6), max(end_mean, 1e-6)):
            return "存在阈值"
        return "单调上升" if np.mean(first_diff) >= 0 else "单调下降"

    sign_series = np.sign(first_diff[np.abs(first_diff) > tol])
    sign_changes = int(np.sum(sign_series[1:] != sign_series[:-1])) if sign_series.size > 1 else 0
    second_diff_mean = float(np.mean(np.diff(y_values, n=2)))

    if sign_changes <= 1:
        if second_diff_mean > 0:
            return "U型"
        if second_diff_mean < 0:
            return "倒U型"
    return "局部转折"


def build_prediction_frame(model_df: pd.DataFrame, alias: str, grid: np.ndarray) -> pd.DataFrame:
    medians = model_df[[RESET_NAME_MAP[var] for var in ALL_VARS]].median()
    prediction_df = pd.DataFrame([medians.to_dict()] * len(grid))
    prediction_df[alias] = grid
    return prediction_df


def test_rcs_variables(
    model_input: pd.DataFrame,
    model_df: pd.DataFrame,
    knots_df: pd.DataFrame,
    ols_model: sm.regression.linear_model.RegressionResultsWrapper,
    rcs_model: sm.regression.linear_model.RegressionResultsWrapper,
    suffix: str = "",
) -> pd.DataFrame:
    rows = []
    knot_lookup = knots_df.set_index("变量名")

    for variable in ALL_VARS:
        alias = RESET_NAME_MAP[variable]

        reduced_formula = build_rcs_formula(knots_df, drop_alias=alias)
        reduced_model = smf.ols(reduced_formula, data=model_df).fit()
        overall_f, overall_p, overall_df = safe_compare_f_test(rcs_model, reduced_model)

        linear_formula = build_rcs_formula(knots_df, linear_overrides={alias})
        linear_model = smf.ols(linear_formula, data=model_df).fit()
        nonlinear_f, nonlinear_p, nonlinear_df = safe_compare_f_test(rcs_model, linear_model)

        grid = np.linspace(float(model_input[variable].min()), float(model_input[variable].max()), 200)
        prediction_df = build_prediction_frame(model_df, alias, grid)
        pred = rcs_model.get_prediction(prediction_df).summary_frame(alpha=0.05)
        mean_curve = pred["mean"].to_numpy(dtype=float)
        shape_label = classify_rcs_shape(grid, mean_curve)

        ols_coef = float(ols_model.params[alias])
        is_key = overall_p < 0.05
        if overall_p < 0.05 and nonlinear_p < 0.05:
            key_type = "关键非线性指标"
        elif overall_p < 0.05:
            key_type = "关键表征指标"
        else:
            key_type = "非关键指标"

        rows.append(
            {
                "变量名": variable,
                "OLS线性系数": ols_coef,
                "OLS方向": "正向" if ols_coef > 0 else ("负向" if ols_coef < 0 else "近零"),
                "总体效应F值": overall_f,
                "总体效应自由度差": overall_df,
                "总体效应P值": overall_p,
                "非线性F值": nonlinear_f,
                "非线性自由度差": nonlinear_df,
                "非线性P值": nonlinear_p,
                "RCS形态标签": shape_label,
                "是否列为关键指标": "是" if is_key else "否",
                "指标类型": key_type,
                "结点5%": float(knot_lookup.loc[variable, "knot_5%"]),
                "结点35%": float(knot_lookup.loc[variable, "knot_35%"]),
                "结点65%": float(knot_lookup.loc[variable, "knot_65%"]),
                "结点95%": float(knot_lookup.loc[variable, "knot_95%"]),
            }
        )
    variable_tests = pd.DataFrame(rows).sort_values(by=["是否列为关键指标", "总体效应P值"], ascending=[False, True])
    variable_tests.to_csv(TABLE_DIR / f"rcs_variable_tests{suffix}.csv", index=False, encoding="utf-8-sig")
    return variable_tests


def plot_rcs_effects(
    model_input: pd.DataFrame,
    model_df: pd.DataFrame,
    knots_df: pd.DataFrame,
    rcs_model: sm.regression.linear_model.RegressionResultsWrapper,
    variable_tests: pd.DataFrame,
    output_dir: Path,
    suffix_label: str = "",
) -> None:
    knot_lookup = knots_df.set_index("变量名")
    test_lookup = variable_tests.set_index("变量名")

    for variable in ALL_VARS:
        alias = RESET_NAME_MAP[variable]
        grid = np.linspace(float(model_input[variable].min()), float(model_input[variable].max()), 200)
        prediction_df = build_prediction_frame(model_df, alias, grid)
        pred = rcs_model.get_prediction(prediction_df).summary_frame(alpha=0.05)

        fig, ax = plt.subplots(figsize=(9, 6))
        ax.plot(grid, pred["mean"], color="#2C7FB8", linewidth=2.4, label="RCS拟合均值")
        ax.fill_between(
            grid,
            pred["mean_ci_lower"],
            pred["mean_ci_upper"],
            color="#A6CEE3",
            alpha=0.35,
            label="95% CI",
        )
        knot_row = knot_lookup.loc[variable]
        for label, knot_col in [("5%", "knot_5%"), ("35%", "knot_35%"), ("65%", "knot_65%"), ("95%", "knot_95%")]:
            ax.axvline(float(knot_row[knot_col]), color="#F58518", linestyle="--", linewidth=1)
            ax.text(
                float(knot_row[knot_col]),
                float(np.max(pred["mean_ci_upper"])),
                label,
                rotation=90,
                va="top",
                ha="right",
                color="#F58518",
            )

        label_row = test_lookup.loc[variable]
        ax.set_xlabel(variable)
        ax.set_ylabel("预测痰湿质积分")
        title_suffix = f"（{suffix_label}）" if suffix_label else ""
        ax.set_title(f"{variable} 的 RCS偏效应曲线{title_suffix}")
        ax.text(
            0.02,
            0.98,
            (
                f"形态：{label_row['RCS形态标签']}\n"
                f"总体效应P={label_row['总体效应P值']:.4f}\n"
                f"非线性P={label_row['非线性P值']:.4f}"
            ),
            transform=ax.transAxes,
            va="top",
            ha="left",
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.9},
        )
        ax.legend(frameon=True)
        fig.tight_layout()
        fig.savefig(output_dir / f"rcs_effect_{sanitize_filename(variable)}.png", dpi=220, bbox_inches="tight")
        plt.close(fig)


def generate_severity_summary_markdown(
    fit_table: pd.DataFrame,
    variable_tests: pd.DataFrame,
    knots_df: pd.DataFrame,
    summary_path: Path,
    data_source: str,
    effect_dir_label: str,
    suffix: str = "",
) -> None:
    key_indicators = variable_tests[variable_tests["是否列为关键指标"] == "是"]
    if key_indicators.empty:
        key_text = "当前全模型下未识别出总体效应显著的关键指标，说明现有 9 个指标对痰湿质积分的解释信号整体较弱。"
    else:
        key_text = "识别出的关键指标如下：" + "、".join(key_indicators["变量名"].tolist()) + "。"

    rcs_row = fit_table.loc[fit_table["模型"] == "RCS全变量模型"].iloc[0]
    ols_row = fit_table.loc[fit_table["模型"] == "OLS基线模型"].iloc[0]
    gain_text = (
        f"RCS相较OLS的调整R2变化为 {rcs_row['调整R2'] - ols_row['调整R2']:.4f}，"
        f"AIC变化为 {rcs_row['AIC'] - ols_row['AIC']:.4f}，"
        f"RMSE变化为 {rcs_row['RMSE'] - ols_row['RMSE']:.4f}。"
    )

    lines = [
        "# 第1题体质严重度表征模型结果汇总",
        "",
        "## 建模口径",
        "- 因变量：`痰湿质`",
        f"- 数据源：`{data_source}`",
        "- 自变量：7 个代谢/血常规指标 + `ADL总分` + `IADL总分`",
        "- OLS 作为线性基线模型，RCS 作为函数形态与非线性检验工具",
        "- RCS 全部变量统一使用 4 结点，结点分位点固定为 `5% / 35% / 65% / 95%`",
        "",
        "## 模型拟合对比",
        df_to_markdown(fit_table),
        "",
        gain_text,
        "",
        "## 变量总体效应与非线性检验",
        df_to_markdown(
            variable_tests[
                [
                    "变量名",
                    "OLS线性系数",
                    "总体效应P值",
                    "非线性P值",
                    "RCS形态标签",
                    "是否列为关键指标",
                    "指标类型",
                ]
            ]
        ),
        "",
        key_text,
        "",
        "## RCS结点表",
        df_to_markdown(knots_df),
        "",
        "## 输出文件",
        f"- OLS结果表：`q1_diagnostics/tables/ols_severity_model_summary{suffix}.csv`、`q1_diagnostics/tables/ols_severity_coefficients{suffix}.csv`",
        f"- RCS拟合对比：`q1_diagnostics/tables/rcs_model_fit_comparison{suffix}.csv`",
        f"- 变量检验：`q1_diagnostics/tables/rcs_variable_tests{suffix}.csv`",
        f"- 结点表：`q1_diagnostics/tables/rcs_knots_table{suffix}.csv`",
        f"- RCS偏效应图：`{effect_dir_label}`",
        "",
        "## 解释提醒",
        "- 该模型体系用于表征痰湿质严重度的函数形态和关键指标，不应表述为高精度预测模型。",
        "- 若总体效应不显著，仅可作为形态观察，不宜写成明确的实质性影响结论。",
    ]
    summary_path.write_text("\n".join(lines), encoding="utf-8")


def run_severity_workflow(
    model_input: pd.DataFrame,
    data_source: str,
    suffix: str,
    effect_dir: Path,
    summary_path: Path,
    effect_dir_label: str,
    summary_label: str,
) -> None:
    ols_model, rcs_model, fit_table, knots_df, model_df = fit_severity_models(
        model_input=model_input,
        data_source=data_source,
        suffix=suffix,
    )
    variable_tests = test_rcs_variables(
        model_input=model_input,
        model_df=model_df,
        knots_df=knots_df,
        ols_model=ols_model,
        rcs_model=rcs_model,
        suffix=suffix,
    )
    plot_rcs_effects(
        model_input=model_input,
        model_df=model_df,
        knots_df=knots_df,
        rcs_model=rcs_model,
        variable_tests=variable_tests,
        output_dir=effect_dir,
        suffix_label=summary_label,
    )
    generate_severity_summary_markdown(
        fit_table=fit_table,
        variable_tests=variable_tests,
        knots_df=knots_df,
        summary_path=summary_path,
        data_source=data_source,
        effect_dir_label=effect_dir_label,
        suffix=suffix,
    )


def write_shape_summary(shapes: list[ShapeAssessment]) -> pd.DataFrame:
    shape_df = pd.DataFrame([shape.__dict__ for shape in shapes]).rename(
        columns={
            "variable": "变量",
            "source": "数据源",
            "ols_slope": "OLS斜率",
            "ols_intercept": "OLS截距",
            "lowess_deviation_ratio": "LOWESS偏离比",
            "boxplot_groups": "箱线图分组数",
            "boxplot_median_trend": "箱线图中位数趋势",
            "shape_judgment": "形态判断",
        }
    )
    shape_df.to_csv(TABLE_DIR / "shape_assessment.csv", index=False, encoding="utf-8-sig")
    return shape_df


def generate_summary_markdown(
    validation: pd.DataFrame,
    shape_df: pd.DataFrame,
    boxplot_df: pd.DataFrame,
    pearson_df: pd.DataFrame,
    spearman_df: pd.DataFrame,
    reset_df: pd.DataFrame,
    vif_df: pd.DataFrame,
) -> None:
    lines = [
        "# 第1题前置检验结果汇总",
        "",
        "## 数据口径",
        "- 图形诊断、Pearson、Spearman 使用 `Q1_D1_异常值标记数据.csv`。",
        "- RESET 检验使用 `Q1_D2_缩尾数据.csv`。",
        "- 因变量统一为 `痰湿质`。",
        "",
        "## 基础检查",
        df_to_markdown(validation),
        "",
        "## 形态判断汇总",
        df_to_markdown(shape_df[["变量", "形态判断", "LOWESS偏离比", "箱线图中位数趋势"]]),
        "",
        "## 箱线图分组情况",
        df_to_markdown(boxplot_df),
        "",
        "## Pearson 结果",
        df_to_markdown(pearson_df),
        "",
        "## Spearman 结果",
        df_to_markdown(spearman_df),
        "",
        "## RESET 结果",
        df_to_markdown(reset_df),
        "",
        "## 多重共线性诊断",
        "- 数据源：`Q1_D2_缩尾数据.csv`",
        "- 变量范围：7 个代谢/血常规指标 + `ADL总分` + `IADL总分`",
        "- 主表报告 `VIF` 和 `Tolerance`，阈值规则为 `VIF<5` 视为无明显风险。",
        "",
        df_to_markdown(vif_df),
        "",
        "## 输出文件",
        "- 散点图 + LOWESS：`q1_diagnostics/figures/scatter_lowess/`",
        "- 箱线图：`q1_diagnostics/figures/boxplots/`",
        "- 热力图：`q1_diagnostics/figures/heatmaps/`",
        "- 结果表：`q1_diagnostics/tables/`",
    ]
    (OUTPUT_DIR / "Q1_前置检验结果汇总.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    configure_plotting()
    ensure_output_dirs()

    d1 = load_csv(D1_PATH)
    d2 = load_csv(D2_PATH)
    d2_preprocessed = load_csv(D2_PREPROCESSED_PATH)
    validation = validate_inputs(d1, d2)
    shapes = scatter_and_lowess_plots(d1)
    shape_df = write_shape_summary(shapes)
    boxplot_df = boxplots(d1)
    boxplot_df.to_csv(TABLE_DIR / "boxplot_group_summary.csv", index=False, encoding="utf-8-sig")
    pearson_df = pearson_analysis(d1)
    spearman_df = spearman_analysis(d1)
    reset_df, _ = reset_analysis(d2)
    vif_df = multicollinearity_analysis(d2)
    generate_summary_markdown(validation, shape_df, boxplot_df, pearson_df, spearman_df, reset_df, vif_df)
    run_severity_workflow(
        model_input=d2,
        data_source="Q1_D2_缩尾数据.csv",
        suffix="",
        effect_dir=RCS_DIR,
        summary_path=SEVERITY_SUMMARY_PATH,
        effect_dir_label="q1_diagnostics/figures/rcs_effects/",
        summary_label="原尺度版",
    )
    run_severity_workflow(
        model_input=d2_preprocessed,
        data_source="Q1_D2_预处理结果.csv",
        suffix="_preprocessed",
        effect_dir=RCS_PREPROCESSED_DIR,
        summary_path=SEVERITY_PREPROCESSED_SUMMARY_PATH,
        effect_dir_label="q1_diagnostics/figures/rcs_effects_preprocessed/",
        summary_label="预处理版",
    )

    print("Q1 diagnostics completed.")
    print(f"Output directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
