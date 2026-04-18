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
from scipy import stats
from statsmodels.stats.diagnostic import linear_reset


ROOT = Path(__file__).resolve().parent
D1_PATH = ROOT / "Q1_D1_异常值标记数据.csv"
D2_PATH = ROOT / "Q1_D2_缩尾数据.csv"
OUTPUT_DIR = ROOT / "q1_diagnostics"
SCATTER_DIR = OUTPUT_DIR / "figures" / "scatter_lowess"
BOXPLOT_DIR = OUTPUT_DIR / "figures" / "boxplots"
HEATMAP_DIR = OUTPUT_DIR / "figures" / "heatmaps"
TABLE_DIR = OUTPUT_DIR / "tables"

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
    for path in [SCATTER_DIR, BOXPLOT_DIR, HEATMAP_DIR, TABLE_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig")


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
) -> None:
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
    validation = validate_inputs(d1, d2)
    shapes = scatter_and_lowess_plots(d1)
    shape_df = write_shape_summary(shapes)
    boxplot_df = boxplots(d1)
    boxplot_df.to_csv(TABLE_DIR / "boxplot_group_summary.csv", index=False, encoding="utf-8-sig")
    pearson_df = pearson_analysis(d1)
    spearman_df = spearman_analysis(d1)
    reset_df, _ = reset_analysis(d2)
    generate_summary_markdown(validation, shape_df, boxplot_df, pearson_df, spearman_df, reset_df)

    print("Q1 diagnostics completed.")
    print(f"Output directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
