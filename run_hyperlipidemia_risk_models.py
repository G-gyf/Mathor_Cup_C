from __future__ import annotations

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
from patsy import bs, cr  # noqa: F401
from scipy import stats
from sklearn.metrics import auc, brier_score_loss, roc_auc_score, roc_curve
from statsmodels.stats.multitest import multipletests


ROOT = Path(__file__).resolve().parent
WORKBOOK_PATH = ROOT / "附件1：样例数据.xlsx"
OUTPUT_ROOT = ROOT / "q1_diagnostics" / "高血脂风险_模型A_B"
TABLE_DIR = OUTPUT_ROOT / "tables"
FIG_DIR = OUTPUT_ROOT / "figures"
SCREEN_FIG_DIR = FIG_DIR / "screening_distributions"
MODEL_A_RCS_DIR = FIG_DIR / "model_a_rcs"
MODEL_B_RCS_DIR = FIG_DIR / "model_b_rcs"
SUMMARY_PATH = OUTPUT_ROOT / "高血脂风险_模型A_B_结果汇总.md"

OUTCOME_COL = "高血脂症二分类标签"
CONTINUOUS_VARS = [
    "HDL-C（高密度脂蛋白）",
    "LDL-C（低密度脂蛋白）",
    "TG（甘油三酯）",
    "TC（总胆固醇）",
    "空腹血糖",
    "血尿酸",
    "BMI",
    "ADL总分",
    "IADL总分",
]
CATEGORICAL_VARS = ["年龄组", "性别", "吸烟史", "饮酒史"]
MODEL_A_VARS = CONTINUOUS_VARS + CATEGORICAL_VARS
DIRECT_LIPIDS = [
    "HDL-C（高密度脂蛋白）",
    "LDL-C（低密度脂蛋白）",
    "TG（甘油三酯）",
    "TC（总胆固醇）",
]
MODEL_B_VARS = [var for var in MODEL_A_VARS if var not in DIRECT_LIPIDS]

ALIAS_MAP = {
    OUTCOME_COL: "hyperlipidemia",
    "HDL-C（高密度脂蛋白）": "hdl_c",
    "LDL-C（低密度脂蛋白）": "ldl_c",
    "TG（甘油三酯）": "tg",
    "TC（总胆固醇）": "tc",
    "空腹血糖": "fbg",
    "血尿酸": "uric_acid",
    "BMI": "bmi",
    "ADL总分": "adl_total",
    "IADL总分": "iadl_total",
    "年龄组": "age_group",
    "性别": "sex",
    "吸烟史": "smoking",
    "饮酒史": "drinking",
}

MODEL_A_NAME = "模型A_含直接血脂指标"
MODEL_B_NAME = "模型B_不含直接血脂指标"


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
    for path in [OUTPUT_ROOT, TABLE_DIR, FIG_DIR, SCREEN_FIG_DIR, MODEL_A_RCS_DIR, MODEL_B_RCS_DIR]:
        path.mkdir(parents=True, exist_ok=True)


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


def load_source_data() -> pd.DataFrame:
    columns = [OUTCOME_COL] + MODEL_A_VARS
    df = pd.read_excel(WORKBOOK_PATH, usecols=columns)
    df = df.dropna(subset=[OUTCOME_COL]).copy()
    for column in CATEGORICAL_VARS + [OUTCOME_COL]:
        df[column] = df[column].astype(int)
    return df


def apply_d2_winsorization(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    clipped = df.copy()
    rows = []
    for variable in CONTINUOUS_VARS:
        q1 = float(df[variable].quantile(0.25))
        q3 = float(df[variable].quantile(0.75))
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        before = clipped[variable].copy()
        clipped[variable] = clipped[variable].clip(lower=lower, upper=upper)
        modified = int((before != clipped[variable]).sum())
        rows.append(
            {
                "变量": variable,
                "Q1": q1,
                "Q3": q3,
                "IQR": iqr,
                "下界": lower,
                "上界": upper,
                "缩尾修改数": modified,
            }
        )
    threshold_df = pd.DataFrame(rows)
    return clipped, threshold_df


def format_mean_std(series: pd.Series) -> str:
    return f"{series.mean():.2f} ± {series.std(ddof=1):.2f}"


def format_median_iqr(series: pd.Series) -> str:
    return f"{series.median():.2f} ({series.quantile(0.25):.2f}, {series.quantile(0.75):.2f})"


def cohen_d(group0: pd.Series, group1: pd.Series) -> float:
    var0 = group0.var(ddof=1)
    var1 = group1.var(ddof=1)
    pooled = np.sqrt(((len(group0) - 1) * var0 + (len(group1) - 1) * var1) / (len(group0) + len(group1) - 2))
    if pooled == 0:
        return 0.0
    return float((group1.mean() - group0.mean()) / pooled)


def rank_biserial_from_u(u_stat: float, n0: int, n1: int) -> float:
    return float((2 * u_stat) / (n0 * n1) - 1)


def cramer_v(contingency: pd.DataFrame, chi2_stat: float) -> float:
    n = contingency.to_numpy().sum()
    if n == 0:
        return 0.0
    r, c = contingency.shape
    denom = min(r - 1, c - 1)
    if denom <= 0:
        return 0.0
    return float(np.sqrt(chi2_stat / (n * denom)))


def normality_test(series: pd.Series) -> float:
    if series.nunique() < 3:
        return np.nan
    try:
        return float(stats.shapiro(series).pvalue)
    except Exception:
        return np.nan


def plot_distribution_diagnostics(df: pd.DataFrame, variable: str, output_dir: Path) -> None:
    group0 = df.loc[df[OUTCOME_COL] == 0, variable].dropna()
    group1 = df.loc[df[OUTCOME_COL] == 1, variable].dropna()

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    sns.histplot(group0, kde=True, ax=axes[0, 0], color="#4C78A8")
    axes[0, 0].set_title(f"{variable} - 未确诊组直方图")
    stats.probplot(group0, dist="norm", plot=axes[0, 1])
    axes[0, 1].set_title(f"{variable} - 未确诊组Q-Q图")
    sns.histplot(group1, kde=True, ax=axes[1, 0], color="#E45756")
    axes[1, 0].set_title(f"{variable} - 确诊组直方图")
    stats.probplot(group1, dist="norm", plot=axes[1, 1])
    axes[1, 1].set_title(f"{variable} - 确诊组Q-Q图")
    fig.tight_layout()
    fig.savefig(output_dir / f"{sanitize_filename(variable)}_分布诊断.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def summarize_category_distribution(contingency: pd.DataFrame, group_value: int) -> str:
    total = int(contingency.loc[group_value].sum())
    parts = []
    for level, count in contingency.loc[group_value].items():
        pct = 100 * count / total if total else 0.0
        parts.append(f"{level}: {int(count)}({pct:.1f}%)")
    return "；".join(parts)


def run_univariate_screening(
    df: pd.DataFrame,
    candidate_vars: list[str],
    model_name: str,
    make_plots: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    normality_rows = []
    continuous_candidates = [var for var in candidate_vars if var in CONTINUOUS_VARS]

    for variable in continuous_candidates:
        if make_plots:
            plot_distribution_diagnostics(df, variable, SCREEN_FIG_DIR)

        group0 = df.loc[df[OUTCOME_COL] == 0, variable].dropna()
        group1 = df.loc[df[OUTCOME_COL] == 1, variable].dropna()
        shapiro0 = normality_test(group0)
        shapiro1 = normality_test(group1)
        both_normal = bool(
            pd.notna(shapiro0)
            and pd.notna(shapiro1)
            and shapiro0 > 0.05
            and shapiro1 > 0.05
        )
        levene_p = float(stats.levene(group0, group1).pvalue)

        if both_normal and levene_p > 0.05:
            test_name = "独立样本t检验"
            stat, p_value = stats.ttest_ind(group0, group1, equal_var=True)
            effect_type = "Cohen's d"
            effect_value = cohen_d(group0, group1)
            desc0 = format_mean_std(group0)
            desc1 = format_mean_std(group1)
            desc_style = "均值±标准差"
        else:
            test_name = "Mann-Whitney U检验"
            stat, p_value = stats.mannwhitneyu(group0, group1, alternative="two-sided")
            effect_type = "Rank-biserial r"
            effect_value = rank_biserial_from_u(float(stat), len(group0), len(group1))
            desc0 = format_median_iqr(group0)
            desc1 = format_median_iqr(group1)
            desc_style = "中位数(IQR)"

        rows.append(
            {
                "模型": model_name,
                "变量": variable,
                "变量类型": "连续变量",
                "未确诊组描述": desc0,
                "确诊组描述": desc1,
                "描述方式": desc_style,
                "未确诊组_Shapiro_p": shapiro0,
                "确诊组_Shapiro_p": shapiro1,
                "Levene_p": levene_p,
                "检验方法": test_name,
                "统计量": float(stat),
                "原始p值": float(p_value),
                "效应量类型": effect_type,
                "效应量": float(effect_value),
            }
        )
        normality_rows.append(
            {
                "模型": model_name,
                "变量": variable,
                "未确诊组_Shapiro_p": shapiro0,
                "确诊组_Shapiro_p": shapiro1,
                "是否双侧近似正态": both_normal,
            }
        )

    for variable in [var for var in candidate_vars if var in CATEGORICAL_VARS]:
        contingency = pd.crosstab(df[OUTCOME_COL], df[variable])
        expected = stats.chi2_contingency(contingency, correction=False)[3]

        if contingency.shape == (2, 2) and (expected < 5).any():
            odds_ratio, p_value = stats.fisher_exact(contingency)
            chi2_stat = stats.chi2_contingency(contingency, correction=False)[0]
            test_name = "Fisher精确检验"
            effect_type = "Odds ratio"
            effect_value = float(odds_ratio)
            stat_value = float(odds_ratio)
        else:
            chi2_stat, p_value, _, _ = stats.chi2_contingency(contingency, correction=False)
            test_name = "卡方检验"
            effect_type = "Cramer's V"
            effect_value = cramer_v(contingency, float(chi2_stat))
            stat_value = float(chi2_stat)

        rows.append(
            {
                "模型": model_name,
                "变量": variable,
                "变量类型": "分类变量",
                "未确诊组描述": summarize_category_distribution(contingency, 0),
                "确诊组描述": summarize_category_distribution(contingency, 1),
                "描述方式": "频数(构成比)",
                "未确诊组_Shapiro_p": np.nan,
                "确诊组_Shapiro_p": np.nan,
                "Levene_p": np.nan,
                "检验方法": test_name,
                "统计量": stat_value,
                "原始p值": float(p_value),
                "效应量类型": effect_type,
                "效应量": float(effect_value),
            }
        )

    table = pd.DataFrame(rows)
    reject, q_values, _, _ = multipletests(table["原始p值"], alpha=0.05, method="fdr_bh")
    table["FDR校正后p值"] = q_values
    table["FDR显著"] = np.where(reject, "是", "否")
    normality_df = pd.DataFrame(normality_rows)
    return table, normality_df


def prepare_model_frame(df: pd.DataFrame, candidate_vars: list[str]) -> pd.DataFrame:
    columns = [OUTCOME_COL] + candidate_vars
    model_df = df[columns].rename(columns=ALIAS_MAP).copy()
    model_df["age_group"] = model_df["age_group"].astype(int).astype("category")
    for column in ["sex", "smoking", "drinking", "hyperlipidemia"]:
        if column in model_df.columns:
            model_df[column] = model_df[column].astype(int)
    return model_df


def model_term(variable: str) -> str:
    alias = ALIAS_MAP[variable]
    if variable == "年龄组":
        return f"C({alias})"
    return alias


def build_logit_formula(variables: list[str], spline_specs: dict[str, dict[str, float]] | None = None) -> str:
    spline_specs = spline_specs or {}
    terms = []
    for variable in variables:
        if variable in spline_specs:
            spec = spline_specs[variable]
            alias = ALIAS_MAP[variable]
            terms.append(
                f"cr({alias}, knots=({spec['k35']:.12g}, {spec['k65']:.12g}), "
                f"lower_bound={spec['k05']:.12g}, upper_bound={spec['k95']:.12g})"
            )
        else:
            terms.append(model_term(variable))
    return "hyperlipidemia ~ " + " + ".join(terms) if terms else "hyperlipidemia ~ 1"


def fit_logistic_model(formula: str, model_df: pd.DataFrame):
    return smf.glm(formula=formula, data=model_df, family=sm.families.Binomial()).fit()


def backward_aic_selection(
    df: pd.DataFrame,
    candidate_vars: list[str],
    model_name: str,
) -> tuple[list[str], pd.DataFrame, object, pd.DataFrame]:
    model_df = prepare_model_frame(df, candidate_vars)
    current_vars = candidate_vars.copy()
    current_formula = build_logit_formula(current_vars)
    current_model = fit_logistic_model(current_formula, model_df)

    path_rows = [
        {
            "模型": model_name,
            "步骤": 0,
            "操作": "初始模型",
            "删除变量": "",
            "AIC": float(current_model.aic),
            "剩余变量": "、".join(current_vars),
            "公式": current_formula,
        }
    ]
    step = 0

    while len(current_vars) > 1:
        candidates = []
        for variable in current_vars:
            reduced_vars = [var for var in current_vars if var != variable]
            reduced_formula = build_logit_formula(reduced_vars)
            reduced_model = fit_logistic_model(reduced_formula, model_df)
            candidates.append((variable, reduced_model, float(reduced_model.aic), reduced_formula, reduced_vars))

        best_variable, best_model, best_aic, best_formula, best_vars = min(candidates, key=lambda item: item[2])
        if best_aic < float(current_model.aic):
            step += 1
            current_vars = best_vars
            current_model = best_model
            path_rows.append(
                {
                    "模型": model_name,
                    "步骤": step,
                    "操作": "删除变量",
                    "删除变量": best_variable,
                    "AIC": best_aic,
                    "剩余变量": "、".join(current_vars),
                    "公式": best_formula,
                }
            )
        else:
            break

    path_df = pd.DataFrame(path_rows)
    return current_vars, path_df, current_model, model_df


def logistic_or_table(model, model_name: str) -> pd.DataFrame:
    params = model.params
    conf = model.conf_int()
    rows = []
    for param in params.index:
        if param == "Intercept":
            continue
        coef = float(params[param])
        rows.append(
            {
                "模型": model_name,
                "参数": param,
                "系数": coef,
                "OR": float(np.exp(coef)),
                "下限95CI": float(np.exp(conf.loc[param, 0])),
                "上限95CI": float(np.exp(conf.loc[param, 1])),
                "p值": float(model.pvalues[param]),
            }
        )
    return pd.DataFrame(rows)


def compute_knots(df: pd.DataFrame, variables: Iterable[str]) -> dict[str, dict[str, float]]:
    result = {}
    for variable in variables:
        series = df[variable]
        result[variable] = {
            "k05": float(series.quantile(0.05)),
            "k35": float(series.quantile(0.35)),
            "k65": float(series.quantile(0.65)),
            "k95": float(series.quantile(0.95)),
        }
    return result


def likelihood_ratio_test(full_model, reduced_model) -> tuple[float, float, float]:
    lr_stat = float(2 * (full_model.llf - reduced_model.llf))
    df_diff = float(full_model.df_model - reduced_model.df_model)
    p_value = float(stats.chi2.sf(lr_stat, df_diff))
    return lr_stat, df_diff, p_value


def classify_curve_shape(x_values: np.ndarray, y_values: np.ndarray) -> str:
    if len(x_values) < 5:
        return "近似线性"
    curve_range = float(np.max(y_values) - np.min(y_values))
    if curve_range < 1e-6:
        return "平台"
    diff = np.diff(y_values)
    tol = max(curve_range * 0.01, np.quantile(np.abs(diff), 0.2), 1e-6)
    pos = diff > tol
    neg = diff < -tol
    if not neg.any() or not pos.any():
        straight = np.linspace(y_values[0], y_values[-1], len(y_values))
        deviation = float(np.mean(np.abs(y_values - straight)) / max(curve_range, 1e-8))
        if deviation < 0.05:
            return "近似线性"
        return "单调上升" if np.mean(diff) >= 0 else "单调下降"
    signs = np.sign(diff[np.abs(diff) > tol])
    sign_changes = int(np.sum(signs[1:] != signs[:-1])) if signs.size > 1 else 0
    second_mean = float(np.mean(np.diff(y_values, n=2)))
    if sign_changes <= 1:
        if second_mean > 0:
            return "U型"
        if second_mean < 0:
            return "倒U型"
    return "局部转折"


def run_rcs_workflow(
    df: pd.DataFrame,
    retained_vars: list[str],
    linear_model,
    model_df: pd.DataFrame,
    model_name: str,
    output_dir: Path,
) -> tuple[object, pd.DataFrame, dict[str, dict[str, float]], pd.DataFrame]:
    retained_continuous = [var for var in retained_vars if var in CONTINUOUS_VARS]
    knots = compute_knots(df, retained_continuous)
    test_rows = []
    nonlinear_vars = []

    for variable in retained_continuous:
        reduced_formula = build_logit_formula(retained_vars)
        reduced_model = fit_logistic_model(reduced_formula, model_df)
        spline_formula = build_logit_formula(retained_vars, spline_specs={variable: knots[variable]})
        spline_model = fit_logistic_model(spline_formula, model_df)
        lr_stat, df_diff, p_value = likelihood_ratio_test(spline_model, reduced_model)
        nonlinear = p_value < 0.05
        if nonlinear:
            nonlinear_vars.append(variable)
        test_rows.append(
            {
                "模型": model_name,
                "变量": variable,
                "线性模型公式": reduced_formula,
                "样条模型公式": spline_formula,
                "LR统计量": lr_stat,
                "自由度差": df_diff,
                "p值": p_value,
                "是否显著非线性": "是" if nonlinear else "否",
            }
        )

    final_spline_specs = {var: knots[var] for var in nonlinear_vars}
    final_formula = build_logit_formula(retained_vars, spline_specs=final_spline_specs)
    final_model = fit_logistic_model(final_formula, model_df)
    test_df = pd.DataFrame(test_rows).sort_values(by="p值")

    knots_rows = []
    for variable, knot_spec in knots.items():
        knots_rows.append(
            {
                "模型": model_name,
                "变量": variable,
                "knot_5%": knot_spec["k05"],
                "knot_35%": knot_spec["k35"],
                "knot_65%": knot_spec["k65"],
                "knot_95%": knot_spec["k95"],
                "最终形式": "RCS样条" if variable in nonlinear_vars else "线性",
            }
        )
    knots_df = pd.DataFrame(knots_rows)

    final_profile = {}
    for variable in retained_vars:
        alias = ALIAS_MAP[variable]
        if variable in CONTINUOUS_VARS:
            final_profile[alias] = float(df[variable].median())
        elif variable == "年龄组":
            final_profile[alias] = int(df[variable].mode().iloc[0])
        else:
            final_profile[alias] = int(df[variable].mode().iloc[0])

    for variable in nonlinear_vars:
        alias = ALIAS_MAP[variable]
        grid = np.linspace(float(df[variable].min()), float(df[variable].max()), 200)
        pred_df = pd.DataFrame([final_profile] * len(grid))
        pred_df[alias] = grid
        pred_df["age_group"] = pred_df["age_group"].astype("category")
        prediction = final_model.get_prediction(pred_df).summary_frame()
        shape = classify_curve_shape(grid, prediction["mean"].to_numpy(dtype=float))

        fig, ax = plt.subplots(figsize=(9, 6))
        ax.plot(grid, prediction["mean"], color="#2C7FB8", linewidth=2.4, label="预测概率")
        ax.fill_between(
            grid,
            prediction["mean_ci_lower"],
            prediction["mean_ci_upper"],
            color="#A6CEE3",
            alpha=0.35,
            label="95% CI",
        )
        for label, knot_value in [
            ("5%", knots[variable]["k05"]),
            ("35%", knots[variable]["k35"]),
            ("65%", knots[variable]["k65"]),
            ("95%", knots[variable]["k95"]),
        ]:
            ax.axvline(knot_value, color="#F58518", linestyle="--", linewidth=1)
            ax.text(knot_value, float(np.max(prediction["mean_ci_upper"])), label, rotation=90, va="top", ha="right")
        p_value = float(test_df.loc[test_df["变量"] == variable, "p值"].iloc[0])
        ax.set_xlabel(variable)
        ax.set_ylabel("高血脂确诊预测概率")
        ax.set_title(f"{model_name}：{variable} 的RCS概率曲线")
        ax.text(
            0.02,
            0.98,
            f"非线性检验p={p_value:.4f}\n形态：{shape}",
            transform=ax.transAxes,
            va="top",
            ha="left",
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.9},
        )
        ax.legend(frameon=True)
        fig.tight_layout()
        fig.savefig(output_dir / f"{sanitize_filename(variable)}_RCS概率曲线.png", dpi=220, bbox_inches="tight")
        plt.close(fig)

        test_df.loc[test_df["变量"] == variable, "曲线形态"] = shape

    if test_df.empty:
        note = pd.DataFrame([{"模型": model_name, "说明": "最终线性模型未保留连续变量，未执行RCS检验。"}])
        test_df = note.rename(columns={"说明": "线性模型公式"})

    return final_model, test_df, final_spline_specs, knots_df


def final_model_parameter_table(model, model_name: str) -> pd.DataFrame:
    rows = []
    conf = model.conf_int()
    for param in model.params.index:
        if param == "Intercept":
            continue
        coef = float(model.params[param])
        rows.append(
            {
                "模型": model_name,
                "参数": param,
                "系数": coef,
                "标准误": float(model.bse[param]),
                "z值": float(model.tvalues[param]),
                "p值": float(model.pvalues[param]),
                "下限95CI": float(conf.loc[param, 0]),
                "上限95CI": float(conf.loc[param, 1]),
            }
        )
    return pd.DataFrame(rows)


def compute_midrank(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x)
    sorted_x = x[order]
    n = len(x)
    midranks = np.zeros(n, dtype=float)
    i = 0
    while i < n:
        j = i
        while j < n and sorted_x[j] == sorted_x[i]:
            j += 1
        midranks[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    out = np.empty(n, dtype=float)
    out[order] = midranks
    return out


def fast_delong(predictions_sorted_transposed: np.ndarray, positive_count: int) -> tuple[np.ndarray, np.ndarray]:
    m = positive_count
    n = predictions_sorted_transposed.shape[1] - m
    positives = predictions_sorted_transposed[:, :m]
    negatives = predictions_sorted_transposed[:, m:]
    k = predictions_sorted_transposed.shape[0]
    tx = np.empty((k, m))
    ty = np.empty((k, n))
    tz = np.empty((k, m + n))
    for row in range(k):
        tx[row, :] = compute_midrank(positives[row, :])
        ty[row, :] = compute_midrank(negatives[row, :])
        tz[row, :] = compute_midrank(predictions_sorted_transposed[row, :])
    aucs = tz[:, :m].sum(axis=1) / (m * n) - (m + 1.0) / (2.0 * n)
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    covariance = sx / m + sy / n
    return aucs, covariance


def delong_test(y_true: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray) -> tuple[float, float, float]:
    order = np.argsort(-y_true)
    label_1_count = int(np.sum(y_true))
    preds = np.vstack([pred_a, pred_b])[:, order]
    aucs, covariance = fast_delong(preds, label_1_count)
    contrast = np.array([1.0, -1.0])
    variance = float(contrast @ covariance @ contrast.T)
    variance = max(variance, 1e-12)
    z_value = float(np.abs(np.diff(aucs))[0] / np.sqrt(variance))
    p_value = float(2 * stats.norm.sf(z_value))
    return float(aucs[0]), float(aucs[1]), p_value


def classification_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    youden = tpr - fpr
    best_idx = int(np.argmax(youden))
    threshold = float(thresholds[best_idx])
    y_pred = (y_prob >= threshold).astype(int)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    sensitivity = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    return {
        "AUC": float(roc_auc_score(y_true, y_prob)),
        "Youden最优阈值": threshold,
        "灵敏度": float(sensitivity),
        "特异度": float(specificity),
        "Brier评分": float(brier_score_loss(y_true, y_prob)),
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn,
    }


def fit_model_pipeline(
    df: pd.DataFrame,
    candidate_vars: list[str],
    model_name: str,
    rcs_dir: Path,
) -> dict[str, object]:
    retained_vars, path_df, linear_model, model_df = backward_aic_selection(df, candidate_vars, model_name)
    linear_or_table = logistic_or_table(linear_model, model_name)
    final_model, rcs_test_df, spline_specs, knots_df = run_rcs_workflow(
        df=df,
        retained_vars=retained_vars,
        linear_model=linear_model,
        model_df=model_df,
        model_name=model_name,
        output_dir=rcs_dir,
    )
    final_params = final_model_parameter_table(final_model, model_name)
    prediction = final_model.predict(model_df)
    metrics = classification_metrics(model_df["hyperlipidemia"].to_numpy(dtype=int), prediction.to_numpy(dtype=float))

    return {
        "retained_vars": retained_vars,
        "selection_path": path_df,
        "linear_model": linear_model,
        "linear_or_table": linear_or_table,
        "final_model": final_model,
        "rcs_tests": rcs_test_df,
        "rcs_spline_specs": spline_specs,
        "rcs_knots": knots_df,
        "final_params": final_params,
        "model_df": model_df,
        "predictions": prediction,
        "metrics": metrics,
    }


def build_summary_markdown(
    source_df: pd.DataFrame,
    threshold_df: pd.DataFrame,
    model_a_screen: pd.DataFrame,
    model_b_screen: pd.DataFrame,
    model_a_results: dict[str, object],
    model_b_results: dict[str, object],
    performance_df: pd.DataFrame,
    delong_df: pd.DataFrame,
) -> None:
    model_a_sig = model_a_screen.loc[model_a_screen["FDR显著"] == "是", "变量"].tolist()
    model_b_sig = model_b_screen.loc[model_b_screen["FDR显著"] == "是", "变量"].tolist()
    model_a_nonlinear = model_a_results["rcs_tests"]
    model_b_nonlinear = model_b_results["rcs_tests"]

    model_a_nonlinear_vars = (
        model_a_nonlinear.loc[model_a_nonlinear.get("是否显著非线性", pd.Series(dtype=str)) == "是", "变量"].tolist()
        if "是否显著非线性" in model_a_nonlinear.columns
        else []
    )
    model_b_nonlinear_vars = (
        model_b_nonlinear.loc[model_b_nonlinear.get("是否显著非线性", pd.Series(dtype=str)) == "是", "变量"].tolist()
        if "是否显著非线性" in model_b_nonlinear.columns
        else []
    )

    lines = [
        "# 高血脂风险模型A/B结果汇总",
        "",
        "## 数据口径",
        f"- 样本量：`{len(source_df)}`，结局分布：未确诊 `{int((source_df[OUTCOME_COL] == 0).sum())}`，确诊 `{int((source_df[OUTCOME_COL] == 1).sum())}`",
        "- 连续变量按 `D2` 规则缩尾，分类变量保留原编码。",
        "- 不纳入 `痰湿质`，也不纳入任何体质积分。",
        "- 单变量初筛用于描述性筛查，不作为多变量初始入模门槛。",
        "",
        "## D2缩尾阈值",
        df_to_markdown(threshold_df),
        "",
        "## 单变量筛查结果",
        f"- 模型A FDR显著变量：{'、'.join(model_a_sig) if model_a_sig else '无'}",
        f"- 模型B FDR显著变量：{'、'.join(model_b_sig) if model_b_sig else '无'}",
        "",
        "## 模型A",
        f"- 向后AIC保留变量：{'、'.join(model_a_results['retained_vars'])}",
        f"- 非线性显著变量：{'、'.join(model_a_nonlinear_vars) if model_a_nonlinear_vars else '无'}",
        "",
        df_to_markdown(model_a_results["selection_path"][["步骤", "操作", "删除变量", "AIC", "剩余变量"]]),
        "",
        "## 模型B",
        f"- 向后AIC保留变量：{'、'.join(model_b_results['retained_vars'])}",
        f"- 非线性显著变量：{'、'.join(model_b_nonlinear_vars) if model_b_nonlinear_vars else '无'}",
        "",
        df_to_markdown(model_b_results["selection_path"][["步骤", "操作", "删除变量", "AIC", "剩余变量"]]),
        "",
        "## 模型性能比较",
        df_to_markdown(performance_df),
        "",
        "## DeLong检验",
        df_to_markdown(delong_df),
        "",
        "## 输出文件",
        "- 表格：`q1_diagnostics/高血脂风险_模型A_B/tables/`",
        "- 分布诊断图：`q1_diagnostics/高血脂风险_模型A_B/figures/screening_distributions/`",
        "- 模型A样条图：`q1_diagnostics/高血脂风险_模型A_B/figures/model_a_rcs/`",
        "- 模型B样条图：`q1_diagnostics/高血脂风险_模型A_B/figures/model_b_rcs/`",
        "",
        "## 解释提醒",
        "- 模型A是含定义性血脂信息的主模型，模型B是剥离直接血脂指标后的辅助模型。",
        "- 性能指标为开发集表观性能，AUC差异采用DeLong检验。",
        "- 灵敏度和特异度基于各模型自身的Youden最优阈值。",
    ]
    SUMMARY_PATH.write_text("\n".join(lines), encoding="utf-8")


def save_outputs(
    source_df: pd.DataFrame,
    d2_df: pd.DataFrame,
    threshold_df: pd.DataFrame,
    model_a_screen: pd.DataFrame,
    model_a_normality: pd.DataFrame,
    model_b_screen: pd.DataFrame,
    model_b_normality: pd.DataFrame,
    model_a_results: dict[str, object],
    model_b_results: dict[str, object],
) -> None:
    source_df.to_csv(TABLE_DIR / "risk_model_source_data.csv", index=False, encoding="utf-8-sig")
    d2_df.to_csv(TABLE_DIR / "risk_model_d2_data.csv", index=False, encoding="utf-8-sig")
    threshold_df.to_csv(TABLE_DIR / "risk_model_d2_thresholds.csv", index=False, encoding="utf-8-sig")

    model_a_screen.to_csv(TABLE_DIR / "model_a_univariate_screening.csv", index=False, encoding="utf-8-sig")
    model_a_normality.to_csv(TABLE_DIR / "model_a_normality_results.csv", index=False, encoding="utf-8-sig")
    model_b_screen.to_csv(TABLE_DIR / "model_b_univariate_screening.csv", index=False, encoding="utf-8-sig")
    model_b_normality.to_csv(TABLE_DIR / "model_b_normality_results.csv", index=False, encoding="utf-8-sig")

    model_a_results["selection_path"].to_csv(TABLE_DIR / "model_a_backward_aic_path.csv", index=False, encoding="utf-8-sig")
    model_b_results["selection_path"].to_csv(TABLE_DIR / "model_b_backward_aic_path.csv", index=False, encoding="utf-8-sig")
    model_a_results["linear_or_table"].to_csv(TABLE_DIR / "model_a_linear_or_table.csv", index=False, encoding="utf-8-sig")
    model_b_results["linear_or_table"].to_csv(TABLE_DIR / "model_b_linear_or_table.csv", index=False, encoding="utf-8-sig")
    model_a_results["rcs_tests"].to_csv(TABLE_DIR / "model_a_rcs_tests.csv", index=False, encoding="utf-8-sig")
    model_b_results["rcs_tests"].to_csv(TABLE_DIR / "model_b_rcs_tests.csv", index=False, encoding="utf-8-sig")
    model_a_results["rcs_knots"].to_csv(TABLE_DIR / "model_a_rcs_knots.csv", index=False, encoding="utf-8-sig")
    model_b_results["rcs_knots"].to_csv(TABLE_DIR / "model_b_rcs_knots.csv", index=False, encoding="utf-8-sig")
    model_a_results["final_params"].to_csv(TABLE_DIR / "model_a_final_parameters.csv", index=False, encoding="utf-8-sig")
    model_b_results["final_params"].to_csv(TABLE_DIR / "model_b_final_parameters.csv", index=False, encoding="utf-8-sig")

    preds_a = pd.DataFrame(
        {
            "样本序号": np.arange(1, len(model_a_results["model_df"]) + 1),
            "高血脂症二分类标签": model_a_results["model_df"]["hyperlipidemia"].astype(int).values,
            "模型A预测概率": model_a_results["predictions"].values,
        }
    )
    preds_b = pd.DataFrame(
        {
            "样本序号": np.arange(1, len(model_b_results["model_df"]) + 1),
            "高血脂症二分类标签": model_b_results["model_df"]["hyperlipidemia"].astype(int).values,
            "模型B预测概率": model_b_results["predictions"].values,
        }
    )
    preds_a.to_csv(TABLE_DIR / "model_a_predictions.csv", index=False, encoding="utf-8-sig")
    preds_b.to_csv(TABLE_DIR / "model_b_predictions.csv", index=False, encoding="utf-8-sig")


def plot_roc_comparison(y_true: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray) -> None:
    fpr_a, tpr_a, _ = roc_curve(y_true, pred_a)
    fpr_b, tpr_b, _ = roc_curve(y_true, pred_b)
    auc_a = auc(fpr_a, tpr_a)
    auc_b = auc(fpr_b, tpr_b)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr_a, tpr_a, label=f"{MODEL_A_NAME} (AUC={auc_a:.3f})", linewidth=2.2)
    ax.plot(fpr_b, tpr_b, label=f"{MODEL_B_NAME} (AUC={auc_b:.3f})", linewidth=2.2)
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set_xlabel("1 - 特异度")
    ax.set_ylabel("灵敏度")
    ax.set_title("模型A与模型B ROC曲线比较")
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "model_a_vs_model_b_roc.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    configure_plotting()
    ensure_output_dirs()

    source_df = load_source_data()
    d2_df, threshold_df = apply_d2_winsorization(source_df)

    model_a_screen, model_a_normality = run_univariate_screening(
        d2_df,
        MODEL_A_VARS,
        MODEL_A_NAME,
        make_plots=True,
    )
    model_b_screen, model_b_normality = run_univariate_screening(
        d2_df,
        MODEL_B_VARS,
        MODEL_B_NAME,
        make_plots=False,
    )

    model_a_results = fit_model_pipeline(d2_df, MODEL_A_VARS, MODEL_A_NAME, MODEL_A_RCS_DIR)
    model_b_results = fit_model_pipeline(d2_df, MODEL_B_VARS, MODEL_B_NAME, MODEL_B_RCS_DIR)

    y_true = model_a_results["model_df"]["hyperlipidemia"].to_numpy(dtype=int)
    pred_a = model_a_results["predictions"].to_numpy(dtype=float)
    pred_b = model_b_results["predictions"].to_numpy(dtype=float)

    perf_rows = []
    for model_name, result in [(MODEL_A_NAME, model_a_results), (MODEL_B_NAME, model_b_results)]:
        metrics = result["metrics"]
        perf_rows.append(
            {
                "模型": model_name,
                "AUC": metrics["AUC"],
                "Youden最优阈值": metrics["Youden最优阈值"],
                "灵敏度": metrics["灵敏度"],
                "特异度": metrics["特异度"],
                "Brier评分": metrics["Brier评分"],
            }
        )
    performance_df = pd.DataFrame(perf_rows)
    performance_df.to_csv(TABLE_DIR / "model_performance_comparison.csv", index=False, encoding="utf-8-sig")

    auc_a, auc_b, delong_p = delong_test(y_true, pred_a, pred_b)
    delong_df = pd.DataFrame(
        [
            {
                "模型A_AUC": auc_a,
                "模型B_AUC": auc_b,
                "DeLong检验p值": delong_p,
                "是否存在显著差异": "是" if delong_p < 0.05 else "否",
            }
        ]
    )
    delong_df.to_csv(TABLE_DIR / "model_auc_delong_test.csv", index=False, encoding="utf-8-sig")

    save_outputs(
        source_df=source_df,
        d2_df=d2_df,
        threshold_df=threshold_df,
        model_a_screen=model_a_screen,
        model_a_normality=model_a_normality,
        model_b_screen=model_b_screen,
        model_b_normality=model_b_normality,
        model_a_results=model_a_results,
        model_b_results=model_b_results,
    )
    plot_roc_comparison(y_true, pred_a, pred_b)
    build_summary_markdown(
        source_df=source_df,
        threshold_df=threshold_df,
        model_a_screen=model_a_screen,
        model_b_screen=model_b_screen,
        model_a_results=model_a_results,
        model_b_results=model_b_results,
        performance_df=performance_df,
        delong_df=delong_df,
    )

    print("Hyperlipidemia risk models completed.")
    print(f"Output directory: {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
