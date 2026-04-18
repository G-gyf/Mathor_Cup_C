from __future__ import annotations

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.sm_exceptions import ConvergenceWarning


ROOT = Path(__file__).resolve().parent
WORKBOOK_PATH = ROOT / "附件1：样例数据.xlsx"
OUTPUT_DIR = ROOT / "q1_diagnostics" / "tables"

OUTCOME_COL = "高血脂症二分类标签"
PREDICTOR_COLS = [
    "平和质",
    "气虚质",
    "阳虚质",
    "阴虚质",
    "痰湿质",
    "湿热质",
    "血瘀质",
    "气郁质",
    "特禀质",
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


def fit_logit(y: pd.Series, x: pd.DataFrame) -> tuple[str, object]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        warnings.simplefilter("ignore", category=RuntimeWarning)
        try:
            model = sm.Logit(y, x)
            result = model.fit(disp=False, maxiter=300)
            return "Logit", result
        except Exception:
            glm = sm.GLM(y, x, family=sm.families.Binomial())
            result = glm.fit(maxiter=300)
            return "GLM-Binomial", result


def build_vif_table(x_raw: pd.DataFrame) -> pd.DataFrame:
    x_vif = sm.add_constant(x_raw)
    rows = []
    for idx, col in enumerate(x_raw.columns, start=1):
        vif = float(variance_inflation_factor(x_vif.values, idx))
        rows.append(
            {
                "变量": col,
                "VIF": vif,
                "Tolerance": 1.0 / vif if vif != 0 else np.nan,
                "是否严重共线性(VIF>=5)": "是" if vif >= 5 else "否",
            }
        )
    return pd.DataFrame(rows).sort_values("VIF", ascending=False).reset_index(drop=True)


def build_coef_table(result: object, predictors: list[str]) -> pd.DataFrame:
    conf_int = result.conf_int()
    rows = []
    for var in predictors:
        beta = float(result.params[var])
        se = float(result.bse[var])
        z_value = float(result.tvalues[var])
        p_value = float(result.pvalues[var])
        lower = float(conf_int.loc[var, 0])
        upper = float(conf_int.loc[var, 1])
        rows.append(
            {
                "变量": var,
                "回归系数β": beta,
                "标准误": se,
                "z值": z_value,
                "p值": p_value,
                "OR": float(np.exp(beta)),
                "OR下限95CI": float(np.exp(lower)),
                "OR上限95CI": float(np.exp(upper)),
                "方向": "正向" if beta > 0 else ("负向" if beta < 0 else "零"),
                "显著性(p<0.05)": "是" if p_value < 0.05 else "否",
            }
        )
    return pd.DataFrame(rows).sort_values(["显著性(p<0.05)", "p值"], ascending=[False, True]).reset_index(drop=True)


def build_model_summary(model_type: str, result: object, y: pd.Series, x: pd.DataFrame) -> pd.DataFrame:
    pred_prob = result.predict(x)
    pred_label = (pred_prob >= 0.5).astype(int)
    tp = int(((pred_label == 1) & (y == 1)).sum())
    tn = int(((pred_label == 0) & (y == 0)).sum())
    fp = int(((pred_label == 1) & (y == 0)).sum())
    fn = int(((pred_label == 0) & (y == 1)).sum())
    accuracy = float((pred_label == y).mean())

    rows = [
        {"指标": "模型类型", "数值": model_type},
        {"指标": "样本量", "数值": int(len(y))},
        {"指标": "因变量", "数值": OUTCOME_COL},
        {"指标": "自变量个数", "数值": int(len(PREDICTOR_COLS))},
        {"指标": "截距项是否纳入", "数值": "是"},
        {"指标": "高血脂症确诊人数", "数值": int(y.sum())},
        {"指标": "高血脂症未确诊人数", "数值": int((1 - y).sum())},
        {"指标": "AIC", "数值": float(result.aic)},
        {"指标": "BIC", "数值": float(result.bic) if hasattr(result, "bic") else np.nan},
        {"指标": "对数似然", "数值": float(result.llf)},
        {"指标": "Pseudo_R2", "数值": float(result.prsquared) if hasattr(result, "prsquared") else np.nan},
        {"指标": "LLR_pvalue", "数值": float(result.llr_pvalue) if hasattr(result, "llr_pvalue") else np.nan},
        {"指标": "分类阈值", "数值": 0.5},
        {"指标": "Accuracy", "数值": accuracy},
        {"指标": "TP", "数值": tp},
        {"指标": "TN", "数值": tn},
        {"指标": "FP", "数值": fp},
        {"指标": "FN", "数值": fn},
    ]
    return pd.DataFrame(rows)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_excel(WORKBOOK_PATH, usecols=[OUTCOME_COL] + PREDICTOR_COLS).dropna().copy()
    y = df[OUTCOME_COL].astype(int)
    x_raw = df[PREDICTOR_COLS].astype(float)
    x = sm.add_constant(x_raw, has_constant="add")

    model_type, result = fit_logit(y, x)
    vif_df = build_vif_table(x_raw)
    coef_df = build_coef_table(result, PREDICTOR_COLS)
    summary_df = build_model_summary(model_type, result, y, x)

    coef_path = OUTPUT_DIR / "constitution_metabolic_multivariable_logit_coefficients.csv"
    summary_path = OUTPUT_DIR / "constitution_metabolic_multivariable_logit_summary.csv"
    vif_path = OUTPUT_DIR / "constitution_metabolic_multivariable_logit_vif.csv"
    xlsx_path = OUTPUT_DIR / "constitution_metabolic_multivariable_logit_results.xlsx"

    coef_df.to_csv(coef_path, index=False, encoding="utf-8-sig")
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    vif_df.to_csv(vif_path, index=False, encoding="utf-8-sig")

    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        coef_df.to_excel(writer, sheet_name="coefficients", index=False)
        summary_df.to_excel(writer, sheet_name="model_summary", index=False)
        vif_df.to_excel(writer, sheet_name="vif", index=False)

    print(coef_path)
    print(summary_path)
    print(vif_path)
    print(xlsx_path)


if __name__ == "__main__":
    main()
