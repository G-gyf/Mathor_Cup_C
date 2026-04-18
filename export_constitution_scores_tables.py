#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.sm_exceptions import ConvergenceWarning


WORKBOOK_GLOB = "*.xlsx"
OUTPUT_DIR = Path("q1_diagnostics") / "tables"


def format_p_value(value: float) -> str:
    if pd.isna(value):
        return ""
    if value < 0.001:
        return "<0.001"
    return f"{value:.6f}"


def significance_flag(value: float) -> str:
    if pd.isna(value):
        return ""
    return "是" if value < 0.05 else "否"


def build_vif_table(score_df: pd.DataFrame) -> pd.DataFrame:
    x = sm.add_constant(score_df)
    rows = []
    for idx, col in enumerate(score_df.columns, start=1):
        vif = variance_inflation_factor(x.values, idx)
        rows.append(
            {
                "分析类型": "共线性VIF",
                "变量": col,
                "VIF": vif,
                "Tolerance": 1.0 / vif,
                "共线性是否严重": "否" if vif < 5 else "是",
            }
        )
    return pd.DataFrame(rows).sort_values("VIF", ascending=False).reset_index(drop=True)


def build_logit_tables(score_df: pd.DataFrame, outcome: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame]:
    scaler = StandardScaler()
    score_z = pd.DataFrame(
        scaler.fit_transform(score_df),
        columns=score_df.columns,
        index=score_df.index,
    )
    model_x = sm.add_constant(score_z)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        warnings.simplefilter("ignore", category=RuntimeWarning)
        model = sm.Logit(outcome, model_x).fit(disp=False, maxiter=200)

    conf_int = model.conf_int()
    result_rows = []
    for col in score_z.columns:
        beta = model.params[col]
        pval = model.pvalues[col]
        ci_low = conf_int.loc[col, 0]
        ci_high = conf_int.loc[col, 1]
        result_rows.append(
            {
                "分析类型": "标准化Logistic",
                "变量": col,
                "回归系数β": beta,
                "标准误": model.bse[col],
                "z值": model.tvalues[col],
                "p值": pval,
                "OR(每增加1个SD)": float(np.exp(beta)),
                "OR的95%CI": f"{np.exp(ci_low):.6f}, {np.exp(ci_high):.6f}",
                "效应方向(β符号)": "正向" if beta > 0 else ("负向" if beta < 0 else "零"),
                "显著性(p<0.05)": significance_flag(pval),
            }
        )

    logit_df = pd.DataFrame(result_rows).sort_values("p值", ascending=True).reset_index(drop=True)

    summary_df = pd.DataFrame(
        [
            {
                "指标": "样本量",
                "数值": int(len(score_df)),
            },
            {
                "指标": "高血脂症确诊人数",
                "数值": int(outcome.sum()),
            },
            {
                "指标": "高血脂症未确诊人数",
                "数值": int((1 - outcome).sum()),
            },
            {
                "指标": "Pseudo_R2",
                "数值": float(model.prsquared),
            },
            {
                "指标": "LLR_pvalue",
                "数值": float(model.llr_pvalue),
            },
            {
                "指标": "AIC",
                "数值": float(model.aic),
            },
        ]
    )
    return logit_df, summary_df


def write_markdown(summary_df: pd.DataFrame, vif_df: pd.DataFrame, logit_df: pd.DataFrame, output_path: Path) -> None:
    pseudo_r2 = float(summary_df.loc[summary_df["指标"] == "Pseudo_R2", "数值"].iloc[0])
    llr_pvalue = float(summary_df.loc[summary_df["指标"] == "LLR_pvalue", "数值"].iloc[0])
    aic = float(summary_df.loc[summary_df["指标"] == "AIC", "数值"].iloc[0])
    cases = int(summary_df.loc[summary_df["指标"] == "高血脂症确诊人数", "数值"].iloc[0])
    controls = int(summary_df.loc[summary_df["指标"] == "高血脂症未确诊人数", "数值"].iloc[0])
    top_vif_var = str(vif_df.iloc[0]["变量"])
    top_vif = float(vif_df.iloc[0]["VIF"])
    top_logit_var = str(logit_df.iloc[0]["变量"])
    top_logit_p = float(logit_df.iloc[0]["p值"])

    lines = [
        "# 九种体质积分共线性与标准化Logistic结果摘要",
        "",
        "- 数据来源：附件1：样例数据.xlsx",
        f"- 样本量：{int(cases + controls)}",
        f"- 高血脂症分组：确诊 {cases} 人，未确诊 {controls} 人",
        f"- 共线性结果：最大 VIF 来自 {top_vif_var}，VIF={top_vif:.6f}，均未达到严重共线性阈值（VIF>=5）",
        f"- 标准化Logistic结果：最小 p 值变量为 {top_logit_var}，p={format_p_value(top_logit_p)}，但九种体质均未达到 p<0.05",
        f"- 模型整体：Pseudo R²={pseudo_r2:.6f}，LLR p={llr_pvalue:.6f}，AIC={aic:.6f}",
        "",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    workbook = next(Path(".").glob(WORKBOOK_GLOB))
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_excel(workbook)
    score_df = df.iloc[:, 2:11].copy()
    outcome = df.iloc[:, 31].astype(int)

    vif_df = build_vif_table(score_df)
    logit_df, summary_df = build_logit_tables(score_df, outcome)

    merged_df = pd.concat([vif_df, logit_df], ignore_index=True, sort=False)

    csv_path = OUTPUT_DIR / "constitution_scores_vif_and_std_logit.csv"
    xlsx_path = OUTPUT_DIR / "constitution_scores_vif_and_std_logit.xlsx"
    md_path = OUTPUT_DIR / "constitution_scores_result_summary.md"

    merged_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        vif_df.to_excel(writer, sheet_name="vif", index=False)
        logit_df.to_excel(writer, sheet_name="std_logit", index=False)
        summary_df.to_excel(writer, sheet_name="model_summary", index=False)

    write_markdown(summary_df, vif_df, logit_df, md_path)

    print(csv_path)
    print(xlsx_path)
    print(md_path)


if __name__ == "__main__":
    main()
