from __future__ import annotations

import math
from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "附件1：样例数据.xlsx"
OUTPUT_DIR = ROOT / "q3_diagnostics"
MONTHS = 6
COST_LIMIT = 2000.0


@dataclass(frozen=True)
class Patient:
    sample_id: int
    s0: float
    activity_total: float
    age_group: int


@dataclass
class MonthlyRecord:
    month: int
    s_start: float
    s_end: float
    treatment_level: int
    intensity: int
    frequency: int
    monthly_decline_rate: float
    monthly_tcm_cost: float
    monthly_exercise_cost: float


@dataclass
class Solution:
    final_score: float
    total_cost: float
    monthly_records: list[MonthlyRecord]
    first_level1_month: int | None
    first_level2_or_lower_month: int | None


def load_patients() -> list[Patient]:
    df = pd.read_excel(DATA_PATH)
    subgroup = df[df["体质标签"] == 5].copy()
    patients = []
    for _, row in subgroup.iterrows():
        patients.append(
            Patient(
                sample_id=int(row["样本ID"]),
                s0=float(row["痰湿质"]),
                activity_total=float(row["活动量表总分（ADL总分+IADL总分）"]),
                age_group=int(row["年龄组"]),
            )
        )
    return patients


def round_half_up_score(score: float) -> int:
    return int(Decimal(str(score)).quantize(Decimal("1"), rounding=ROUND_HALF_UP))


def treatment_level(score: float) -> int:
    rounded_score = round_half_up_score(score)
    if rounded_score >= 62:
        return 3
    if rounded_score >= 59:
        return 2
    return 1


def monthly_tcm_cost(score: float) -> float:
    level = treatment_level(score)
    return {1: 30.0, 2: 80.0, 3: 130.0}[level]


def exercise_unit_cost(intensity: int) -> float:
    return {1: 3.0, 2: 5.0, 3: 8.0}[intensity]


def feasible_intensities(age_group: int, activity_total: float) -> list[int]:
    if age_group in (1, 2):
        age_allowed = {1, 2, 3}
    elif age_group in (3, 4):
        age_allowed = {1, 2}
    else:
        age_allowed = {1}

    if activity_total < 40:
        activity_allowed = {1}
    elif activity_total < 60:
        activity_allowed = {1, 2}
    else:
        activity_allowed = {1, 2, 3}

    return sorted(age_allowed & activity_allowed)


def monthly_decline_rate(intensity: int, frequency: int) -> float:
    if frequency < 5:
        return 0.0
    return (intensity - 1) * 0.03 + (frequency - 5) * 0.01


def score_bin(score: float) -> str:
    if score >= 62:
        return "高(>=62)"
    if score >= 59:
        return "中(59-61)"
    return "低(<=58)"


def activity_bin(activity_total: float) -> str:
    if activity_total < 40:
        return "低(<40)"
    if activity_total < 60:
        return "中(40-59)"
    return "高(>=60)"


def age_bin(age_group: int) -> str:
    if age_group in (1, 2):
        return "年轻(1-2)"
    if age_group in (3, 4):
        return "老年(3-4)"
    return "高龄(5)"


def compare_solutions(candidate: Solution, incumbent: Solution | None) -> bool:
    if incumbent is None:
        return True
    eps = 1e-9
    if candidate.final_score < incumbent.final_score - eps:
        return True
    if candidate.final_score > incumbent.final_score + eps:
        return False
    if candidate.total_cost < incumbent.total_cost - eps:
        return True
    if candidate.total_cost > incumbent.total_cost + eps:
        return False

    cand_l1 = candidate.first_level1_month if candidate.first_level1_month is not None else math.inf
    inc_l1 = incumbent.first_level1_month if incumbent.first_level1_month is not None else math.inf
    if cand_l1 < inc_l1:
        return True
    if cand_l1 > inc_l1:
        return False

    cand_l2 = candidate.first_level2_or_lower_month if candidate.first_level2_or_lower_month is not None else math.inf
    inc_l2 = incumbent.first_level2_or_lower_month if incumbent.first_level2_or_lower_month is not None else math.inf
    if cand_l2 < inc_l2:
        return True
    if cand_l2 > inc_l2:
        return False

    cand_load = sum(r.intensity * r.frequency for r in candidate.monthly_records)
    inc_load = sum(r.intensity * r.frequency for r in incumbent.monthly_records)
    return cand_load < inc_load


def simulate_stage(
    start_month: int,
    start_score: float,
    intensity: int,
    frequency: int,
) -> tuple[int, float, list[MonthlyRecord]]:
    current_score = start_score
    current_level = treatment_level(current_score)
    records: list[MonthlyRecord] = []

    month = start_month
    while month <= MONTHS:
        rate = monthly_decline_rate(intensity, frequency)
        next_score = current_score * (1.0 - rate)
        record = MonthlyRecord(
            month=month,
            s_start=current_score,
            s_end=next_score,
            treatment_level=current_level,
            intensity=intensity,
            frequency=frequency,
            monthly_decline_rate=rate,
            monthly_tcm_cost=monthly_tcm_cost(current_score),
            monthly_exercise_cost=4.0 * frequency * exercise_unit_cost(intensity),
        )
        records.append(record)
        next_level = treatment_level(next_score)
        if month == MONTHS or next_level != current_level:
            return month + 1, next_score, records
        current_score = next_score
        month += 1

    return MONTHS + 1, current_score, records


def solve_patient(patient: Patient) -> Solution:
    allowed_intensities = feasible_intensities(patient.age_group, patient.activity_total)
    best_solution: Solution | None = None

    def dfs(
        month: int,
        score: float,
        total_cost: float,
        records: list[MonthlyRecord],
        first_l1: int | None,
        first_l2: int | None,
    ) -> None:
        nonlocal best_solution

        if month > MONTHS:
            candidate = Solution(
                final_score=score,
                total_cost=total_cost,
                monthly_records=records.copy(),
                first_level1_month=first_l1,
                first_level2_or_lower_month=first_l2,
            )
            if compare_solutions(candidate, best_solution):
                best_solution = candidate
            return

        for intensity in allowed_intensities:
            for frequency in range(1, 11):
                next_month, next_score, stage_records = simulate_stage(month, score, intensity, frequency)
                stage_cost = sum(item.monthly_tcm_cost + item.monthly_exercise_cost for item in stage_records)
                new_total_cost = total_cost + stage_cost
                if new_total_cost > COST_LIMIT + 1e-9:
                    continue

                new_records = records + stage_records
                new_first_l1 = first_l1
                new_first_l2 = first_l2
                for item in stage_records:
                    if new_first_l2 is None and treatment_level(item.s_end) <= 2:
                        new_first_l2 = item.month
                    if new_first_l1 is None and treatment_level(item.s_end) == 1:
                        new_first_l1 = item.month

                dfs(
                    month=next_month,
                    score=next_score,
                    total_cost=new_total_cost,
                    records=new_records,
                    first_l1=new_first_l1,
                    first_l2=new_first_l2,
                )

    dfs(
        month=1,
        score=patient.s0,
        total_cost=0.0,
        records=[],
        first_l1=None,
        first_l2=None,
    )
    if best_solution is None:
        raise RuntimeError(f"未找到可行方案: sample_id={patient.sample_id}")
    return best_solution


def stage_summary(monthly_records: list[MonthlyRecord]) -> list[dict[str, Any]]:
    if not monthly_records:
        return []
    stages: list[dict[str, Any]] = []
    current = {
        "stage_index": 1,
        "start_month": monthly_records[0].month,
        "end_month": monthly_records[0].month,
        "treatment_level": monthly_records[0].treatment_level,
        "intensity": monthly_records[0].intensity,
        "frequency": monthly_records[0].frequency,
        "stage_start_score": monthly_records[0].s_start,
        "stage_end_score": monthly_records[0].s_end,
        "stage_cost": monthly_records[0].monthly_tcm_cost + monthly_records[0].monthly_exercise_cost,
    }
    for record in monthly_records[1:]:
        same_action = (
            record.treatment_level == current["treatment_level"]
            and record.intensity == current["intensity"]
            and record.frequency == current["frequency"]
        )
        if same_action:
            current["end_month"] = record.month
            current["stage_end_score"] = record.s_end
            current["stage_cost"] += record.monthly_tcm_cost + record.monthly_exercise_cost
        else:
            stages.append(current)
            current = {
                "stage_index": current["stage_index"] + 1,
                "start_month": record.month,
                "end_month": record.month,
                "treatment_level": record.treatment_level,
                "intensity": record.intensity,
                "frequency": record.frequency,
                "stage_start_score": record.s_start,
                "stage_end_score": record.s_end,
                "stage_cost": record.monthly_tcm_cost + record.monthly_exercise_cost,
            }
    stages.append(current)
    return stages


def monthly_records_to_frame(solution: Solution) -> pd.DataFrame:
    rows = []
    cumulative_cost = 0.0
    for record in solution.monthly_records:
        month_cost = record.monthly_tcm_cost + record.monthly_exercise_cost
        cumulative_cost += month_cost
        rows.append(
            {
                "month": record.month,
                "S_start": record.s_start,
                "S_end": record.s_end,
                "treatment_level": record.treatment_level,
                "intensity": record.intensity,
                "frequency": record.frequency,
                "monthly_decline_rate": record.monthly_decline_rate,
                "monthly_tcm_cost": record.monthly_tcm_cost,
                "monthly_exercise_cost": record.monthly_exercise_cost,
                "monthly_total_cost": month_cost,
                "cumulative_cost": cumulative_cost,
            }
        )
    return pd.DataFrame(rows)


def stages_to_columns(stages: list[dict[str, Any]], max_stages: int = 6) -> dict[str, Any]:
    row: dict[str, Any] = {}
    for idx in range(1, max_stages + 1):
        if idx <= len(stages):
            stage = stages[idx - 1]
            row[f"stage{idx}_months"] = f"{stage['start_month']}-{stage['end_month']}"
            row[f"stage{idx}_treatment_level"] = stage["treatment_level"]
            row[f"stage{idx}_intensity"] = stage["intensity"]
            row[f"stage{idx}_frequency"] = stage["frequency"]
            row[f"stage{idx}_start_score"] = stage["stage_start_score"]
            row[f"stage{idx}_end_score"] = stage["stage_end_score"]
            row[f"stage{idx}_cost"] = stage["stage_cost"]
        else:
            row[f"stage{idx}_months"] = ""
            row[f"stage{idx}_treatment_level"] = ""
            row[f"stage{idx}_intensity"] = ""
            row[f"stage{idx}_frequency"] = ""
            row[f"stage{idx}_start_score"] = ""
            row[f"stage{idx}_end_score"] = ""
            row[f"stage{idx}_cost"] = ""
    return row


def build_patient_result_row(patient: Patient, solution: Solution) -> dict[str, Any]:
    stages = stage_summary(solution.monthly_records)
    first_record = solution.monthly_records[0]
    row = {
        "sample_id": patient.sample_id,
        "S0": patient.s0,
        "P": patient.activity_total,
        "A": patient.age_group,
        "score_bin": score_bin(patient.s0),
        "activity_bin": activity_bin(patient.activity_total),
        "age_bin": age_bin(patient.age_group),
        "feasible_intensities": ",".join(str(x) for x in feasible_intensities(patient.age_group, patient.activity_total)),
        "initial_treatment_level": treatment_level(patient.s0),
        "initial_intensity": first_record.intensity,
        "initial_frequency": first_record.frequency,
        "S6": solution.final_score,
        "total_cost": solution.total_cost,
        "first_level2_or_lower_month": solution.first_level2_or_lower_month,
        "first_level1_month": solution.first_level1_month,
        "n_stages": len(stages),
    }
    row.update(stages_to_columns(stages))
    return row


def group_mode(series: pd.Series) -> Any:
    if series.empty:
        return ""
    modes = series.mode(dropna=True)
    if modes.empty:
        return ""
    return modes.iloc[0]


def build_matching_rules(patient_df: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        patient_df.groupby(["score_bin", "activity_bin", "age_bin"], dropna=False)
        .apply(
            lambda g: pd.Series(
                {
                    "n": len(g),
                    "mean_S6": g["S6"].mean(),
                    "mean_total_cost": g["total_cost"].mean(),
                    "common_initial_intensity": group_mode(g["initial_intensity"]),
                    "common_initial_frequency": group_mode(g["initial_frequency"]),
                    "common_initial_action": f"I={group_mode(g['initial_intensity'])}, f={group_mode(g['initial_frequency'])}",
                    "common_initial_treatment_level": group_mode(g["initial_treatment_level"]),
                    "common_stage_count": group_mode(g["n_stages"]),
                    "mean_first_level2_or_lower_month": g["first_level2_or_lower_month"].dropna().mean() if g["first_level2_or_lower_month"].notna().any() else None,
                    "mean_first_level1_month": g["first_level1_month"].dropna().mean() if g["first_level1_month"].notna().any() else None,
                    "low_support_flag": len(g) < 5,
                }
            )
        )
        .reset_index()
        .sort_values(["score_bin", "activity_bin", "age_bin"])
        .reset_index(drop=True)
    )
    return grouped


def sample_summary_md(sample_rows: pd.DataFrame) -> str:
    lines = ["# 样本1/2/3最优方案汇总", ""]
    for _, row in sample_rows.iterrows():
        lines.extend(
            [
                f"## 样本 {int(row['sample_id'])}",
                f"- 初始痰湿积分 `S0={row['S0']:.2f}`，活动总分 `P={row['P']:.2f}`，年龄组 `A={int(row['A'])}`",
                f"- 初始调理等级 `C0={int(row['initial_treatment_level'])}`",
                f"- 最优初始动作：`I={int(row['initial_intensity'])}, f={int(row['initial_frequency'])}`",
                f"- 6个月末痰湿积分：`{row['S6']:.4f}`",
                f"- 总成本：`{row['total_cost']:.2f}` 元",
                f"- 首次降至2级或以下月份：`{row['first_level2_or_lower_month']}`",
                f"- 首次降至1级月份：`{row['first_level1_month']}`",
                "",
            ]
        )
    return "\n".join(lines)


def overall_summary_md(patient_df: pd.DataFrame, rules_df: pd.DataFrame) -> str:
    lines = [
        "# 第三问结果汇总",
        "",
        f"- 痰湿质患者总数：`{len(patient_df)}`",
        f"- 平均6个月末痰湿积分：`{patient_df['S6'].mean():.4f}`",
        f"- 平均总成本：`{patient_df['total_cost'].mean():.2f}` 元",
        f"- 达到1级调理的患者比例：`{patient_df['first_level1_month'].notna().mean():.4f}`",
        "",
        "## 分层匹配规律预览",
        "```csv",
        rules_df.to_csv(index=False).strip(),
        "```",
    ]
    return "\n".join(lines)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    patients = load_patients()

    patient_rows: list[dict[str, Any]] = []
    monthly_dir = OUTPUT_DIR / "monthly_plans"
    monthly_dir.mkdir(parents=True, exist_ok=True)
    stages_dir = OUTPUT_DIR / "stage_plans"
    stages_dir.mkdir(parents=True, exist_ok=True)

    for patient in patients:
        solution = solve_patient(patient)
        patient_rows.append(build_patient_result_row(patient, solution))

        monthly_df = monthly_records_to_frame(solution)
        monthly_df.insert(0, "sample_id", patient.sample_id)
        monthly_df.to_csv(monthly_dir / f"sample_{patient.sample_id}_monthly_plan.csv", index=False, encoding="utf-8-sig")

        stages_df = pd.DataFrame(stage_summary(solution.monthly_records))
        if not stages_df.empty:
            stages_df.insert(0, "sample_id", patient.sample_id)
        else:
            stages_df = pd.DataFrame(columns=["sample_id", "stage_index", "start_month", "end_month", "treatment_level", "intensity", "frequency", "stage_start_score", "stage_end_score", "stage_cost"])
        stages_df.to_csv(stages_dir / f"sample_{patient.sample_id}_stage_plan.csv", index=False, encoding="utf-8-sig")

    patient_df = pd.DataFrame(patient_rows).sort_values("sample_id").reset_index(drop=True)
    patient_df.to_csv(OUTPUT_DIR / "patient_optimal_plans.csv", index=False, encoding="utf-8-sig")

    rules_df = build_matching_rules(patient_df)
    rules_df.to_csv(OUTPUT_DIR / "strata_matching_rules.csv", index=False, encoding="utf-8-sig")

    sample_rows = patient_df[patient_df["sample_id"].isin([1, 2, 3])].copy().sort_values("sample_id")
    sample_rows.to_csv(OUTPUT_DIR / "sample_1_2_3_optimal_plans.csv", index=False, encoding="utf-8-sig")
    (OUTPUT_DIR / "sample_1_2_3_summary.md").write_text(sample_summary_md(sample_rows), encoding="utf-8")
    (OUTPUT_DIR / "q3_result_summary.md").write_text(overall_summary_md(patient_df, rules_df), encoding="utf-8")

    print("[INFO] Q3 optimization completed.")


if __name__ == "__main__":
    main()
