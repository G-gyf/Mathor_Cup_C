from __future__ import annotations

import math
from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parent
DATA_PATH = next(ROOT.glob("*.xlsx"))
BASELINE_DIR = ROOT / "q3_diagnostics"
OUTPUT_DIR = ROOT / "q3_diagnostics_improved"
MONTHS = 6
COST_LIMIT = 2000.0
EPS = 1e-9
NEAR_OPTIMAL_RATIO = 1.03
SCORE_BUCKET_DECIMALS = 2
MAX_BUCKET_STATES = 6


@dataclass(frozen=True)
class Patient:
    sample_id: int
    s0: float
    activity_total: float
    age_group: int


@dataclass(frozen=True)
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
class Candidate:
    score: float
    total_cost: float
    total_load: int
    first_level1_month: int | None
    first_level2_or_lower_month: int | None
    monthly_records: list[MonthlyRecord]
    action_sequence: tuple[tuple[int, int], ...]


@dataclass
class Solution:
    final_score: float
    total_cost: float
    total_load: int
    monthly_records: list[MonthlyRecord]
    first_level1_month: int | None
    first_level2_or_lower_month: int | None
    action_sequence: tuple[tuple[int, int], ...]
    s6_min: float


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
    return {1: 30.0, 2: 80.0, 3: 130.0}[treatment_level(score)]


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
    freq_term = 0.05 * math.log(frequency - 4) / math.log(6)
    return (intensity - 1) * 0.03 + freq_term


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


def action_sort_key(action_sequence: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    return action_sequence


def month_for_compare(value: int | None) -> float:
    return math.inf if value is None else float(value)


def candidate_better(
    candidate: Candidate,
    incumbent: Candidate | None,
) -> bool:
    if incumbent is None:
        return True
    if candidate.total_load != incumbent.total_load:
        return candidate.total_load < incumbent.total_load
    if abs(candidate.total_cost - incumbent.total_cost) > EPS:
        return candidate.total_cost < incumbent.total_cost
    if month_for_compare(candidate.first_level1_month) != month_for_compare(incumbent.first_level1_month):
        return month_for_compare(candidate.first_level1_month) < month_for_compare(incumbent.first_level1_month)
    if month_for_compare(candidate.first_level2_or_lower_month) != month_for_compare(incumbent.first_level2_or_lower_month):
        return month_for_compare(candidate.first_level2_or_lower_month) < month_for_compare(incumbent.first_level2_or_lower_month)
    return action_sort_key(candidate.action_sequence) < action_sort_key(incumbent.action_sequence)


def dominates(left: Candidate, right: Candidate) -> bool:
    return (
        left.score <= right.score + EPS
        and left.total_cost <= right.total_cost + EPS
        and left.total_load <= right.total_load
        and month_for_compare(left.first_level1_month) <= month_for_compare(right.first_level1_month)
        and month_for_compare(left.first_level2_or_lower_month) <= month_for_compare(right.first_level2_or_lower_month)
        and (
            left.score < right.score - EPS
            or left.total_cost < right.total_cost - EPS
            or left.total_load < right.total_load
            or month_for_compare(left.first_level1_month) < month_for_compare(right.first_level1_month)
            or month_for_compare(left.first_level2_or_lower_month) < month_for_compare(right.first_level2_or_lower_month)
        )
    )


def candidate_rank_key(candidate: Candidate) -> tuple[Any, ...]:
    return (
        candidate.total_cost,
        candidate.total_load,
        month_for_compare(candidate.first_level1_month),
        month_for_compare(candidate.first_level2_or_lower_month),
        action_sort_key(candidate.action_sequence),
    )


def compress_candidates(candidates: list[Candidate]) -> list[Candidate]:
    buckets: dict[tuple[Any, ...], list[Candidate]] = {}
    ordered = sorted(
        candidates,
        key=lambda c: (
            round(c.score, SCORE_BUCKET_DECIMALS),
            treatment_level(c.score),
            month_for_compare(c.first_level1_month),
            month_for_compare(c.first_level2_or_lower_month),
            candidate_rank_key(c),
        ),
    )
    for candidate in ordered:
        bucket_key = (
            round(candidate.score, SCORE_BUCKET_DECIMALS),
            treatment_level(candidate.score),
            candidate.first_level1_month,
            candidate.first_level2_or_lower_month,
        )
        bucket = buckets.setdefault(bucket_key, [])
        bucket.append(candidate)
        bucket.sort(key=candidate_rank_key)
        if len(bucket) > MAX_BUCKET_STATES:
            del bucket[MAX_BUCKET_STATES:]
    compressed: list[Candidate] = []
    for bucket in buckets.values():
        compressed.extend(bucket)
    return compressed


def advance_one_month(candidate: Candidate, intensity: int, frequency: int, month: int) -> Candidate | None:
    start_score = candidate.score
    level = treatment_level(start_score)
    rate = monthly_decline_rate(intensity, frequency)
    end_score = start_score * (1.0 - rate)
    monthly_tcm = monthly_tcm_cost(start_score)
    monthly_exercise = 4.0 * frequency * exercise_unit_cost(intensity)
    new_total_cost = candidate.total_cost + monthly_tcm + monthly_exercise
    if new_total_cost > COST_LIMIT + EPS:
        return None

    first_l2 = candidate.first_level2_or_lower_month
    if first_l2 is None and treatment_level(end_score) <= 2:
        first_l2 = month

    first_l1 = candidate.first_level1_month
    if first_l1 is None and treatment_level(end_score) == 1:
        first_l1 = month

    record = MonthlyRecord(
        month=month,
        s_start=start_score,
        s_end=end_score,
        treatment_level=level,
        intensity=intensity,
        frequency=frequency,
        monthly_decline_rate=rate,
        monthly_tcm_cost=monthly_tcm,
        monthly_exercise_cost=monthly_exercise,
    )
    return Candidate(
        score=end_score,
        total_cost=new_total_cost,
        total_load=candidate.total_load + intensity * frequency,
        first_level1_month=first_l1,
        first_level2_or_lower_month=first_l2,
        monthly_records=candidate.monthly_records + [record],
        action_sequence=candidate.action_sequence + ((intensity, frequency),),
    )


def prune_candidates(candidates: list[Candidate]) -> list[Candidate]:
    compressed = compress_candidates(candidates)
    ordered = sorted(
        compressed,
        key=lambda c: (
            c.score,
            c.total_cost,
            c.total_load,
            month_for_compare(c.first_level1_month),
            month_for_compare(c.first_level2_or_lower_month),
            action_sort_key(c.action_sequence),
        ),
    )
    frontier: list[Candidate] = []
    for cand in ordered:
        if any(dominates(existing, cand) for existing in frontier):
            continue
        frontier = [existing for existing in frontier if not dominates(cand, existing)]
        frontier.append(cand)
    return frontier


def solve_patient(patient: Patient) -> Solution:
    allowed_intensities = feasible_intensities(patient.age_group, patient.activity_total)
    initial = Candidate(
        score=patient.s0,
        total_cost=0.0,
        total_load=0,
        first_level1_month=None,
        first_level2_or_lower_month=None,
        monthly_records=[],
        action_sequence=(),
    )
    frontier = [initial]
    for month in range(1, MONTHS + 1):
        expanded: list[Candidate] = []
        for candidate in frontier:
            for intensity in allowed_intensities:
                for frequency in range(1, 11):
                    next_candidate = advance_one_month(candidate, intensity, frequency, month)
                    if next_candidate is not None:
                        expanded.append(next_candidate)
        frontier = prune_candidates(expanded)
        if not frontier:
            raise RuntimeError(f"未找到可行方案: sample_id={patient.sample_id}, month={month}")

    s6_min = min(candidate.score for candidate in frontier)
    near_optimal = [candidate for candidate in frontier if candidate.score <= s6_min * NEAR_OPTIMAL_RATIO + EPS]

    best_candidate: Candidate | None = None
    for candidate in near_optimal:
        if candidate_better(candidate, best_candidate):
            best_candidate = candidate
    if best_candidate is None:
        raise RuntimeError(f"未找到近优方案: sample_id={patient.sample_id}")

    return Solution(
        final_score=best_candidate.score,
        total_cost=best_candidate.total_cost,
        total_load=best_candidate.total_load,
        monthly_records=best_candidate.monthly_records,
        first_level1_month=best_candidate.first_level1_month,
        first_level2_or_lower_month=best_candidate.first_level2_or_lower_month,
        action_sequence=best_candidate.action_sequence,
        s6_min=s6_min,
    )


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
    cumulative_load = 0
    for record in solution.monthly_records:
        month_cost = record.monthly_tcm_cost + record.monthly_exercise_cost
        cumulative_cost += month_cost
        cumulative_load += record.intensity * record.frequency
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
                "cumulative_load": cumulative_load,
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
        "S6_min": solution.s6_min,
        "near_optimal_upper_bound": solution.s6_min * NEAR_OPTIMAL_RATIO,
        "total_cost": solution.total_cost,
        "total_load": solution.total_load,
        "first_level2_or_lower_month": solution.first_level2_or_lower_month,
        "first_level1_month": solution.first_level1_month,
        "n_stages": len(stages),
        "action_sequence": " | ".join(f"I={i},f={f}" for i, f in solution.action_sequence),
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
                    "mean_S6_min": g["S6_min"].mean(),
                    "mean_total_cost": g["total_cost"].mean(),
                    "mean_total_load": g["total_load"].mean(),
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
    lines = ["# 样本1/2/3改进版最优方案汇总", ""]
    for _, row in sample_rows.iterrows():
        lines.extend(
            [
                f"## 样本 {int(row['sample_id'])}",
                f"- 初始痰湿积分 `S0={row['S0']:.2f}`，活动总分 `P={row['P']:.2f}`，年龄组 `A={int(row['A'])}`",
                f"- 初始调理等级 `C0={int(row['initial_treatment_level'])}`",
                f"- 改进版初始动作：`I={int(row['initial_intensity'])}, f={int(row['initial_frequency'])}`",
                f"- 6个月末痰湿积分：`{row['S6']:.4f}`，纯疗效最优基准 `S6_min={row['S6_min']:.4f}`",
                f"- 总运动负荷：`{int(row['total_load'])}`，总成本：`{row['total_cost']:.2f}` 元",
                f"- 首次降至2级或以下月份：`{row['first_level2_or_lower_month']}`",
                f"- 首次降至1级月份：`{row['first_level1_month']}`",
                f"- 月度动作序列：`{row['action_sequence']}`",
                "",
            ]
        )
    return "\n".join(lines)


def overall_summary_md(patient_df: pd.DataFrame, rules_df: pd.DataFrame) -> str:
    lines = [
        "# 第三问改进版结果汇总",
        "",
        f"- 痰湿质患者总数：`{len(patient_df)}`",
        f"- 平均6个月末痰湿积分：`{patient_df['S6'].mean():.4f}`",
        f"- 平均纯疗效最优基准：`{patient_df['S6_min'].mean():.4f}`",
        f"- 平均总运动负荷：`{patient_df['total_load'].mean():.2f}`",
        f"- 平均总成本：`{patient_df['total_cost'].mean():.2f}` 元",
        f"- 达到1级调理的患者比例：`{patient_df['first_level1_month'].notna().mean():.4f}`",
        "",
        "## 分层匹配规律预览",
        "```csv",
        rules_df.to_csv(index=False).strip(),
        "```",
    ]
    return "\n".join(lines)


def build_comparison_table(improved_df: pd.DataFrame, baseline_df: pd.DataFrame) -> pd.DataFrame:
    baseline_df = baseline_df.copy()

    def baseline_total_load(row: pd.Series) -> int:
        total = 0
        for idx in range(1, 7):
            months = row.get(f"stage{idx}_months", "")
            intensity = row.get(f"stage{idx}_intensity", "")
            frequency = row.get(f"stage{idx}_frequency", "")
            if pd.isna(months) or months == "":
                continue
            start_month, end_month = str(months).split("-")
            duration = int(end_month) - int(start_month) + 1
            total += duration * int(float(intensity)) * int(float(frequency))
        return total

    baseline_df["baseline_total_load"] = baseline_df.apply(baseline_total_load, axis=1)

    baseline = baseline_df[
        [
            "sample_id",
            "initial_intensity",
            "initial_frequency",
            "S6",
            "total_cost",
            "first_level1_month",
            "baseline_total_load",
        ]
    ].copy()
    baseline["baseline_initial_action"] = baseline.apply(
        lambda row: f"I={int(row['initial_intensity'])}, f={int(row['initial_frequency'])}",
        axis=1,
    )
    baseline = baseline.rename(
        columns={
            "S6": "baseline_S6",
            "total_cost": "baseline_total_cost",
            "first_level1_month": "baseline_first_level1_month",
        }
    )

    improved = improved_df[
        [
            "sample_id",
            "initial_intensity",
            "initial_frequency",
            "S6",
            "total_cost",
            "total_load",
            "first_level1_month",
        ]
    ].copy()
    improved["improved_initial_action"] = improved.apply(
        lambda row: f"I={int(row['initial_intensity'])}, f={int(row['initial_frequency'])}",
        axis=1,
    )
    improved = improved.rename(
        columns={
            "S6": "improved_S6",
            "total_cost": "improved_total_cost",
            "total_load": "improved_total_load",
            "first_level1_month": "improved_first_level1_month",
        }
    )

    merged = baseline.merge(improved, on="sample_id", how="inner")
    merged["delta_S6"] = merged["improved_S6"] - merged["baseline_S6"]
    merged["delta_total_cost"] = merged["improved_total_cost"] - merged["baseline_total_cost"]
    merged["delta_total_load"] = merged["improved_total_load"] - merged["baseline_total_load"]
    return merged[
        [
            "sample_id",
            "baseline_initial_action",
            "improved_initial_action",
            "baseline_S6",
            "improved_S6",
            "delta_S6",
            "baseline_total_cost",
            "improved_total_cost",
            "delta_total_cost",
            "baseline_total_load",
            "improved_total_load",
            "delta_total_load",
            "baseline_first_level1_month",
            "improved_first_level1_month",
        ]
    ].sort_values("sample_id")


def overall_comparison_md(comparison_df: pd.DataFrame, sample_df: pd.DataFrame) -> str:
    lines = [
        "# 第三问改进版与基准版对照",
        "",
        f"- 样本总数：`{len(comparison_df)}`",
        f"- 平均 S6 变化：`{comparison_df['delta_S6'].mean():.4f}`",
        f"- 平均总成本变化：`{comparison_df['delta_total_cost'].mean():.2f}` 元",
        f"- 平均总负荷变化：`{comparison_df['delta_total_load'].mean():.2f}`",
        "",
        "## 样本1/2/3对照",
        "```csv",
        sample_df.to_csv(index=False).strip(),
        "```",
    ]
    return "\n".join(lines)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    monthly_dir = OUTPUT_DIR / "monthly_plans"
    monthly_dir.mkdir(parents=True, exist_ok=True)
    stages_dir = OUTPUT_DIR / "stage_plans"
    stages_dir.mkdir(parents=True, exist_ok=True)

    patients = load_patients()
    patient_rows: list[dict[str, Any]] = []

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
            stages_df = pd.DataFrame(
                columns=[
                    "sample_id",
                    "stage_index",
                    "start_month",
                    "end_month",
                    "treatment_level",
                    "intensity",
                    "frequency",
                    "stage_start_score",
                    "stage_end_score",
                    "stage_cost",
                ]
            )
        stages_df.to_csv(stages_dir / f"sample_{patient.sample_id}_stage_plan.csv", index=False, encoding="utf-8-sig")

    patient_df = pd.DataFrame(patient_rows).sort_values("sample_id").reset_index(drop=True)
    patient_df.to_csv(OUTPUT_DIR / "patient_optimal_plans.csv", index=False, encoding="utf-8-sig")

    rules_df = build_matching_rules(patient_df)
    rules_df.to_csv(OUTPUT_DIR / "strata_matching_rules.csv", index=False, encoding="utf-8-sig")

    sample_rows = patient_df[patient_df["sample_id"].isin([1, 2, 3])].copy().sort_values("sample_id")
    sample_rows.to_csv(OUTPUT_DIR / "sample_1_2_3_optimal_plans.csv", index=False, encoding="utf-8-sig")
    (OUTPUT_DIR / "sample_1_2_3_summary.md").write_text(sample_summary_md(sample_rows), encoding="utf-8")
    (OUTPUT_DIR / "q3_result_summary.md").write_text(overall_summary_md(patient_df, rules_df), encoding="utf-8")

    baseline_df = pd.read_csv(BASELINE_DIR / "patient_optimal_plans.csv")
    comparison_df = build_comparison_table(patient_df, baseline_df)
    comparison_df.to_csv(OUTPUT_DIR / "sample_1_2_3_vs_baseline.csv", index=False, encoding="utf-8-sig")
    sample_comparison = comparison_df[comparison_df["sample_id"].isin([1, 2, 3])].copy()
    (OUTPUT_DIR / "overall_vs_baseline_summary.md").write_text(
        overall_comparison_md(comparison_df, sample_comparison),
        encoding="utf-8",
    )

    print("[INFO] Q3 improved optimization completed.")


if __name__ == "__main__":
    main()
