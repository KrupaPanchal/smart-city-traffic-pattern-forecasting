"""
Initial dataset acquisition and inspection for the Metro Interstate Traffic Volume project.

Run:
    pip install pandas numpy ucimlrepo
    python src/inspect_dataset.py

This script generates reports/dataset_profile.txt. It does not delete or overwrite raw data.
"""
from pathlib import Path
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
REPORTS.mkdir(parents=True, exist_ok=True)
REPORT_PATH = REPORTS / "dataset_profile.txt"

def main() -> int:
    try:
        from ucimlrepo import fetch_ucirepo
    except ImportError:
        print("Missing dependency. Install it with: pip install pandas numpy ucimlrepo")
        return 1

    print("Requesting Metro Interstate Traffic Volume from the UCI repository...")
    try:
        dataset = fetch_ucirepo(id=492)
        features = dataset.data.features.copy()
        target = dataset.data.targets.copy()
        if isinstance(target, pd.Series):
            target = target.to_frame()
        df = pd.concat([features.reset_index(drop=True), target.reset_index(drop=True)], axis=1)
    except Exception as exc:
        print(f"Could not fetch the dataset from UCI: {exc}")
        print("Download the file from https://archive.ics.uci.edu/dataset/492/metro+interstate+traffic+volume")
        print("Then adapt this script to load the local CSV with pd.read_csv(path).")
        return 2

    lines = []
    lines.append("METRO INTERSTATE TRAFFIC VOLUME — INITIAL DATA PROFILE")
    lines.append("=" * 58)
    lines.append(f"Rows: {df.shape[0]}")
    lines.append(f"Columns: {df.shape[1]}")
    lines.append("")
    lines.append("COLUMN NAMES")
    lines.extend(str(c) for c in df.columns)
    lines.append("")
    lines.append("DATA TYPES")
    lines.append(df.dtypes.to_string())
    lines.append("")
    lines.append("FIRST FIVE ROWS")
    lines.append(df.head().to_string(index=False))
    lines.append("")
    lines.append("MISSING VALUES BY COLUMN")
    lines.append(df.isna().sum().to_string())
    lines.append("")
    lines.append("DUPLICATE ROWS")
    lines.append(str(int(df.duplicated().sum())))
    lines.append("")
    lines.append("NUMERIC DESCRIPTIVE STATISTICS")
    lines.append(df.describe(include="all").transpose().to_string())
    lines.append("")

    if "date_time" in df.columns:
        parsed = pd.to_datetime(df["date_time"], errors="coerce")
        lines.append("DATE/TIME CHECK")
        lines.append(f"Values failing datetime parsing: {int(parsed.isna().sum())}")
        lines.append(f"Minimum timestamp: {parsed.min()}")
        lines.append(f"Maximum timestamp: {parsed.max()}")
        lines.append(f"Duplicate timestamps: {int(parsed.duplicated().sum())}")
        lines.append("")

    if "traffic_volume" in df.columns:
        numeric_target = pd.to_numeric(df["traffic_volume"], errors="coerce")
        lines.append("TARGET CHECK: traffic_volume")
        lines.append(f"Non-numeric target values: {int(numeric_target.isna().sum())}")
        lines.append(f"Negative target values: {int((numeric_target < 0).sum())}")
        lines.append("")

    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"Inspection complete. Profile saved to: {REPORT_PATH}")
    print(f"Rows: {df.shape[0]} | Columns: {df.shape[1]}")
    print("Review the profile before deciding whether to drop, impute, or transform any values.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
