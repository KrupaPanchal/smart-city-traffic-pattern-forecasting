"""Generate EDA charts for the UCI Metro Interstate Traffic Volume dataset.

Install: pip install pandas numpy matplotlib seaborn ucimlrepo
Run:     python src/run_eda.py

The script attempts to fetch UCI dataset 492. If fetching fails, download the CSV from
the official UCI page and edit the loading section to use pd.read_csv(local_path).
"""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "figures"
REPORTS = ROOT / "reports"
FIGURES.mkdir(exist_ok=True)
REPORTS.mkdir(exist_ok=True)
sns.set_theme()

def main():
    try:
        from ucimlrepo import fetch_ucirepo
        ds = fetch_ucirepo(id=492)
        X = ds.data.features.copy()
        y = ds.data.targets.copy()
        if isinstance(y, pd.Series):
            y = y.to_frame()
        df = pd.concat([X.reset_index(drop=True), y.reset_index(drop=True)], axis=1)
    except Exception as exc:
        print(f"Unable to fetch dataset automatically: {exc}")
        print("Download from https://archive.ics.uci.edu/dataset/492/metro+interstate+traffic+volume")
        print("Then edit this script to load the downloaded CSV with pd.read_csv(...).")
        return

    if "date_time" in df.columns:
        df["date_time"] = pd.to_datetime(df["date_time"], errors="coerce")
        df["hour"] = df["date_time"].dt.hour
        df["weekday"] = df["date_time"].dt.day_name()
        df["weekday_num"] = df["date_time"].dt.dayofweek
        df["month"] = df["date_time"].dt.month
        df["is_weekend"] = df["weekday_num"] >= 5

    if "traffic_volume" not in df.columns:
        print("Target column 'traffic_volume' not found. Inspect dataset columns first.")
        print(df.columns.tolist())
        return

    # Record a profile before plotting.
    profile = [
        f"Rows: {df.shape[0]}",
        f"Columns: {df.shape[1]}",
        "",
        "Missing values:",
        df.isna().sum().to_string(),
        "",
        f"Duplicate rows: {int(df.duplicated().sum())}",
        "",
        "Numeric summary:",
        df.describe(include="all").transpose().to_string(),
    ]
    (REPORTS / "eda_profile.txt").write_text("\n".join(profile), encoding="utf-8")

    # Target distribution
    plt.figure(figsize=(9, 5))
    sns.histplot(df["traffic_volume"].dropna(), bins=40, kde=True)
    plt.title("Distribution of Hourly Traffic Volume")
    plt.xlabel("Traffic volume")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(FIGURES / "traffic_volume_distribution.png", dpi=160)
    plt.close()

    # Box plot
    plt.figure(figsize=(8, 4))
    sns.boxplot(x=df["traffic_volume"])
    plt.title("Spread of Hourly Traffic Volume")
    plt.xlabel("Traffic volume")
    plt.tight_layout()
    plt.savefig(FIGURES / "traffic_volume_boxplot.png", dpi=160)
    plt.close()

    # By hour
    if "hour" in df:
        by_hour = df.groupby("hour", dropna=True)["traffic_volume"].agg(["mean", "median"]).reset_index()
        plt.figure(figsize=(9, 5))
        sns.lineplot(data=by_hour, x="hour", y="mean", marker="o", label="Mean")
        sns.lineplot(data=by_hour, x="hour", y="median", marker="o", label="Median")
        plt.title("Traffic Volume by Hour of Day")
        plt.xlabel("Hour (0–23)")
        plt.ylabel("Traffic volume")
        plt.xticks(range(0, 24, 2))
        plt.tight_layout()
        plt.savefig(FIGURES / "traffic_by_hour.png", dpi=160)
        plt.close()

    # Weekday × hour heatmap
    if "weekday_num" in df and "hour" in df:
        pivot = df.pivot_table(index="weekday_num", columns="hour", values="traffic_volume", aggfunc="mean")
        plt.figure(figsize=(12, 5))
        sns.heatmap(pivot, cmap="viridis")
        plt.title("Mean Traffic Volume by Weekday and Hour")
        plt.xlabel("Hour of day")
        plt.ylabel("Weekday (0=Monday)")
        plt.tight_layout()
        plt.savefig(FIGURES / "weekday_hour_heatmap.png", dpi=160)
        plt.close()

    # Temperature scatter
    if "temp" in df:
        sample = df[["temp", "traffic_volume"]].dropna()
        if len(sample) > 12000:
            sample = sample.sample(12000, random_state=42)
        plt.figure(figsize=(8, 5))
        sns.scatterplot(data=sample, x="temp", y="traffic_volume", alpha=0.25, s=15)
        plt.title("Temperature and Traffic Volume")
        plt.xlabel("Temperature (source units: Kelvin)")
        plt.ylabel("Traffic volume")
        plt.tight_layout()
        plt.savefig(FIGURES / "temperature_vs_traffic.png", dpi=160)
        plt.close()

    # Weather category boxplot
    if "weather_main" in df:
        plt.figure(figsize=(10, 5))
        order = df.groupby("weather_main")["traffic_volume"].median().sort_values().index
        sns.boxplot(data=df, x="weather_main", y="traffic_volume", order=order)
        plt.title("Traffic Volume by Main Weather Category")
        plt.xlabel("Weather category")
        plt.ylabel("Traffic volume")
        plt.xticks(rotation=35, ha="right")
        plt.tight_layout()
        plt.savefig(FIGURES / "traffic_by_weather.png", dpi=160)
        plt.close()

    print(f"EDA profile saved to {REPORTS / 'eda_profile.txt'}")
    print(f"Charts saved in {FIGURES}")
    print("Review charts and write observations based on the actual results; do not infer causation from correlation.")

if __name__ == "__main__":
    main()
