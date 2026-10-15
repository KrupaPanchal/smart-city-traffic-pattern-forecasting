"""
Week 5: Model Evaluation, Reporting, and Optimization Recommendations
Dataset: UCI Metro Interstate Traffic Volume (ID 492)

Install:
    pip install -r requirements.txt

Run from the project root:
    python src/evaluate_models.py

Outputs:
    outputs/validation_metrics.csv
    outputs/test_metrics.csv
    outputs/test_predictions.csv
    outputs/optimization_recommendations.txt
    outputs/actual_vs_predicted.png
    outputs/residual_distribution.png
    outputs/residuals_over_time.png
    outputs/mae_by_hour.png (if supported)

Do not report metrics until this script completes successfully and the outputs are reviewed.
"""
from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeRegressor

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
OUT.mkdir(parents=True, exist_ok=True)
RANDOM_STATE = 42


def load_dataset():
    """Retrieve data from the UCI Machine Learning Repository."""
    try:
        from ucimlrepo import fetch_ucirepo
        dataset = fetch_ucirepo(id=492)
        X = dataset.data.features.copy()
        y = dataset.data.targets.copy()
        if isinstance(y, pd.Series):
            y = y.to_frame()
        return pd.concat(
            [X.reset_index(drop=True), y.reset_index(drop=True)], axis=1
        )
    except Exception as exc:
        local = ROOT / "data" / "Metro_Interstate_Traffic_Volume.csv"
        if local.exists():
            print(f"UCI fetch did not work ({exc}); loading local file: {local}")
            return pd.read_csv(local)
        raise RuntimeError(
            "Could not retrieve the dataset automatically. Check your internet connection or download "
            "the CSV from https://archive.ics.uci.edu/dataset/492/metro+interstate+traffic+volume "
            "and save it as data/Metro_Interstate_Traffic_Volume.csv."
        ) from exc


def prepare_data(df):
    """Validate essential fields, sort by timestamp, and engineer features."""
    if "date_time" not in df.columns or "traffic_volume" not in df.columns:
        raise ValueError(
            "Expected columns 'date_time' and 'traffic_volume'. Actual columns: "
            + ", ".join(map(str, df.columns))
        )
    d = df.copy()
    d["date_time"] = pd.to_datetime(d["date_time"], errors="coerce")
    d["traffic_volume"] = pd.to_numeric(d["traffic_volume"], errors="coerce")
    initial_rows = len(d)
    invalid = int(d["date_time"].isna().sum() + d["traffic_volume"].isna().sum())
    d = d.dropna(subset=["date_time", "traffic_volume"])
    d = d.sort_values("date_time").drop_duplicates().reset_index(drop=True)

    d["hour"] = d["date_time"].dt.hour
    d["day_of_week"] = d["date_time"].dt.dayofweek
    d["month"] = d["date_time"].dt.month
    d["is_weekend"] = (d["day_of_week"] >= 5).astype(int)
    d["hour_sin"] = np.sin(2 * np.pi * d["hour"] / 24)
    d["hour_cos"] = np.cos(2 * np.pi * d["hour"] / 24)
    d["weekday_sin"] = np.sin(2 * np.pi * d["day_of_week"] / 7)
    d["weekday_cos"] = np.cos(2 * np.pi * d["day_of_week"] / 7)
    return d, initial_rows, invalid


def make_pipeline(model, numeric_features, categorical_features):
    numeric_pipe = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    prep = ColumnTransformer([
        ("num", numeric_pipe, numeric_features),
        ("cat", cat_pipe, categorical_features),
    ], remainder="drop")
    return Pipeline([("preprocess", prep), ("model", model)])


def metrics(y_true, y_pred):
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)),
    }


def main():
    warnings.filterwarnings("default")
    df_raw = load_dataset()
    df, rows_before, invalid_count = prepare_data(df_raw)
    if len(df) < 100:
        raise RuntimeError(f"Only {len(df)} usable rows were found; inspect the dataset before modeling.")

    profile = {
        "source": "UCI Machine Learning Repository, dataset ID 492",
        "rows_before_validation": int(rows_before),
        "rows_after_validation_and_exact_duplicate_removal": int(len(df)),
        "invalid_datetime_or_target_values_counted_before_drop": int(invalid_count),
        "columns": list(df.columns),
        "missing_values_by_column_before_pipeline": {
            str(k): int(v) for k, v in df.isna().sum().items()
        },
        "start_time": str(df["date_time"].min()),
        "end_time": str(df["date_time"].max()),
    }
    (OUT / "data_profile.json").write_text(json.dumps(profile, indent=2), encoding="utf-8")

    numeric_options = [
        "temp", "rain_1h", "snow_1h", "clouds_all",
        "hour", "day_of_week", "month", "is_weekend",
        "hour_sin", "hour_cos", "weekday_sin", "weekday_cos"
    ]
    categorical_options = ["holiday", "weather_main", "weather_description"]
    numeric = [c for c in numeric_options if c in df.columns]
    categorical = [c for c in categorical_options if c in df.columns]
    features = numeric + categorical
    X = df[features].copy()
    y = df["traffic_volume"].copy()
    time_col = df["date_time"].copy()

    n = len(df)
    fit_end = int(n * 0.70)
    val_end = int(n * 0.80)
    # Chronological train/validation/test partitions: 70% / 10% / 20%.
    X_fit, y_fit = X.iloc[:fit_end], y.iloc[:fit_end]
    X_val, y_val = X.iloc[fit_end:val_end], y.iloc[fit_end:val_end]
    X_fitval, y_fitval = X.iloc[:val_end], y.iloc[:val_end]
    X_test, y_test = X.iloc[val_end:], y.iloc[val_end:]
    test_time = time_col.iloc[val_end:].reset_index(drop=True)

    configs = {
        "MeanBaseline": DummyRegressor(strategy="mean"),
        "LinearRegression": LinearRegression(),
        "DecisionTree_depth12_leaf20": DecisionTreeRegressor(
            max_depth=12, min_samples_leaf=20, random_state=RANDOM_STATE
        ),
        "DecisionTree_depth20_leaf10": DecisionTreeRegressor(
            max_depth=20, min_samples_leaf=10, random_state=RANDOM_STATE
        ),
        "RandomForest_120_depth18_leaf5": RandomForestRegressor(
            n_estimators=120, max_depth=18, min_samples_leaf=5,
            random_state=RANDOM_STATE, n_jobs=-1
        ),
        "RandomForest_150_depth26_leaf3": RandomForestRegressor(
            n_estimators=150, max_depth=26, min_samples_leaf=3,
            random_state=RANDOM_STATE, n_jobs=-1
        ),
    }

    val_rows = []
    fitted_on_fit = {}
    print(f"Usable rows: {n}; fitting candidates on {len(X_fit)} rows, validating on {len(X_val)} rows.")
    for name, estimator in configs.items():
        pipe = make_pipeline(estimator, numeric, categorical)
        pipe.fit(X_fit, y_fit)
        pred_val = pipe.predict(X_val)
        row = {"model": name, **metrics(y_val, pred_val)}
        val_rows.append(row)
        fitted_on_fit[name] = pipe
        print(f"Validated {name}: RMSE={row['RMSE']:.3f}, MAE={row['MAE']:.3f}, R2={row['R2']:.4f}")

    val_results = pd.DataFrame(val_rows).sort_values("RMSE")
    val_results.to_csv(OUT / "validation_metrics.csv", index=False)

    non_baseline = val_results[val_results["model"] != "MeanBaseline"]
    if non_baseline.empty:
        best_name = "MeanBaseline"
    else:
        best_name = str(non_baseline.iloc[0]["model"])
    chosen_estimator = configs[best_name]

    # Final test only compares the validation-selected model against the mean baseline.
    chosen_pipe = make_pipeline(chosen_estimator, numeric, categorical)
    baseline_pipe = make_pipeline(DummyRegressor(strategy="mean"), numeric, categorical)
    chosen_pipe.fit(X_fitval, y_fitval)
    baseline_pipe.fit(X_fitval, y_fitval)
    chosen_pred = chosen_pipe.predict(X_test)
    baseline_pred = baseline_pipe.predict(X_test)

    chosen_metrics = metrics(y_test, chosen_pred)
    baseline_metrics = metrics(y_test, baseline_pred)
    test_rows = [
        {"model": "MeanBaseline", **baseline_metrics,
         "test_rows": int(len(y_test)), "test_start": str(test_time.min()), "test_end": str(test_time.max())},
        {"model": best_name, **chosen_metrics,
         "test_rows": int(len(y_test)), "test_start": str(test_time.min()), "test_end": str(test_time.max())},
    ]
    pd.DataFrame(test_rows).to_csv(OUT / "test_metrics.csv", index=False)

    predictions = pd.DataFrame({
        "date_time": test_time,
        "actual_traffic_volume": y_test.reset_index(drop=True),
        "baseline_prediction": baseline_pred,
        "selected_model_prediction": chosen_pred,
        "selected_model_residual": y_test.to_numpy() - chosen_pred,
    })
    predictions.to_csv(OUT / "test_predictions.csv", index=False)

    # Diagnostic plots
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(y_test, chosen_pred, alpha=0.25, s=12)
    low = min(float(y_test.min()), float(np.min(chosen_pred)))
    high = max(float(y_test.max()), float(np.max(chosen_pred)))
    ax.plot([low, high], [low, high], linestyle="--")
    ax.set(title=f"Actual vs Predicted — {best_name}", xlabel="Actual traffic volume", ylabel="Predicted traffic volume")
    fig.tight_layout(); fig.savefig(OUT / "actual_vs_predicted.png", dpi=160); plt.close(fig)

    residuals = y_test.to_numpy() - chosen_pred
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(residuals, bins=40)
    ax.set(title="Distribution of Test Residuals", xlabel="Residual (actual - predicted)", ylabel="Count")
    fig.tight_layout(); fig.savefig(OUT / "residual_distribution.png", dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(test_time, residuals, linewidth=0.8)
    ax.axhline(0, linestyle="--")
    ax.set(title="Test Residuals Over Time", xlabel="Timestamp", ylabel="Residual")
    fig.tight_layout(); fig.savefig(OUT / "residuals_over_time.png", dpi=160); plt.close(fig)

    # Error by hour can reveal systematic time-of-day weaknesses.
    test_hours = df["hour"].iloc[val_end:].reset_index(drop=True)
    by_hour = pd.DataFrame({
        "hour": test_hours,
        "actual": y_test.reset_index(drop=True),
        "predicted": chosen_pred,
    })
    by_hour["absolute_error"] = (by_hour["actual"] - by_hour["predicted"]).abs()
    hour_err = by_hour.groupby("hour", as_index=False)["absolute_error"].mean()
    hour_err.to_csv(OUT / "mae_by_hour.csv", index=False)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(hour_err["hour"], hour_err["absolute_error"])
    ax.set(title="Mean Absolute Error by Hour of Day", xlabel="Hour of day", ylabel="Mean absolute error")
    fig.tight_layout(); fig.savefig(OUT / "mae_by_hour.png", dpi=160); plt.close(fig)

    # Save RF importances only when the chosen model is a Random Forest.
    if best_name.startswith("RandomForest"):
        try:
            feature_names = chosen_pipe.named_steps["preprocess"].get_feature_names_out()
            importances = chosen_pipe.named_steps["model"].feature_importances_
            imp = pd.DataFrame({"feature": feature_names, "importance": importances}).sort_values("importance", ascending=False)
            imp.to_csv(OUT / "feature_importance.csv", index=False)
        except Exception as exc:
            print(f"Could not export feature importance: {exc}")

    # Automatic evidence-based recommendations.
    baseline_rmse = baseline_metrics["RMSE"]
    selected_rmse = chosen_metrics["RMSE"]
    if baseline_rmse > 0:
        pct = (baseline_rmse - selected_rmse) / baseline_rmse * 100
    else:
        pct = 0.0
    recs = [
        "MODEL EVALUATION SUMMARY",
        "=" * 28,
        f"Validation-selected model: {best_name}",
        f"Final test period: {test_time.min()} to {test_time.max()}",
        f"Test rows: {len(y_test)}",
        "",
        "FINAL TEST METRICS",
        f"Mean baseline — MAE {baseline_metrics['MAE']:.3f}, RMSE {baseline_metrics['RMSE']:.3f}, R2 {baseline_metrics['R2']:.4f}",
        f"Selected model — MAE {chosen_metrics['MAE']:.3f}, RMSE {chosen_metrics['RMSE']:.3f}, R2 {chosen_metrics['R2']:.4f}",
        f"RMSE change vs mean baseline: {pct:.2f}% reduction (positive is better).",
        "",
        "RECOMMENDATIONS",
    ]
    if selected_rmse < baseline_rmse:
        recs.append("- The selected model improves on the mean baseline RMSE for this test period; inspect residual plots and validate across additional time windows before claiming robust performance.")
    else:
        recs.append("- The selected model did not improve on the mean baseline RMSE on this test period. Prefer the simpler baseline until features or validation design are improved.")
    if chosen_metrics["R2"] < 0:
        recs.append("- Negative test R-squared indicates performance worse than a mean benchmark under the R2 definition; investigate distribution shift, leakage assumptions, and feature design.")
    recs.extend([
        "- Review MAE-by-hour output to find periods with larger errors; consider improved calendar/interaction features if justified.",
        "- Use only past-derived traffic lags for future prediction tasks and create them without using future observations.",
        "- For advance forecasting, use weather forecasts available at prediction time rather than same-hour observed weather.",
        "- Any further tuning must use training/validation periods only; retain the final test period for the final estimate.",
        "- The data is from one monitored highway location, so generalization to other roads is not established.",
    ])
    (OUT / "optimization_recommendations.txt").write_text("\n".join(recs), encoding="utf-8")

    print("\nValidation comparison:")
    print(val_results.to_string(index=False))
    print("\nFinal test comparison:")
    print(pd.DataFrame(test_rows).to_string(index=False))
    print(f"\nOutputs saved in: {OUT}")
    print("Use the CSV outputs to update the Week 5 report. Do not substitute external benchmark figures for these project results.")


if __name__ == "__main__":
    main()
