"""
Week 4: Predictive Modeling and Feature Engineering
Dataset: UCI Metro Interstate Traffic Volume (ID 492)

Install:
    pip install -r requirements.txt

Run from the project root:
    python src/train_models.py

The script fetches the UCI dataset, sorts observations chronologically, engineers
time features, and compares a mean baseline with several regression models.
It does not claim that the models are production-ready. Inspect outputs and
limitations before interpreting results.
"""
from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd
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
OUTPUTS = ROOT / "outputs"
OUTPUTS.mkdir(parents=True, exist_ok=True)
RANDOM_STATE = 42


def load_dataset() -> pd.DataFrame:
    """Load the dataset from the UCI repository interface."""
    try:
        from ucimlrepo import fetch_ucirepo
    except ImportError as exc:
        raise RuntimeError(
            "Missing ucimlrepo. Install dependencies with: pip install -r requirements.txt"
        ) from exc

    dataset = fetch_ucirepo(id=492)
    features = dataset.data.features.copy()
    target = dataset.data.targets.copy()
    if isinstance(target, pd.Series):
        target = target.to_frame()
    df = pd.concat(
        [features.reset_index(drop=True), target.reset_index(drop=True)], axis=1
    )
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Parse timestamp and create compact calendar features."""
    if "date_time" not in df.columns:
        raise ValueError("Expected a 'date_time' column in the dataset.")
    if "traffic_volume" not in df.columns:
        raise ValueError("Expected a 'traffic_volume' target column.")

    result = df.copy()
    result["date_time"] = pd.to_datetime(result["date_time"], errors="coerce")
    result["traffic_volume"] = pd.to_numeric(
        result["traffic_volume"], errors="coerce"
    )
    result = result.dropna(subset=["date_time", "traffic_volume"])
    result = result.sort_values("date_time").drop_duplicates().reset_index(drop=True)

    # Calendar features are derived from the observation timestamp.
    result["hour"] = result["date_time"].dt.hour
    result["day_of_week"] = result["date_time"].dt.dayofweek
    result["month"] = result["date_time"].dt.month
    result["is_weekend"] = (result["day_of_week"] >= 5).astype(int)

    # Cyclical hour encoding keeps 23:00 close to 00:00.
    result["hour_sin"] = np.sin(2 * np.pi * result["hour"] / 24)
    result["hour_cos"] = np.cos(2 * np.pi * result["hour"] / 24)

    # The source timestamp is retained for ordering but excluded from model inputs.
    return result


def build_preprocessor(numeric_features, categorical_features):
    numeric_pipe = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median"))]
    )
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipe, numeric_features),
            ("categorical", categorical_pipe, categorical_features),
        ],
        remainder="drop",
    )


def main():
    warnings.filterwarnings("default")
    try:
        raw = load_dataset()
    except Exception as exc:
        print(f"Could not retrieve dataset from UCI: {exc}")
        print("Check internet access or download data from:")
        print("https://archive.ics.uci.edu/dataset/492/metro+interstate+traffic+volume")
        print("If using a local CSV, update load_dataset() to read the downloaded file.")
        raise SystemExit(1)

    df = engineer_features(raw)
    if len(df) < 20:
        raise ValueError("Too few usable rows after timestamp/target validation.")

    # Save a profile before model training.
    profile = {
        "rows_after_basic_validation": int(len(df)),
        "columns": list(df.columns),
        "missing_values_by_column": {
            str(k): int(v) for k, v in df.isna().sum().items()
        },
        "duplicate_rows_after_basic_validation": int(df.duplicated().sum()),
        "min_timestamp": str(df["date_time"].min()),
        "max_timestamp": str(df["date_time"].max()),
        "target": "traffic_volume",
        "split": "chronological 80% train / 20% test",
    }
    (OUTPUTS / "data_profile.json").write_text(
        json.dumps(profile, indent=2), encoding="utf-8"
    )

    # Features and target. Exclude timestamp, raw target, and unused free-text.
    candidate_numeric = [
        "temp", "rain_1h", "snow_1h", "clouds_all",
        "hour", "day_of_week", "month", "is_weekend", "hour_sin", "hour_cos"
    ]
    candidate_categorical = ["holiday", "weather_main", "weather_description"]
    numeric_features = [c for c in candidate_numeric if c in df.columns]
    categorical_features = [c for c in candidate_categorical if c in df.columns]
    feature_cols = numeric_features + categorical_features
    if not feature_cols:
        raise ValueError("No usable features found. Inspect the dataset schema.")

    X = df[feature_cols].copy()
    y = df["traffic_volume"].copy()

    # Chronological holdout: earliest rows train, latest rows test.
    split_idx = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    test_times = df["date_time"].iloc[split_idx:].reset_index(drop=True)

    numeric = [c for c in numeric_features if c in X_train.columns]
    categorical = [c for c in categorical_features if c in X_train.columns]
    preprocessor = build_preprocessor(numeric, categorical)

    models = {
        "DummyMeanBaseline": DummyRegressor(strategy="mean"),
        "LinearRegression": LinearRegression(),
        "DecisionTree": DecisionTreeRegressor(
            max_depth=12, min_samples_leaf=20, random_state=RANDOM_STATE
        ),
        "RandomForest": RandomForestRegressor(
            n_estimators=200,
            max_depth=18,
            min_samples_leaf=5,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }

    results = []
    predictions = pd.DataFrame({
        "date_time": test_times,
        "actual_traffic_volume": y_test.reset_index(drop=True),
    })

    fitted_models = {}
    for name, estimator in models.items():
        pipe = Pipeline(
            steps=[
                ("preprocess", build_preprocessor(numeric, categorical)),
                ("model", estimator),
            ]
        )
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        fitted_models[name] = pipe
        results.append({
            "model": name,
            "MAE": float(mean_absolute_error(y_test, pred)),
            "RMSE": float(np.sqrt(mean_squared_error(y_test, pred))),
            "R2": float(r2_score(y_test, pred)),
            "train_rows": int(len(X_train)),
            "test_rows": int(len(X_test)),
            "test_start": str(test_times.min()),
            "test_end": str(test_times.max()),
        })
        predictions[name + "_prediction"] = pred

    metrics = pd.DataFrame(results).sort_values("RMSE")
    metrics.to_csv(OUTPUTS / "model_metrics.csv", index=False)
    predictions.to_csv(OUTPUTS / "test_predictions.csv", index=False)

    # Feature importance is saved for the Random Forest pipeline.
    rf = fitted_models["RandomForest"]
    try:
        feature_names = rf.named_steps["preprocess"].get_feature_names_out()
        importances = rf.named_steps["model"].feature_importances_
        importance_df = pd.DataFrame({
            "feature": feature_names,
            "importance": importances,
        }).sort_values("importance", ascending=False)
        importance_df.to_csv(OUTPUTS / "random_forest_feature_importance.csv", index=False)
    except Exception as exc:
        print(f"Feature-importance export skipped: {exc}")

    print("\nModel comparison (lower MAE/RMSE is better; higher R2 is generally better):")
    print(metrics.to_string(index=False))
    print("\nOutputs saved to:", OUTPUTS)
    print("Caution: the test set is the latest 20% of timestamps; interpret results in that context.")
    print("Weather observations from the same hour may not be known in advance in a real deployment.")


if __name__ == "__main__":
    main()
