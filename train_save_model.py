import os

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd


def load_and_prepare(csv_path):
    df = pd.read_csv(csv_path)

    target_col = "Life expectancy "
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in input CSV")

    df = df.dropna(subset=[target_col])

    if "Status" in df.columns:
        df["Status"] = df["Status"].map({"Developed": 0, "Developing": 1})

    if "Year" in df.columns:
        if np.issubdtype(df["Year"].dtype, np.datetime64):
            df["Year"] = df["Year"].dt.year
        df["Year"] = pd.to_numeric(df["Year"], errors="coerce")

    X = df.drop(columns=[target_col, "Country"], errors="ignore")
    X = X.select_dtypes(include=[np.number])
    y = df[target_col].astype(float)
    return X, y


def _train_val_split(X, y, test_size=0.25, random_state=42):
    rng = np.random.RandomState(random_state)
    n = len(X)
    idx = np.arange(n)
    rng.shuffle(idx)
    split = int(n * (1 - test_size))
    tr, va = idx[:split], idx[split:]
    return X.iloc[tr], X.iloc[va], y.iloc[tr], y.iloc[va]


def train_model(X, y):
    X_train, X_test, y_train, y_test = _train_val_split(X, y)

    medians = X_train.median(numeric_only=True)
    X_train = X_train.fillna(medians)
    X_test = X_test.fillna(medians)

    train_data = lgb.Dataset(X_train, label=y_train, feature_name=list(X_train.columns))
    valid_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

    params = {
        "objective": "regression",
        "metric": "rmse",
        "verbosity": -1,
        "seed": 42,
    }

    model = lgb.train(
        params,
        train_data,
        num_boost_round=1000,
        valid_sets=[train_data, valid_data],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50),
            lgb.log_evaluation(period=100),
        ],
    )

    y_pred = model.predict(X_test, num_iteration=model.best_iteration)
    rmse = float(np.sqrt(np.mean((y_pred - y_test.values) ** 2)))
    y_mean = float(np.mean(y_test.values))
    r2 = float(1 - np.sum((y_pred - y_test.values) ** 2) / np.sum((y_test.values - y_mean) ** 2))

    impute_medians = {k: float(v) for k, v in medians.items() if pd.notna(v)}

    return model, rmse, r2, impute_medians


def main():
    data_path = os.getenv("DATA_PATH", "Life Expectancy Data.csv")
    model_dir = os.getenv("MODEL_DIR", "models")
    os.makedirs(model_dir, exist_ok=True)

    X, y = load_and_prepare(data_path)
    model, rmse, r2, impute_medians = train_model(X, y)

    model_path = os.path.join(model_dir, "lightgbm_life_expectancy.pkl")
    feature_list = X.columns.tolist()
    numeric_ranges = {
        col: {
            "min": float(X[col].min(skipna=True)),
            "max": float(X[col].max(skipna=True)),
            "median": float(X[col].median(skipna=True)),
        }
        for col in feature_list
    }
    metadata = {
        "train_rows": int(X.shape[0]),
        "feature_count": int(X.shape[1]),
        "rmse": rmse,
        "r2": r2,
        "feature_ranges": numeric_ranges,
        "impute_medians": impute_medians,
    }

    joblib.dump({"model": model, "features": feature_list, "metadata": metadata}, model_path)

    print(f"Saved model to: {model_path}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R2: {r2:.4f}")


if __name__ == "__main__":
    main()
