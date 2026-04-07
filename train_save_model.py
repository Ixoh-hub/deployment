import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import lightgbm as lgb
import joblib


def load_and_prepare(csv_path):
    df = pd.read_csv(csv_path)

    # keep only required columns
    target_col = 'Life expectancy '
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in input CSV")

    df = df.dropna(subset=[target_col])

    # fix status
    if 'Status' in df.columns:
        df['Status'] = df['Status'].map({'Developed': 0, 'Developing': 1})
    
    # convert Year if datetime
    if 'Year' in df.columns:
        if np.issubdtype(df['Year'].dtype, np.datetime64):
            df['Year'] = df['Year'].dt.year
        df['Year'] = pd.to_numeric(df['Year'], errors='coerce')

    # features (drop country+target)
    X = df.drop(columns=[target_col, 'Country'], errors='ignore')
    X = X.select_dtypes(include=[np.number])

    y = df[target_col].astype(float)
    return X, y


def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42)

    model = Pipeline(
        steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('regressor', lgb.LGBMRegressor(
                objective='regression',
                n_estimators=1000,
                random_state=42,
                verbose=-1,
            )),
        ]
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse = np.sqrt(np.mean((y_pred - y_test) ** 2))
    r2 = 1 - np.sum((y_pred - y_test) ** 2) / np.sum((y_test - np.mean(y_test)) ** 2)

    return model, rmse, r2


def main():
    data_path = os.getenv('DATA_PATH', 'Life Expectancy Data.csv')
    model_dir = os.getenv('MODEL_DIR', 'models')
    os.makedirs(model_dir, exist_ok=True)

    X, y = load_and_prepare(data_path)
    model, rmse, r2 = train_model(X, y)

    model_path = os.path.join(model_dir, 'lightgbm_life_expectancy.pkl')
    numeric_ranges = {
        col: {
            'min': float(X[col].min(skipna=True)),
            'max': float(X[col].max(skipna=True)),
            'median': float(X[col].median(skipna=True)),
        }
        for col in X.columns
    }
    metadata = {
        'train_rows': int(X.shape[0]),
        'feature_count': int(X.shape[1]),
        'rmse': float(rmse),
        'r2': float(r2),
        'feature_ranges': numeric_ranges,
    }

    joblib.dump({'model': model, 'features': X.columns.tolist(), 'metadata': metadata}, model_path)

    print(f"Saved model to: {model_path}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R2: {r2:.4f}")


if __name__ == '__main__':
    main()
