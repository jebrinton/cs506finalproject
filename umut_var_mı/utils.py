import numpy as np
import pandas as pd

def build_sequences(values, lookback):
    X, y = [], []
    for i in range(len(values) - lookback):
        X.append(values[i : i + lookback].flatten())
        y.append(values[i + lookback, 0])
    return np.array(X), np.array(y)

def add_date_features(df):
    df["month"] = df.index.month
    df["day_of_year"] = df.index.dayofyear
    df["sin_month"] = np.sin(2 * np.pi * df["month"] / 12)
    df["cos_month"] = np.cos(2 * np.pi * df["month"] / 12)
    df["sin_day"] = np.sin(2 * np.pi * df["day_of_year"] / 365)
    df["cos_day"] = np.cos(2 * np.pi * df["day_of_year"] / 365)
    return df

