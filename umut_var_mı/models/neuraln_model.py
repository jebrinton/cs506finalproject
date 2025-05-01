import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error
from utils import build_sequences, add_date_features

LOOKBACK = 30 # Set model-wide constant

def fit_model(train_df, response_var):
    df = train_df.copy()
    df = add_date_features(df)
    
    feature_cols = [response_var] + ['sin_month', 'cos_month', 'sin_day', 'cos_day']
    
    scaler = MinMaxScaler()
    values = scaler.fit_transform(df[feature_cols])
    X, y = build_sequences(values, LOOKBACK)

    model = MLPRegressor(
        hidden_layer_sizes=(128, 64, 32),
        activation="relu",
        solver="adam",
        alpha=0.001,
        learning_rate_init=0.001,
        max_iter=5000,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
        random_state=42
    )
    model.fit(X, y)
    return model, scaler, df, feature_cols

def forecast_model(model_bundle, test_df, response_var):
    model, scaler, train_df, feature_cols = model_bundle
    df = test_df.copy()
    df = add_date_features(df)

    all_df = pd.concat([train_df, df])
    all_vals = scaler.transform(all_df[feature_cols])
    
    test_date_feats = all_vals[-len(df):, 1:]
    last_window = all_vals[-len(df)-LOOKBACK:-len(df)].copy()

    preds_scaled = []
    for i in range(len(df)):
        inp = last_window.flatten().reshape(1, -1)
        p = model.predict(inp)[0]
        preds_scaled.append(p)
        nxt = np.concatenate([[p], test_date_feats[i]])
        last_window = np.vstack([last_window[1:], nxt])

    full_pred_scaled = np.hstack([np.array(preds_scaled).reshape(-1,1), test_date_feats])
    full_pred = scaler.inverse_transform(full_pred_scaled)
    preds = full_pred[:, 0]
    return pd.Series(preds, index=df.index)