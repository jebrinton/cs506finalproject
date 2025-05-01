# %% libraries
import pandas as pd
from models import sarima_model, neuraln_model
from sklearn.model_selection import train_test_split
from config import TRAIN_YEARS, TEST_YEARS, RESPONSE_VAR, SAMPLE_FACTOR
import os
os.makedirs("out", exist_ok=True)  # creates 'out/' if it doesn't exist


MODELS = {
    "sarima": sarima_model,
    "nn": neuraln_model
}

# %% cleaning
df = pd.read_csv("bavim.csv", index_col="DATE")
df = df.interpolate(method='linear')
df.drop(columns=["STATION"], inplace=True)
df.index = pd.to_datetime(df.index)

df = df.iloc[::SAMPLE_FACTOR]

# %% modeling
cities = df["NAME"].unique()

for city in cities:
    city_df = df[df["NAME"] == city].copy()
    city_series = city_df[RESPONSE_VAR]
    train_steps = int(TRAIN_YEARS * 365 / SAMPLE_FACTOR)
    test_steps = int(TEST_YEARS * 365 / SAMPLE_FACTOR)
    train = city_series.iloc[:train_steps]
    test = city_series.iloc[train_steps:train_steps + test_steps]

    for model_name, model_module in MODELS.items():
        if model_name == "nn":
            # For NN: use full DataFrame, fit and forecast with helper functions
            train = city_df.iloc[:train_steps]
            test = city_df.iloc[train_steps:train_steps + test_steps]

            model_bundle = model_module.fit_model(train, RESPONSE_VAR)
            forecast = model_module.forecast_model(model_bundle, test, RESPONSE_VAR)
        # model name == "sarima"
        else: 
            city_series = city_df[RESPONSE_VAR]
            train = city_series.iloc[:train_steps]
            test = city_series.iloc[train_steps:train_steps + test_steps]

            model = model_module.fit_model(train)
            forecast = model_module.forecast_model(model, len(test))

        forecast.to_csv(f"out/predictions_{model_name}_{city}.csv")
