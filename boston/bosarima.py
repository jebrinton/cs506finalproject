# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.statespace.sarimax import SARIMAX

# 1 sample per 30 days = less than 1 second fit on my laptop
# sample per 10 days =  16.5 second fit (pretty significant increase)
# sample per  8 days =  30.1 second fit
# sample per  5 days = 187.8 second fit
SAMPLE_FACTOR = 30

TEST_YEARS = 0.1
TRAIN_YEARS = 40 - TEST_YEARS

RESPONSE_VAR = "TMAX"

# %%
# load data

# df = pd.read_csv("boston_weather_data.csv")

# # Use 'time' as index
# df['time'] = pd.to_datetime(df['time'])
# df.set_index('time', inplace=True)

# # Fill in NaN
# df = df.interpolate(method='linear')

# # Some sample datasets to make dev time faster
# baby_df = df[:12]
# sampled_df = df.iloc[::SAMPLE_FACTOR]

# %%
# the same cell as before but for weather.csv
# Storing the dataset into the variable dataset and making the "DATE" column the index column.
dataset = pd.read_csv("weather.csv", index_col="DATE")

# Remove non-numerical data
dataset = dataset.select_dtypes(include=['number', 'datetime'])

# For each column, store the percentage of missing values.
miss_pct = dataset.apply(pd.isnull).sum()/dataset.shape[0]

# List columns with low miss_pct (good columns); feel free to tweak the threshold of how low we want missing values to be.
good_columns = dataset.columns[miss_pct < 0.3]
dataset = dataset[good_columns].copy()
dataset.apply(pd.isnull).sum()/dataset.shape[0]

# Since the "PRCP" column has missin values, we fill the missing values with zero since
# if there was no PRCP recorded for that day then that must mean that it did not rain therefore PRCP would be zero
dataset["PRCP"] = dataset["PRCP"].fillna(0)

# Fill in other NaNs
dataset = dataset.interpolate(method='linear')

# confirming no NaNs left
dataset.apply(pd.isnull).sum() / dataset.shape[0]

# Change dtype from object to datetime.
dataset.index = pd.to_datetime(dataset.index)

sampled_df = dataset.iloc[::SAMPLE_FACTOR]

# %%
train_steps = int(TRAIN_YEARS * 365 / SAMPLE_FACTOR)
test_steps = int(TEST_YEARS * 365 / SAMPLE_FACTOR)

sampled_train = sampled_df.iloc[:train_steps]
sampled_test = sampled_df.iloc[train_steps:train_steps + test_steps]

# %%
import matplotlib.pyplot as plt

def plot_tavg(df):
    plt.figure(figsize=(12, 6))
    plt.plot(df["TMAX"], label='Temperatura promedio')
    plt.title('Serie temporal de temperatura promedio')
    plt.xlabel('Fecha')
    plt.ylabel('Temperatura (°F)')
    plt.legend()
    plt.show()

# plot_tavg(sampled_df)

# %%
# fit SARIMAX
from statsmodels.tsa.statespace.sarimax import SARIMAX

# autoregressive order—number of past values used to predict current value
p = 1
# differencing order—remove noise so we can focus on overall trend
d = 1
# moving average order—take into account past errors
q = 1

# these are same as before but for the seasonal trend
P = 1
D = 1
Q = 1
# number of entries that form a complete season (year)
s = (int) (365 / SAMPLE_FACTOR)

# fit model
model = SARIMAX(sampled_train[RESPONSE_VAR], order=(p, d, q), seasonal_order=(P, D, Q, s))
sarima = model.fit()

# %%
# Forecast
predictions = sarima.forecast(test_steps)
freq_title = f"{SAMPLE_FACTOR}D"
pred_index = pd.date_range(start=sampled_train.index[-1], periods=test_steps+1, freq=freq_title)[1:]
pred_series = pd.Series(predictions.values, index=pred_index)

# Confidence range
forecast = sarima.get_forecast(test_steps)
lower_conf_int = forecast.conf_int().iloc[:, 0]
upper_conf_int = forecast.conf_int().iloc[:, 1]

plt.figure(figsize=(14, 7))
plt.plot(sampled_train[RESPONSE_VAR], label='Recorded Max Temp', color='blue')
plt.plot(sampled_test[RESPONSE_VAR], label='Recorded Max Temp', color='green')

plt.plot(pred_series, label='SARIMA Prediction', color='orange', linestyle='--')
# Confidence range
# plt.fill_between(
#     pred_series.index,
#     lower_conf_int,
#     upper_conf_int,
#     color='orange',
#     alpha=0.2,
#     label='Confidence Range'
# )

plt.title('Maximum Temp Prediction with SARIMAX')
plt.xlabel('Date')
plt.ylabel('Temp (°F)')
plt.legend()
plt.grid(True)

plt.show()

# %%
diff = np.abs(pred_series - sampled_test[RESPONSE_VAR])
print(f"Mean Average Error ({TEST_YEARS} years): {diff.mean().round(4)}ºF")
# %%
