# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.statespace.sarimax import SARIMAX

# 1 sample per 30 days = less than 1 second fit on my laptop
# sample per 10 days =  16.5 second fit (pretty significant increase)
# sample per  5 days = 187.8 second fit
SAMPLE_FACTOR = 1

# %%
# load data
import pandas as pd

df = pd.read_csv("boston_weather_data.csv")

# Use 'time' as index
df['time'] = pd.to_datetime(df['time'])
df.set_index('time', inplace=True)

# Fill in NaN
df = df.interpolate(method='linear')

# Some sample datasets to make dev time faster
baby_df = df[:12]
sampled_df = df.iloc[::SAMPLE_FACTOR]

# %%
import matplotlib.pyplot as plt

def plot_tavg(df):
    plt.figure(figsize=(12, 6))
    plt.plot(df['tavg'], label='Temperatura promedio')
    plt.title('Serie temporal de temperatura promedio')
    plt.xlabel('Fecha')
    plt.ylabel('Temperatura (°C)')
    plt.legend()
    plt.show()

plot_tavg(sampled_df)

# %%
# fit SARIMAX
from statsmodels.tsa.statespace.sarimax import SARIMAX

# autoregressive order—number of past values used to predict current value
p = 1
# differencing order
d = 1
# moving average order
q = 1

# these are same as before but for the seasonal trend
P = 1
D = 1
Q = 1
# number of entries that form a complete season (year)
s = (int) (365 / SAMPLE_FACTOR)

# fit model and predict 5 years
model = SARIMAX(sampled_df["tavg"], order=(p, d, q), seasonal_order=(P, D, Q, s))
sarima = model.fit()

# %%
# Forecast
years = 5
steps = (int) (years * 365 / SAMPLE_FACTOR)
predictions = sarima.forecast(steps)
freq_title = f"{SAMPLE_FACTOR}D"
pred_index = pd.date_range(start=sampled_df.index[-1], periods=steps+1, freq=freq_title)[1:]
pred_series = pd.Series(predictions.values, index=pred_index)

# Confidence range
forecast = sarima.get_forecast(steps)
conf_int = forecast.conf_int()

plt.figure(figsize=(14, 7))
plt.plot(sampled_df['tavg'], label='Recorded Temp', color='blue')
plt.plot(pred_series, label='SARIMA Prediction', color='orange', linestyle='--')
# Confidence range
plt.fill_between(
    pred_series.index, 
    conf_int['lower tavg'], 
    conf_int['upper tavg'], 
    color='orange', 
    alpha=0.2, 
    label='Confidence Range'
)

plt.title('Average Temp Prediction with SARIMA')
plt.xlabel('Date')
plt.ylabel('Temp (°C)')
plt.legend()
plt.grid(True)

plt.show()

# %%
