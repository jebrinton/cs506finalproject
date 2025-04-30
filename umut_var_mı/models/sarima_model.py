from config import TRAIN_YEARS, TEST_YEARS, RESPONSE_VAR, SAMPLE_FACTOR
import pandas as pd
import numpy as np
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
def fit_model(train_series):
    model = SARIMAX(train_series, order=(p, d, q), seasonal_order=(P, D, Q, s))
    return model.fit()

def forecast_model(model, steps):
    return model.forecast(steps)