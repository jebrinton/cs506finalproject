# %%
# load data
import pandas as pd
import numpy as np

df = pd.read_csv("boston_weather_data.csv")

# Use 'time' as index
df['time'] = pd.to_datetime(df['time'])
df.set_index('time', inplace=True)

# Fill in NaN
df = df.interpolate(method='linear')

# get the average difference in max tempurature between days
print(df)
# %%
diff = np.diff(df['tmax'].values)

print("MAE if predicted tomorrow's max with today's max: ", diff.sum().round(4))
