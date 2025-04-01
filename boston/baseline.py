# %%
# load data
import pandas as pd
import numpy as np

# Storing the dataset into the variable dataset and making the "DATE" column the index column.
dataset = pd.read_csv("weather.csv", index_col="DATE")

# For each column, store the percentage of missing values.
miss_pct = dataset.apply(pd.isnull).sum()/dataset.shape[0]

# List columns with low miss_pct (good columns); feel free to tweak the threshold of how low we want missing values to be.
good_columns = dataset.columns[miss_pct < 0.01]
dataset = dataset[good_columns].copy()

# Fill the missing values in PRCP column with zero (assuming default is no rain)
dataset["PRCP"] = dataset["PRCP"].fillna(0)

# Otherwise, fill in NaN with linear interpolation
dataset = dataset.infer_objects(copy=False)
dataset = dataset.interpolate(method='linear')

# get the average difference in max temperature between days
print(dataset)
# %%
temp_diff = np.diff(dataset['TMAX'].values)
prcp_diff = np.diff((dataset["PRCP"] > 0).astype(int))

print("MAE if predicting tomorrow's max with today's max: ", np.mean(np.abs(temp_diff)))
print("% error if predicting tomorrow's rain with today's rain: ", np.mean(np.abs(prcp_diff)))

# %%
