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
dataset.apply(pd.isnull).sum()/dataset.shape[0]

# Since the "PRCP" column has missin values, we fill the missing values with zero since
# if there was no PRCP recorded for that day then that must mean that it did not rain therefore PRCP would be zero
dataset["PRCP"] = dataset["PRCP"].fillna(0)

dataset.apply(pd.isnull).sum() / dataset.shape[0]

# Fill in NaN
df = df.interpolate(method='linear')

# get the average difference in max tempurature between days
print(df)
# %%
diff = np.diff(df['TMAX'].values)

print("MAE if predicted tomorrow's max with today's max: ", diff.sum().round(4))
