# CS506 Final Project
## **Team Members**  
- **Afitab Iyigun** 
- **Jacob Brinton** 
- **Bhimraj Bhuller** 
- **Badr Qattan**
- **Wen Zhang**  

## **Project Overview**  

### Presentations Video ###
Youtube Link: https://youtu.be/k0vcuHoykxI

### **Objective**  
Our project aims to develop a predictive model for weather and climate change for 6 cities around the world, and compare our models' performances for the 6 cities.

### **Goals**  
- Utilize historical climate data to make **long-term weather predictions**.  
- Explore and compare various **statistical and machine learning (ML) models** for forecasting.  
- Evaluate model accuracy and incorporate feedback from **weather experts at Boston University**.

### Timeline

We initially started off forecasting the maximum temperature and precipitation in Boston using classification and regression models, guessing tomorrow’s max temperature and precipitation using today’s temperature and rain indicator. *(Section Previous Models: Maximum Temperature Regression and Precipitation Classification)* However, we soon discovered that these models were not performing great as they failed to account for a critical component—seasonality. With such regression and classification models, you can imagine if we were forced to extend our predictions from tomorrow’s temperature and rain to next week’s and next month’s, they would become more and more divergent from the actual weather.

Hence, to account for seasonality, we used a time series model: SARIMA, which stands for Seasonal Auto-Regressive Integrated Moving Average. *(Section Previous Models: SARIMA Time Series)* 

To improve our prediction accuracies, we created a feedforward neural network by adding seasonality factor with engineered features. *(Section Final Model: Neural Network Temperature Forecasting)* 

---

## **Data Collection & Preparation**  

### **Primary Data Sources**  
- The primary data source is **cdc.noaa.gov/cdo-web/search**.
- All 6 cities were retrieved from the above link, and combined into one .csv file.
- The 6 cities are: Boston, Buenos Aires, Darwin, New York, Madrid, Vladivostok.
  
### **Key Features of the Dataset**  
- **Temperature:** Maximum (°F)  
- **Precipitation:** Rainfall (mm or inches)
  
### **Data Cleaning** 
- The DATE column was converted to datetime format and fixed as the index.
- All other non-numeric columns (e.g. Station Name) were dropped.
- Dropping columns that have less that 99% of data and filling in NaNs in by using context from adjacent days (linear interpolation).

## Previous Models

### **Maximum Temperature Regression and Precipitation Classification**

#### **Response Variable Creation**
*Baseline Question:* Can we predict tomorrow's weather with today's weather? Can we predict tomorrow's max temperature or precipitation with today's data?
- **tmrw_temp:** Maximum temperature of the next day (regression based), created by shifting current day's maximum temperature. 
- **tmrw_rain:** Binary indicator (0 or 1) for rain occurrence the next day (classification based).
 
#### **Model Selection and Performance**
Implemented two types of predictive models: a regression and classification model. 
- Predicting **tmrw_temp**, employed Ridge Regression, K-Nearest Neighbours (KNN) Regression, Random Forest regression, XGBoost (XGB) Regression, Gradient Boosting (GB) Regression. 
- Predicting **tmrw_rain**, employed Logistic Regression, KNN Classifier, Random Forest Classifier, XGB Classifier, and Gb Classifier.

#### **Backtesting and Performance Evaluation**

**Backtest Function:** Backtesting is a way to verify that our future weather forecasting would be correct by validating it on past data. We performed backtesting on our dataset by taking in a temperature regression model and a rain classification model (as tmrw_rain is a binary indication of if it rains tomorrow or not). For each iteration, we trained our models on all preceding data and test on the next 90 days. 

**Parameter Finetuning:** For KNN Classifier and Regressor models, Performed exhuastive hyperparemeter tuning with **GridSearchCV**, which used cross-validation, to find the best n_neighbors. 

**Results:** Each of these methods resulted in pretty similar performance. We calculated the Mean Average Error (MAE) for each model.

| Model            | MAE (max temp) | MAE (rain) |
|------------------|-------|-----------|
| Ridge Regression | 6.231 | 0.349     |
| K-Nearest Neighbors | 6.616 | 0.360     |
| Random Forest | 6.267 | 0.364     |
| XGBoost | 6.093 | 0.345     |
| Gradient Boosting | 6.085 | 0.347     |

As mentioned above, the accuracy is pretty similar, with XGBoost and Gradient Boosting with the highest performance.


### **SARIMA Time Series**

#### **ARIMA Model Architecture**

We used statsmodels' SARIMA model. To explain SARIMA, the following is a break down of what ARIMA models are:

1. The *AR* stands for the autoregressive order; it is the number of past values we should take into account when predicting each new value. 
2. The integrated part (*I*) is used to take out noise: it forces the time series to become stationary and subtract out anomalies. 
3. The *MA*, or moving average part, looks at how the present value is related to past errors.

#### **Adding Seasonalitys**

The issue with ARIMA for this is that it doesn't take into account seasonality. To account for this:

We add the *S* of SARIMA - a new set of the 3 ARIMA parameters, except they take effect based off the number of steps in a season, the 4th parameter. (So we are now at 7 hyperparameters).
The reason we have an *X* in SARIMAX is due to the eXogenous features such as precipitation or windspeed—features related to what we care about, but they will just aid, we won't actually predict them.

#### **Model Training and Performance**

The SARIMA model was applied to a dataset spanning several decades with the following training and prediction periods. Then, the MAE was calculated for the real temperature data. 

1. Training on 10 years and predicting the next 30 years, the MAE was 20.65.
2. Training on 20 years and predicting the next 20 years, the MAE was 18.23.
3. Training on 30 years and predicting the next 10 years, the MAE was 12.32.
4. Training on 39 years and predicting the next year, the MAE was 6.10.
5. Training on 39 years and predicting a tenth of a year (0.1), the MAE was 2.07.

---

## Final Model: Neural Network Temperature Forecasting

### Model Architecture

This model uses a feedforward neural network (specifically `MLPRegressor` from `scikit-learn`) to forecast future daily maximum temperatures (`TMAX`). The model learns using historical data and cyclical date features to improve seasonal awareness.

#### Key Components:
1. **Lookback Window (30 days)**: The model uses the past 30 days of data to predict the next day.
2. **Date-Based Features**: To capture seasonality, four cyclical features are added:
   - sin(month), cos(month)
   - sin(day_of_year), cos(day_of_year)
3. **MLPRegressor Hyperparameters**:
   - Hidden layers (e.g., (256, 128))
   - Regularization strength (`alpha`)
   - Learning rate (`learning_rate_init`)

---

### Adding Seasonality with Engineered Features

While MLP does not inherently model seasonality like SARIMA, we incorporated periodic patterns using cyclical transformations of month and day-of-year. These served as proxies for seasonal cycles, helping our model distinguish between, say, July and January.

---

### Model Training and Performance

The model was trained on 45 years of historical data and evaluated on the most recent 10 years using a grid search across multiple hyperparameters for each city out of the 6 cities.

Here is how the model performed for each city:

- Boston : MAE=7.2285
- Buenos Aires : MAE=5.3350
- Darwin: MAE=2.3335
- New York: MAE=5.9301
- Madrid : MAE=5.8432
- Vladivostok : MAE=6.6314

Conclusion: our model performed best for the city of Darwin, since the TMAX does not flucuate as much as the other cities we experimented on.

---

### Future Forecasting

Using the best model from above, we retrained on the **entire historical dataset** and forecasted daily maximum temperatures for the next **20 years**.

- **Recursive Forecasting**: Predictions are made day-by-day, using each new prediction as part of the next input.
- **Future Features**: Future sin/cos month and day values were precomputed to guide the network in handling seasonal context.

---

### Output

- **Plot**: Displays full temperature history + forecasted values until the year 2045.
- **CSV**: Forecasted temperatures saved in `NN_Predictions.csv`.

---

### Observations and Adjustments

- **Forecast Accuracy**: Unlike SARIMA, the NN doesn’t degrade drastically with long-term forecasts due to its use of engineered cyclical features and deep learning flexibility.
- **Training Stability**: Early stopping prevents overfitting during training.
- **Leap Years**: Not explicitly handled, though cyclical features may indirectly account for this.

---

## Reproducibility

Run `pip install -r requirements.txt` on the terminal

Then run `python NN_6cities.ipynb`

## Link

https://youtu.be/JMkNOV8DOMc

---
