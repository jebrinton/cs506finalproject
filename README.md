# **CS506 Final Project Proposal**  
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
Our project aims to develop a predictive model for weather and climate change in a given city. Additionally, the model will identify a "Twin City"—a city with similar weather and climate patterns.  

### **Goals**  
- Utilize historical climate data to make **long-term weather predictions**.  
- Explore and compare various **statistical and machine learning (ML) models** for forecasting.  
- Evaluate model accuracy and incorporate feedback from **weather experts at Boston University**.  

---

## **Data Collection & Preparation**  

### **Primary Data Sources**  
- The primary data source is the **Boston 1970-2025 Daily Summaries Dataset**. 
- Initially, we engaged with the NOAA Daily Meteorological Summaries database, focusing on global major station daily weather summaries. However, we encountered challenges related to data handling and processing when utilizing station-based datasets. To refine our approach and enhance our model training and testing processes, we have decided to shift our focus toward a city-specific daily summary dataset, beginning with Boston.

### **Key Features of the Dataset**  
- **Temperature:** Maximum & minimum (°F or °C)  
- **Precipitation:** Rainfall (mm or inches)  
- **Snowfall & Snow Depth:** Measured in mm or inches  
- **Wind Speed:** Average daily speed (m/s or mph)  
- **Cloudiness:** Recorded via ceilometer and manual observations  

### **Data Cleaning**
- The DATE column was converted to datetime format and fixed as the index.
- All other non-numeric columns (e.g. Station Name) were dropped.
- Dropping columns that have less that 99% of data and filling in NaNs in by using context from adjacent days (linear interpolation).


## **Maximum Temperature Regression and Precipitation Classification**

### **Response Variable Creation**
*Baseline Question:* Can we predict tomorrow's weather with today's weather? Can we predict tomorrow's max temperature or precipitation with today's data?
- **tmrw_temp:** Maximum temperature of the next day (regression based), created by shifting current day's maximum temperature. 
- **tmrw_rain:** Binary indicator (0 or 1) for rain occurrence the next day (classification based).
 
### **Model Selection and Performance**
Implemented two types of predictive models: a regression and classification model. 
- Predicting **tmrw_temp**, employed Ridge Regression, K-Nearest Neighbours (KNN) Regression, Random Forest regression, XGBoost (XGB) Regression, Gradient Boosting (GB) Regression. 
- Predicting **tmrw_rain**, employed Logistic Regression, KNN Classifier, Random Forest Classifier, XGB Classifier, and Gb Classifier.

### **Backtesting and Performance Evaluation**

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


## **SARIMA Time Series**

### **ARIMA Model Architecture**

We used statsmodels' SARIMAX model. To explain SARIMAX, the following is a break down of what ARIMA models are:

1. The *AR* stands for the autoregressive order; it is the number of past values we should take into account when predicting each new value. 
2. The integrated part (*I*) is used to take out noise: it forces the time series to become stationary and subtract out anomalies. 
3. The *MA*, or moving average part, looks at how the present value is related to past errors.

### **Adding Seasonality and Exogenous Variables**

The issue with ARIMA for this is that it doesn't take into account seasonality. To account for this:

We add the *S* of SARIMAX - a new set of the 3 ARIMA parameters, except they take effect based off the number of steps in a season, the 4th parameter. (So we are now at 7 hyperparameters).
The reason we have an *X* in SARIMAX is due to the eXogenous features such as precipitation or windspeed—features related to what we care about, but they will just aid, we won't actually predict them.

### **Model Training and Performance**

The SARIMAX model was applied to a dataset spanning several decades with the following training and prediction periods. Then, the MAE was calculated for the real temperature data. 

1. Training on 10 years and predicting the next 30 years, the MAE was 20.65.
2. Training on 20 years and predicting the next 20 years, the MAE was 18.23.
3. Training on 30 years and predicting the next 10 years, the MAE was 12.32.
4. Training on 39 years and predicting the next year, the MAE was 6.10.
5. Training on 39 years and predicting a tenth of a year (0.1), the MAE was 2.07.

### **Observations and Adjustments**
Forecast Accuracy: wE observed that the forecast accuracy decreases as the prediction horizon increases. This degradation in performance over longer forecast periods can be attributed to the increasing uncertainty and potential changes in underlying patterns over time.

Leap Years: The model might need adjustments to accommodate the effects of leap years in long-term forecasts, as these add an extra day periodically that could slightly alter seasonal patterns.
