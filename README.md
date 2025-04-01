
# **CS506 Final Project Proposal**  
## **Team Members**  
- **Afitab Iyigun** (afitab@bu.edu) 
- **Jacob Brinton** (jbrin@bu.edu)
- **Bhimraj Bhuller** ()
- **Badr Qattan**
- **Wen Zhang**  

## **Project Overview**  
### **Objective**  
Our project aims to develop a predictive model for weather and climate change in a given city. Additionally, the model will identify a "Twin City"—a city with similar weather and climate patterns.  

### **Goals**  
- Utilize historical climate data to make **long-term weather predictions**.  
- Explore and compare various **statistical and machine learning (ML) models** for forecasting.  
- Evaluate model accuracy and incorporate feedback from **weather experts at Boston University**.  

---

## **Data Collection & Preparation**  
### **Primary Data Sources**  
We will leverage datasets from reputable sources, including:  
- 
*(Note: Initially, we engaged with the NOAA Daily Meteorological Summaries database, focusing on global major station daily weather summaries. However, we encountered challenges related to data handling and processing when utilizing station-based datasets. To refine our approach and enhance our model training and testing processes, we have decided to shift our focus toward a city-specific daily summary dataset, beginning with Boston.)*
- ****

### **Key Features of the Dataset**  
- **Temperature:** Maximum & minimum (°F or °C)  
- **Precipitation:** Rainfall (mm or inches)  
- **Snowfall & Snow Depth:** Measured in mm or inches  
- **Wind Speed:** Average daily speed (m/s or mph)  
- **Cloudiness:** Recorded via ceilometer and manual observations  

### **Data Cleaning**
- The DATE column was converted to datetime format and fixed as the index. 
- Dropping columns that have less that 97% of data and filling in NaNs. Specifically for precipitation data, filling in NaN values that correspond to no rain with 0. 
- 

## **Maximum Temperature Regression and Precipitation Classification **

### **Response Variable Creation**
- **tmrw_temp:** Maximum temperature of the next day (regression based), created by shifting current day's maximum temperature. 
- **tmrw_rain:** Binary indicator (0 or 1) for rain occurrence the next day (classification based).
 
### **Model Selection and Performance**
Implemented two types of predictive models: a regression and classification model. 
- Predicting **tmrw_temp**, employed Ridge Regression, K-Nearest Neighbours (KNN) Regression, Random Forest regression, XGBoost (XGB) Regression, Gradient Boosting (GB) Regression. 
- Predicting **tmrw_rain**, employed Logistic Regression, KNN Classifier, Random Forest Classifier, XGB Classifier, and Gb Classifier. 

Performed exhuastive hyperparemeter tuning with **GridSearchCV**, which used cross-validation, to find the best n_neighbors for KNN Classifier and Regressor models. 

### **Backtesting and Performance Evaluation**


### **SARIMA Time Series**
We used statsmodels' SARIMAX model. To explain SARIMAX, perhaps it would be first helpful to break down what ARIMA models are:

The *AR* stands for the autoregressive order; it is the number of past values we should take into account when predicting each new value. The integrated part (*I*) is used to take out noise: it forces the time series to become stationary and subtract out anomalies. The *MA*, or moving average part, looks at how the present value is related to past errors.

The issue with ARIMA for this is that it doesn't take into account seasonality. For that, we add the *S* of SARIMAX; this adds a new set of the 3 ARIMA parameters, except they take effect based off the number of steps in a season, the 4th parameter. (So we are now at 7 hyperparameters). The reason we have an *X* in SARIMAX is due to eXogenous features such as precipitation or windspeed—features related to what we care about, but they will just aid, we won't actually predict them.

The cleaning process for SARIMAX was much of the same as described previously. We split the data into a training set of 30 years and forecasted 5 years. Then, the mean average error was calculated for the real temperature data 

Training on 39 years and predicting the next year, the MAE was 6.10.
Training on 30 years and predicting the next 10 years, the MAE was 12.32.
Training on 10 years and predicting the next 30 years, the MAE was 20.65.

The seasons start to get off while forecasting larger amounts of time. One small issue that might been needed to get resolved is the effect of leap years. Mainly, though, this is to be expected.

---

## **Data Modeling**  
### **Model Architecture & Learning Approach**  
We will explore multiple forecasting techniques, including:  
- **Autoregressive Integrated Moving Average (ARIMA):**  
  - **p:** Number of past observations  
  - **d:** Differencing to remove trends  
  - **q:** Moving average  
- **Machine Learning Models:**  
  - **XGBoost** and other ML techniques introduced in class  


## **Data Visualization & User Interaction**  
### **Geospatial Representation**  


## **Test Plan & Model Evaluation**  
### **Training Strategy**
- We will train the model on the first 80% of the years, holding out 20% for testing.
- On the current dataset we're looking at, this means we will use the first 8 years for training and 2 years for testing.
- We will implement **weighted training**, giving **recent data** higher importance in prediction accuracy.

### **Evaluation Metrics**  
- **Time Series Plots:** Compare actual vs. predicted values.  
- **Uncertainty Bands:** Visualize **confidence intervals** for forecast reliability.  
- **Performance Analysis:** Use **statistical significance testing** to assess model robustness.  

---
