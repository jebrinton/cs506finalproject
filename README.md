
# **CS506 Final Project Proposal**  (Please scroll below to view the Midterm Report which was due 03/31)
## **Team Members**  
- **Afitab Iyigun**  
- **Jacob Brinton**  
- **Bhimraj Bhuller**
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
- **NOAA Daily Meteorological Summaries** (Global major station data)  
- **Copernicus Climate Data**  
- **NASA Climate Data**  
- **Berkeley Earth Climate Data** (Records dating back to 1750)  
- **ECMWF Reanalysis Data** (Hourly climate data for temperature, wind, and humidity from 1950 onward)  

### **Key Features of the Dataset**  
- **Temperature:** Maximum & minimum (°F or °C)  
- **Precipitation:** Rainfall (mm or inches)  
- **Snowfall & Snow Depth:** Measured in mm or inches  
- **Wind Speed:** Average daily speed (m/s or mph)  
- **Cloudiness:** Recorded via ceilometer and manual observations  

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
  - **Deep Learning (CNN-based models) for severe weather events (e.g., storms, heatwaves, floods)**  
- **Comparative Analysis:**  
  - Evaluate model performance (e.g., **ARIMA vs. XGBoost**) to determine the most effective forecasting approach.  

---

## **Data Visualization & User Interaction**  
### **Geospatial Representation**  
- **Interactive Maps:**  
  - Users select an **origin city** for prediction.  
  - A **slider control** allows users to visualize predictions over different time horizons.  
  - Animated visualizations will illustrate **temperature trends** (cooling/warming effects).  
- **Twin City Identification:**  
  - The model will highlight **the city with the most similar predicted future climate**.  
  - Option to shade an entire **regional climate similarity zone** instead of a single city match.  

---

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

# Midterm Report (Updated on 03/31)

## Introduction

This midterm report details the methodology, processes, and results of our weather prediction model. Our model attempts to forecast maximum daily temperature (`TMAX`) and precipitation (`PRCP`). Our analysis includes data cleaning, feature engineering, training of various ML models, and performance evaluation across different time horizons.  

---

## Data Overview

Our dataset comprises daily weather records in Boston, indexed by date. The dataset contains three main variables:

- **TMAX**: Maximum daily temperature
- **TMIN**: Minimum daily temperature
- **PRCP**: Precipitation (rainfall) in mm  

---

## Data Preprocessing and Cleaning

We implemented the following preprocessing steps:

- **Handling missing values**: Columns with more than 1% missing data were removed, and missing values in `PRCP` were replaced with 0.
- **Handling of date**: The `DATE` column was converted to datetime format and set as the index.
- **Feature correlation analysis**: We observed a high correlation between `TMAX` and `TMIN`. Additionally, `TMAX` was strongly correlated with `tmrw_temp`, justifying its predictive relevance.
- **Creation of response variables** (with scaling applied where necessary for model stability):
  - `tmrw_temp`: Maximum temperature of the next day (response variable for regression), created using a one-day shift in `TMAX`.
  - `tmrw_rain`: A probability value between 0 and 1 indicating the likelihood of rain occurring the next day. It was converted to a binary indicator (0 or 1) where 0 represents no rain and 1 represents rain.
- **Brief visualization**: We created time-series plots for `TMAX` and `PRCP` over a 10-year period.

---

## Model Selection and Training

We implemented two types of predictive models: a regression model for `tmrw_temp` and a classification model for `tmrw_rain`.

- **Regression Models for `tmrw_temp`**:
  - Ridge Regression
  - K-Nearest Neighbors (KNN) Regression
  - Random Forest (RF) Regression
  - XGBoost (XGB) Regression
  - Gradient Boosting Regression (GBR)
  
- **Classification Models for `tmrw_rain`**:
  - Ridge Regression
  - K-Nearest Neighbors (KNN) Classifier
  - Random Forest Classifier
  - XGBoost Classifier
  - Gradient Boosting Classifier

---

## Backtesting and Performance Evaluation + Visualization

We tested model performance over different time horizons: 10-year and 30-year data samples.

- **Mean Absolute Error (MAE) for Temperature Prediction**:
  - Random Forest, XGBoost, and Gradient Boosting performed best.
  - Increasing training data from 10 years to 30 years improved accuracy.

- **Rain Prediction Accuracy Evaluation**:
  - Random Forest, XGBoost, and Gradient Boosting performed best.
  - We created a time-series step plot comparing predicted and actual rain occurrences between Jan 2024 and Jan 2025.

- **Additional Visualization**:
  - MAE bar charts comparing model performance for 10-year and 30-year training periods.

---

## Possible Extensions

To further improve our project, we plan to:

- Include additional meteorological features such as humidity and wind speed to enhance predictive accuracy.
- Explore using Singular Value Decomposition (SVD) to create feature combinations, as interpretability is not our primary focus.



