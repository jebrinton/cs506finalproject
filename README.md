
# **CS506 Final Project Proposal**  
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
