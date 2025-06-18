# Smart Agriculture System using IoT and Machine Learning

This project implements a Smart Agriculture System that uses IoT sensors and Machine Learning models to intelligently predict whether irrigation is needed, based on real-time environmental data like temperature, humidity, soil moisture, and motion detection. The system aims to optimize water usage and promote sustainable farming practices.

# Key Features

  Real-time data collection using IoT sensors

  Predicts irrigation needs using ML models

  Compares performance of multiple classifiers

  Visualizations: confusion matrix, decision tree, feature importance

  Scalable and suitable for automation in smart farms

# Dataset Used

Filename: Smart_agriculture_dataset.xlsx

# Features:

Temperature (°C)

Humidity (%)

SoilMoisture (scaled)

Motion_Detected (0 or 1)


# Target:

Irrigation_Needed (1 = Yes, 0 = No)

Source: Simulated/collected sensor data representing agricultural field conditions.


# Technologies Used

Programming Language: Python

# Libraries:

pandas, numpy – Data handling

matplotlib, seaborn – Visualization

scikit-learn – ML models and evaluation metrics


# Hardware (for IoT implementation):

NodeMCU / ESP8266

DHT11 / DHT22 (Temperature + Humidity)

Soil Moisture Sensor

PIR Motion Sensor

Relay Module + Water Pump

# Step-by-Step Process

1. Data Collection

Simulated IoT sensor values collected in Excel format

Features: Temperature, Humidity, SoilMoisture, Motion_Detected


2. Data Preprocessing

Selected relevant features

Split into train-test sets using train_test_split

Applied StandardScaler for Logistic & Linear Regression


3. Model Building

Trained the following models:

Logistic Regression

Naive Bayes

Decision Tree

Linear Regression (for probability estimation)


Used accuracy_score, classification_report, confusion_matrix for evaluation


4. Visualization

Confusion Matrix heatmaps for classification models

Decision Tree diagram

Feature Importance bar chart

Line plot of Actual vs Predicted (Linear Regression)


5. IoT Integration (Prototype Concept)

Data can be collected from physical sensors connected to NodeMCU

Prediction results can be used to automatically trigger irrigation via relay and motor


# Model Evaluation Summary

Model	Accuracy (%)	Remarks

Logistic Regression	~78%	Good baseline model
Naive Bayes	~87%	Fast and simple, less interpretive
Decision Tree	~98%	Highly interpretable, very accurate
Linear Regression	- (uses MSE/R²)	Used for probabilistic prediction.

# Visualizations Included

✅ Confusion Matrix Heatmaps

🌳 Decision Tree Plot

📊 Feature Importance Bar Graph

📈 Actual vs Predicted Line Plot (Linear Regression)


# Applications

Precision irrigation control

Water resource optimization

Smart greenhouse management

Real-time smart farming dashboard

# Team Roles

Name	Responsibility

Reena S	IoT hardware setup and sensor integration
Deebeka M	Data collection, cleaning, and preprocessing
Janani R	Machine Learning model development and testing
Nandhini S S	Visualization, evaluation, and result interpretation
Deepika G H	Report writing and presentation preparation


# References

Scikit-learn Documentation

“Deep Learning with Python” by F. Chollet

“Smart Farming: A Review on IoT-based Technologies,” IJACSA, 2020

Raspberry Pi IoT Projects


# How to Run the Project

1. Clone or download the code and dataset.

2. Ensure all required Python libraries are installed (pip install -r requirements.txt).

3. Run the script (.py or Jupyter notebook).

4. View the printed metrics and generated plots.


