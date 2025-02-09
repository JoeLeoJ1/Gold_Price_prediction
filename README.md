# Gold_Price_prediction

This project demonstrates the use of machine learning models to predict gold prices based on historical data. The code leverages various libraries like pandas, scikit-learn, and XGBoost for data preprocessing, model training, and evaluation.

## Table of Contents

1.  [Introduction](#introduction)
2.  [Dataset](#dataset)
3.  [Data Preprocessing](#data-preprocessing)
4.  [Model Training](#model-training)
5.  [Model Evaluation](#model-evaluation)
6.  [Feature Importance](#feature-importance)
7.  [Model Saving](#model-saving)
8.  [Dependencies](#dependencies)
9.  [Usage](#usage)
10. [Contributing](#contributing)
11. [License](#license)

## Introduction

Gold prices are influenced by various factors such as economic conditions, geopolitical events, and market sentiment. This project aims to build a predictive model using historical gold price data and related features.

## Dataset

The project utilizes a dataset named "gold_price_data.csv" containing historical gold prices and relevant features. The dataset includes columns like:

*   Date
*   SPX (S&P 500 index)
*   GLD (Gold price)
*   USO (United States Oil Fund)
*   SLV (Silver price)
*   EUR/USD (Euro to US Dollar exchange rate)

## Data Preprocessing

The code performs the following data preprocessing steps:

1.  **Loading the Dataset:** Reads the "gold_price_data.csv" file into a pandas DataFrame.
2.  **Data Exploration:** Prints information about the dataset, including data types and missing values.
3.  **Correlation Analysis:** Calculates and visualizes correlations between features using a heatmap.
4.  **Data Cleaning:** Drops irrelevant columns (e.g., SLV) and sets the "Date" column as the index.
5.  **Data Visualization:** Plots the gold price trend over time and the distribution of data across columns.
6.  **Data Transformation:** Applies transformations (e.g., square root) to address skewness in features.
7.  **Outlier Handling:** Identifies and handles outliers using percentiles.
8.  **Data Splitting:** Splits the data into training and testing sets for model training and evaluation.
9.  **Feature Scaling:** Scales features using StandardScaler to ensure consistent ranges.
10. **Missing Value Imputation:** Imputes missing values using SimpleImputer with the mean strategy.


## Model Training

The code trains three different machine learning models:

1.  **Lasso Regression:** A linear regression model with L1 regularization. Uses GridSearchCV to find the optimal alpha parameter.
2.  **Random Forest:** An ensemble learning method that combines multiple decision trees. Uses GridSearchCV to find the optimal hyperparameters (n_estimators and max_depth).
3.  **XGBoost:** A gradient boosting algorithm known for its high accuracy. Trains the model using default parameters.


## Model Evaluation

The code evaluates the performance of each model using the R-squared metric on both the training and testing data. The R-squared score measures the goodness of fit of the model.


## Feature Importance

The code calculates and visualizes the importance of each feature in the Random Forest model. This helps understand which features contribute most to the prediction of gold prices.


## Model Saving

The trained XGBoost model is saved to a file named "model.pkl" using the pickle library. This allows for reusing the model without retraining.



## Dependencies

The code requires the following libraries:

*   numpy
*   pandas
*   seaborn
*   matplotlib
*   scikit-learn
*   xgboost
*   eli5
*   pickle

You can install them using pip:
