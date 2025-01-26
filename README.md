# House Price Prediction Project
## Overview
This project aims to predict the sale price of houses based on various features using machine learning algorithms. The model is built and evaluated using multiple regression techniques and optimized for accuracy. The project utilizes several libraries such as Scikit-learn, XGBoost, CatBoost, and more for data preprocessing, model training, evaluation, and feature importance analysis.

## Steps in the Project
### 1. Data Import and Initial Exploration
The dataset is loaded from a CSV file named data.csv.
Initial exploration of the data includes displaying the first and last rows, checking the data types, and identifying missing values and duplicates.
### 2. Data Preprocessing
The dataset is preprocessed by handling missing values, encoding categorical features, and scaling numerical features.
Categorical columns are encoded using LabelEncoder, and missing values for numerical columns are imputed with the mean using SimpleImputer.
Numerical features are standardized using StandardScaler.
### 3. Feature Engineering
Feature importance is derived from models like Random Forest and XGBoost to identify key predictors for house prices.
Visualizations are provided for feature importance, skewness, and kurtosis of numerical features to understand their distribution.
### 4. Model Training and Evaluation
The project explores multiple models, including:
Linear Regression
Ridge Regression
Lasso Regression
Random Forest Regressor
CatBoost Regressor
XGBRegressor
GridSearchCV is used for hyperparameter tuning to find the optimal model configuration.
Models are evaluated based on metrics such as MSE, RMSE, and R².
### 5. Model Saving
The best-performing models (Gradient Boosting Regressor, Ridge, Random Forest, etc.) are saved as pickle files for future use.
### 6. Feature Importance Visualization
Feature importance is plotted to identify which features contribute the most to the prediction of house prices.
### 7. Hyperparameter Tuning
Hyperparameter tuning is performed using GridSearchCV for models like Ridge, Lasso, Random Forest, CatBoost, and XGBoost to enhance their performance.
### 8. Final Model
The final models, after tuning, are saved in pickle files:
house_price_gradient_boosting_model.pkl
house_price_ridge_model.pkl
Preprocessors in house_price_preprocessors.pkl
### 9. Model Performance
After training, the models are evaluated on a test set, and their performance is displayed. Below are the results:
Gradient Boosting Regressor - Performance on test set evaluated using MSE, RMSE, and R².
Ridge Regression - Performance of the best-tuned model.
Other models are similarly evaluated and compared.
### 10. Feature Importance Analysis
Visualizations are created for feature importance, providing insights into which features most impact the prediction.
## Libraries Used
Pandas and NumPy: For data manipulation and numerical operations.
Scikit-learn: For data preprocessing, model building, and evaluation.
Matplotlib and Seaborn: For visualizing data and results.
XGBoost and CatBoost: For advanced boosting techniques to improve prediction accuracy.
Imbalanced-learn: For handling class imbalances using SMOTE if needed.

## Installation
Ensure you have the necessary libraries installed:

pip install pandas numpy scikit-learn xgboost catboost imbalanced-learn matplotlib seaborn

## Conclusion
This project successfully builds a house price prediction model that can be used for forecasting future housing prices based on various factors. The project also demonstrates the power of different machine learning algorithms, preprocessing techniques, and model evaluation strategies to obtain accurate predictions.
