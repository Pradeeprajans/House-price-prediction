# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.model_selection import GridSearchCV
import warnings

warnings.filterwarnings('ignore')

# Import the dataset
df = pd.read_csv("data.csv")

# Display dataset information
print(df.head())
print(df.info())

# Drop unwanted columns
df.drop("id", axis=1, inplace=True)

# Define the target variable and features
X = df.drop(columns=["SalePrice"])
y = df["SalePrice"]

# Split the dataset into training and test sets
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Identify numerical and categorical columns
num_cols = [c for c in X.columns if X[c].dtype in ['float64', 'int64']]
ohe_cols = [c for c in X.columns if X[c].dtype == 'object']

# Handle missing values and scaling for numerical columns
numeric_imputer = IterativeImputer(max_iter=100, random_state=42)
x_train_numeric = numeric_imputer.fit_transform(x_train[num_cols])
x_test_numeric = numeric_imputer.transform(x_test[num_cols])

scaler = MinMaxScaler()
x_train_numeric = scaler.fit_transform(x_train_numeric)
x_test_numeric = scaler.transform(x_test_numeric)

# Handle missing values and one-hot encoding for categorical columns
categorical_imputer = SimpleImputer(strategy="constant", fill_value="Unknown")
x_train_categorical = categorical_imputer.fit_transform(x_train[ohe_cols])
x_test_categorical = categorical_imputer.transform(x_test[ohe_cols])

encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)
x_train_categorical = encoder.fit_transform(x_train_categorical)
x_test_categorical = encoder.transform(x_test_categorical)

# Convert back to DataFrame
x_train_numeric = pd.DataFrame(x_train_numeric, columns=num_cols)
x_test_numeric = pd.DataFrame(x_test_numeric, columns=num_cols)
x_train_categorical = pd.DataFrame(x_train_categorical, columns=encoder.get_feature_names_out(ohe_cols))
x_test_categorical = pd.DataFrame(x_test_categorical, columns=encoder.get_feature_names_out(ohe_cols))

# Concatenate processed numerical and categorical features
x_train_processed = pd.concat([x_train_numeric, x_train_categorical], axis=1)
x_test_processed = pd.concat([x_test_numeric, x_test_categorical], axis=1)

# Apply SMOTE to balance the dataset (if needed)
# smote = SMOTE(random_state=42)
# x_train_smote, y_train_smote = smote.fit_resample(x_train_processed, y_train)

# Model: Gradient Boosting Regressor
final_model = GradientBoostingRegressor(learning_rate=0.1, n_estimators=100, max_depth=7, random_state=42)
final_model.fit(x_train_processed, y_train)

# Save the final model to a pickle file
with open("house_price_gradient_boosting_model.pkl", "wb") as f:
    pickle.dump(final_model, f)

print("Model saved to house_price_gradient_boosting_model.pkl")

# Model Evaluation: Predict on the test set
y_pred = final_model.predict(x_test_processed)

# Calculate evaluation metrics
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R²: {r2:.2f}")

# Hyperparameter tuning using GridSearchCV for Ridge regression as an example
ridge_params = {'alpha': np.linspace(0.1, 10, 50)}
ridge_search = GridSearchCV(Ridge(), ridge_params, scoring="neg_mean_squared_error", cv=5, verbose=3)
ridge_search.fit(x_train_processed, y_train)

# Best model from grid search
ridge_best = ridge_search.best_estimator_
ridge_preds_tuned = ridge_best.predict(x_test_processed)

# Evaluate the tuned Ridge model
mse_ridge = mean_squared_error(y_test, ridge_preds_tuned)
rmse_ridge = np.sqrt(mse_ridge)
r2_ridge = r2_score(y_test, ridge_preds_tuned)

print(f"\nTuned Ridge Regression Performance:")
print(f"  MSE: {mse_ridge:.2f}")
print(f"  RMSE: {rmse_ridge:.2f}")
print(f"  R²: {r2_ridge:.2f}")

# Save the best Ridge model
with open("house_price_ridge_model.pkl", "wb") as f:
    pickle.dump(ridge_best, f)

print("Tuned Ridge Model saved to house_price_ridge_model.pkl")

# Feature Importance from the GradientBoostingRegressor
feature_importance = final_model.feature_importances_

# Create a DataFrame to display the feature importances
feature_importance_df = pd.DataFrame({
    "Feature": x_train_processed.columns,
    "Importance": feature_importance
})

# Sort by importance
feature_importance_df = feature_importance_df.sort_values(by="Importance", ascending=False)

# Plot Feature Importance
plt.figure(figsize=(10, 8))
sns.barplot(x="Importance", y="Feature", data=feature_importance_df)
plt.title("Feature Importance - Gradient Boosting Regressor")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.show()

# Save the preprocessors for future use
preprocessors = {
    'numeric_imputer': numeric_imputer,
    'scaler': scaler,
    'categorical_imputer': categorical_imputer,
    'encoder': encoder
}

with open('house_price_preprocessors.pkl', 'wb') as f:
    pickle.dump(preprocessors, f)

print("Preprocessors saved to house_price_preprocessors.pkl")
