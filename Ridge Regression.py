import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt

file_path = r"C:\Users\192052\Desktop\Explainable Course\CS2_35.csv"
try:
    df = pd.read_csv(file_path)
except FileNotFoundError:
    print(f"Error: The file at {file_path} was not found. Please check the path.")
    exit()

X = df[['CCCT', 'CVCT', 'R']]
y = df['capacity']

split_index = int(len(df) * 0.8)
X_train_raw, X_test_raw = X.iloc[:split_index], X.iloc[split_index:]
y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

test_indices = y_test.index

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train_raw)
X_test = scaler.transform(X_test_raw)

ridge_model = Ridge(alpha=0.1)
ridge_model.fit(X_train, y_train)

y_pred = ridge_model.predict(X_test)

print("--- Model Performance Metrics ---")

mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error (MSE): {mse:.4f}")

rmse = np.sqrt(mse)
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")

mae = mean_absolute_error(y_test, y_pred)
print(f"Mean Absolute Error (MAE): {mae:.4f}")

r2 = r2_score(y_test, y_pred)
print(f"R-squared (R²): {r2:.4f}")

print("\n--- Model Coefficients (Impact of each feature) ---")
feature_names = X.columns
coefficients = ridge_model.coef_

for feature, coef in zip(feature_names, coefficients):
    print(f"{feature}: {coef:.4f}")

plt.figure(figsize=(12, 6))
plt.plot(test_indices, y_test.values, label='True Capacity', marker='o', linestyle='-', markersize=5)
plt.plot(test_indices, y_pred, label='Predicted Capacity', marker='x', linestyle='--', markersize=5)
plt.xlabel("Original Sample Index (from DataFrame)")
plt.ylabel("Capacity")
plt.title("Ridge Regression: True vs. Predicted Capacity")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

results_df = pd.DataFrame({
    'Original Sample Index': test_indices,
    'True Capacity': y_test.values,
    'Predicted Capacity': y_pred
})

output_excel_path = r"C:\Users\192052\Desktop\Explainable Course\Ridge_Regression_Predictions_With_Index.xlsx"

results_df.to_excel(output_excel_path, index=False)
print(f"\nTrue and predicted values (with original sample index) saved to: {output_excel_path}")