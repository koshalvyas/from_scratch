from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import linreg
import numpy as np



# Load
data = load_diabetes()
X, y = data.data, data.target

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# IMPORTANT: Linear Regression is sensitive to feature scaling!
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

lr = 0.01
W,b = linreg.train(X_train, y_train, lr, 2000)
y_pred = linreg.predict(X_test, W, b)

# Mean Squared Error
mse = np.mean((y_pred - y_test)**2)

# R² Score (how well the model explains variance)
ss_res = np.sum((y_test - y_pred)**2)
ss_tot = np.sum((y_test - np.mean(y_test))**2)
r2 = 1 - (ss_res / ss_tot)

print(f"MSE  : {mse:.4f}")
print(f"R²   : {r2:.4f}")



sk_model = LinearRegression()
sk_model.fit(X_train, y_train)
sk_pred = sk_model.predict(X_test)
sk_mse = np.mean((sk_pred - y_test)**2)
sk_r2 = 1 - (np.sum((y_test - sk_pred)**2) / np.sum((y_test - np.mean(y_test))**2))

print(f"\n--- sklearn benchmark ---")
print(f"MSE  : {sk_mse:.4f}")
print(f"R²   : {sk_r2:.4f}")