import pandas as pd
import numpy as np

df = pd.read_csv(r"C:\Users\Tanmay\Desktop\C0D3\IML lab\Lab-06\dataset.csv")



df["date"] = pd.to_datetime(df["date"], errors="coerce")
df["Day_Number"] = (df["date"] - df["date"].min()).dt.days

feature_names = ["humidity", "wind_speed", "meanpressure", "Day_Number"]
X = df[feature_names].values
y = df["meantemp"].values


np.random.seed(42)
indices = np.arange(len(X))
np.random.shuffle(indices)

X = X[indices]
y = y[indices]

split_index = int(0.8 * len(X))
X_train = X[:split_index]
X_test = X[split_index:]
y_train = y[:split_index]
y_test = y[split_index:]


mean_train = X_train.mean(axis=0)
std_train = X_train.std(axis=0)
std_train[std_train == 0] = 1

X_train_scaled = (X_train - mean_train) / std_train
X_test_scaled = (X_test - mean_train) / std_train


ones_train = np.ones((X_train_scaled.shape[0], 1))
X_train_b = np.hstack((ones_train, X_train_scaled))


B = np.linalg.pinv(X_train_b) @ y_train

print("Intercept and coefficients:")
print(B)

ones_test = np.ones((X_test_scaled.shape[0], 1))
X_test_b = np.hstack((ones_test, X_test_scaled))
y_hat = X_test_b @ B

#evaluation 
ss_total = np.sum((y_test - y_test.mean()) ** 2)
ss_residual = np.sum((y_test - y_hat) ** 2)
r2 = 1 - (ss_residual / ss_total)
mse = np.mean((y_test - y_hat) ** 2)
rmse = np.sqrt(mse)
mae = np.mean(np.abs(y_test - y_hat))

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))
print("R2 Score:", r2)
print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)
print("Features used:", feature_names)

"""
new_sample = np.array([80, 3.5, 1015, 10])
new_sample_scaled = (new_sample - mean_train) / std_train
new_x = np.hstack(([1], new_sample_scaled))
y_pred = new_x @ B
print("Predicted value:", y_pred)"""

