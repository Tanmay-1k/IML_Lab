#WAP to implement linear regression model on the given dataset also find the accuracy , f1 and plot the ROC curve

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, f1_score, roc_auc_score, roc_curve

df = pd.read_csv(
    r"Lab-03\daily-minimum-temperatures-in-me.csv",
    na_values=['?']
)

df['Daily minimum temperatures'] = pd.to_numeric(
    df['Daily minimum temperatures'], errors='coerce'
)

df.dropna(inplace=True)

df['Date'] = pd.to_datetime(df['Date'])
df['Day'] = df['Date'].dt.day

X = df[['Day']]
y = df['Daily minimum temperatures']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

r2 = r2_score(y_test, y_pred)
print("R2:", r2)

y_test_cls = np.where(y_test >= 10, 1, 0)
y_pred_cls = np.where(y_pred >= 10, 1, 0)

f1 = f1_score(y_test_cls, y_pred_cls)
print("F1:", f1)

roc_auc = roc_auc_score(y_test_cls, y_pred)
print("ROC AUC:", roc_auc)

fpr, tpr, _ = roc_curve(y_test_cls, y_pred)

plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()
