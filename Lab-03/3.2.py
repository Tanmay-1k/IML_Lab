#WAP to implement multiple linear regression model and find out the accuracy , f1 and roc curve.Also compare the performance of linear regression model and multiple linear regression model
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv(r"Lab-03\daily-minimum-temperatures-in-me.csv")

df['Daily minimum temperatures'] = pd.to_numeric(
    df['Daily minimum temperatures'], errors='coerce'
)
df.dropna(inplace=True)

df['Date'] = pd.to_datetime(df['Date'])
df['Day'] = df['Date'].dt.day

x = df['Day'].values
y = df['Daily minimum temperatures'].values

n = int(0.8 * len(x))
x_tr, x_te = x[:n], x[n:]
y_tr, y_te = y[:n], y[n:]

m = np.sum((x_tr - x_tr.mean()) * (y_tr - y_tr.mean())) / np.sum((x_tr - x_tr.mean())**2)
c = y_tr.mean() - m * x_tr.mean()

y_pred = m * x_te + c

y_true_cls = (y_te >= 10).astype(int)
y_pred_cls = (y_pred >= 10).astype(int)

score = np.mean(y_true_cls == y_pred_cls)

print("Accuracy:", score)

plt.scatter(x_te, y_te)
plt.plot(x_te, y_pred)
plt.show()
