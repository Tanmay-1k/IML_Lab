import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

df = pd.read_csv(r"C:\Users\Tanmay\Desktop\C0D3\IML lab\Lab-04\daily-minimum-temperatures-in-me (1).csv")

df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
df = df.sort_values(by='Date')

df['Day_Number'] = (df['Date'] - df['Date'].min()).dt.days
df['Labels'] = (df['Daily minimum temperatures'] >= 10).astype(int)

df['doy'] = df['Date'].dt.dayofyear

X = df[['doy']]
y = df['Labels']   

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print("Accuracy:", acc)
print("F1:", f1)
print("Confusion matrix:\n", cm)

ConfusionMatrixDisplay(cm).plot()
plt.show()