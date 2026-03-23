import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv(r"C:\Users\Tanmay\Desktop\C0D3\IML lab\Lab-04\daily-minimum-temperatures-in-me (1).csv")

df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
df = df.sort_values(by='Date')

df['Labels'] = (df['Daily minimum temperatures'] >= 5).astype(int)
df['doy'] = df['Date'].dt.dayofyear

def sigmoid(y):
    return 1/(1+np.exp(-y))

np.random.seed(42)
idx = np.random.permutation(len(df))
split = int(0.8 * len(df))

train = df.iloc[idx[:split]].copy()
test  = df.iloc[idx[split:]].copy()

w = 0.05
b = -5

train['Y'] = w*train['doy'] + b
train['Prob'] = sigmoid(train['Y'])
train['Predicted'] = (train['Prob'] >= 0.5).astype(int)

test['Y'] = w*test['doy'] + b
test['Prob'] = sigmoid(test['Y'])
test['Predicted'] = (test['Prob'] >= 0.5).astype(int)

tp = ((test['Labels'] == 1) & (test['Predicted'] == 1)).sum()
fp = ((test['Labels'] == 0) & (test['Predicted'] == 1)).sum()
tn = ((test['Labels'] == 0) & (test['Predicted'] == 0)).sum()
fn = ((test['Labels'] == 1) & (test['Predicted'] == 0)).sum()



accuracy = tp/(tp+fn+fp+tn)
print("Accuracy:", accuracy)

precision = tp/(tp+fp) if (tp+fp)>0 else 0
recall = tp/(tp+fn) if (tp+fn)>0 else 0
f1 = 2*precision*recall/(precision+recall) if (precision+recall)>0 else 0

print("Precision:", precision)
print("Recall:", recall)
print("F1:", f1)

cm = np.array([[tn, fp],
               [fn, tp]])

plt.imshow(cm)
plt.title("Confusion Matrix (DOY)")
plt.colorbar()
plt.xticks([0,1], ['Pred 0','Pred 1'])
plt.yticks([0,1], ['True 0','True 1'])

for i in range(2):
    for j in range(2):
        plt.text(j, i, cm[i,j], ha='center', va='center', color='white')

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()