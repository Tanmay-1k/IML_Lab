import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    confusion_matrix,ConfusionMatrixDisplay
)

# 1D datasets
y_true = np.array([1, 0, 1, 1, 0, 0, 1])

y_pred = np.array([1, 0, 0, 1, 0, 1, 1])


accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

print("Accuracy :", accuracy)
print("Precision:", precision)
print("Recall   :", recall)
print("F1 Score :", f1)


cm = confusion_matrix(y_true,y_pred)
print("\nConfusion Matrix\n",cm)

print(type(cm))

# taking values
tn=cm[0][0]
fp=cm[0][1]
fn=cm[1][0]
tp=cm[1][1]

print("\nTrue Negatives: ",tn)
print("\nTrue Positives: ",tp)
print("\nFalse Negatives: ",fn)
print("\nFalse Positives: ",fp)



# Plotting
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
disp.plot(cmap="Blues")
plt.title("Confusion Matrix ")
plt.show()
