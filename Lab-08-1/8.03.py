import pandas as pd
import numpy as np

# ----------- BASIC FUNCTIONS -----------

def train_test_split(X, y, test_size=0.2):
    idx = np.random.permutation(len(X))
    split = int(len(X)*(1-test_size))
    return X[idx[:split]], X[idx[split:]], y[idx[:split]], y[idx[split:]]

def minmax_scale(X):
    return (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0) + 1e-8)

def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)

# ----------- SIMPLE SVM (LINEAR, HINGE LOSS) -----------

def train_svm(X, y, lr=0.001, epochs=1000):
    y = np.where(y==0, -1, 1)   # convert to -1,1
    w = np.zeros(X.shape[1])
    b = 0

    for _ in range(epochs):
        for i in range(len(X)):
            condition = y[i] * (np.dot(X[i], w) + b) >= 1
            if condition:
                w -= lr * (2 * 0.01 * w)
            else:
                w -= lr * (2 * 0.01 * w - np.dot(X[i], y[i]))
                b -= lr * y[i]
    return w, b

def predict_svm(X, w, b):
    return np.where(np.dot(X, w) + b >= 0, 1, 0)

# ================= TITANIC =================
df = pd.read_csv("Lab-08-1/titanic.csv")
df = df[['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked','Survived']]

df['Age'].fillna(df['Age'].mean(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
df['Sex'] = df['Sex'].map({'male':1,'female':0})
df['Embarked'] = df['Embarked'].astype('category').cat.codes

X = df.drop('Survived', axis=1).values
y = df['Survived'].values

X = minmax_scale(X)
Xtr, Xte, ytr, yte = train_test_split(X, y)

w, b = train_svm(Xtr, ytr)
print("Titanic Accuracy:", accuracy(yte, predict_svm(Xte, w, b)))


# ================= DIABETES =================
df = pd.read_csv("Lab-08-1/diabetes_dataset.csv")

X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

X = minmax_scale(X)
Xtr, Xte, ytr, yte = train_test_split(X, y)

w, b = train_svm(Xtr, ytr)
print("Diabetes Accuracy:", accuracy(yte, predict_svm(Xte, w, b)))


# ================= SOCIAL NETWORK ADS =================
df = pd.read_csv("Lab-08-1/Social_Network_Ads.csv")

df['Gender'] = df['Gender'].map({'Male':1,'Female':0})

X = df.drop('Gender', axis=1).values
y = df['Gender'].values

X = minmax_scale(X)
Xtr, Xte, ytr, yte = train_test_split(X, y)

w, b = train_svm(Xtr, ytr)
print("Social Ads Accuracy:", accuracy(yte, predict_svm(Xte, w, b)))