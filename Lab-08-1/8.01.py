import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# ---------- TITANIC ----------
df = pd.read_csv("Lab-08-1/titanic.csv")
df = df[['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked','Survived']]
df['Age'].fillna(df['Age'].mean(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
df['Sex'] = LabelEncoder().fit_transform(df['Sex'])
df['Embarked'] = LabelEncoder().fit_transform(df['Embarked'])

X, y = df.drop('Survived', axis=1), df['Survived']
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
Xtr, Xte = MinMaxScaler().fit_transform(Xtr), MinMaxScaler().fit(Xtr).transform(Xte)

print("Titanic:", accuracy_score(yte, SVC(kernel='rbf').fit(Xtr, ytr).predict(Xte)))


# ---------- DIABETES ----------
df = pd.read_csv("Lab-08-1/diabetes_dataset.csv")
X, y = df.iloc[:, :-1], df.iloc[:, -1]
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)

model = SVC(kernel='linear').fit(Xtr, ytr)
y_pred = model.predict(Xte)

print("\nDiabetes:", accuracy_score(yte, y_pred))
print(classification_report(yte, y_pred))


# ---------- SOCIAL NETWORK ADS ----------
df = pd.read_csv("Lab-08-1/Social_Network_Ads.csv")
df['Gender'] = df['Gender'].map({'Male':1,'Female':0})

X, y = df.drop('Gender', axis=1), df['Gender']
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2)

model = SVC(kernel='rbf').fit(Xtr, ytr)
print("\nSocial Ads:", accuracy_score(yte, model.predict(Xte)))