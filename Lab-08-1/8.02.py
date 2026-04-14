import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

def run(path, target, prep=None):
    df = pd.read_csv(path)
    if prep: df = prep(df)
    X, y = df.drop(target, axis=1), df[target]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    sc = StandardScaler()
    model = GaussianNB().fit(sc.fit_transform(Xtr), ytr)
    return accuracy_score(yte, model.predict(sc.transform(Xte)))

# Diabetes
print("Diabetes Accuracy:", run(r"Lab-08-1\diabetes_dataset.csv", "Outcome"))

# Social Ads
print("Social Ads Accuracy:", run(
    r"Lab-08-1\Social_Network_Ads.csv", "Purchased",
    lambda df: df.assign(Gender=df["Gender"].map({"Male":1,"Female":0}))
))

# Titanic
print("Titanic Accuracy:", run(
    r"Lab-08-1\titanic.csv", "Survived",
    lambda df: df[["Pclass","Sex","Age","SibSp","Parch","Fare","Embarked","Survived"]]
        .assign(Age=lambda x: x["Age"].fillna(x["Age"].mean()),
                Embarked=lambda x: x["Embarked"].fillna(x["Embarked"].mode()[0]))
        .pipe(lambda x: x.assign(
            Sex=LabelEncoder().fit_transform(x["Sex"]),
            Embarked=LabelEncoder().fit_transform(x["Embarked"])
        ))
))