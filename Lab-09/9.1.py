
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC




def load_datasets():
    df1 = pd.read_csv( r"Lab-09\diabetes_dataset.csv")
    df2 = pd.read_csv( r"Lab-09\Social_Network_Ads.csv")
    df3 = pd.read_csv(r"Lab-09\titanic.csv")

    df2["Gender"] = df2["Gender"].map({"Male": 1, "Female": 0})

    df3 = df3[["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked", "Survived"]].copy()
    df3["Age"] = df3["Age"].fillna(df3["Age"].mean())
    df3["Embarked"] = df3["Embarked"].fillna(df3["Embarked"].mode()[0])
    df3["Sex"] = LabelEncoder().fit_transform(df3["Sex"])
    df3["Embarked"] = LabelEncoder().fit_transform(df3["Embarked"])

    X1, y1 = df1.iloc[:, :-1], df1.iloc[:, -1]
    X2, y2 = df2.drop("Purchased", axis=1), df2["Purchased"]
    X3, y3 = df3.drop("Survived", axis=1), df3["Survived"]

    return {
        "Diabetes": (X1, y1),
        "Social Network Ads": (X2, y2),
        "Titanic": (X3, y3),
    }


def evaluate_models(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42,
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "KNN": KNeighborsClassifier(),
        "SVM": SVC(kernel="rbf"),
        "Naive Bayes": GaussianNB(),
    }

    rows = []
    for model_name, model in models.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)

        rows.append(
            {
                "Model": model_name,
                "Accuracy": accuracy_score(y_test, y_pred),
                "F1 Score": f1_score(y_test, y_pred, zero_division=0),
                "Recall": recall_score(y_test, y_pred, zero_division=0),
                "Precision": precision_score(y_test, y_pred, zero_division=0),
            }
        )

    return rows


datasets = load_datasets()
results = []

for dataset_name, (X, y) in datasets.items():
        for row in evaluate_models(X, y):
            results.append({"Dataset": dataset_name, **row})

        result_df = pd.DataFrame(results)
        print(result_df.to_string(index=False))



