import pandas as pd, numpy as np, matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE


def load_data():
    d1 = pd.read_csv("Lab-09/diabetes_dataset.csv")
    d2 = pd.read_csv("Lab-09/Social_Network_Ads.csv")
    d3 = pd.read_csv("Lab-09/titanic.csv")

    d2["Gender"] = d2["Gender"].map({"Male":1,"Female":0})

    d3 = d3[["Pclass","Sex","Age","SibSp","Parch","Fare","Embarked","Survived"]]
    d3["Age"] = d3["Age"].fillna(d3["Age"].mean())
    d3["Embarked"] = d3["Embarked"].fillna(d3["Embarked"].mode()[0])

    for c in ["Sex","Embarked"]:
        d3[c] = LabelEncoder().fit_transform(d3[c])

    return {
        "Diabetes": (d1.iloc[:,:-1], d1.iloc[:,-1], 3),
        "Social Ads": (d2.drop("Purchased", axis=1), d2["Purchased"], 2),
        "Titanic": (d3.drop("Survived", axis=1), d3["Survived"], 5)
    }


def bisect_kmeans(X, k):
    clusters = [np.arange(len(X))]
    while len(clusters) < k:
        c = clusters.pop(np.argmax([np.var(X[i],0).sum()*len(i) for i in clusters]))
        best = min(
            (KMeans(2,n_init=1).fit_predict(X[c]) for _ in range(5)),
            key=lambda l: sum(np.var(X[c][l==i],0).sum()*sum(l==i) for i in [0,1])
        )
        clusters += [c[best==0], c[best==1]]

    labels = np.zeros(len(X),int)
    for i,c in enumerate(clusters): labels[c]=i
    return labels


def plot(X, labels, title):
    X2 = TSNE(n_components=2, random_state=42).fit_transform(X)
    markers = ['o','s','^','D','x','*','P','H']

    for i in np.unique(labels):
        plt.scatter(X2[labels==i,0], X2[labels==i,1],
                    marker=markers[i%len(markers)], label=f"C{i}")

    plt.title(title); plt.legend(); plt.grid(); plt.show()


def main():
    for name,(X,_,k) in load_data().items():
        X = StandardScaler().fit_transform(X)
        labels = bisect_kmeans(X,k)
        print(f"{name} | Silhouette: {silhouette_score(X,labels):.4f}")
        plot(X, labels, name)


if __name__ == "__main__":
    main()