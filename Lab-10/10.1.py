import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import k_means
from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelEncoder, StandardScaler


def load_datasets():
    df1 = pd.read_csv(r"Lab-09\diabetes_dataset.csv")
    df2 = pd.read_csv(r"Lab-09\Social_Network_Ads.csv")
    df3 = pd.read_csv(r"Lab-09\titanic.csv")

    # Preprocess Social Network Ads
    df2["Gender"] = df2["Gender"].map({"Male": 1, "Female": 0})

    # Preprocess Titanic
    df3 = df3[["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked", "Survived"]].copy()
    df3["Age"] = df3["Age"].fillna(df3["Age"].mean())
    df3["Embarked"] = df3["Embarked"].fillna(df3["Embarked"].mode()[0])
    df3["Sex"] = LabelEncoder().fit_transform(df3["Sex"])
    df3["Embarked"] = LabelEncoder().fit_transform(df3["Embarked"])

    X1, y1 = df1.iloc[:, :-1], df1.iloc[:, -1]
    X2, y2 = df2.drop("Purchased", axis=1), df2["Purchased"]
    X3, y3 = df3.drop("Survived", axis=1), df3["Survived"]

    return {
        "Diabetes": (X1, y1, 3),
        "Social Network Ads": (X2, y2, 2),
        "Titanic": (X3, y3, 5),
    }


def plot_clusters(X, title, n_clusters):
    scaler = StandardScaler()
    tsne = TSNE(n_components=2, random_state=42)

    X_scaled = scaler.fit_transform(X)
    X_tsne = tsne.fit_transform(X_scaled)

    kmeans_model = k_means(X_tsne, n_clusters=n_clusters, random_state=42)
    centers, labels, _ = kmeans_model

    print(f"{title} K-Means Cluster Centers:\n{centers}\n")

    # Markers for B/W differentiation
    markers = ['o', 's', '^', 'D', 'P', 'X', '*']
    unique_labels = sorted(set(labels))

    for i, label in enumerate(unique_labels):
        plt.scatter(
            X_tsne[labels == label, 0],
            X_tsne[labels == label, 1],
            marker=markers[i % len(markers)],
            color=str(i / len(unique_labels)),  # grayscale
            edgecolors='black',
            linewidths=0.5,
            label=f"Cluster {label}"
        )

    # Plot centroids (important for reports)
    plt.scatter(
        centers[:, 0],
        centers[:, 1],
        marker='X',
        s=200,
        color='black',
        label='Centroids'
    )

    plt.title(f"{title} K-Means Clusters")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.legend()
    plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.5)
    plt.show()


# -------------------- MAIN --------------------
datasets = load_datasets()

for name, (X, y, k) in datasets.items():
    plot_clusters(X, name, k)