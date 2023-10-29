import numpy as np
from classical_utils import get_cluster_labels, compute_accuracy, euclidean_distance
from plotly.offline import plot
import plotly.graph_objects as go
from dataset import Kmeans_dataset


class DeltaKMeans:
    def __init__(
        self,
        n_clusters,
        delta_decay=None,
        max_iter=50,
        delta_init=0.2,
        random_state=None,
    ):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.delta_init = delta_init
        self.delta_decay = delta_decay
        self.random_state = random_state
        self.centroids_ = None
        self.labels_ = None
        self.accuracy_train = []

    def initialize_centroids_plusplus(self, X):
        """
        k-means++ initialization method.
        """
        n_samples, n_features = X.shape
        centroids = np.zeros((self.n_clusters, n_features))

        # Choose the first centroid randomly
        centroids[0] = X[np.random.choice(n_samples)]

        for k in range(1, self.n_clusters):
            # Compute the squared distances from the previous centroids
            squared_distances = np.min(
                np.sum((X[:, np.newaxis] - centroids[:k]) ** 2, axis=2), axis=1
            )
            # Compute the probabilities
            probs = squared_distances / np.sum(squared_distances)
            # Choose the next centroid
            centroids[k] = X[np.random.choice(n_samples, p=probs)]

        self.centroids_ = centroids

    def fit(self, X_train, X_test, y_test):
        rng = np.random.default_rng(self.random_state)
        self.initialize_centroids_plusplus(X_train)
        delta = self.delta_init
        labels = None

        for _ in range(self.max_iter):
            # Calcul des labels en utilisant euclidean_distance
            labels = np.argmin(
                np.array(
                    [
                        [
                            euclidean_distance(x, centroid)
                            for centroid in self.centroids_
                        ]
                        for x in X_train
                    ]
                ),
                axis=1,
            )

            new_centroids = np.array(
                [X_train[labels == i].mean(0) for i in range(self.n_clusters)]
            )

            # Ajout de bruit avec moyenne 0 et écart-type \sqrt{delta/2}
            noise = rng.normal(
                loc=0, scale=np.sqrt(delta / 2), size=self.centroids_.shape
            )
            new_centroids += noise

            if np.all(new_centroids == self.centroids_):
                break

            self.centroids_ = new_centroids
            if self.delta_decay is not None:
                delta *= self.delta_decay

            # Calcul de l'accuracy
            y_pred_test = self.predict(X_test)
            y_mapped_test = get_cluster_labels(y_pred_test, y_test)

            self.accuracy_train.append(compute_accuracy(y_mapped_test, y_test))

        print(
            f"\nAccuracies delta KMeans: {[round(acc, 3) for acc in self.accuracy_train]}"
        )

        self.labels_ = labels
        return self

    def predict(self, X):
        # Utilisation de euclidean_distance pour la prédiction
        return np.argmin(
            np.array(
                [
                    [euclidean_distance(x, centroid) for centroid in self.centroids_]
                    for x in X
                ]
            ),
            axis=1,
        )

    def plot_data_with_labels(self, X_train, return_fig=False):
        """
        Plot data with labels, and centroids with same color as labels
        """
        fig = go.Figure()

        AssociatedColor = {
            0: "red",
            1: "blue",
            2: "green",
            3: "black",
            4: "purple",
            5: "orange",
            6: "pink",
            7: "brown",
            8: "gray",
        }

        colors = [AssociatedColor[label] for label in self.labels_]

        # Ajout des points d'entraînement
        fig.add_trace(
            go.Scatter(
                x=X_train[:, 0],
                y=X_train[:, 1],
                mode="markers",
                marker=dict(size=5, color=colors),
                showlegend=False,
            )
        )

        # Ajout des centroïdes
        fig.add_trace(
            go.Scatter(
                x=self.centroids_[:, 0],
                y=self.centroids_[:, 1],
                mode="markers",
                marker=dict(size=10, color="black", symbol="cross"),
                name="Centroïdes",
            )
        )

        fig.update_layout(title=f"delta_K-Means")

        if return_fig:
            return fig

        plot(fig, filename=f"images/delta_kmeans_clusters.html")

    def plot_accuracy(self):
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=np.arange(1, len(self.accuracy_train) + 1),
                y=self.accuracy_train,
                mode="lines+markers",
                marker=dict(size=5, color="blue"),
                line=dict(color="blue", width=1),
                name="Accuracy",
            )
        )
        fig.update_layout(
            title=f"Accuracy classic delta-KMeans, delta_init = {self.delta_init}"
        )
        plot(fig, filename="images/accuracy_delta_kmeans.html")


if __name__ == "__main__":
    # ======================== Parameters ======================== #
    n_clusters = 3
    n_features = 2
    n_samples = 200
    random_state = 42
    cluster_std = 0.1
    test_size = 0.1

    delta_init = 0.05

    # ======================== Data ======================== #
    data = Kmeans_dataset(
        source="random blobs",
        n_samples=n_samples,
        n_features=n_features,
        n_clusters=n_clusters,
        cluster_std=cluster_std,
        random_state=random_state,
        test_size=test_size,
    )

    X_train, X_test, y_train, y_test = data.get_dataset()

    # data.plot_dataset()

    # ======================== Model ======================== #

    # Delta KMeans
    delta_kmeans = DeltaKMeans(
        n_clusters=n_clusters, random_state=random_state, delta_init=delta_init
    )

    # ======================== Training ======================== #
    print("Start training Delta KMeans...")
    delta_kmeans.fit(X_train, X_test, y_test)
    print("Training done.")

    # ======================== Results ======================== #
    delta_kmeans.plot_data_with_labels(X_train)
