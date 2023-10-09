import numpy as np
import plotly.graph_objects as go
from plotly.offline import plot
from classical_utils import get_cluster_labels, compute_accuracy, euclidean_distance


class ClassicKMeans:
    def __init__(self, n_clusters, max_iter=50, random_state=None):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.random_state = random_state
        self.centroids_ = None
        self.labels_ = None
        self.accuracy_train = []

    def initialize_centroids(self, X):
        n_samples, n_features = X.shape
        rng = np.random.default_rng(self.random_state)

        centroids_idx = rng.choice(n_samples, size=self.n_clusters, replace=False)
        self.centroids_ = X[centroids_idx, :]

    def fit(self, X_train, X_test, y_test):
        self.initialize_centroids(X_train)
        labels = None

        for _ in range(self.max_iter):
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
                [X_train[labels == i].mean(axis=0) for i in range(self.n_clusters)]
            )

            if np.all(new_centroids == self.centroids_):
                break

            self.centroids_ = new_centroids

            # Calcul de l'accuracy
            y_pred_test = self.predict(X_test)
            y_mapped_test = get_cluster_labels(y_pred_test, y_test)

            self.accuracy_train.append(compute_accuracy(y_mapped_test, y_test))

        self.labels_ = labels
        return self

    def predict(self, X):
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

        fig.update_layout(title=f"K-Means")

        if return_fig:
            return fig

        plot(fig, filename=f"images/kmeans_clusters.html")

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
        fig.update_layout(title="Accuracy classic KMeans")
        plot(fig, filename="images/accuracy.html")
