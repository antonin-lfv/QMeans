import numpy as np
import plotly.graph_objects as go
from plotly.offline import plot
from utils import get_cluster_labels, compute_accuracy, distance_centroids_parallel
from tqdm import tqdm


class QKMeans:
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

    def fit(self, X_train, X_test, y_test, backend=None, shots=1024):
        self.initialize_centroids(X_train)
        labels = None

        for _ in tqdm(range(self.max_iter)):
            distances = np.array(
                list(
                    map(
                        lambda x: distance_centroids_parallel(
                            x, self.centroids_, backend, shots
                        ),
                        X_train,
                    )
                )
            )

            labels = np.argmin(distances, axis=1)

            new_centroids = np.array(
                [X_train[labels == i].mean(axis=0) for i in range(self.n_clusters)]
            )

            if np.all(new_centroids == self.centroids_):
                break

            self.centroids_ = new_centroids

            # Calcul de l'accuracy
            y_pred_test = self.predict(X_test, backend, shots)
            y_mapped_test = get_cluster_labels(y_pred_test, y_test)
            self.accuracy_train.append(compute_accuracy(y_mapped_test, y_test))
            print(f"Accuracy: {round(self.accuracy_train[-1],3)}")

        self.labels_ = labels
        return self

    def predict(self, X, backend=None, shots=1024):
        distances = np.array(
            list(
                map(
                    lambda x: distance_centroids_parallel(
                        x, self.centroids_, backend, shots
                    ),
                    X,
                )
            )
        )
        return np.argmin(distances, axis=1)

    def plot_clusters(self, X_train, X_test):
        centroids_x = self.centroids_[:, 0]
        centroids_y = self.centroids_[:, 1]

        fig = go.Figure()

        # Ajout des points d'entraînement
        fig.add_trace(
            go.Scatter(
                x=X_train[:, 0],
                y=X_train[:, 1],
                mode="markers",
                marker=dict(size=5, color="orange"),
                name="Points d'entraînement",
            )
        )

        # Ajout des points de test
        fig.add_trace(
            go.Scatter(
                x=X_test[:, 0],
                y=X_test[:, 1],
                mode="markers",
                marker=dict(size=5, color="green"),
                name="Points de test",
            )
        )

        fig.add_trace(
            go.Scatter(
                x=centroids_x,
                y=centroids_y,
                mode="markers",
                marker=dict(size=10, color="red", symbol="cross"),
                name="Centroïdes",
            )
        )

        fig.update_layout(title="Q-Means")

        plot(fig, filename="q_kmeans_clusters.html")

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
        fig.update_layout(title="Accuracy quantum KMeans")
        plot(fig, filename="accuracy_q_means.html")


# main
if __name__ == "__main__":
    ...
