import numpy as np
from utils import get_cluster_labels, compute_accuracy
from plotly.offline import plot
import plotly.graph_objects as go
from fastdist import fastdist


def euclidean_distance(X, Y):
    """
    Calcule la distance euclidienne entre deux vecteurs X et Y
    """
    return fastdist.euclidean(X, Y)


class DeltaKMeans:
    def __init__(
        self,
        n_clusters,
        delta_decay=None,
        max_iter=50,
        delta_init=0.5,
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

    def initialize_centroids(self, X):
        n_samples = X.shape[0]
        rng = np.random.default_rng(self.random_state)

        centroids_idx = [rng.choice(n_samples)]
        for _ in range(1, self.n_clusters):
            distance = np.sqrt(((X - X[centroids_idx][:, np.newaxis]) ** 2).sum(axis=2))
            min_distances = np.min(
                distance, axis=0
            )  # distances minimales pour chaque point
            probs = min_distances / np.sum(min_distances)  # normalisation des distances
            centroids_idx.append(rng.choice(n_samples, p=probs))

        return X[centroids_idx, :]

    def fit(self, X_train, X_test, y_test):
        rng = np.random.default_rng(self.random_state)
        self.centroids_ = self.initialize_centroids(X_train)
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

    def plot_clusters(self, X_train, X_test):
        # Extraire les coordonnées des centroïdes
        centroids_x = self.centroids_[:, 0]
        centroids_y = self.centroids_[:, 1]

        # Créer la figure
        fig = go.Figure()

        # Ajouter les points d'entraînement
        fig.add_trace(
            go.Scatter(
                x=X_train[:, 0],
                y=X_train[:, 1],
                mode="markers",
                marker=dict(size=5, color="orange"),
                name="Points d'entraînement",
            )
        )

        # Ajouter les points de test
        fig.add_trace(
            go.Scatter(
                x=X_test[:, 0],
                y=X_test[:, 1],
                mode="markers",
                marker=dict(size=5, color="green"),
                name="Points de test",
            )
        )

        # Ajouter les centroïdes
        fig.add_trace(
            go.Scatter(
                x=centroids_x,
                y=centroids_y,
                mode="markers",
                marker=dict(size=10, color="red", symbol="cross"),
                name="Centroïdes",
            )
        )

        # Titre
        fig.update_layout(title="delta-KMeans")

        # Afficher la figure
        plot(fig, filename="images/delta_kmeans.html")

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
            title=f"Accuracy classic KMeans, delta_init = {self.delta_init}"
        )
        plot(fig, filename="images/accuracy_delta_kmeans.html")
