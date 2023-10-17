import numpy as np
import plotly.graph_objects as go
from plotly.offline import plot
from classical_utils import get_cluster_labels, compute_accuracy
from quantum_utils_CLOUD import (
    distance_centroids_parallel,
    transform_distances_matrix_to_bit_matrix,
    apply_quantum_find_min,
)
from tqdm import tqdm
import random


class QMeans:
    def __init__(
        self, n_clusters, max_iter=5, random_state=None, init_method="kmeans++"
    ):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.random_state = random_state
        self.centroids_ = None
        self.labels_ = None
        self.accuracy_train = []
        self.init_method = init_method

    def initialize_centroids_maximin(self, X):
        """
        Méthode Maximin: Cette méthode vise à maximiser la distance minimale entre les centroïdes, en commençant
        par un centroïde choisi au hasard, puis en sélectionnant les centroïdes suivants de manière à maximiser
        la distance minimale par rapport aux centroïdes déjà choisis. L'optimisation mentionnée dans le texte
        que vous avez partagé consiste à réduire le nombre de calculs de distance nécessaires en conservant
        une trace des distances minimales déjà calculées et en évitant les calculs inutiles.

        url article : https://borgelt.net/papers/data_20.pdf
        """
        n_samples, n_features = X.shape
        centroids = np.zeros((self.n_clusters, n_features))

        # Choisissez le premier centroïde aléatoirement
        centroids[0] = X[np.random.choice(n_samples)]

        for k in range(1, self.n_clusters):
            # Calculez les distances entre chaque point et le centroïde le plus proche
            distances = np.min(
                np.linalg.norm(X[:, np.newaxis] - centroids[:k], axis=2), axis=1
            )
            # Choisissez le prochain centroïde comme le point le plus éloigné du centroïde le plus proche
            centroids[k] = X[np.argmax(distances)]

        self.centroids_ = centroids

    def initialize_centroids_k_plusplus(self, X):
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

    def initialize_centroids_q_plusplus(self, points, backend, shots=4096):
        """
        QMeans++ initialization method.
        Using the quantum algorithm to find the minimum distance between a point and a set of points.
        """
        print("Start QMeans++ initialization method")

        # Initialize centroids as an empty numpy array with shape (0, num_features)
        centroids = np.empty((0, points.shape[1]))

        # Step 1 : Choose the first centroid randomly
        first_centroid = random.choice(points)
        centroids = np.append(centroids, [first_centroid], axis=0)

        # Step 2: Choose the other centroids
        for _ in range(self.n_clusters - 1):
            # compute the distances between each point and the closest centroid
            distances = np.array(
                list(
                    map(
                        lambda x: min(
                            distance_centroids_parallel(x, centroids, backend, shots)
                        ),
                        points,
                    )
                )
            )

            # NOTE : We can't use minimum quantum algorithm for searching the minimum distance between
            # a point and a set of points because we will lose the information of the real distance

            # Compute the probabilities
            probs = distances / np.sum(distances)

            # Choose the next centroid
            next_centroid = points[np.random.choice(a=len(points), p=probs)]
            centroids = np.append(centroids, [next_centroid], axis=0)

        self.centroids_ = centroids
        print("End QMeans++ initialization method")

    def fit(self, X_train, X_test, y_test, backend=None, shots=4096):
        if self.init_method == "kmeans++":
            self.initialize_centroids_k_plusplus(X_train)
        elif self.init_method == "qmeans++":
            self.initialize_centroids_q_plusplus(X_train, backend, shots)
        elif self.init_method == "maximin":
            self.initialize_centroids_maximin(X_train)
        else:
            raise ValueError(
                "Invalid init_method, must be 'kmeans++', 'qmeans++' or 'maximin'"
            )

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

            # Quantize the distances matrix
            bit_matrix = transform_distances_matrix_to_bit_matrix(distances)

            # We compute the quantum minimum for each point to have the labels
            labels = np.apply_along_axis(apply_quantum_find_min, axis=1, arr=bit_matrix)

            new_centroids = np.array(
                [X_train[labels == i].mean(axis=0) for i in range(self.n_clusters)]
            )

            if np.all(new_centroids == self.centroids_):
                # if centroids do not change anymore, stop
                break

            self.centroids_ = new_centroids

            # Calcul de l'accuracy
            y_pred_test = self.predict(X_test, backend, shots)
            y_mapped_test = get_cluster_labels(y_pred_test, y_test)
            self.accuracy_train.append(compute_accuracy(y_mapped_test, y_test))
            print(f"Accuracy: {round(self.accuracy_train[-1],3)}")

            if self.accuracy_train[-1] >= 0.95 and len(self.accuracy_train) >= 4:
                # if accuracy is greater than 95%, stop
                break

            if (
                len(self.accuracy_train) >= 3
                and self.accuracy_train[-1]
                == self.accuracy_train[-2]
                == self.accuracy_train[-3]
            ):
                # if accuracy does not change anymore, stop
                break

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

        fig.update_layout(title=f"Q-Means")

        if return_fig:
            return fig

        plot(fig, filename=f"images/q_kmeans_clusters.html")

    def plot_data_with_clusters(self, X_train):
        """
        Plot data with labels, and centroids with same color as labels
        """
        fig = go.Figure()

        # Ajout des points d'entraînement
        fig.add_trace(
            go.Scatter(
                x=X_train[:, 0],
                y=X_train[:, 1],
                mode="markers",
                marker=dict(size=5, color="blue"),
                showlegend=False,
            )
        )

        # Ajout des centroïdes
        fig.add_trace(
            go.Scatter(
                x=self.centroids_[:, 0],
                y=self.centroids_[:, 1],
                mode="markers",
                marker=dict(size=10, color="red", symbol="cross"),
                name="Centroïdes",
            )
        )

        fig.update_layout(title=f"Q-Means")

        plot(fig)

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
        plot(fig, filename="images/accuracy_q_means.html")


# main
if __name__ == "__main__":
    ...
