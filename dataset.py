from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np

import plotly.graph_objects as go
from plotly.offline import plot


class Kmeans_dataset:
    def __init__(
        self,
        source,
        n_clusters,
        n_features,
        n_samples,
        random_state,
        cluster_std,
        test_size,
    ):
        """
        Create a dataset for KMeans.
        Source can be "random blobs", "random moon", "random circle", "iris", "aniso" or "varied"
        """
        self.source = source
        self.n_clusters = n_clusters
        self.n_features = n_features
        self.n_samples = n_samples
        self.random_state = random_state
        self.cluster_std = cluster_std
        self.test_size = test_size

        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

        self.set_dataset()

    def set_dataset(self):
        if self.source == "random blobs":
            X, y = datasets.make_blobs(
                n_samples=self.n_samples,
                n_features=self.n_features,
                centers=self.n_clusters,
                random_state=self.random_state,
                cluster_std=self.cluster_std,
                center_box=(0.0, 1.0),
            )
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X, y, test_size=self.test_size, random_state=self.random_state
            )

        elif self.source == "random moon":
            X, y = datasets.make_moons(
                n_samples=self.n_samples,
                shuffle=True,
                noise=0.1,
                random_state=self.random_state,
            )
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X, y, test_size=self.test_size, random_state=self.random_state
            )

        elif self.source == "random circle":
            X, y = datasets.make_circles(
                n_samples=self.n_samples,
                shuffle=True,
                noise=0.1,
                random_state=self.random_state,
                factor=0.5,
            )
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X, y, test_size=self.test_size, random_state=self.random_state
            )

        elif self.source == "iris":
            iris = datasets.load_iris()
            X = iris.data
            # keep only 2 features
            X = X[:, :2]
            y = iris.target
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X, y, test_size=self.test_size, random_state=self.random_state
            )

        elif self.source == "aniso":
            X, y = datasets.make_blobs(
                n_samples=self.n_samples,
                random_state=self.random_state,
                cluster_std=self.cluster_std,
                centers=4,  # always 3 clusters
            )
            transformation = np.array([[0.7, -0.6], [-0.3, 0.8]])
            X_aniso = np.dot(X, transformation)
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X_aniso, y, test_size=self.test_size, random_state=self.random_state
            )

        elif self.source == "varied":
            assert self.n_clusters in [
                3,
                4,
            ], "n_clusters must be 3 or 4 for this source"
            X, y = datasets.make_blobs(
                n_samples=self.n_samples,
                centers=self.n_clusters,
                cluster_std=[0.11, 0.07, 0.1, 0.04]
                if self.n_clusters == 4
                else [0.1, 0.05, 0.12],
                random_state=self.random_state,
                center_box=(0.0, 1.0),
            )
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X, y, test_size=self.test_size, random_state=self.random_state
            )

        else:
            raise ValueError(
                "Invalid source, please choose between 'random blobs', 'random moon', 'random circle', "
                "'iris', 'aniso' or 'varied'"
            )

    def get_dataset(self):
        return self.X_train, self.X_test, self.y_train, self.y_test

    def plot_dataset(self, show_split=True):
        """
        Plot the dataset

        Parameters
        - show_split: bool, default=True : if True, show the split between train and test data (different colors)
        """
        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=self.X_train[:, 0],
                y=self.X_train[:, 1],
                mode="markers",
                marker=dict(size=5, color="orange" if show_split else "blue"),
                name="Points d'entraînement",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=self.X_test[:, 0],
                y=self.X_test[:, 1],
                mode="markers",
                marker=dict(size=5, color="blue"),
                name="Points de test",
            )
        )

        fig.update_layout(
            showlegend=False,
            template="plotly_white",
        )

        plot(fig, filename="images/kmeans_data.html")


if __name__ == "__main__":
    # Parameters
    source = "random blobs"
    n_clusters = 3
    n_features = 2
    n_samples = 150
    random_state = 301
    cluster_std = 1.5
    test_size = 0.1

    # Chargement des données
    dataset = Kmeans_dataset(
        source,
        n_clusters,
        n_features,
        n_samples,
        random_state,
        cluster_std,
        test_size,
    )
    X_train, X_test, y_train, y_test = dataset.get_dataset()
    dataset.plot_dataset()
