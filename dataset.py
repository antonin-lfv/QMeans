from sklearn import datasets
from sklearn.model_selection import train_test_split

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
            )
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X, y, test_size=self.test_size, random_state=self.random_state
            )

        elif self.source == "iris":
            iris = datasets.load_iris()
            X = iris.data
            y = iris.target
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X, y, test_size=self.test_size, random_state=self.random_state
            )

        else:
            raise ValueError(
                "Invalid source, please choose between 'random blobs', 'random moon', 'random circle' and 'iris'"
            )

    def get_dataset(self):
        return self.X_train, self.X_test, self.y_train, self.y_test

    def plot_dataset(self):
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=self.X_train[:, 0],
                y=self.X_train[:, 1],
                mode="markers",
                marker=dict(size=5, color="orange"),
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
        plot(fig, filename="images/kmeans_data.html")


if __name__ == "__main__":
    # Parameters
    source = "random circle"
    n_clusters = 2
    n_features = 2
    n_samples = 100
    random_state = 17
    cluster_std = 0.05
    test_size = 0.2

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
