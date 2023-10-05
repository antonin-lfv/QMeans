from delta_KMeans import DeltaKMeans
from KMeans_file import ClassicKMeans
import plotly.graph_objects as go
from plotly.offline import plot
import numpy as np
from dataset import Kmeans_dataset


if __name__ == "__main__":
    print("Running main_classical.py")

    # Parameters
    n_clusters = 3
    n_features = 2
    n_samples = 150
    random_state = 12
    cluster_std = 0.8

    # Chargement des données
    data = Kmeans_dataset(
        source="random blobs",
        n_samples=n_samples,
        n_features=n_features,
        n_clusters=n_clusters,
        cluster_std=cluster_std,
        random_state=random_state,
        test_size=0.2,
    )

    X_train, X_test, y_train, y_test = data.get_dataset()

    # Plot dataset
    data.plot_dataset()

    # Comparaison des accuracy
    fig = go.Figure()
    # On teste delta_init = 0.2, 0.3, 0.4 et 0.5 en pointillés
    delta_KMeans_02 = DeltaKMeans(
        n_clusters=n_clusters, random_state=random_state, delta_init=0.2
    ).fit(X_train, X_test, y_test)
    delta_KMeans_03 = DeltaKMeans(
        n_clusters=n_clusters, random_state=random_state, delta_init=0.3
    ).fit(X_train, X_test, y_test)
    delta_KMeans_04 = DeltaKMeans(
        n_clusters=n_clusters, random_state=random_state, delta_init=0.4
    ).fit(X_train, X_test, y_test)
    delta_KMeans_05 = DeltaKMeans(
        n_clusters=n_clusters, random_state=random_state, delta_init=0.5
    ).fit(X_train, X_test, y_test)

    fig.add_trace(
        go.Scatter(
            x=np.arange(len(delta_KMeans_02.accuracy_train)),
            y=delta_KMeans_02.accuracy_train,
            mode="lines",
            name="δ-KMeans (δ = 0.2)",
            line=dict(dash="dot", width=2),
            opacity=0.8,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=np.arange(len(delta_KMeans_03.accuracy_train)),
            y=delta_KMeans_03.accuracy_train,
            mode="lines",
            name="δ-KMeans (δ = 0.3)",
            line=dict(dash="dot", width=2),
            opacity=0.8,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=np.arange(len(delta_KMeans_04.accuracy_train)),
            y=delta_KMeans_04.accuracy_train,
            mode="lines",
            name="δ-KMeans (δ = 0.4)",
            line=dict(dash="dot", width=2),
            opacity=0.8,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=np.arange(len(delta_KMeans_05.accuracy_train)),
            y=delta_KMeans_05.accuracy_train,
            mode="lines",
            name="δ-KMeans (δ = 0.5)",
            line=dict(dash="dot", width=2),
            opacity=0.8,
        )
    )

    classic_kmeans = ClassicKMeans(
        n_clusters=n_clusters, random_state=random_state
    ).fit(X_train, X_test, y_test)
    fig.add_trace(
        go.Scatter(
            x=np.arange(len(classic_kmeans.accuracy_train)),
            y=classic_kmeans.accuracy_train,
            mode="lines",
            name="Classic KMeans",
            line=dict(width=3),
        )
    )

    # Ajouter un seul encadré (annotation) en bas à droite de la figure pour afficher la dernière accuracy de chaque
    # méthode
    fig.add_annotation(
        x=0.1,
        y=0.1,
        text=f"δ-KMeans (δ = 0.2) : {delta_KMeans_02.accuracy_train[-1]*100:.2f} %"
        + "<br>"
        + f"δ-KMeans (δ = 0.3) : {delta_KMeans_03.accuracy_train[-1]*100:.2f} %"
        + "<br>"
        + f"δ-KMeans (δ = 0.4) : {delta_KMeans_04.accuracy_train[-1]*100:.2f} %"
        + "<br>"
        + f"δ-KMeans (δ = 0.5) : {delta_KMeans_05.accuracy_train[-1]*100:.2f} %"
        + "<br>"
        + f"Classic KMeans : {classic_kmeans.accuracy_train[-1]*100:.2f} %",
        align="left",
        showarrow=False,
        yshift=5,
        bordercolor="black",
        borderwidth=1,
        borderpad=1,
        xref="paper",
        yref="paper",
        # size of the annotation text
        font=dict(size=20),
    )

    fig.update_layout(
        title="Accuracy en fonction du nombre d'itérations",
        xaxis_title="Nombre d'itérations",
        yaxis_title="Accuracy (%)",
        # legend text size
        font=dict(size=15),
    )

    plot(fig, filename="images/accuracy_comparison.html")
