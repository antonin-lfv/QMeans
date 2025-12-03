from dataset import Kmeans_dataset
from KMeans_code.delta_KMeans import DeltaKMeans
from KMeans_code.KMeans_file import ClassicKMeans
from QMeans_code.my_qmeans import QMeans
from config import IBM_QUANTUM_API_TOKEN, IONQ_API_TOKEN
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_ionq import IonQProvider

from plotly.subplots import make_subplots
from plotly.offline import plot

import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)
# QiskitRuntimeService.save_account(channel="ibm_cloud", token=IBM_QUANTUM_API_TOKEN, instance='ibm-q/open/main')


if __name__ == "__main__":
    print("[INFO]: Starting\n")

    # ======================== Data ======================== #

    source = "varied"
    n_clusters = 4
    n_features = 2
    n_samples = 80
    random_state = 999
    cluster_std = 0.045
    test_size = 0.15

    delta_init = 0.0005

    service = "IonQ"
    if service == "IBM":
        print("Using IBM Quantum Service")
        service = QiskitRuntimeService('ibm_quantum_platform')
        backend = service.backend("simulator_mps")

    else:
        print("Using IonQ Quantum Service")
        provider = IonQProvider(IONQ_API_TOKEN)
        backend = provider.get_backend("ionq_simulator")

    shots = 2048

    data = Kmeans_dataset(
        source=source,
        n_samples=n_samples,
        n_features=n_features,
        n_clusters=n_clusters,
        cluster_std=cluster_std,
        random_state=random_state,
        test_size=test_size,
    )

    X_train, X_test, y_train, y_test = data.get_dataset()

    data.plot_dataset(show_split=False)

    # ======================== Model ======================== #

    # Classic KMeans
    classic_kmeans = ClassicKMeans(n_clusters=n_clusters, random_state=random_state)

    # Delta KMeans
    delta_kmeans = DeltaKMeans(
        n_clusters=n_clusters, random_state=random_state, delta_init=delta_init
    )

    # QMeans
    qmeans = QMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        init_method="kmeans++",
        backend=backend,
    )

    # ======================== Training ======================== #
    print("Start training Classic KMeans...")
    classic_kmeans.fit(X_train, X_test, y_test)

    print("Start training Delta KMeans...")
    delta_kmeans.fit(X_train, X_test, y_test)

    print("Start training QMeans...")
    qmeans.fit(X_train, X_test, y_test, shots=shots)

    print("Training done.")

    # ======================== Subplots results ======================== #
    classic_kmeans_fig = classic_kmeans.plot_data_with_labels(X_train, return_fig=True)

    delta_kmeans_fig = delta_kmeans.plot_data_with_labels(X_train, return_fig=True)

    qmeans_fig = qmeans.plot_data_with_labels(X_train, return_fig=True)

    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=("KMeans", "Delta KMeans", "QMeans"),
        shared_yaxes=True,
    )

    fig.add_trace(classic_kmeans_fig.data[0], row=1, col=1)
    fig.add_trace(classic_kmeans_fig.data[1], row=1, col=1)

    fig.add_trace(delta_kmeans_fig.data[0], row=1, col=2)
    fig.add_trace(delta_kmeans_fig.data[1], row=1, col=2)

    fig.add_trace(qmeans_fig.data[0], row=1, col=3)
    fig.add_trace(qmeans_fig.data[1], row=1, col=3)

    fig.update_layout(
        showlegend=False,
        template="plotly_white",
    )

    # Ratio 1:1 des subplots
    fig.update_yaxes(scaleanchor="x", scaleratio=1, row=1, col=1)
    fig.update_yaxes(scaleanchor="x", scaleratio=1, row=1, col=2)
    fig.update_yaxes(scaleanchor="x", scaleratio=1, row=1, col=3)

    plot(fig, filename="images/comparaisons.html")
