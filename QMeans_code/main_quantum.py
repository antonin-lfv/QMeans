from QMeans_code.my_qmeans import QMeans
from qiskit_ionq import IonQProvider
from qiskit_ibm_runtime import QiskitRuntimeService

from config import IBM_QUANTUM_API_TOKEN, IONQ_API_TOKEN
from dataset import Kmeans_dataset

if __name__ == "__main__":
    provider = "IBM"
    if provider == "IBM":
        # Load IBMQ account
        service = QiskitRuntimeService()
        backend = service.backend("simulator_mps")
    else:
        provider = IonQProvider(IONQ_API_TOKEN)
        backend = provider.get_backend("ionq_simulator")

    # Parameters
    n_clusters = 3
    n_features = 2
    n_samples = 100
    random_state = 17
    cluster_std = 0.12

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

    # model
    print("Start training...")
    model = QMeans(
        n_clusters=n_clusters, max_iter=10, random_state=random_state, backend=backend
    )
    model.fit(X_train, X_test, y_test, shots=4096)
    model.plot_data_with_labels(X_train)
    model.plot_accuracy()
