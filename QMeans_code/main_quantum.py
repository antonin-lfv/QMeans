from QMeans_code.my_qmeans import QMeans
from qiskit_ibm_provider import IBMProvider
from qiskit_ionq import IonQProvider

from config import IBM_QUANTUM_API_TOKEN, IONQ_API_TOKEN
from dataset import Kmeans_dataset

if __name__ == "__main__":
    provider = "IBM"
    if provider == "IBM":
        # Save IBMQ account
        # IBMProvider.save_account(IBM_QUANTUM_API_TOKEN)
        # Load IBMQ account
        provider = IBMProvider()
        # Print available backends
        # print(provider.backends())
        # Choose backend
        backend = provider.get_backend("ibmq_qasm_simulator")
    else:
        # Save IonQ account
        provider = IonQProvider(IONQ_API_TOKEN)
        # print available backends
        print(provider.backends())
        # Choose backend
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
    model = QMeans(n_clusters=n_clusters, max_iter=10, random_state=random_state)
    model.fit(X_train, X_test, y_test, backend=backend, shots=4096)
    model.plot_clusters(X_train, X_test)
    model.plot_accuracy()

    # Discussion
    # - Si il y a que 2 entiers dans la liste, et qu'il y a un zero, il trouve jamais le bon minimum
    # - On est limité par la taille du circuit
    # - Obligé de convertir les distances en entiers, puis les convertir en petit entiers pour que ça
    # rentre dans le circuit
