from QMeans_code.my_qmeans import QKMeans
from sklearn.model_selection import train_test_split
from qiskit_ibm_provider import IBMProvider
from config import IBM_QUANTUM_API_TOKEN
from sklearn import datasets

# Save IBMQ account
# IBMProvider.save_account(IBM_QUANTUM_API_TOKEN)
# Load IBMQ account
provider = IBMProvider()
# Print available backends
# print(provider.backends())
# Choose backend
backend = provider.get_backend("ibmq_qasm_simulator")

# Parameters
n_clusters = 2
n_features = 2
n_samples = 110
random_state = 12
cluser_std = 0.5

# Chargement des données
X, y = datasets.make_blobs(
    n_samples=n_samples,
    n_features=n_features,
    centers=n_clusters,
    random_state=random_state,
    cluster_std=cluser_std,
)

# train and test data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=random_state
)

# model
model = QKMeans(n_clusters=2, max_iter=10, random_state=random_state)
model.fit(X_train, X_test, y_test, backend=backend)
model.plot_clusters(X_train, X_test)
model.plot_accuracy()
