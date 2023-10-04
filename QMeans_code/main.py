from QMeans_code.my_qmeans import QKMeans
from sklearn.model_selection import train_test_split
from qiskit_ibm_provider import IBMProvider
from config import IBM_QUANTUM_API_TOKEN
from sklearn import datasets
import plotly.graph_objects as go
from plotly.offline import plot

# Save IBMQ account
# IBMProvider.save_account(IBM_QUANTUM_API_TOKEN)
# Load IBMQ account
provider = IBMProvider()
# Print available backends
# print(provider.backends())
# Choose backend
backend = provider.get_backend("ibmq_qasm_simulator")

# Parameters
n_clusters = 3
n_features = 2
n_samples = 100
random_state = 17
cluser_std = 0.12

# Chargement des données
X, y = datasets.make_blobs(
    n_samples=n_samples,
    n_features=n_features,
    centers=n_clusters,
    random_state=random_state,
    cluster_std=cluser_std,
    center_box=(0.0, 1.0),
)

# train and test data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=random_state
)

# plot data
fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=X_train[:, 0],
        y=X_train[:, 1],
        mode="markers",
        marker=dict(size=5, color="orange"),
        name="Points d'entraînement",
    )
)
fig.add_trace(
    go.Scatter(
        x=X_test[:, 0],
        y=X_test[:, 1],
        mode="markers",
        marker=dict(size=5, color="blue"),
        name="Points de test",
    )
)
plot(fig)

# model
print("Start training...")
model = QKMeans(n_clusters=n_clusters, max_iter=10, random_state=random_state)
model.fit(X_train, X_test, y_test, backend=backend, shots=4096)
model.plot_clusters(X_train, X_test)
model.plot_accuracy()


# Discussion
# - Si il y a que 2 entiers dans la liste, et qu'il y a un zero, il trouve jamais le bon minimum
# - On est limité par la taille du circuit
# - Obligé de convertir les distances en entiers, puis les convertir en petit entiers pour que ça
# rentre dans le circuit
