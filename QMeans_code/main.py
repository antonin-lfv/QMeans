from my_qmeans import QKMeans
import pandas as pd
from sklearn import preprocessing
import numpy as np
from sklearn.model_selection import train_test_split
from qiskit_ibm_provider import IBMProvider
from config import IBM_QUANTUM_API_TOKEN

# Save IBMQ account
# IBMProvider.save_account(IBM_QUANTUM_API_TOKEN)
# Load IBMQ account
provider = IBMProvider()
# Print available backends
print(provider.backends())
# Choose backend
backend = provider.get_backend("ibmq_qasm_simulator")

# Parameters
random_state = 12

# Load data
df = pd.read_csv(
    filepath_or_buffer="kmeans_data.csv",
    usecols=["Feature 1", "Feature 2", "Class"],
)
df["Class"] = pd.Categorical(df["Class"])
df["Class"] = df["Class"].cat.codes
data = df.values[:, 0:2]
y = df.values[:, 2].astype(np.int64)
X = preprocessing.maxabs_scale(data)
print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")

# train and test data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=random_state
)

# model
model = QKMeans(n_clusters=3, max_iter=10, random_state=random_state)
model.fit(X_train, X_test, y_test, backend=backend)
model.plot_clusters(X_train, X_test)

# main
if __name__ == "__main__":
    ...
