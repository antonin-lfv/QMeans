from dataset import Kmeans_dataset
from KMeans_code.delta_KMeans import DeltaKMeans
from KMeans_code.KMeans_file import ClassicKMeans
from QMeans_code.my_qmeans import QMeans

# ======================== Data ======================== #

n_clusters = 3
n_features = 2
n_samples = 150
random_state = 17
cluster_std = 0.1

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

data.plot_dataset()

# ======================== Model ======================== #

# Classic KMeans
classic_kmeans = ClassicKMeans(n_clusters=n_clusters, random_state=random_state)

# Delta KMeans
delta_kmeans = DeltaKMeans(n_clusters=n_clusters, random_state=random_state)

# QMeans
qmeans = QMeans(n_clusters=n_clusters, random_state=random_state)


# ======================== Training ======================== #

# ======================== Subplots results ======================== #
