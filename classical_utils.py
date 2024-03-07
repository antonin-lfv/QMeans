from scipy.stats import mode
import numpy as np
from fastdist import fastdist


def get_cluster_labels(y_pred, y_true):
    """
    Pour chaque cluster, trouve le label le plus fréquent.
    """
    labels = np.zeros_like(y_pred)
    for i in range(max(y_pred) + 1):  # Pour chaque cluster
        mask = y_pred == i
        labels[mask] = mode(y_true[mask], keepdims=True)[
            0
        ]  # Associer le label le plus fréquent
    return labels


def compute_accuracy(y_pred, y_true):
    """
    Calcule l'accuracy des labels prédits par rapport aux labels réels.
    """
    return np.sum(y_pred == y_true) / float(y_true.size)


def euclidean_distance(X, Y):
    """
    Calcule la distance euclidienne entre deux vecteurs X et Y
    """
    return fastdist.euclidean(X, Y)
