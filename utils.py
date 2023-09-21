from scipy.stats import mode
import numpy as np
from qiskit import QuantumRegister, ClassicalRegister
from qiskit import QuantumCircuit
from qiskit import execute
from numpy import pi
from fastdist import fastdist
from qiskit_ibm_provider import IBMProvider
from tqdm import tqdm


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


# ======================================================================================================================


def _encode_feature(x):
    """ "
    We map data feature values to \theta and \\phi values using this equation:
        \\phi = (x + 1) \frac{\\pi}{2},
    where \\phi is the phase and \theta the angle
    """
    return (x + 1) * pi / 2


def _binary_combinations(n):
    """
    Returns all possible combinations of length n binary numbers as strings
    """
    combinations = []
    for i in range(2**n):
        bin_value = str(bin(i)).split("b")[1]
        while len(bin_value) < n:
            bin_value = "0" + bin_value

        combinations.append(bin_value)

    return combinations


def _binary_combinations_pos(n, index):
    """
    Returns all possible combinations of binary numbers where bit index=1
    """
    combinations_pos = []
    for bin_number in _binary_combinations(n):
        if bin_number[n - index - 1] == "1":
            combinations_pos.append(bin_number)

    return combinations_pos


def distance_centroids_parallel(point, centroids, backend, shots=1024):
    """
    Estimates distances using quantum computer specified by backend
    Computes it in parallel for all centroids

    Parameters:
        point: point to measure distance from
        centroids: list of centroids
        backend (IBMProvider backend): backend to use
        shots (int): number of shots to use
    """
    k = len(centroids)
    x_point, y_point = point[0], point[1]

    # Calculating theta and phi values
    phi_list = []
    theta_list = []
    for i in range(k):
        phi_list.append(_encode_feature(centroids[i][0]))
        theta_list.append(_encode_feature(centroids[i][1]))

    phi_input = _encode_feature(x_point)
    theta_input = _encode_feature(y_point)

    # We need 3 quantum registers, of size k one for a data point (input),
    # one for each centroid and one for each ancillary
    qreg_input = QuantumRegister(k, name="qreg_input")
    qreg_centroid = QuantumRegister(k, name="qreg_centroid")
    qreg_psi = QuantumRegister(k, name="qreg_psi")

    # Create a one bit ClassicalRegister to hold the result
    # of the measurements
    creg = ClassicalRegister(k, "creg")

    # Create the quantum circuit containing our registers
    qc = QuantumCircuit(qreg_input, qreg_centroid, qreg_psi, creg, name="qc")

    if not backend:
        raise Exception("No backend specified")

    for i in range(k):
        # Encode the point to measure and centroid
        qc.u(theta_list[i], phi_list[i], 0, qreg_centroid[i])
        qc.u(theta_input, phi_input, 0, qreg_input[i])

        # Apply a Hadamard to the ancillaries
        qc.h(qreg_psi[i])

        # Perform controlled swap
        qc.cswap(qreg_psi[i], qreg_input[i], qreg_centroid[i])

        # Apply second Hadamard to ancillary
        qc.h(qreg_psi[i])

        # Measure ancillary
        qc.measure(qreg_psi[i], creg[i])

    # Register and execute job
    job = execute(qc, backend=backend, shots=shots)
    result = job.result().get_counts(qc)

    distance_centroids = [0] * k
    for i in range(k):
        keys_centroid_k = _binary_combinations_pos(k, i)
        for key in keys_centroid_k:
            if key in result:
                distance_centroids[i] += result[key]

    return distance_centroids


def distances_for_multiple_examples_tests(
    num_examples=100, num_centroids=5, verbose=True, shots=1024
):
    """
    Test the distance_centroids_parallel function on multiple examples
    :param num_examples: int, number of examples
    :param num_centroids: int, number of centroids
    :param verbose: if True, print the results of each example
    :param shots: int, number of shots to use

    Examples:
    # num_examples=100, shots=2048
        Les tests sont terminés, 19 erreurs ont été trouvées sur 100 exemples.
        - 81.0 % de réussite !

    """
    nb_failures = 0
    for i in tqdm(range(num_examples)):
        if verbose:
            print(f"\n === Example {i + 1}")
        # Génération aléatoire d'un point et d'une liste de centroids
        point = np.random.rand(2).tolist()  # Point dans [0, 1] x [0, 1]
        centroids = [np.random.rand(2).tolist() for _ in range(num_centroids)]

        # Utilisation du simulateur comme backend
        provider = IBMProvider()
        backend = provider.get_backend("ibmq_qasm_simulator")

        # Calculer les distances quantiques
        quantum_distances = distance_centroids_parallel(
            point, centroids, backend, shots=shots
        )

        # Calculer les distances euclidiennes
        euclidean_distances = [
            euclidean_distance(np.array(point), np.array(centroid))
            for centroid in centroids
        ]

        if verbose:
            print(f"\nQuantum distances: {quantum_distances}")
            print(f"Euclidean distances: {euclidean_distances}")

        # Vérifier si l'indice du minimum est le même
        if np.argmin(quantum_distances) != np.argmin(euclidean_distances):
            if verbose:
                print(
                    f"\nError: the indices of the minimum distances are not the same for example {i + 1}"
                )
            nb_failures += 1
        else:
            if verbose:
                print(
                    f"\nIndices of the minimum distances are the same for example {i + 1}"
                )

    print(
        f"\n Tests finished, {nb_failures} errors were found out of {num_examples} "
        f"examples.\n - {(num_examples-nb_failures) / num_examples * 100} % success rate !"
    )


if __name__ == "__main__":
    distances_for_multiple_examples_tests(num_examples=100, verbose=False, shots=2048)
