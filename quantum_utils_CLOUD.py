from qiskit import (
    QuantumCircuit,
    execute,
    QuantumRegister,
    ClassicalRegister,
)
from qiskit_ibm_provider import IBMProvider
from qiskit.tools.visualization import circuit_drawer, plot_histogram

import plotly.graph_objects as go
from plotly.offline import plot
import matplotlib.pyplot as plt
import math
import random
from tqdm import tqdm
import numpy as np
from math import pi
import time

from classical_utils import euclidean_distance

# ======================================================================================================================
# Global functions
# ======================================================================================================================


def rank_and_convert(row):
    """
    Trouve les valeurs uniques, les trie, et retourne les indices de chaque élément.
    Convertit ensuite ces indices en bitstrings.
    """
    unique_values, inverse_indices = np.unique(row, return_inverse=True)
    bit_row = np.vectorize(int_to_bits)(inverse_indices + 1, 3)
    return bit_row


def transform_distances_matrix_to_bit_matrix(matrix):
    """
    Transforme une matrice de distances en une matrice de bits.
    Pour cela, on commence par trier les valeurs uniques de la matrice, puis on convertit chaque valeur en bits.
    On obtient ainsi une matrice de bits.
    """
    bit_matrix = np.apply_along_axis(rank_and_convert, axis=1, arr=matrix)
    return bit_matrix


def apply_quantum_find_min(row):
    """
    Applique la fonction quantum_find_min à une ligne de la matrice de bits.
    """
    # Convertir la ligne en liste de chaînes de bits
    list_of_bits = row.tolist()
    # Appeler quantum_find_min et retourner seulement l'index
    index = quantum_find_min(list_of_bits, only_index=True, shots=100)
    return index


# ======================================================================================================================
# Find the minimum of a list of integers using quantum computing and bitstrings comparator
# ======================================================================================================================


def qubits_needed(n):
    if n < 0:
        raise ValueError("Le nombre doit être non négatif")
    elif n == 0:
        return 1  # Un qubit est nécessaire pour représenter 0
    else:
        return math.ceil(math.log2(n + 1))


def encode(bit, plot_circuit=False):
    qr = QuantumRegister(1, "number")
    qc = QuantumCircuit(qr)
    if bit == "1":
        qc.x(qr[0])

    if plot_circuit:
        qc.draw(output="mpl")
        plt.show()
    return qc


def encode_bitstring(bitstring, plot_circuit=False):
    bits = len(bitstring)
    qr = QuantumRegister(bits, "bit")
    qc = QuantumCircuit(qr)
    for i in range(bits):
        if bitstring[i] == "1":
            qc.x(qr[i])

    if plot_circuit:
        qc.draw(output="mpl")
        plt.show()
    return qc


def bit_compare():
    qr = QuantumRegister(2, "bits")
    aux = QuantumRegister(2, "aux")

    qc = QuantumCircuit(qr, aux)
    qc.x(qr[1])
    qc.mcx(qr, aux[0])
    qc.x(qr[0])
    qc.x(qr[1])
    qc.mcx(qr, aux[1])
    qc.x(qr[0])

    return qc


def bits_to_int(bits):
    """
    Convert a string of bits to an integer
    :param bits: string of bits, e.g. "101"
    :return: integer representation of bits, e.g. 5
    """
    return int(bits, 2)


def int_to_bits(integer):
    """
    Convert an integer to a bitstring using the minimum number of bits
    :param integer: integer to convert
    :return: bitstring representation of integer
    """
    return format(integer, "b")


def compare_bitstring(
    bitstring_a, bitstring_b, plot_circuit=False, return_time=False, shots=1024
):
    """
    Compare two bitstrings using quantum computing.

    Parameters:
        bitstring_a: first bitstring to compare, e.g. "101"
        bitstring_b: second bitstring to compare, e.g. "010"
        plot_circuit: if True, plot the circuit
        return_time: if True, return the circuit time
        shots: number of shots to perform

    Example:
        >>> bitstring_a = "101"
        >>> bitstring_b = "010"
        >>> counts = compare_bitstring(bitstring_a, bitstring_b, plot_circuit=False)
        >>> print(counts)
        {'01': 1024}

    Return:
        counts: dict, counts of the different results of the comparison
        time: float, time of the circuit execution (if return_time=True)

    Interpretation of the results:
        - If '01' has higher score, then the second bitstring is smaller
        - If '10' has higher score, then the first bitstring is smaller
        - If '00' has higher score, then the bitstrings are equal
    """
    bits = max(
        qubits_needed(bits_to_int(bitstring_a)), qubits_needed(bits_to_int(bitstring_b))
    )
    qra = QuantumRegister(bits, "a")
    qrb = QuantumRegister(bits, "b")
    qraux = QuantumRegister(2 * bits, "aux")
    qrint = QuantumRegister(bits - 1, "int")
    cr = ClassicalRegister(2)

    qc = QuantumCircuit(qra, qrb, qraux, qrint, cr)

    for i in range(bits):
        qc.append(encode(bitstring_a[i]).to_instruction(), [qra[i]])
        qc.append(encode(bitstring_b[i]).to_instruction(), [qrb[i]])
        qc.append(
            bit_compare().to_instruction(),
            [qra[i], qrb[i], qraux[2 * i], qraux[2 * i + 1]],
        )

        if i < bits - 2:
            qc.x(qraux[2 * i])
            qc.x(qraux[2 * i + 1])
            qc.mcx([qraux[2 * i], qraux[2 * i + 1]], qrint[i])
            qc.x(qraux[2 * i])
            qc.x(qraux[2 * i + 1])

    for i in range(0, bits - 1):
        qc.mcx([qraux[2 * (-i - 1)], qrint[-i]], qraux[2 * (-i)])
        qc.mcx([qraux[2 * (-i - 1) + 1], qrint[-i]], qraux[2 * (-i) + 1])

    qc.measure(qraux[0], cr[0])
    qc.measure(qraux[1], cr[1])

    # Do the simulation, returning the result
    result = execute(qc, backend, shots=shots).result()
    circuit_time = result.results[0]._metadata["metadata"]["sample_measure_time"]

    # get the probability distribution
    counts = result.get_counts()

    if plot_circuit:
        circuit_drawer(
            qc,
            output="mpl",
        )
        plt.show()

    if return_time:
        return counts, circuit_time

    return counts


# Test compare_bitstring
# a = "11"
# b = "01"
# counts, time_c = compare_bitstring(a, b, plot_circuit=False, return_time=True)
# print(f"First bitstring: '{a}', that is {bits_to_int(a)}")
# print(f"Second bitstring: '{b}', that is {bits_to_int(b)}")
# print(counts)

# if '01' has higher score, then the second bitstring is smaller
# if '10' has higher score, then the first bitstring is smaller
# if '00' has higher score, then the bitstrings are equal


def accuracy_compare_bitstring(n_times=100, bits=4, shots=10):
    """
    Compare n times 2 random bitstrings and compute the accuracy of the comparison

    Results:
        - 70% accuracy with 100 tests, 4 bits, 1024 shots
        - 67% accuracy with 100 tests, 4 bits, 2048 shots
        - 74% accuracy with 100 tests, 4 bits, 100 shots
        - 78% accuracy with 100 tests, 4 bits, 10 shots
        - 73% accuracy with 100 tests, 4 bits, 1 shots

        - 74% accuracy with 100 tests, 3 bits, 1 shots

        - 72% accuracy with 100 tests, 5 bits, 1 shots

        - 69% accuracy with 100 tests, 6 bits, 1 shots
    """
    nb_success = 0
    for _ in tqdm(range(n_times)):
        a = int_to_bits(random.randint(0, 2**bits - 1))
        b = int_to_bits(random.randint(0, 2**bits - 1))
        counts = compare_bitstring(a, b, shots=shots)
        # get the most frequent result
        result = max(counts, key=counts.get)
        if result == "01" and bits_to_int(a) > bits_to_int(b):
            nb_success += 1
        elif result == "10" and bits_to_int(a) < bits_to_int(b):
            nb_success += 1
        elif result == "00" and bits_to_int(a) == bits_to_int(b):
            nb_success += 1

    print(f"Accuracy: {round((nb_success / n_times)*100, 3)}")


def compare_bitstring_compare_time():
    """
    Compare the time of the classical and quantum compare_bitstring functions with different bits size (mean)
    """
    bits_size = [1, 2, 3, 4, 5, 6]
    N = 100  # test N times for each bits size
    classical_pourcentage = []
    quantum_pourcentage = []
    for j in tqdm(range(N)):
        time_classical = []
        time_quantum = []
        for i in bits_size:
            b1 = int_to_bits(2**i - 1)
            b2 = int_to_bits(2**i - 2)

            start_time = time.time()
            _ = min(b1, b2)
            end_time = time.time()
            time_classical.append(end_time - start_time)

            _, time_q = compare_bitstring(b1, b2, return_time=True)
            time_quantum.append(time_q)

        # Compute the pourcentage of difference between the first and the last time for both classical and quantum
        classical_pourcentage.append(
            (time_classical[-1] - time_classical[0])
            / max(time_classical[0], 0.00001)
            * 100
        )
        quantum_pourcentage.append(
            (time_quantum[-1] - time_quantum[0]) / time_quantum[0] * 100
        )

    print(f"Classical pourcentage mean: {np.mean(classical_pourcentage)}")
    print(f"Quantum pourcentage mean: {np.mean(quantum_pourcentage)}")


# compare_bitstring_compare_time()


def quantum_find_min(list_of_bits, shots=100, only_index=False) -> (int, int):
    """
    Find the minimum bitstring in a list of bitstrings and return its index
    :param list_of_bits: list of bitstrings to compare e.g. ["0101", "0100", "0110", "0010", "1001"]
    :param shots: number of shots to perform
    :param only_index: if True, return only the index of the minimum bitstring
    :return: value of the min and index of it in list_of_bits or only the index
    """
    min_index = 0
    min_value = bits_to_int(list_of_bits[0])
    for i in range(1, len(list_of_bits)):
        # print(f"Comparing {list_of_bits[min_index]} and {list_of_bits[i]}")
        counts = compare_bitstring(
            list_of_bits[min_index], list_of_bits[i], shots=shots
        )
        if "01" not in counts:
            counts["01"] = 0
        if "10" not in counts:
            counts["10"] = 0
        if counts["01"] > counts["10"]:
            min_index = i
            min_value = bits_to_int(list_of_bits[i])

    if only_index:
        return min_index

    return min_value, min_index


def quantum_find_max(list_of_bits, shots=1024, only_index=False) -> (int, int):
    """
    Find the maximum bitstring in a list of bitstrings and return its index
    :param list_of_bits: list of bitstrings to compare e.g. ["0101", "0100", "0110", "0010", "1001"]
    :param shots: number of shots to perform
    :param only_index: if True, return only the index of the maximum bitstring
    :return: value of the max and index of it in list_of_bits or only the index
    """
    max_index = 0
    max_value = bits_to_int(list_of_bits[0])
    for i in range(1, len(list_of_bits)):
        # print(f"Comparing {list_of_bits[min_index]} and {list_of_bits[i]}")
        counts = compare_bitstring(
            list_of_bits[max_index], list_of_bits[i], shots=shots
        )
        if "01" not in counts:
            counts["01"] = 0
        if "10" not in counts:
            counts["10"] = 0
        if counts["01"] < counts["10"]:
            max_index = i
            max_value = bits_to_int(list_of_bits[i])

    if only_index:
        return max_index

    return max_value, max_index


# Test quantum_find_min
def test_quantum_find_min():
    bitstrings = ["1101", "0010", "1110", "0110", "0101", "1111", "1011"]
    min_value, min_index = quantum_find_min(bitstrings, shots=100)
    print(f"Bitstring : {[bits_to_int(bitstring) for bitstring in bitstrings]}")
    print(
        f"The minimum bitstring is: {bitstrings[min_index]}, with value {min_value} and index {min_index}"
    )
    return min_value, min_index


# test_quantum_find_max
def test_quantum_find_max():
    bitstrings = ["1101", "0010", "1110", "0110", "0101", "1111", "1011"]
    max_value, max_index = quantum_find_max(bitstrings, shots=4096)
    print(f"Bitstring : {[bits_to_int(bitstring) for bitstring in bitstrings]}")
    print(
        f"The maximum bitstring is: {bitstrings[max_index]}, with value {max_value} and index {max_index}"
    )
    return max_value, max_index


def get_success_rate_min(nb_bits=3, list_size=3, nb_tests=50, shots=5):
    """
    Get the success rate of quantum_find_min with random bitstrings
    :param nb_bits: number of bits to represent each integer
    :param list_size: number of integers in the list
    :param nb_tests: number of tests to perform
    :param shots: number of shots to perform
    """
    random.seed(0)
    nb_success = 0
    for _ in tqdm(range(nb_tests)):
        list_of_ints = [random.randint(0, 2**nb_bits - 1) for _ in range(list_size)]
        list_of_bits = [int_to_bits(integer) for integer in list_of_ints]
        min_value, min_index = quantum_find_min(list_of_bits, shots=shots)
        if min_value == min(list_of_ints):
            nb_success += 1

    print(f"Success rate find min: {nb_success / nb_tests}")
    return nb_success / nb_tests


# get_success_rate_min(4, 5, 50, 5)


# Success rate with find_min
# Pourcentage de réussite avec une liste de 5 elements sur 2 bits : 0.72 (50 tests)
# Pourcentage de réussite avec une liste de 3 elements sur 2 bits : 0.74 (50 tests)
# Pourcentage de réussite avec une liste de 5 elements sur 3 bits : 0.82 (50 tests)
# Pourcentage de réussite avec une liste de 3 elements sur 3 bits : 0.96 (50 tests)
# Pourcentage de réussite avec une liste de 5 elements sur 4 bits : 0.52 (50 tests)
# Pourcentage de réussite avec une liste de 3 elements sur 4 bits : 0.70 (50 tests)
# Pourcentage de réussite avec une liste de 5 elements sur 5 bits : 0.46 (50 tests)
# Pourcentage de réussite avec une liste de 3 elements sur 5 bits : 0.62 (50 tests)
# Pourcentage de réussite avec une liste de 5 elements sur 6 bits : 0.48 (50 tests)
# Pourcentage de réussite avec une liste de 3 elements sur 6 bits : 0.68 (50 tests)
# Pourcentage de réussite avec une liste de 5 elements sur 7 bits : 0.44 (50 tests)
# Pourcentage de réussite avec une liste de 3 elements sur 7 bits : 0.62 (50 tests)
# Pourcentage de réussite avec une liste de 5 elements sur 8 bits : 0.38 (50 tests)
# Pourcentage de réussite avec une liste de 3 elements sur 8 bits : 0.52 (50 tests)


def plot_success_rate_min():
    nb_tests = 50
    nb_bits = [2, 3, 4, 5, 6, 7, 8]
    success_rate_size_5 = [0.72, 0.82, 0.52, 0.46, 0.48, 0.44, 0.38]
    success_rate_size_3 = [0.74, 0.96, 0.7, 0.62, 0.68, 0.62, 0.52]

    # Using plotly, create a scatter plot of the success rates with 2 lines
    fig = go.Figure()
    # With size 5, nb_bits on x-axis, success rate on y-axis
    fig.add_scatter(
        x=nb_bits,
        y=success_rate_size_5,
        name="Liste de 5 entiers",
        line=dict(color="firebrick", width=2, dash="dot"),
    )
    # With size 3, nb_bits on x-axis, success rate on y-axis
    fig.add_scatter(
        x=nb_bits,
        y=success_rate_size_3,
        name="Liste de 3 entiers",
        line=dict(color="royalblue", width=2, dash="dot"),
    )
    fig.update_layout(
        xaxis_title="Nombre de bits pour représenter un entier",
        yaxis_title=f"Taux de réussite de la recherche du minimum sur {nb_tests} tests",
        template="plotly_white",
        # text size
        font=dict(size=20),
    )

    # legend top right
    fig.update_layout(legend=dict(x=0.8, y=0.9))

    plot(fig, filename="success_rate.html")


# plot_success_rate()


def get_success_rate_max(nb_bits=5, list_size=3, nb_tests=50, shots=4096):
    """
    Get the success rate of quantum_find_max with random bitstrings
    :param nb_bits: number of bits to represent each integer
    :param list_size: number of integers in the list
    :param nb_tests: number of tests to perform
    :param shots: number of shots to perform
    """
    random.seed(0)
    nb_success = 0
    for _ in tqdm(range(nb_tests)):
        list_of_ints = [random.randint(0, 2**nb_bits - 1) for _ in range(list_size)]
        list_of_bits = [int_to_bits(integer) for integer in list_of_ints]
        max_value, max_index = quantum_find_max(list_of_bits, shots=shots)
        if max_value == max(list_of_ints):
            nb_success += 1

    print(
        f"# Pourcentage de réussite avec une liste de {list_size} elements sur {nb_bits} "
        f"bits : {nb_success / nb_tests} ({nb_tests} tests)"
    )
    return nb_success / nb_tests


# get_success_rate_max(4, 5, 50, 4096)


# Success rate with find_max
# Pourcentage de réussite avec une liste de 5 elements sur 4 bits : 0.64 (50 tests)
# Pourcentage de réussite avec une liste de 3 elements sur 4 bits : 0.82 (50 tests)


# ===== New version =====


def bit_compare_new():
    qr = QuantumRegister(2, "bits")
    aux = QuantumRegister(2, "aux")

    qc = QuantumCircuit(qr, aux)
    qc.x(qr[1])
    qc.mcx(qr, aux[0])
    qc.x(qr[0])
    qc.x(qr[1])
    qc.mcx(qr, aux[1])

    return qc


def reverse_bit_compare_new():
    qr = QuantumRegister(2, "bits")
    aux = QuantumRegister(2, "aux")

    qc = QuantumCircuit(qr, aux)
    qc.mcx(qr, aux[1])
    qc.x(qr[0])
    qc.x(qr[1])
    qc.mcx(qr, aux[0])
    qc.x(qr[1])

    return qc


def compare_integers(n_bits, a="101", b="010", plot_circuit=False):
    """
    Compare two integers using quantum computing.

    Parameters:
        n_bits: min number of bits to represent each integer
        a: first integer to compare, e.g. "101"
        b: second integer to compare, e.g. "010"
        plot_circuit: if True, plot the circuit
    """
    # Ajuster les chaînes de bits pour qu'elles aient une longueur n_bits
    a = a.rjust(n_bits, "0")
    b = b.rjust(n_bits, "0")

    # Création des registres quantiques
    bits_a = QuantumRegister(n_bits, "bits_a")
    bits_b = QuantumRegister(n_bits, "bits_b")
    # Utiliser un seul qubit auxiliaire pour le résultat final
    aux = QuantumRegister(
        3, "aux"
    )  # Deux pour la comparaison, un pour le résultat final
    qc = QuantumCircuit(bits_a, bits_b, aux)

    # Ajouter un registre classique pour la mesure
    cr = ClassicalRegister(1, "cr")
    qc.add_register(cr)

    # Utiliser un qubit auxiliaire supplémentaire pour indiquer une différence découverte
    diff_found = QuantumRegister(1, "diff_found")
    qc.add_register(diff_found)

    # Encodage des entiers à comparer
    qc.append(encode_bitstring(a).to_instruction(), bits_a)
    qc.append(encode_bitstring(b).to_instruction(), bits_b)

    # Boucle pour comparer chaque bit
    for i in range(n_bits):
        # 1) Appliquer la comparaison bit par bit
        qc_compare = bit_compare_new()
        qc.append(qc_compare.to_instruction(), [bits_a[i], bits_b[i], aux[0], aux[1]])

        """
        # Sans verification
        # 2) Appliquer la porte X contrôlée conditionnellement
        # Si aux[1] (le résultat de la comparaison actuelle) est à 1, et aux[2] (le résultat final) est à 0, mettre aux[2] à 1
        qc.x(aux[1])
        qc.mcx([aux[1], aux[0]], aux[2])
        qc.x(aux[1])  # Restaurer l'état original de aux[1] si nécessaire
        """

        # Avec verification
        # 2) Si aucune différence n'a été trouvée auparavant, effectuer la comparaison
        qc.x(diff_found[0])
        qc.x(aux[1])
        qc.mcx(
            [diff_found[0], aux[0], aux[1]], aux[2]
        )  # Mise à jour conditionnelle du qubit de résultat
        qc.mcx([aux[0], aux[1], aux[2]], diff_found[0])
        qc.x(aux[0])
        qc.x(aux[1])
        qc.mcx([diff_found[0], aux[0], aux[1]], aux[2])
        # Si une différence est trouvée dans cette comparaison, mettre diff_found à 1
        qc.mcx([aux[0], aux[1], aux[2]], diff_found[0])
        qc.x(aux[0])
        qc.x(diff_found[0])

        # 3) Inverser la comparaison pour réinitialiser les qubits auxiliaires
        qc_reverse_compare = reverse_bit_compare_new()
        qc.append(
            qc_reverse_compare.to_instruction(),
            [bits_a[i], bits_b[i], aux[0], aux[1]],
        )

    # Mesure du qubit aux[2] qui contient le résultat final de la comparaison
    qc.measure(aux[2], cr[0])

    # Afficher le circuit si demandé
    if plot_circuit:
        qc.draw(output="mpl")
        plt.show()

    return qc


"""
provider = IBMProvider()
backend = provider.get_backend("ibmq_qasm_simulator")

a = '111'
b = '000'
qubits_needed_num = max(len(a), len(b))
qc = compare_integers(qubits_needed_num, a, b, plot_circuit=True)
result = execute(qc, backend, shots=1).result()
counts = result.get_counts()
result = max(counts, key=counts.get)
if result == "1":
    print(
        f"The first integer {bits_to_int(a)} ({a}) is greater than {bits_to_int(b)} ({b})"
    )
elif result == "0":
    print(
        f"The first integer {bits_to_int(a)} ({a}) is smaller or equal than {bits_to_int(b)} ({b})"
    )
"""

# comparairing a with b:
# If '1' has higher score, then a > b
# If '0' has higher score, then a <= b


def test_compare_integers(number_of_bits):
    """
    2 entiers sur 1 bits: 100% d'accuracy
    2 entiers sur 2 bits: 100% d'accuracy
    2 entiers sur 3 bits: 99% d'accuracy
    2 entiers sur 4 bits: 70% d'accuracy
    2 entiers sur 5 bits: 72% d'accuracy
    """
    number_of_tests = 100
    nb_success = 0
    for _ in tqdm(range(number_of_tests)):
        a = int_to_bits(random.randint(0, 2**number_of_bits - 1))
        b = int_to_bits(random.randint(0, 2**number_of_bits - 1))
        qc = compare_integers(number_of_bits, a, b, plot_circuit=False)
        result = execute(qc, backend, shots=1).result()
        counts = result.get_counts()
        # get the most frequent result
        result = max(counts, key=counts.get)
        if result == "1" and bits_to_int(a) > bits_to_int(b):
            nb_success += 1
        elif result == "0" and bits_to_int(a) <= bits_to_int(b):
            nb_success += 1
        else:
            print(f"Error : response is {result} with a='{a}' and b='{b}'")
    print(f"Accuracy: {round((nb_success / number_of_tests)*100, 3)}")
    return round((nb_success / number_of_tests) * 100, 3)


def test_all_possibilities(number_of_bits):
    # test all comparison of int on number_of_bits
    number_of_tests = 2**number_of_bits
    nb_success = 0
    for i in tqdm(range(number_of_tests)):
        a = int_to_bits(i)
        for j in range(number_of_tests):
            b = int_to_bits(j)
            qc = compare_integers(number_of_bits, a, b, plot_circuit=False)
            result = execute(qc, backend, shots=1).result()
            counts = result.get_counts()
            # get the most frequent result
            result = max(counts, key=counts.get)
            if result == "1" and bits_to_int(a) > bits_to_int(b):
                nb_success += 1
            elif result == "0" and bits_to_int(a) <= bits_to_int(b):
                nb_success += 1
            else:
                print(f"Error : response is {result} with a='{a}' and b='{b}'")


provider = IBMProvider()
backend = provider.get_backend("simulator_mps")
# accuracy = test_compare_integers(1)
test_all_possibilities(1)

# ======================================================================================================================
# Compute distance two vectors using quantum computing
# ======================================================================================================================


def _encode_feature(x):
    """
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


def distance_centroids_parallel(
    point, centroids, backend, shots=1024, return_time=False
):
    """
    Estimates distances using quantum computer specified by backend
    Computes it in parallel for all centroids

    Parameters:
        point: point to measure distance from, e.g. [0.1, 0.2]
        centroids: list of centroids, e.g. [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]] (3 centroids so list of length 3)
        backend (IBMProvider backend): backend to use
        shots (int): number of shots to use
        return_time (bool): if True, return the circuit time

    Example:
        >>> point = [0.1, 0.2]
        >>> centroids = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]
        >>> dist = distance_centroids_parallel(point, centroids, backend, shots=1024)
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
    job_result = job.result()
    result = job_result.get_counts(qc)

    distance_centroids = [0] * k
    for i in range(k):
        keys_centroid_k = _binary_combinations_pos(k, i)
        for key in keys_centroid_k:
            if key in result:
                distance_centroids[i] += result[key]

    if return_time:
        circuit_time = job_result.results[0]._metadata["metadata"][
            "sample_measure_time"
        ]
        return distance_centroids, circuit_time

    return distance_centroids


def distances_for_multiple_examples_tests(
    num_examples=300, num_centroids=3, verbose=True, shots=1024
):
    """
    Test the distance_centroids_parallel function on multiple examples
    :param num_examples: int, number of examples
    :param num_centroids: int, number of centroids
    :param verbose: if True, print the results of each example
    :param shots: int, number of shots to use

    Examples:
    # num_examples=100, shots=2048, num_centroids=5
        Les tests sont terminés, 19 erreurs ont été trouvées sur 100 exemples.
        - 81.0 % de réussite !

    # num_examples=300, shots=4096, num_centroids=5
         Tests finished, 40 errors were found out of 200 examples.
         - 88.0 % success rate !
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


# distances_for_multiple_examples_tests(num_examples=300, verbose=False, shots=4096)


def quantum_vs_classical_time_distances_compute():
    """
    Compare the time of the classical and quantum distance_centroids_parallel
    """
    data_point = [0.1, 0.2]
    centroids = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8]]
    # - Classical time: 0.00019216537475585938 seconds
    # - Quantum time: 0.002270349 seconds, backend: ibmq_qasm_simulator

    # classical
    start_time = time.time()
    for centroid in centroids:
        _ = euclidean_distance(np.array(data_point), np.array(centroid))
    end_time = time.time()
    classical_time = end_time - start_time
    print(f"- Classical time: {classical_time} seconds")

    # quantum
    backend_name = "ibm_brisbane"
    backend = provider.get_backend(backend_name)
    _, quantum_time = distance_centroids_parallel(
        data_point, centroids, backend, shots=512, return_time=True
    )
    print(f"- Quantum time: {quantum_time} seconds, backend: {backend_name}")


# quantum_vs_classical_time_distances_compute()


if __name__ == "__main__":
    # IBMQ account
    provider = IBMProvider()
    backend = provider.get_backend(
        "simulator_mps"
    )  # simulator_mps, ibmq_qasm_simulator
