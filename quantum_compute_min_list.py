from qiskit import QuantumCircuit, execute, QuantumRegister, ClassicalRegister
from qiskit_ibm_provider import IBMProvider
from qiskit.tools.visualization import circuit_drawer

import plotly.graph_objects as go
from plotly.offline import plot
import matplotlib.pyplot as plt
import math
import random
from tqdm import tqdm


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


def encode_bitstring(bitstring, plot_circuit=True):
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


def int_to_bits(integer, num_bits=4):
    """
    Convert an integer to a bitstring
    :param integer: integer to convert
    :param num_bits: number of bits to represent the integer
    :return: bitstring representation of integer
    """
    return format(integer, f"0{num_bits}b")


def compare_bitstring(bitstring_a, bitstring_b, plot_circuit=False, shots=1024):
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

        if i < bits - 1:
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

    # get the probability distribution
    counts = result.get_counts()

    if plot_circuit:
        circuit_drawer(qc, output="mpl")
        plt.show()

    return counts


# IBMQ account
provider = IBMProvider()
backend = provider.get_backend("simulator_mps")

# Test compare_bitstring
# b1 = "11"
# b2 = "01"
# counts = compare_bitstring(b1, b2, plot_circuit=True)
# print(f"First bitstring: '{b1}', that is {bits_to_int(b1)}")
# print(f"Second bitstring: '{b2}', that is {bits_to_int(b2)}")
# print(counts)

# if '01' has higher score, then the second bitstring is smaller
# if '10' has higher score, then the first bitstring is smaller
# if '00' has higher score, then the bitstrings are equal


def quantum_find_min(list_of_bits, shots=1024, only_index=False) -> (int, int):
    """
    Find the minimum bitstring in a list of bitstrings and return its index
    :param list_of_bits: list of bitstrings to compare e.g. ["0101", "0100", "0110", "0010", "1001"]
    :param shots: number of shots to perform
    :param only_index: if True, return only the index of the minimum bitstring
    :return: index of minimum bitstring
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


# Test quantum_find_min
def test_quantum_find_min():
    bitstrings = ["1101", "0010", "1110", "0110", "0101", "1111", "1011"]
    min_value, min_index = quantum_find_min(bitstrings, shots=4096)
    print(f"Bitstring : {[bits_to_int(bitstring) for bitstring in bitstrings]}")
    print(
        f"The minimum bitstring is: {bitstrings[min_index]}, with value {min_value} and index {min_index}"
    )
    return min_value, min_index


def get_success_rate(nb_bits=5, list_size=3, nb_tests=50, shots=4096):
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
        list_of_bits = [int_to_bits(integer, nb_bits) for integer in list_of_ints]
        min_value, min_index = quantum_find_min(list_of_bits, shots=shots)
        if min_value == min(list_of_ints):
            nb_success += 1

    print(f"Success rate: {nb_success / nb_tests}")
    return nb_success / nb_tests


# get_success_rate(4, 5, 50, 4096)


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


def plot_success_rate():
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
