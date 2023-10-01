from qiskit import QuantumCircuit, execute, QuantumRegister, ClassicalRegister
from qiskit_ibm_provider import IBMProvider

import matplotlib.pyplot as plt
import math


def qubits_needed(n):
    if n < 0:
        raise ValueError("Le nombre doit être non négatif")
    elif n == 0:
        return 1  # Un qubit est nécessaire pour représenter 0
    else:
        return math.ceil(math.log2(n + 1))


def encode(bit):
    qr = QuantumRegister(1, "number")
    qc = QuantumCircuit(qr)
    if bit == "1":
        qc.x(qr[0])
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


def int_to_bits(integer, num_bits):
    """
    Convert an integer to a bitstring
    :param integer: integer to convert
    :param num_bits: number of bits to represent the integer
    :return: bitstring representation of integer
    """
    return format(integer, f"0{num_bits}b")


def compare_bitstring(bitstring_a, bitstring_b, plot_circuit=False):
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

    # Tell Qiskit how to simulate our circuit
    backend = provider.get_backend("ibmq_qasm_simulator")

    # Do the simulation, returning the result
    result = execute(qc, backend, shots=1024).result()

    # get the probability distribution
    counts = result.get_counts()

    if plot_circuit:
        qc.draw(output="mpl")
        plt.show()

    return counts


# IBMQ account
provider = IBMProvider()
backend = provider.get_backend("ibmq_qasm_simulator")

# Test compare_bitstring
# b1 = "101"
# b2 = "100"
# counts = compare_bitstring(b1, b2, plot_circuit=True)
# print(f"First bitstring: {b1}, that is {bits_to_int(b1)}")
# print(f"Second bitstring: {b2}, that is {bits_to_int(b2)}")
# print(counts)

# if '01' has higher score, then the second bitstring is smaller
# if '10' has higher score, then the first bitstring is smaller
# if '00' has higher score, then the bitstrings are equal


def quantum_find_min(list_of_bits) -> (int, int):
    """
    Find the minimum bitstring in a list of bitstrings and return its index
    :param list_of_bits: list of bitstrings to compare e.g. ["0101", "0100", "0110", "0010", "1001"]
    :return: index of minimum bitstring
    """
    min_index = 0
    min_value = bits_to_int(list_of_bits[0])
    for i in range(1, len(list_of_bits)):
        print(f"Comparing {list_of_bits[min_index]} and {list_of_bits[i]}")
        counts = compare_bitstring(list_of_bits[min_index], list_of_bits[i])
        if "01" not in counts:
            counts["01"] = 0
        if "10" not in counts:
            counts["10"] = 0
        if counts["01"] > counts["10"]:
            min_index = i
            min_value = bits_to_int(list_of_bits[i])
    return min_value, min_index


# Test quantum_find_min
bitstrings = ["1101", "1010", "1110", "0110", "0101", "1111", "1011"]
min_value, min_index = quantum_find_min(bitstrings)
print(f"Bitstring : {[bits_to_int(bitstring) for bitstring in bitstrings]}")
print(
    f"The minimum bitstring is: {bitstrings[min_index]}, with value {min_value} and index {min_index}"
)
