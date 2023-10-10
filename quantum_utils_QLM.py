from qat.lang.AQASM import Program, QRoutine, H, X, CNOT, CCNOT
from qat.lang.AQASM.bits import QRegister
from config import QLM_HOSTNAME
from qat.qlmaas import QLMaaSConnection

# test if conn variable is defined
if "conn" not in locals():
    conn = QLMaaSConnection(hostname=QLM_HOSTNAME, authentication="password")
# Get remote QPU
LinAlg = conn.get_qpu("qat.qpus:LinAlg")
qpu = LinAlg()

import math
import numpy as np
import time

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


# ======================================================================================================================
# Find the minimum of a list of integers using quantum computing
# ======================================================================================================================


def qubits_needed(n):
    if n < 0:
        raise ValueError("Le nombre doit être non négatif")
    elif n == 0:
        return 1  # Un qubit est nécessaire pour représenter 0
    else:
        return math.ceil(math.log2(n + 1))


def bits_to_int(bits):
    """
    Convert a string of bits to an integer
    :param bits: string of bits, e.g. "101"
    :return: integer representation of bits, e.g. 5
    """
    return int(bits, 2)


def extract_intermediate_measurements(result):
    output_str = ""
    for sample in result.raw_data:
        for im in sample.intermediate_measurements:
            cbits_str = "".join(["1" if bit else "0" for bit in im.cbits])
            output_str += cbits_str
    return output_str


def toffoli(prog, control1, control2, target):
    prog.apply(H, target)
    prog.apply(CNOT, control2, target)
    prog.apply(H, target)
    prog.apply(CNOT, control1, control2)
    prog.apply(H, target)
    prog.apply(CNOT, control2, target)
    prog.apply(H, target)
    prog.apply(CNOT, control1, control2)


def int_to_bits(integer, num_bits=4):
    """
    Convert an integer to a bitstring
    :param integer: integer to convert
    :param num_bits: number of bits to represent the integer
    :return: bitstring representation of integer
    """
    return format(integer, f"0{num_bits}b")


def submit_and_wait_for_result(qpu, circuit, timeout=60):
    """
    Soumet un job au QPU et attend que le travail soit terminé.

    Paramètres:
        qpu: L'instance du QPU sur lequel le job sera exécuté.
        circuit: Le circuit quantique à exécuter.
        timeout: Le délai d'attente maximal en secondes (par défaut à 60 secondes).

    Retourne:
        Le résultat du job si réussi, None sinon.
    """
    # Soumettre le job
    job = qpu.submit(circuit.to_job())

    # Enregistrer le moment du début
    start_time = time.time()

    # Attendre que le job soit terminé
    while job.get_status() != "done":
        # Vérifier le délai d'attente
        if time.time() - start_time > timeout:
            print("Le travail n'a pas terminé dans le délai imparti.")
            return None
        # Attendre avant de vérifier à nouveau
        time.sleep(1)

    # Récupérer et retourner le résultat
    return job.get_result()


def encode(bit):
    routine = QRoutine()
    qb = routine.new_wires(1)
    if bit == 1:
        routine.apply(X, qb)
    return routine


def bit_compare():
    routine = QRoutine()
    a, b, aux1, aux2 = routine.new_wires(4)
    routine.apply(CNOT, a, aux1)
    routine.apply(CNOT, b, aux1)
    routine.apply(CNOT, a, aux2)
    routine.apply(CNOT, b, aux2)
    routine.apply(X, aux2)
    return routine


def compare_bitstring(bitstring_a, bitstring_b):
    bits = max(
        qubits_needed(bits_to_int(bitstring_a)), qubits_needed(bits_to_int(bitstring_b))
    )
    prog = Program()
    qra = prog.qalloc(bits, class_type=QRegister)
    qrb = prog.qalloc(bits, class_type=QRegister)
    qraux = prog.qalloc(2 * bits, class_type=QRegister)
    qrint = prog.qalloc(bits - 1, class_type=QRegister)
    cr = prog.calloc(2)

    for i in range(bits):
        prog.apply(encode(bitstring_a[i]), qra[i])
        prog.apply(encode(bitstring_b[i]), qrb[i])
        prog.apply(bit_compare(), qra[i], qrb[i], qraux[2 * i], qraux[2 * i + 1])

        if i < bits - 1:
            prog.apply(X, qraux[2 * i])
            prog.apply(X, qraux[2 * i + 1])
            prog.apply(CCNOT, qraux[2 * i], qraux[2 * i + 1], qrint[i])
            prog.apply(X, qraux[2 * i])
            prog.apply(X, qraux[2 * i + 1])

    for i in range(0, bits - 1):
        prog.apply(CCNOT, qraux[2 * (-i - 1)], qrint[-i], qraux[2 * (-i)])
        prog.apply(CCNOT, qraux[2 * (-i - 1) + 1], qrint[-i], qraux[2 * (-i) + 1])

    # Mesurer qraux[0] et qraux[1]
    prog.measure(qraux[0], cr[0])
    prog.measure(qraux[1], cr[1])

    circuit = prog.to_circ()

    result = submit_and_wait_for_result(qpu, circuit)

    return extract_intermediate_measurements(result)


# Test compare_bitstring
b1 = "0110"
b2 = "0100"
# encode_bitstring(b1)
result = compare_bitstring(b1, b2)
# Afficher les résultats
# print(f"Résultats du circuit : {result}")

# if '01' has higher score, then the second bitstring is smaller
# if '10' has higher score, then the first bitstring is smaller
# if '00' has higher score, then the bitstrings are equal


def quantum_find_min(list_of_bits, only_index=False):
    """
    Find the minimum bitstring in a list of bitstrings and return its index
    :param list_of_bits: list of bitstrings to compare e.g. ["0101", "0100", "0110", "0010", "1001"]
    :param only_index: if True, return only the index of the minimum bitstring
    :return: value of the min and index of it in list_of_bits or only the index
    """
    if not list_of_bits:
        return None

    min_bitstring = list_of_bits[0]
    min_index = 0

    for i, bitstring in enumerate(list_of_bits[1:], 1):
        print(f"Comparing {bits_to_int(min_bitstring)} and {bits_to_int(bitstring)}")
        comparison_result = compare_bitstring(min_bitstring, bitstring)

        # Interprétez le résultat de la comparaison ici
        # Par exemple, si '10' cela signifie que min_bitstring est plus petit
        # Si '01' cela signifie que bitstring est plus petit

        if comparison_result == "01":
            print(
                f"{bits_to_int(bitstring)} is smaller than {bits_to_int(min_bitstring)}"
            )
            min_bitstring = bitstring
            min_index = i
        elif comparison_result == "10":
            print(
                f"{bits_to_int(min_bitstring)} is smaller than {bits_to_int(bitstring)}"
            )
        else:
            print(
                f"{bits_to_int(bitstring)} and {bits_to_int(min_bitstring)} are equal"
            )

    if only_index:
        return min_index
    else:
        return min_bitstring, min_index


# Test quantum_find_min
list_of_bits = ["0101", "0100", "0110", "0010", "1001"]
list_of_ints = [bits_to_int(bitstring) for bitstring in list_of_bits]
min_bitstring, min_index = quantum_find_min(list_of_bits)
print(f"Quantum Index of minimum bitstring: {min_index}")
print(f"Real Index of minimum bitstring: {list_of_ints.index(min(list_of_ints))}")
