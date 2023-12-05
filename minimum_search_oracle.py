import math
from math import sqrt
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, execute
from qiskit_ibm_provider import IBMProvider
import matplotlib.pyplot as plt
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


# ======================================================================================================================
# Find the minimum of a list of integers using quantum computing and Grover algorithm
# ======================================================================================================================


def compare_bitstring_modified(
    bitstring_a, bitstring_b, qc, qra, qrb, qraux, qrint, oraux
):
    bits = max(
        qubits_needed(bits_to_int(bitstring_a)), qubits_needed(bits_to_int(bitstring_b))
    )

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

    # Si qraux[0] est à 1 (bitstring_a est plus petit), marquez le qubit oraux
    qc.cx(control_qubit=qraux[0], target_qubit=oraux[0])

    # Réinitialisation de qraux pour la prochaine comparaison
    # qc.reset(qraux)  # impossible car reset n'est pas disponible sur simulateur

    # Réinitialiser qraux sans utiliser 'reset'
    for i in range(2 * bits):
        qc.cx(qraux[i], oraux[0])  # Utilisez oraux[0] comme un qubit de "contrôle"
        qc.cx(oraux[0], qraux[i])  # Réinitialise qraux[i] si nécessaire
        qc.cx(qraux[i], oraux[0])  # Remettre oraux[0] dans son état original


def grover_oracle(
    circuit, candidate, list_bitstrings, qr, qra, qrb, qraux, qrint, oraux
):
    # Initialiser un qubit auxiliaire pour vérifier si le candidat est le plus petit
    circuit.x(oraux[0])  # Initialise à |1>
    circuit.h(oraux[0])  # Transforme en superposition

    for bitstring in list_bitstrings:
        if bitstring != candidate:
            # Comparer le candidat avec le bitstring
            compare_bitstring_modified(
                candidate, bitstring, circuit, qra, qrb, qraux, qrint, oraux
            )

            # Ici, oraux[0] est marqué par compare_bitstring_modified si le candidat est plus petit
            # Pour un oracle, nous voulons appliquer une porte si le candidat est plus petit que TOUS les bitstrings
            # Utilisez une porte logique appropriée ici si nécessaire

    # Appliquez la porte de phase conditionnelle si le candidat est plus petit que tous les bitstrings
    circuit.cz(oraux[0], oraux[1])

    # Réinitialiser les qubits auxiliaires à la fin
    circuit.h(oraux[0])
    circuit.x(oraux[0])


def apply_diffusion_operator(circuit, qr):
    # Appliquer les portes Hadamard à tous les qubits
    circuit.h(qr)
    # Appliquer les portes X à tous les qubits
    circuit.x(qr)
    # Appliquer une porte Z multi-contrôlée
    # Pour un nombre de qubits > 2, utilisez mcx avec un qubit auxiliaire ou mcmt (multi-controlled multi-target)
    circuit.h(qr[-1])  # Transformation pour une porte Z contrôlée
    circuit.mcx(qr[:-1], qr[-1])  # Porte Z contrôlée
    circuit.h(qr[-1])  # Retour à la base d'origine
    # Réappliquer les portes X
    circuit.x(qr)
    # Réappliquer les portes Hadamard
    circuit.h(qr)


def grover_find_min(list_bitstrings, shots=4096, iterations=None, show_info=True):
    # candidate
    candidate = list_bitstrings[0]

    # Déterminer le nombre de qubits nécessaires
    num_qubits = max(qubits_needed(bits_to_int(bs)) for bs in list_bitstrings)

    # Créer les registres quantiques et classiques nécessaires
    qr = QuantumRegister(
        num_qubits, "qr"
    )  # Assurez-vous que la taille correspond à la taille des bitstrings
    qra = QuantumRegister(num_qubits, "qra")
    qrb = QuantumRegister(num_qubits, "qrb")
    qraux = QuantumRegister(
        2 * num_qubits, "qraux"
    )  # Taille ajustée en fonction de votre circuit
    qrint = QuantumRegister(
        num_qubits - 1, "qrint"
    )  # Taille ajustée en fonction de votre circuit
    oraux = QuantumRegister(2, "oraux")  # Qubits auxiliaires pour l'oracle
    cr = ClassicalRegister(num_qubits, "cr")  # Pour la mesure

    # Créer le circuit
    circuit = QuantumCircuit(qr, qra, qrb, qraux, qrint, oraux, cr)

    # Initialisation de l'état de superposition
    circuit.h(qr)

    if not iterations:
        # Nombre d'itérations (approximativement sqrt(N))
        iterations = int(sqrt(len(list_bitstrings)))
        # iterations = 20

    print(f"Iterations : {iterations}") if show_info else None

    # Appliquer l'oracle et l'opérateur de diffusion plusieurs fois
    for iteration in range(iterations):
        # Oracle
        grover_oracle(
            circuit, candidate, list_bitstrings, qr, qra, qrb, qraux, qrint, oraux
        )
        # Opérateur de diffusion
        apply_diffusion_operator(circuit, qr)

    # Mesurer les qubits
    circuit.measure(qr, cr)

    # Exécution du circuit
    result = execute(circuit, backend, shots=shots).result()
    counts = result.get_counts(circuit)
    print(f"List of bitstrings : {list_bitstrings}") if show_info else None
    print(
        f"Lits of ints : {[bits_to_int(bitstring) for bitstring in list_bitstrings]}"
    ) if show_info else None
    # print(f"Counts : {counts}")
    # Keep only the keys in counts that are in list_bitstrings
    counts = {key: counts[key] for key in counts if key in list_bitstrings}
    # show the 2 most frequent results if enough shots
    if len(counts) > 1:
        first_min, second_min = sorted(counts, key=counts.get, reverse=True)[:2]
        print(
            f"First min : {first_min}, with value {bits_to_int(first_min)}"
        ) if show_info else None
        print(
            f"Second min : {second_min}, with value {bits_to_int(second_min)}"
        ) if show_info else None
    elif len(counts) == 1:
        first_min = list(counts.keys())[0]
        print(
            f"First min : {first_min}, with value {bits_to_int(first_min)}"
        ) if show_info else None
    else:
        first_min = None
        print("No min found") if show_info else None

    print(
        f"Expected min : {min([bits_to_int(bitstring) for bitstring in list_bitstrings])}"
    ) if show_info else None

    if first_min:
        return bits_to_int(first_min)
    else:
        return None


if __name__ == "__main__":
    # IBMQ account
    provider = IBMProvider()
    backend = provider.get_backend(
        "simulator_mps"
    )  # simulator_mps, ibmq_qasm_simulator

    list_bitstrings = [
        "111",
        "101",
        "010",
        "001",
        "100",
        "011",
        "110",
        "111",
        "100",
        "101",
        "110",
        "111",
        "000",
        "001",
        "010",
        "011",
        "100",
        "101",
        "110",
        "111",
        "010",
    ]

    # test k times the grover_find_min function with list_bitstrings for iterations between sqrt(N) and N
    total_success = 0
    k = 100
    res = {
        k: None
        for k in range(int(sqrt(len(list_bitstrings))), len(list_bitstrings) + 1)
    }

    for it in tqdm(range(int(sqrt(len(list_bitstrings))), len(list_bitstrings) + 1)):
        total_success = 0
        for _ in range(k):
            min_calculated = grover_find_min(
                list_bitstrings, shots=2096, iterations=it, show_info=False
            )
            if min_calculated == min(
                [bits_to_int(bitstring) for bitstring in list_bitstrings]
            ):
                total_success += 1
        res[it] = total_success / k

    # show results
    for k, v in res.items():
        print(f"Number of iterations : {k}, success rate : {v}")

    """
    len_list = 2

    # Test the quantum_find_min function 100 times with random bitstrings
    total_success = 0
    for i in range(10):
        list_int = random.sample(range(1, 2**3), len_list)
        num_bits = max([qubits_needed(integer) for integer in list_int])
        list_bitstrings = [int_to_bits(integer, num_bits) for integer in list_int]

        min_calculated = grover_find_min(list_bitstrings, shots=20)
        if min_calculated == min(list_int):
            total_success += 1

    print(f"Success rate: {total_success / 100}")
    """
