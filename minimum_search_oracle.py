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
    diff_found = QuantumRegister(2, "diff_found")
    qc.add_register(diff_found)
    qc.x(diff_found[0])  # Initialiser à 1
    qc.x(diff_found[1])  # Initialiser à 1

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
        qc.x(aux[1])
        qc.mcx(
            [diff_found[0], diff_found[1], aux[0], aux[1]], aux[2]
        )  # Mise à jour conditionnelle du qubit de résultat
        qc.mcx([aux[0], aux[1], aux[2]], diff_found[0])
        qc.x(aux[1])
        # ----------------------------------
        qc.x(aux[0])
        qc.mcx([aux[0], aux[1]], diff_found[1])
        qc.x(aux[0])
        qc.x(diff_found[0])
        qc.x(diff_found[1])
        qc.mcx([diff_found[1], diff_found[0]], aux[2])
        qc.x(diff_found[1])
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


provider = IBMProvider()
backend = provider.get_backend("ibmq_qasm_simulator")

"""
a = '1'
b = '0'
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


"""
provider = IBMProvider()
backend = provider.get_backend("simulator_mps")
# accuracy = test_compare_integers(1)
for i in range(1, 5):
    print(f"Number of bits: {i}")
    test_all_possibilities(i)
"""
