from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt
from math import log2, ceil
from qiskit_ionq import IonQProvider
from qiskit_ibm_runtime import QiskitRuntimeService, Sampler, Options
from config import IBM_QUANTUM_API_TOKEN, IONQ_API_TOKEN


# UTILS


def int_to_bits(integer):
    """
    Convert an integer to a bitstring using the minimum number of bits
    :param integer: integer to convert
    :return: bitstring representation of integer
    """
    return format(integer, "b")


def qubits_needed(n):
    if n < 0:
        raise ValueError("Le nombre doit être non négatif")
    elif n == 0:
        return 1  # Un qubit est nécessaire pour représenter 0
    else:
        return ceil(log2(n + 1))


# COMPARAISON DE DEUX ENTIERS


def bit_compare():
    Qr = QuantumRegister(2, "bits")
    Qaux = QuantumRegister(2, "aux")

    qc = QuantumCircuit(Qr, Qaux)
    qc.x(Qr[1])
    qc.mcx(Qr, Qaux[0])
    qc.x(Qr[0])
    qc.x(Qr[1])
    qc.mcx(Qr, Qaux[1])
    qc.x(Qr[0])
    return qc


def reverse_bit_compare():
    qr = QuantumRegister(2, "bits")
    aux = QuantumRegister(2, "aux")

    qc = QuantumCircuit(qr, aux)

    qc.x(qr[0])
    qc.mcx(qr, aux[1])
    qc.x(qr[0])
    qc.x(qr[1])
    qc.mcx(qr, aux[0])
    qc.x(qr[1])

    return qc


def encode(bit, plot_circuit=False):
    qr = QuantumRegister(1, "number")
    qc = QuantumCircuit(qr)
    if bit == "1":
        qc.x(qr[0])

    if plot_circuit:
        qc.draw(output="mpl")
        plt.show()
    return qc


def compare_integers(n_bits=3, a="101", b="010"):
    """

    :param n_bits: int, nombre de bits pour la comparaison
    :param a: str, chaîne de bits représentant le premier entier
    :param b: str, chaîne de bits représentant le deuxième entier

    Architecture du circuit:

    - 2 registres quantiques Qa et Qb pour les deux entiers à comparer (2 * n_bits qubits)
    - 2 registres quantiques Qaux1 et Qaux2 pour les résultats intermédiaires (2 * n_bits qubits)
    - 1 registre quantique Qres pour les résultats de la comparaison (n_bits - 1 qubits)
    - 1 registre quantique Qfin pour le résultat final (1 qubit)
    - 1 registre classique Cout pour le résultat final (classique)
    """
    # Ajuster les chaînes de bits pour qu'elles aient une longueur n_bits
    a = a.rjust(n_bits, "0")
    b = b.rjust(n_bits, "0")

    # valeur à comparer
    Qa = QuantumRegister(n_bits, "a")
    Qb = QuantumRegister(n_bits, "b")
    # sortie comparateur
    Qaux1 = QuantumRegister(n_bits, "aux1")
    Qaux2 = QuantumRegister(n_bits, "aux2")
    # resultat comparaison
    Qres = QuantumRegister(n_bits - 1, "res")
    # resultat final
    Qfin = QuantumRegister(1, "fin")

    # circuit quantique
    qc = QuantumCircuit(Qa, Qb, Qaux1, Qaux2, Qres, Qfin)

    # registre classique pour le résultat
    Cout = ClassicalRegister(1, "cr")
    qc.add_register(Cout)

    # Encodage des entiers à comparer en utilisant encode()
    for i in range(n_bits):
        qc.append(encode(a[i]).to_instruction(), [Qa[i]])
        qc.append(encode(b[i]).to_instruction(), [Qb[i]])

    # circuit original du papier de Oliveira et Ramos
    # 1/ comparaison
    for i in range(n_bits):
        # Appliquer la comparaison bit par bit
        qc_compare = bit_compare()
        qc.append(qc_compare.to_instruction(), [Qa[i], Qb[i], Qaux1[i], Qaux2[i]])

    # 2/ portes iCCNOT (test de non égalité de qubit Qaux1/2)
    for i in range(n_bits - 1):
        qc.x(Qaux1[i])
        qc.x(Qaux2[i])
        qc.mcx([Qaux1[i], Qaux2[i]], Qres[i])
        qc.x(Qaux1[i])
        qc.x(Qaux2[i])

    # 3/ remontée des résultats
    for i in reversed(range(n_bits - 1)):
        qc.mcx([Qaux2[i + 1], Qres[i]], Qaux2[i])

    # modification: retour pour un bit pour l'ordre (à inverser si besoin)
    # 4/ copie de Qaux2[0]: 1=> dans l'ordre, 0 => supérieur ou égal
    qc.mcx([Qaux2[0]], Qfin)

    # 5/ inverse remontée des résultats
    for i in range(n_bits - 1):
        qc.mcx([Qaux2[i + 1], Qres[i]], Qaux2[i])

    # 6/ inverse des iCCNOT
    for i in range(n_bits - 1):
        qc.x(Qaux1[i])
        qc.x(Qaux2[i])
        qc.mcx([Qaux1[i], Qaux2[i]], Qres[i])
        qc.x(Qaux1[i])
        qc.x(Qaux2[i])

    # faire uncompute ici pour libérer les 3n registres temporaires
    # 7/ uncompute avec reverse_bit_compare_new()
    for i in reversed(range(n_bits)):
        qc_compare = reverse_bit_compare()
        qc.append(qc_compare.to_instruction(), [Qa[i], Qb[i], Qaux1[i], Qaux2[i]])

    # Mesure du qubit aux[2] qui contient le résultat final de la comparaison
    qc.measure(Qfin, Cout)

    return qc


# TESTS


def check_all_possibilities(number_of_bits):
    """
    Teste toutes les possibilités pour une comparaison de 2 entiers sur n_bits
    Utilise la fonction compare_integers pour comparer tous les entiers de 0 à 2**n_bits - 1

    Parameters:
        number_of_bits: nombre de bits pour la comparaison, pour que les entiers soient codés sur n_bits
    """
    number_of_tests = 2**number_of_bits
    operator = {"1": "<", "0": ">="}

    for i in range(number_of_tests):  # tqdm(range(number_of_tests)):
        a = format(i, "b")
        for j in range(number_of_tests):
            b = format(j, "b")
            print(
                f'a={a.rjust(number_of_bits, "0")}  b={b.rjust(number_of_bits, "0")} : ',
                end="",
            )
            qc = compare_integers(number_of_bits, a, b)
            qc = transpile(qc, backend)
            result = backend.run(qc, shots=1).result()
            out = [*result.get_counts()]
            print(f"{i} {operator[out[0]]} {j}", end="")
            if ((i < j) and (out[0] == "1")) or ((i >= j) and (out[0] == "0")):
                print(f" ok")
            else:
                print(f" failed")


if __name__ == "__main__":
    platform = "IONQ"
    if platform == "IBM":
        print("IBM")
        # Load saved credentials
        service = QiskitRuntimeService()
        backend = service.backend("simulator_mps")
    elif platform == "IONQ":
        print("IONQ")

        provider = IonQProvider(IONQ_API_TOKEN)
        backend = provider.get_backend("ionq_simulator")

    # Test de toutes les possibilités pour une comparaison de 2 entiers sur 3 bits
    check_all_possibilities(3)
