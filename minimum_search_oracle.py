from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit.circuit.library.standard_gates import ZGate
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt
from math import log2, ceil
from qiskit_ionq import IonQProvider
from qiskit_ibm_runtime import QiskitRuntimeService, Sampler, Options
from config import IBM_QUANTUM_API_TOKEN, IONQ_API_TOKEN

# IMPORTANT : Ce code utilise la nouvelle version de qiskit (1.0.1)


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


def oracle_compare_integers(Qa, Qb, Qaux1, Qaux2, Qres, Qfin, n_bits):
    """
    Crée un oracle pour comparer deux entiers a et b sur n_bits
    Pour cela, il suffit de remplacer la grande porte CX par une porte CZ
    Et de supprimer le qubit de sortie classique
    :param Qa: QuantumRegister, registre quantique pour le premier entier (n_bits qubits préparés)
    :param Qb: QuantumRegister, registre quantique pour le deuxième entier (n_bits qubits préparés)
    :param Qaux1: QuantumRegister, registre quantique pour les résultats intermédiaires (n_bits qubits)
    :param Qaux2: QuantumRegister, registre quantique pour les résultats intermédiaires (n_bits qubits)
    :param Qres: QuantumRegister, registre quantique pour les résultats de la comparaison (n_bits - 1 qubits)
    :param Qfin: QuantumRegister, registre quantique pour le résultat final (1 qubit)
    :param n_bits: int, nombre de bits pour la comparaison
    """
    # circuit quantique
    qc = QuantumCircuit(Qa, Qb, Qaux1, Qaux2, Qres, Qfin)

    # circuit original du papier de Oliveira et Ramos
    # 1/ comparaison Uc
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
    # 4/ On remplace la grande porte CX par une porte CZ
    qc.cz([Qaux2[0]], Qfin)

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
    # 7/ Décalcul avec Uc-1
    for i in reversed(range(n_bits)):
        qc_compare = reverse_bit_compare()
        qc.append(qc_compare.to_instruction(), [Qa[i], Qb[i], Qaux1[i], Qaux2[i]])

    return qc


# PRÉPARATION DES ÉTATS SUPERPOSÉS (GROVER)


def oracle_grover_preparation(n_bits, L, Qa, Qfin):
    """
    Crée un oracle pour la recherche d'éléments en vue de les amplifier
    La superposition de n qubits va créer une superposition de 2**n états, mais notre liste L ne va pas contenir
    tous ces états. Il faut donc créer un oracle pour marquer les états de L dans cette superposition
    Pour cela on va créer un oracle qui va prendre en entrée un n-qubit et un qubit auxiliaire, et qui va
    appliquer une porte Z sur le qubit auxiliaire si l'entrée est dans L (avec une porte MCX)
    Si L[0] = 4 et n_bits = 3, alors l'oracle doit marquer l'état 100 donc la première porte Z doit être appliquée
    sur le 4e qubit si le premier qubit est à 1, le deuxième à 0 et le troisième à 0.

    Parametres:
    :param n_bits: nombre de bits sur lesquels les entiers sont codés
    :param L: liste des entiers à marquer dans la superposition
    :param Qa: QuantumRegister, registre quantique pour les entiers à comparer (n_bits qubits)
    :param Qfin: QuantumRegister, registre quantique du qubit auxiliaire sur lequel appliquer la porte Z (1 qubit)
    """
    # circuit quantique
    qc = QuantumCircuit(Qa, Qfin)

    # Création de l'oracle
    for integer in L:
        # Création de l'oracle
        # On crée un masque pour marquer l'entier dans la superposition
        mask = format(integer, f"0{n_bits}b")
        # On applique une porte Z sur le qubit auxiliaire si l'entrée est dans L
        qc.mcx([Qa[i] for i in range(n_bits) if mask[i] == "1"], Qfin)

    return qc


# OPERATEUR DE DIFFUSION


def diffusion_operator(n_bits, Qa, Qfin):
    """
    Crée un opérateur de diffusion pour amplifier les états marqués par l'oracle
    :param n_bits: nombre de bits sur lesquels les entiers sont codés
    :param Qa: QuantumRegister, registre quantique pour les entiers à comparer (n_bits qubits) (superposés)
    :param Qfin: QuantumRegister, registre quantique du qubit auxiliaire sur lequel appliquer la porte Z (1 qubit)
    """

    qc = QuantumCircuit(Qa, Qfin)

    # Appliquer H à tous les qubits dans Qa
    qc.h(Qa)

    # Appliquer X à tous les qubits dans Qa
    qc.x(Qa)

    # Appliquer une porte Z conditionnelle sur l'état |0...0> en utilisant le qubit auxiliaire
    # Pour cela, on peut utiliser une porte Z multi-contrôlée
    # En Qiskit, on doit créer un ZGate et utiliser la méthode .control() pour obtenir une version contrôlée
    z_gate = ZGate().control(n_bits)
    qc.append(z_gate, Qa[:] + [Qfin[0]])

    # Réappliquer les portes X et H à tous les qubits dans Qa
    qc.x(Qa)
    qc.h(Qa)

    return qc


# MAIN CIRCUIT


def minimum_search_circuit(L, show_circuit=False, transpile_plot=False):
    """
    Circuit pour la recherche du minimum dans une liste d'entiers L

    Parametres:
    :param L: list, liste des entiers parmi lesquels chercher le minimum
    :param show_circuit: bool, afficher le circuit
    :param transpile_plot: bool, afficher le circuit transpilé

    :return: le minimum de L
    """
    # Nombre de bits nécessaires pour représenter les entiers de L
    n_bits = qubits_needed(max(L))

    # résultat
    min_L = None

    # Registre quantique pour les entiers à comparer
    Qa = QuantumRegister(n_bits, "a")
    # Registre quantique pour yi (b)
    Qb = QuantumRegister(n_bits, "b")
    # Registres quantiques pour les résultats intermédiaires (oracle comparaison de P)
    Qaux1 = QuantumRegister(n_bits, "aux1")
    Qaux2 = QuantumRegister(n_bits, "aux2")
    # Registre quantique pour les résultats de la comparaison (oracle comparaison de P)
    Qres = QuantumRegister(n_bits - 1, "res")
    # Qubit auxiliaire initialisé à ket(-)
    Qfin = QuantumRegister(1, "fin")
    # Registre classique pour le résultat (on mesure les n qubits de Qa)
    Cout = ClassicalRegister(n_bits, "cr")

    qc = QuantumCircuit(Qa, Qb, Qaux1, Qaux2, Qres, Qfin, Cout)

    # -- Superposition des n qubits de Qa --
    qc.h(Qa)

    # -- Préparation des états superposés pour la recherche de minimum (porte G répété g fois) --
    # On applique l'oracle pour marquer les états de L dans la superposition
    qc.append(
        oracle_grover_preparation(n_bits, L, Qa, Qfin).to_instruction(), [*Qa, *Qfin]
    )
    # On applique l'opérateur de diffusion pour amplifier les états marqués par l'oracle
    qc.append(diffusion_operator(n_bits, Qa, Qfin).to_instruction(), [*Qa, *Qfin])

    # -- Comparaison des états superposés avec l'entier b (porte P répétée p fois) --
    # On applique l'oracle pour comparer les entiers superposés avec l'entier b
    # TODO
    # On applique l'opérateur de diffusion pour amplifier les états marqués par l'oracle
    # TODO

    # -- Mesure des qubits de Qa --
    qc.measure(Qa, Cout)

    # -- Show circuit --
    if show_circuit:
        if transpile_plot:
            transpile(qc, backend).draw(output="mpl")
        else:
            qc.draw(output="mpl")
        plt.show()

    # -- On regarde le résultat de la mesure pour trouver le minimum --
    # On exécute le circuit
    qc = transpile(qc, backend)
    # TODO

    return min_L


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
    # check_all_possibilities(3)

    minimum_search_circuit([3, 5], show_circuit=True, transpile_plot=False)
