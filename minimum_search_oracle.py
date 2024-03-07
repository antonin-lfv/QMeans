from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_ionq import IonQProvider
import matplotlib.pyplot as plt
from math import log2, ceil
from config import IONQ_API_TOKEN
import plotly.graph_objects as go
from plotly.offline import plot
import random

# IMPORTANT : Ce code utilise la nouvelle version de qiskit (1.0.1)


# UTILS


def int_to_bits(integer):
    """
    Convert an integer to a bitstring using the minimum number of bits
    :param integer: integer to convert
    :return: bitstring representation of integer
    """
    return format(integer, "b")


def bits_to_int(bits):
    """
    Convert a string of bits to an integer
    :param bits: string of bits, e.g. "101"
    :return: integer representation of bits, e.g. 5
    """
    return int(bits, 2)


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
    # 4/ Porte controlée Z sur Qfin si Qaux2[0] est à 1 (Z = H.X.H)
    qc.h(Qfin)
    qc.mcx([Qaux2[0]], Qfin)
    qc.h(Qfin)

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


def show_oracle_compare_integers(n_bits=2):
    """
    Affiche le circuit de l'oracle pour comparer deux entiers a et b sur n_bits
    :param n_bits: int, nombre de bits pour la comparaison
    """
    Qa = QuantumRegister(n_bits, "a")
    Qb = QuantumRegister(n_bits, "b")
    Qaux1 = QuantumRegister(n_bits, "aux1")
    Qaux2 = QuantumRegister(n_bits, "aux2")
    Qres = QuantumRegister(n_bits - 1, "res")
    Qfin = QuantumRegister(1, "fin")

    qc = oracle_compare_integers(Qa, Qb, Qaux1, Qaux2, Qres, Qfin, n_bits)

    qc = transpile(qc, backend)
    qc.draw(output="mpl")
    plt.show()


# PRÉPARATION DES ÉTATS SUPERPOSÉS (GROVER)


def oracle_grover_preparation(n_bits, L, Qa, Qfin):
    """
    Crée un oracle pour la recherche d'éléments en vue de les amplifier
    La superposition de n qubits va créer une superposition de 2**n états, mais notre liste L ne va pas contenir
    tous ces états. Il faut donc créer un oracle pour marquer les états de L dans cette superposition
    Pour cela on va créer un oracle qui va prendre en entrée un n-qubit et un qubit auxiliaire, et qui va
    appliquer une porte Z sur le qubit auxiliaire si l'entrée est dans L (avec une porte MCX)
    Si L[0] = 4 et n_bits = 3, alors l'oracle doit marquer l'état 100 donc la première porte Z doit être appliquée
    sur le 4e qubit si le premier qubit est à 1, le deuxième à 0 (donc égale à 1 après une porte X) et le troisième à 0
    (donc égale à 1 après une porte X).

    Parametres:
    :param n_bits: nombre de bits sur lesquels les entiers sont codés
    :param L: liste des entiers à marquer dans la superposition
    :param Qa: QuantumRegister, registre quantique pour les entiers à comparer (n_bits qubits)
    :param Qfin: QuantumRegister, registre quantique du qubit auxiliaire sur lequel appliquer la porte Z (1 qubit)
    """
    # circuit quantique
    Qa = Qa[::-1]  # Inverser l'ordre des qubits pour correspondre à l'ordre des bits
    qc = QuantumCircuit(Qa, Qfin)

    # On doit supprimer les occurences multiples dans L sinon la porte Z va s'annuler (si le nombre est pair)
    L = list(set(L))

    # Création de l'oracle
    for integer in L:
        # Création de l'oracle
        # On crée un masque pour marquer l'entier dans la superposition
        mask = format(integer, f"0{n_bits}b")

        # On applique des portes NOT (X) sur les qubits qui doivent être à 0 pour qu'il passe à 1
        for i in range(n_bits):
            if mask[i] == "0":
                qc.x(Qa[i])

        # On applique une porte Z sur le qubit auxiliaire si l'entrée est dans L (tout Qa doit être à 1)
        qc.h(Qfin)
        qc.mcx(Qa, Qfin)
        qc.h(Qfin)

        # On applique les portes NOT (X) inverses pour remettre les qubits à leur état initial
        for i in range(n_bits):
            if mask[i] == "0":
                qc.x(Qa[i])

    return qc


def show_oracle_grover_preparation(L, transpile_plot=False):
    """
    Affiche le circuit de l'oracle pour la recherche d'éléments en vue de les amplifier
    """
    n_bits = max(qubits_needed(max(L)), 2)
    Qa = QuantumRegister(n_bits, "a")
    Qfin = QuantumRegister(1, "fin")

    qc = oracle_grover_preparation(n_bits, L, Qa, Qfin)

    if transpile_plot:
        qc = transpile(qc, backend)
    qc.draw(output="mpl")
    plt.show()


# OPERATEUR DE DIFFUSION


def diffusion_operator(Qa):
    """
    Crée un opérateur de diffusion pour amplifier les états marqués par l'oracle
    :param Qa: QuantumRegister, registre quantique pour les entiers à comparer (n_bits qubits) (superposés)
    """
    qc = QuantumCircuit(Qa)

    # Appliquer H à tous les qubits dans Qa
    qc.h(Qa)
    # Appliquer X à tous les qubits dans Qa
    qc.x(Qa)

    # Appliquer une porte Z conditionnelle sur l'état |0...0> (après application des portes X, c'est l'état |1...1>)
    # Pour cela, on applique une porte H sur le dernier qubit pour le préparer à la porte Z conditionnelle
    qc.h(Qa[-1])
    # Appliquer une porte Z conditionnelle (CZ) simulée par MCX avec inversion sur le dernier qubit
    qc.mcx(Qa[:-1], Qa[-1])  # Utilise tous sauf le dernier qubit comme contrôle
    qc.h(Qa[-1])  # Appliquer H sur le dernier qubit pour compléter la simulation

    # Réappliquer les portes X à tous les qubits dans Qa
    qc.x(Qa)
    # Réappliquer les portes H à tous les qubits dans Qa
    qc.h(Qa)

    return qc


def show_diffusion_operator(n_bits=2):
    """
    Affiche le circuit de l'opérateur de diffusion pour amplifier les états marqués par l'oracle
    :param n_bits: int, nombre de bits pour la comparaison
    """
    Qa = QuantumRegister(n_bits, "a")

    qc = diffusion_operator(Qa)

    # qc = transpile(qc, backend)
    qc.draw(output="mpl")
    plt.show()


# MAIN CIRCUIT


def minimum_search_circuit(
    L,
    backend,
    yi=None,
    show_circuit=False,
    transpile_plot=False,
    show_hist=True,
    hist_title=None,
    G=True,
    P=True,
    g=None,
    p=None,
    return_circuit=False,
    show_logs=False,
):
    """
    Circuit pour la recherche du minimum dans une liste d'entiers L

    Parametres:
    :param L: list, liste des entiers parmi lesquels chercher le minimum
    :param yi: int, valeur de yi pour la comparaison (si None, on utilise une valeur aléatoire)
    :param show_circuit: bool, afficher le circuit
    :param transpile_plot: bool, afficher le circuit transpilé (si show_circuit=True)
    :param show_hist: bool, afficher l'histogramme des résultats
    :param hist_title: str, titre de l'histogramme
    :param G: bool, appliquer l'opérateur de préparation des états superposés (G)
    :param P: bool, appliquer l'opérateur de comparaison des entiers superposés avec l'entier b (P)
    :param g: int, nombre d'itérations de G (si None, on utilise la formule)
    :param p: int, nombre d'itérations de P (si None, on utilise la formule)
    :param return_circuit: bool, retourner le circuit
    :param show_logs: bool, afficher les logs

    :return: le minimum de L suivant la valeur yi
    """
    assert len(L) > 1, "La liste doit contenir au moins 2 entiers"
    assert all(isinstance(x, int) for x in L), "La liste doit contenir des entiers"
    assert isinstance(yi, int) or yi is None, "yi doit être un entier ou None"

    # Nombre de bits nécessaires pour représenter les entiers de L
    n_bits = max(qubits_needed(max(L)), 5)

    if show_logs:
        print("--- Données ---")
        print(f"L: {L}")
        print(f"n_bits = {n_bits}\nN = {len(L)}\n")

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

    # -- Initialisation de Qfin (qubit auxiliaire) à ket(-)=H.X|0> --
    qc.x(Qfin)
    qc.h(Qfin)

    # -- Initialisation de Qb (yi) --
    if show_logs:
        print("--- Initialisation ---")
    if yi is None:
        # Si yi n'est pas donné, on choisit une valeur aléatoire
        yi = L[random.randint(0, len(L) - 1)]
        if show_logs:
            print(f"yi (aléatoire): {yi}\n")
    else:
        if show_logs:
            print(f"yi (itération précédente): {yi}\n")

    assert yi < 2**n_bits, f"yi doit être inférieur ou égale à {2**n_bits-1}"

    # On transforme yi en une chaîne de bits (avec n_bits)
    yi = int_to_bits(yi).rjust(n_bits, "0")
    # On encode yi dans Qb
    for i in range(n_bits):
        qc.append(encode(yi[i]).to_instruction(), [Qb[i]])

    if show_logs:
        print(f"--- Oracles ---")

    # -- Préparation des états superposés pour la recherche de minimum (porte G répété g fois) --
    if G:
        if g is None:
            # Theoriquement : g=pi/4*sqrt(2^n/N) où N est le nombre d'éléments dans L et n est le nombre de qubits
            # Empiriquement : g=5 pour N>4 et g=2 pour N<=3
            g = 5 if len(L) > 4 else 2
        if show_logs:
            print(f"Itérations de G: {g}")
        for i in range(g):
            # On applique l'oracle pour marquer les états de L dans la superposition
            qc.append(
                oracle_grover_preparation(n_bits, L, Qa, Qfin).to_instruction(),
                [*Qa, *Qfin],
            )
            # On applique l'opérateur de diffusion pour amplifier les états marqués par l'oracle
            qc.append(diffusion_operator(Qa).to_instruction(), [*Qa])

    # -- Comparaison des états superposés avec l'entier b (porte P répétée p fois) --
    if P:
        if p is None:
            # Theoriquement :
            # Empiriquement : p=2 pour N<=3 et p=2 pour N>4
            p = 2
        if show_logs:
            print(f"Itérations de P: {p}")
        for i in range(p):
            # On applique l'oracle pour comparer les entiers superposés avec l'entier b
            qc.append(
                oracle_compare_integers(
                    Qa, Qb, Qaux1, Qaux2, Qres, Qfin, n_bits
                ).to_instruction(),
                [*Qa, *Qb, *Qaux1, *Qaux2, *Qres, *Qfin],
            )
            # On applique l'opérateur de diffusion pour amplifier les états marqués par l'oracle
            qc.append(diffusion_operator(Qa).to_instruction(), [*Qa])

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
    if show_logs:
        print("\n--- Resultats ---")
    # On exécute le circuit
    qc_transpile = transpile(qc, backend)
    result = backend.run(qc_transpile, shots=2048).result()
    counts = result.get_counts()
    # Inverser les clés, exepmle : {'01': 100} devient {'10': 100}
    counts = {k[::-1]: v for k, v in counts.items()}

    min_L = int(max(counts, key=counts.get), 2)
    if show_logs:
        print(f"Minimum quantique: {min_L}")
        print(f"Vrai minimum: {min(L)}")

    # -- Affichage de la distribution des résultats --
    if show_hist:
        # Utiliser plotly pour un affichage interactif
        # Colorer en deepskyblue les elements de L, en darkgray les autres
        # Colorer en yellow yi
        colors = [
            "deepskyblue" if int(k, 2) in L else "darkgray" for k in counts.keys()
        ]
        colors = [
            colors[i] if k != yi else "yellow" for i, k in enumerate(counts.keys())
        ]
        # Ajouter une annotation pour indiquer le minimum
        # Afficher les probabilités pour chaque entier (au lieu du nombre d'occurences)
        fig = go.Figure(
            data=[
                go.Bar(
                    x=[int(k, 2) for k in counts.keys()],
                    y=[v / sum(counts.values()) for v in counts.values()],
                    marker=dict(color=colors),
                )
            ]
        )

        fig.update_layout(
            title=f"yi = {bits_to_int(yi)}; min_L = {min_L};"
            f"n_bits = {n_bits}; N = {len(L)}"
            f"g = {g}; p = {p}; {hist_title}",
            xaxis_title="Entiers",
            yaxis_title="Nombre d'occurences",
            # font
            font=dict(family="Noto Sans", size=25, color="black"),
        )

        # Ajouter une annotation pour indiquer le minimum
        if P:
            fig.add_annotation(
                x=min_L,
                y=counts[format(min_L, f"0{n_bits}b")] / sum(counts.values()),
                text="minimum",
                showarrow=True,
                arrowhead=1,
            )

        # Ajouter une annotation pour indiquer yi
        fig.add_annotation(
            x=bits_to_int(yi),
            y=counts[format(bits_to_int(yi), f"0{n_bits}b")] / sum(counts.values()),
            text="yi",
            showarrow=True,
            arrowhead=1,
        )

        plot(
            fig, filename=f"minimum_search{f'_{hist_title}' if hist_title else ''}.html"
        )

    if return_circuit:
        return qc

    return min_L


def minimum_search(L, backend, g=None, p=None, show_logs=False, plot_fig=False):
    """
    Recherche du minimum dans une liste L
    Utilise l'algorithme de recherche du minimum quantique

    On choisi au hasard un entier yi parmi les entiers de L
    Puis on relance l'algorithme avec yi = la sortie de l'algorithme précédent
    Et on s'arrête quand le minimum n'est plus dans L
    On peut aussi s'arrêter après un certain nombre d'itérations sqrt(N)

    Parametres:
    :param L: list, liste des entiers parmi lesquels chercher le minimum
    :param backend: backend, backend pour exécuter le circuit
    :param g: int, nombre d'itérations de G
    :param p: int, nombre d'itérations de P
    :param show_logs: bool, afficher les logs
    :param plot_fig: bool, afficher l'évolution de la recherche (la valeur de yi à chaque itération)

    :return: le minimum de L
    """
    minimum_found = False
    history = []
    iteration = 1
    minimum_value = None
    minimum_val = minimum_search_circuit(
        L,  # Liste des entiers parmi lesquels chercher le minimum
        backend,
        G=True,  # Appliquer l'opérateur de préparation des états superposés
        P=True,  # Appliquer l'opérateur de comparaison des entiers superposés avec l'entier b
        show_hist=True,
        hist_title="Première itération",
        show_logs=show_logs,
        g=g,
        p=p,
    )
    while not minimum_found:
        history.append(minimum_val)
        if iteration <= len(L) ** 0.5:
            if minimum_val in L:
                minimum_val = minimum_search_circuit(
                    L,  # Liste des entiers parmi lesquels chercher le minimum
                    backend,
                    yi=minimum_val,  # Valeur de yi pour la comparaison
                    G=True,  # Appliquer l'opérateur de préparation des états superposés
                    P=True,  # Appliquer l'opérateur de comparaison des entiers superposés avec l'entier b
                    show_hist=True,
                    hist_title=f"Itération {iteration}",
                    show_logs=show_logs,
                    g=g,
                    p=p,
                )
                # print(f"Minimum à l'itération {iteration}: {minimum_val}")
            else:
                if len(history) > 1:
                    minimum_found = True
                    # print(f"Minimum trouvé: {history[-2]} car {history[-1]} n'est pas dans L")
                    minimum_value = history[-2]
                else:
                    # On relance l'algorithme avec un nouveau yi
                    history = []
                    minimum_val = minimum_search_circuit(
                        L,  # Liste des entiers parmi lesquels chercher le minimum
                        backend,
                        G=True,  # Appliquer l'opérateur de préparation des états superposés
                        P=True,  # Appliquer l'opérateur de comparaison des entiers superposés avec l'entier b
                        show_hist=True,
                        hist_title="Première itération",
                        show_logs=show_logs,
                        g=g,
                        p=p,
                    )
                    iteration = 0
                    # print(f"Relance de l'algorithme avec un nouveau yi: {minimum_val}")

            iteration += 1

        else:
            minimum_found = True
            # print(f"Nombre d'itérations maximum atteint: {iteration-1}")
            # le minimum est donc la dernière valeur de history
            # print(f"Minimum trouvé: {history[-1]}")
            minimum_value = history[-1]

    if plot_fig:
        # On affiche l'évolution de la recherche (la valeur de yi à chaque itération)
        # pour voir sa vitesse de convergence
        fig = go.Figure(
            data=[
                go.Scatter(
                    x=[i for i in range(len(history))], y=history, mode="lines+markers"
                )
            ]
        )
        fig.update_layout(
            title=f"Evolution de la recherche du minimum",
            xaxis_title="Itération",
            yaxis_title="Valeur de yi",
            # font
            font=dict(family="Noto Sans", size=25, color="black"),
        )
        plot(fig, filename=f"minimum_search_history.html")

        assert minimum_value is not None, "Le minimum n'a pas été trouvé"

    return minimum_value


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
    platform = "IBM"
    if platform == "IBM":
        print("Using IBM platform...\n")
        # Load saved credentials
        service = QiskitRuntimeService()
        backend = service.backend("ibm_brisbane")
        # simulator_mps, ibmq_qasm_simulator, simulator_statevector
    else:
        print("Using IONQ platform...\n")

        provider = IonQProvider(IONQ_API_TOKEN)
        backend = provider.get_backend("ionq_simulator")

    # Test de la recherche du minimum dans une liste L
    L = [8, 12, 29, 16, 10, 3]

    # minimum_val = minimum_search_circuit(L, backend, yi=12, g=3, p=1, show_hist=True)

    # show_oracle_compare_integers(3)
    # show_oracle_grover_preparation(L)
    # show_diffusion_operator(4)

    # Pour des petites listes (N<=3) : g,p = 2, 2
    # Pour des grandes listes (N>4) : g,p = 5, 2

    minimum_search(L, backend, g=5, p=2, plot_fig=True, show_logs=True)
