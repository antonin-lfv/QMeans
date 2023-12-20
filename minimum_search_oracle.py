from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, execute
from qiskit_ibm_provider import IBMProvider
import matplotlib.pyplot as plt


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
        qc.mcx([Qaux1[i + 1], Qres[i]], Qaux1[i])
        qc.mcx([Qaux2[i + 1], Qres[i]], Qaux2[i])

    # modification: retour pour un bit pour l'ordre (à inverser si besoin)
    # 4/ copie de Qaux2[0]: 1=> dans l'ordre, 0 => supérieur ou égal
    qc.mcx([Qaux2[0]], Qfin)

    # faire uncompute ici pour libérer les 3n registres temporaires
    # 5/ uncompute avec reverse_bit_compare_new()
    for i in reversed(range(n_bits)):
        qc_compare = reverse_bit_compare()
        qc.append(qc_compare.to_instruction(), [Qa[i], Qb[i], Qaux1[i], Qaux2[i]])

    # Mesure du qubit aux[2] qui contient le résultat final de la comparaison
    qc.measure(Qfin, Cout)

    return qc


def test_all_possibilities(number_of_bits):
    """
    Teste toutes les possibilités pour une comparaison de 2 entiers sur n_bits

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
            result = execute(qc, backend, shots=1).result()
            out = [*result.get_counts()]
            print(f"{i} {operator[out[0]]} {j}", end="")
            if ((i < j) and (out[0] == "1")) or ((i >= j) and (out[0] == "0")):
                print(f" ok")
            else:
                print(f" failed")


if __name__ == "__main__":
    provider = IBMProvider()
    backend = provider.get_backend("simulator_mps")
    # test_all_possibilities(3)

    # Show the circuit
    qc = compare_integers(4, "1", "0")
    qc.draw(output="mpl")
    plt.show()

    # show bit compare
    qc = bit_compare()
    qc.draw(output="mpl")
    plt.show()

    # show reverse bit compare
    qc = reverse_bit_compare()
    qc.draw(output="mpl")
    plt.show()
