from config import QLM_HOSTNAME
from qat.qlmaas import QLMaaSConnection

conn = QLMaaSConnection(hostname=QLM_HOSTNAME, authentication="password")
# conn.create_config()  # Run une seule fois

from qat.lang.AQASM import Program, H, CNOT


# Créer un programme quantique
prog = Program()

# Créer un registre quantique avec 2 qubits
qbits = prog.qalloc(2)

# Appliquer la porte Hadamard au premier qubit
prog.apply(H, qbits[0])

# Appliquer la porte CNOT entre les deux qubits
prog.apply(CNOT, qbits[0], qbits[1])

# Compiler le programme en un circuit
circuit = prog.to_circ()

# Get remote QPU
LinAlg = conn.get_qpu("qat.qpus:LinAlg")
qpu = LinAlg()

# Exécuter le circuit sur le plugin
job = qpu.submit(circuit.to_job(nbshots=10000))
result = job.get_result()

# Afficher les résultats
print("Résultats du circuit :")
for sample in result.raw_data:
    print(f"État: {sample._state}, Probabilité: {sample.probability:.4f}")
