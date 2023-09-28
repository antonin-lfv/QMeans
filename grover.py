import math
import random
import matplotlib.pyplot as plt

plots = []

# From https://github.com/sid1993/qbitwise/blob/master/Grover's.ipynb


def superposition(qubits):
    states = []
    total_states = int(math.pow(2, qubits))
    amplitude = 1 / math.sqrt(total_states)  # Hadamard Transformation
    for _ in range(0, total_states):
        states.append(amplitude)
    return states


def grover_diffusion(states):
    average = sum(states) / len(states)
    for i in range(0, len(states)):
        states[i] = (average - states[i]) + average  # inversion about mean
    return states


def oracle(states, datalist, key):
    for i in range(0, int(len(datalist))):
        if datalist[i] == key:
            states[i] *= -1  # phase inversion
    return states


def grover_search(qubits, datalist, key):
    states = superposition(qubits)
    num_iterations = math.ceil(math.sqrt(math.pow(2, qubits)))
    probability_states = []
    for _ in range(0, num_iterations):
        states = oracle(states, datalist, key)
        states = grover_diffusion(states)
        probability_states = [states[i] * states[i] for i in range(0, len(states))]
        plots.append(probability_states)
    return probability_states


def grover(datalist, key):
    size_datalist = len(datalist)
    qubits_needed = math.ceil(math.log(size_datalist, 2))
    paddings_required = int(math.pow(2, qubits_needed) - size_datalist)
    # required if the number of data items is not a power of 2.
    for _ in range(0, paddings_required):
        datalist.append(0)
    grover_search(qubits_needed, datalist, key)


datalist = random.sample(range(1, 100), 64)
grover(datalist, 32)
print(datalist)
print(
    "Following plots show the change in probabilities of items after every grover's iteration :"
)
result = {}
iteration = 1
for plot in plots:
    for i in range(len(plot)):
        result[datalist[i]] = plot[i]
    print(
        "----------------------------------------------------------------------------------------------------------"
    )
    print("Iteration ", iteration, " :")
    plt.bar(result.keys(), result.values(), color="b")
    plt.ylabel("Probability")
    plt.xlabel("Items")
    plt.show()
    iteration += 1
    print(
        "----------------------------------------------------------------------------------------------------------"
    )
