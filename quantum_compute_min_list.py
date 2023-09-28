import random
import math

# From https://github.com/sid1993/qbitwise/blob/master/Durr_Hoyer_Modified_Max.ipynb
# Durr-Hoyer's algorithm for finding the maximum of a list of numbers
# Using mathematics instead of qiskit


def superposition(qubits):
    states = []
    # Hadamard Transformation
    total_states = int(math.pow(2, qubits))
    amplitude = 1 / math.sqrt(total_states)
    for _ in range(0, total_states):
        states.append(amplitude)
    return states


def markingOracle(states, datalist, y):
    # marks all states/elements smaller than datalist[y]
    for index in range(len(datalist)):
        if datalist[index] < datalist[y]:
            states[index] *= -1  # phase inversion
    return states


def grover_diffusion(states):
    average = sum(states) / len(states)
    for i in range(0, len(states)):
        states[i] = (average - states[i]) + average
    return states


def oracle(states, datalist, key):
    for i in range(0, int(len(datalist))):
        if datalist[i] > key:
            states[i] *= -1  # phase inversion
    return states


def grover_iteration(states, j, y):
    num_iterations = j
    for _ in range(0, num_iterations):
        states = oracle(states, datalist, y)
        states = grover_diffusion(states)
    probability_states = [states[i] * states[i] for i in range(0, len(states))]
    # print(probability_states)
    max_value = max(probability_states)
    index_with_max_val = []
    for i in range(len(probability_states)):
        if probability_states[i] == max_value:
            index_with_max_val.append(i)
    index_random_state_with_max_val = index_with_max_val[
        random.randint(0, len(index_with_max_val) - 1)
    ]
    return index_random_state_with_max_val


def exponential_search(states, datalist, y):
    m = 1
    alpha = 6 / 5  # (Any value of λ strictly between 1 and 4/3 would do.)
    while True:
        j = random.randint(0, int(m) - 1)  # number of iterations of Grover
        i = grover_iteration(states, j, y)
        # print(datalist[i]," ",datalist[y])
        if datalist[i] > datalist[y]:
            m = min(alpha * m, math.sqrt(len(datalist)))
        else:
            break
    return i


def marked(states):
    for s in states:
        if s < 0:
            return True
    return False


def findMin(datalist):
    size_datalist = len(datalist)
    qubits_needed = math.ceil(math.log(size_datalist, 2))
    paddings_required = int(math.pow(2, qubits_needed) - size_datalist)
    max_value = max(datalist)
    for _ in range(0, paddings_required):
        datalist.append(
            max_value + 1
        )  # Utiliser une valeur sûrement plus grande que toutes les autres

    y_value = random.choice([x for x in datalist if x != max_value + 1])
    y = datalist.index(y_value)
    print("Starting threshold : ", datalist[y])
    print("Exponential search steps :")
    max_iteration = int(
        22.5 * math.sqrt(len(datalist)) + 1.4 * math.log2(len(datalist))
    )
    for _ in range(0, max_iteration):
        states = superposition(qubits_needed)
        states = markingOracle(
            states, datalist, y
        )  # marks all states/elements smaller than datalist[y]
        if marked(states):
            new_y = exponential_search(
                states, datalist, y
            )  # find an element smaller than datalist[y]
            if datalist[new_y] < datalist[y]:
                print(datalist[y], " ---> ", datalist[new_y])
                y = new_y
        else:
            break
    return y


datalist = random.sample(range(1, 10000000), 5000)
# print(datalist)
minIndex = findMin(datalist)
print("Actual min : ", min(datalist), " Quantum Min : ", datalist[minIndex])
