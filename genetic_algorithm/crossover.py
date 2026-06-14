import numpy as np
import random

# NATURAL & REAL NUMBER OPERATORS
def kpoint_crossover(chromosome1, chromosome2, k, p_crossover): # 0 < k <= len(chromosome)
    child1 = chromosome1.copy()
    child2 = chromosome2.copy()

    if random.random() < p_crossover:
        cross_points = []
        while len(cross_points) < k:
            new_point = random.randint(0, len(chromosome1) - 1)
            if new_point not in cross_points:
                cross_points.append(new_point)

        cross_points.sort()

        points = [0] + cross_points + [len(chromosome1)]

        for i in range(len(points) - 1):
            start = points[i]
            end = points[i + 1]

            if i % 2 == 0:
                child1[start:end] = chromosome1[start:end]
                child2[start:end] = chromosome2[start:end]
            else:
                child1[start:end] = chromosome2[start:end]
                child2[start:end] = chromosome1[start:end]
    return child1, child2

def uniform(chromosome1, chromosome2, p_crossover):
    child1 = chromosome1.copy()
    child2 = chromosome2.copy()

    if random.random() <= p_crossover:
        for i in range(len(chromosome1)):
            if random.random() <= 0.5:
                child1[i] = chromosome2[i]
                child2[i] = chromosome1[i]

    return child1, child2

# REAL NUMBER OPERATORS
def intermediate_recombination(chromosome1, chromosome2, p_crossover):
    child1 = chromosome1.copy()
    child2 = chromosome2.copy()

    if random.random() <= p_crossover:
        for i in range(len(chromosome1)):
            alpha = random.random()

            child1[i] = alpha*chromosome1[i] + (1-alpha)*chromosome2[i]
            child2[i] = alpha*chromosome2[i] + (1-alpha)*chromosome1[i]

    return child1, child2

# ORDERING OPERATORS
def partially_mapped(chromosome1, chromosome2, p_crossover):
    child1 = chromosome1.copy()
    child2 = chromosome2.copy()

    if random.random() <= p_crossover:
        swap_point1 = random.randint(0, len(chromosome1) - 1)

        # Choosing random swap point until we find one that does not repeat
        while True:
            swap_point2 = random.randint(0, len(chromosome1) - 1)
            if swap_point1 != swap_point2:
                break

        # Swapping part of the chromosomes
        start, end = sorted([swap_point1, swap_point2])

        child1[start:end] = chromosome2[start:end]
        child2[start:end] = chromosome1[start:end]

        # Creating mapping for both chromosomes
        mapping1 = dict(zip(chromosome2[start:end], chromosome1[start:end]))
        mapping2 = dict(zip(chromosome1[start:end], chromosome2[start:end]))

        # Mapping function
        def resolve(gene, mapping):
            while gene in mapping:
                gene = mapping[gene]
            return gene

        # Correcting child1
        for i in range(len(chromosome1)):
            if start <= i < end:
                continue
            child1[i] = resolve(chromosome1[i], mapping1)

        # Correcting child2
        for i in range(len(chromosome1)):
            if start <= i < end:
                continue
            child2[i] = resolve(chromosome2[i], mapping2)

    return child1, child2

def order(chromosome1, chromosome2, p_crossover):
    child1 = chromosome1.copy()
    child2 = chromosome2.copy()

    if random.random() <= p_crossover:
        swap_point1 = random.randint(0, len(chromosome1) - 1)

        # Choosing random swap point until we find one that does not repeat
        while True:
            swap_point2 = random.randint(0, len(chromosome1) - 1)
            if swap_point1 != swap_point2:
                break

        # Swapping part of the chromosomes
        start, end = sorted([swap_point1, swap_point2])

        child1_segment = chromosome1[start:end]
        child2_segment = chromosome2[start:end]

        child1[start:end] = child1_segment
        child2[start:end] = child2_segment

        # 2. Filling remaining genes in children
        rem_genes1 = [gene for gene in chromosome2 if gene not in child1_segment]
        rem_genes2 = [gene for gene in chromosome1 if gene not in child2_segment]

        j = 0
        for i in range(len(chromosome1)):
            if start <= i < end:
                continue

            child1[i] = rem_genes1[j]
            child2[i] = rem_genes2[j]
            j += 1

    return child1, child2

def cycle(chromosome1, chromosome2, p_crossover):
    if random.random() > p_crossover:
        return chromosome1.copy(), chromosome2.copy()

    child1 = [-1] * len(chromosome1)
    child2 = [-1] * len(chromosome1)

    visited = [False] * len(chromosome1)
    cycle_counter = 0

    # Ensure chromosome2 can be searched properly
    is_list = isinstance(chromosome2, list)

    for start in range(len(chromosome1)):
        if visited[start]:
            continue

        # Finding the cycle_counter
        idx = start
        cycle_indices = []

        while not visited[idx]:
            visited[idx] = True
            cycle_indices.append(idx)

            value = chromosome1[idx]
            if is_list:
                idx = chromosome2.index(value)
            else:
                idx = np.where(chromosome2 == value)[0][0]

        # Assigning the cycle_counter
        if cycle_counter % 2 == 0:
            for i in cycle_indices:
                child1[i] = chromosome1[i]
                child2[i] = chromosome2[i]
        else:
            for i in cycle_indices:
                child1[i] = chromosome2[i]
                child2[i] = chromosome1[i]

        cycle_counter += 1

    return child1, child2