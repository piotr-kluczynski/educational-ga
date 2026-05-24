import numpy as np
import random

# NATURAL NUMBER OPERATORS
def single_point(chromosome, alphabet, p_mutate):
    new_chromosome = []

    for i in range(len(chromosome)):
        if random.random() <= p_mutate:

            curr_alphabet = alphabet.copy()
            curr_alphabet.remove(chromosome[i])

            new_chromosome.append(random.choice(curr_alphabet))
        else:
            new_chromosome.append(chromosome[i])

    return new_chromosome

# REAL NUMBER OPERATORS
def gauss(chromosome, p_mutate, bounds, mutation_scale):
    new_chromosome = chromosome.copy()

    for i in range(len(chromosome)):
        current_bound = bounds[i]
        if random.random() <= p_mutate:
            new_chromosome[i] += np.random.normal(loc=0, scale=mutation_scale)

        new_chromosome[i] = np.clip(new_chromosome[i], current_bound[0], current_bound[1])

    return new_chromosome

# ORDERING OPERATORS
def rotation(chromosome, p_mutate, rotation_step, direction):
    new_chromosome = chromosome.copy()

    if random.random() <= p_mutate:
        n = rotation_step % len(chromosome)
        if direction == 'right':
            new_chromosome = new_chromosome[-n:] + new_chromosome[:-n]
        elif direction == 'left':
            new_chromosome = new_chromosome[n:] + new_chromosome[:n]

    return new_chromosome

def inversion(chromosome, p_mutate):
    new_chromosome = chromosome.copy()

    if random.random() <= p_mutate:
        inv_point1 = random.randint(0, len(new_chromosome)-1)

        # Choosing random inversion point until we find one that does not repeat
        while True:
            inv_point2 = random.randint(0, len(new_chromosome)-1)
            if inv_point1 != inv_point2:
                break

        start, end = sorted([inv_point1, inv_point2])
        segment = new_chromosome[start:end]
        segment = segment[::-1]

        new_chromosome[start:end] = segment

    return new_chromosome