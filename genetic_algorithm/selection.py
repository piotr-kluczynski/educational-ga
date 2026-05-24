import random
import math

def deterministic_sampling(population, total_fitness, pop_size):
    mating_pool = []
    expected = []

    # Calculating expected counts
    for ind in population:
        E = (ind.fitness / total_fitness) * pop_size
        expected.append((ind, E))

        k = math.floor(E)
        mating_pool.extend([ind] * k)

    # Filling remaining slots
    remaining = pop_size - len(mating_pool)

    remainders = sorted(
        expected,
        key=lambda x: x[1] - math.floor(x[1]),
        reverse=True
    )

    i = 0
    while remaining > 0:
        mating_pool.append(remainders[i][0])
        i += 1
        remaining -= 1

    # Creating couples
    random.shuffle(mating_pool)

    couples = []
    for i in range(0, pop_size, 2):
        couples.append((mating_pool[i], mating_pool[i+1]))

    return couples


def stochastic_sampling_with_replacement(population, pop_size):
    couples = []
    weights = [individual.fitness for individual in population]

    for i in range(0, pop_size, 2):
        parent1_id = random.choices(range(len(population)), weights=weights, k=1)[0]

        parent2_id = parent1_id
        while parent2_id == parent1_id:
            parent2_id = random.choices(range(len(population)), weights=weights, k=1)[0]

        couples.append((population[parent1_id], population[parent2_id]))

    return couples

def stochastic_sampling_without_replacement(population, avg_fitness, pop_size):
    couples = []
    weights = [individual.fitness for individual in population]
    offspring_count = [individual.fitness / avg_fitness for individual in population]

    for i in range(0, pop_size, 2):
        available = [idx for idx in range(len(population)) if offspring_count[idx] > 0]

        if len(available) < 2:
            break
        parent1_id = random.choices(available, weights=[weights[idx] for idx in available], k=1)[0]
        offspring_count[parent1_id] -= 1

        available = [idx for idx in range(len(population)) if offspring_count[idx] > 0 and idx != parent1_id]

        if len(available) == 0:
            break
        parent2_id = random.choices(available, weights=[weights[idx] for idx in available], k=1)[0]
        offspring_count[parent2_id] -= 1

        couples.append((population[parent1_id], population[parent2_id]))

    return couples

def remainder_stochastic_sampling_with_replacement(population, total_fitness, pop_size):
    mating_pool = []
    expected = []
    remainders = []

    # Calculating expected counts
    for ind in population:
        E = (ind.fitness / total_fitness) * pop_size
        expected.append((ind, E))

        k = math.floor(E)
        mating_pool.extend([ind] * k)
        remainders.append(E - k)

    # Filling remaining slots
    while len(mating_pool) < pop_size:
        selected = random.choices(population, weights=remainders, k=1)[0]
        mating_pool.append(selected)

    # Creating couples
    random.shuffle(mating_pool)

    couples = []
    for i in range(0, pop_size, 2):
        couples.append((mating_pool[i], mating_pool[i+1]))

    return couples

def remainder_stochastic_sampling_without_replacement(population, total_fitness, pop_size):
    mating_pool = []
    expected = []
    remainders = []

    # Calculating expected counts
    for ind in population:
        E = (ind.fitness / total_fitness) * pop_size
        expected.append((ind, E))

        k = math.floor(E)
        mating_pool.extend([ind] * k)
        remainders.append(E - k)

    # Filling remaining slots
    for i in range(len(population)):

        if len(mating_pool) >= pop_size:
            break

        if random.random() <= remainders[i]:
            mating_pool.append(population[i])

    # Selecting uniformly if we didn't get enough individuals
    while len(mating_pool) < pop_size:
        selected = random.choices(population, weights=remainders, k=1)[0]

        mating_pool.append(selected)

    # Creating couples
    random.shuffle(mating_pool)

    couples = []
    for i in range(0, pop_size, 2):
        couples.append((mating_pool[i], mating_pool[i+1]))

    return couples

def rank(population, pop_size, scaling):
    sorted_population = sorted(population, key=lambda ind: ind.fitness)
    mating_pool = []
    couples = []
    ranks = []

    if scaling == "linear":
        ranks = [i for i in range(1, len(population) + 1)]
    elif scaling == "exponential":
        ranks = [i**2 for i in range(1, len(population) + 1)]

    for i in range(pop_size):
        mating_pool.append(random.choices(sorted_population, ranks, k=1)[0])

    random.shuffle(mating_pool)

    for i in range(0, pop_size, 2):
        couples.append((mating_pool[i], mating_pool[i + 1]))

def tournament(population, pop_size, k=2):
    couples = []
    mating_pool = []

    for i in range(pop_size):
        contestants = random.sample(population, k)
        winner = max(contestants, key=lambda ind: ind.fitness)

        mating_pool.append(winner)

    random.shuffle(mating_pool)

    for i in range(0, pop_size, 2):
        couples.append((mating_pool[i], mating_pool[i+1]))

    return couples

def stochastic_tournament(population, pop_size):
    couples = []
    mating_pool = []
    weights = [individual.fitness for individual in population]

    for i in range(0, pop_size):
        # Finding couple for a tournament
        parent1_id = random.choices(range(len(population)), weights=weights, k=1)[0]

        parent2_id = parent1_id
        while parent2_id == parent1_id:
            parent2_id = random.choices(range(len(population)), weights=weights, k=1)[0]

        # Inserting winner to the mating pool
        parent1 = population[parent1_id]
        parent2 = population[parent2_id]

        mating_pool.append(parent1 if parent1.fitness >= parent2.fitness else parent2)

    # Creating couples
    random.shuffle(mating_pool)

    for i in range(0, pop_size, 2):
        couples.append((mating_pool[i], mating_pool[i+1]))

    return couples