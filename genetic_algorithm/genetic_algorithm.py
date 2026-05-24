import random

from mutation import single_point
from individual import Individual


def initialize_population(chromosome_length, alphabet, domains, encoding, is_haploidal, pop_size, fitness_function, crowding_function):
    population = []

    for _ in range(pop_size):
        chromosome1 = []
        if encoding == "natural":
            for _ in range(chromosome_length):
                value = random.randint(0, len(alphabet) - 1)
                chromosome1.append(alphabet[value])
        elif encoding == "ordering":
            chromosome1 = random.sample(alphabet, k=chromosome_length)
        elif encoding == "real":
            for i in range(chromosome_length):
                value = random.uniform(*domains[i])
                chromosome1.append(value)

        new_individual = Individual(chromosome1)

        if is_haploidal:
            chromosome2 = []
            domination_chromosome = [random.randint(0, 1) for _ in range(chromosome_length)]

            if encoding == "natural":
                for _ in range(chromosome_length):
                    value = random.randint(0, len(alphabet) - 1)
                    chromosome2.append(alphabet[value])
            elif encoding == "ordering":
                chromosome2 = random.sample(alphabet, k=chromosome_length)
            elif encoding == "real":
                for i in range(chromosome_length):
                    value = random.uniform(*domains[i])
                    chromosome2.append(value)

            new_individual.chromosome2 = chromosome2
            new_individual.domination_chromosome = domination_chromosome

        # Calculating initial fitness
        new_individual.raw_fitness = fitness_function(new_individual.get_phenotype())
        new_individual.practical_fitness = new_individual.raw_fitness

        population.append(new_individual)

    # Applying crowding function
    if crowding_function is not None:
        for individual in population:
            individual.practical_fitness = crowding_function(individual, population)

    return population

def next_generation(prev_gen, pop_size, p_mutation,
                    fitness_function, crossover, mutation, selection, crowding_function,
                    is_diploid=False, elite_size=0, replacement=1):
    offsprings = []
    couples = selection(prev_gen, pop_size)

    for i in range(len(couples)):
        parent1, parent2 = couples[i]

        # Creating chromosome1
        child1_chromosome1, child2_chromosome1 = crossover(parent1.chromosome1, parent2.chromosome1)

        child1_chromosome1 = mutation(child1_chromosome1, p_mutation)
        child2_chromosome1 = mutation(child2_chromosome1, p_mutation)

        # Creating chromosome2 and domination_chromosome in case of diploidal
        if is_diploid:
            child1_chromosome2, child2_chromosome2 = crossover(parent1.chromosome2, parent2.chromosome2)
            child1_domination_chromosome, child2_domination_chromosome = crossover(parent1.domination_chromosome, parent2.domination_chromosome)

            child1_chromosome2 = mutation(child1_chromosome2, p_mutation)
            child1_domination_chromosome = single_point(child1_domination_chromosome, [0, 1], p_mutation)
            child2_chromosome2 = mutation(child2_chromosome2, p_mutation)
            child2_domination_chromosome = single_point(child2_domination_chromosome, [0, 1], p_mutation)


        # Creating the children individuals
        if is_diploid:
            child1 = Individual(child1_chromosome1, child1_chromosome2, child1_domination_chromosome)
            child2 = Individual(child2_chromosome1, child2_chromosome2, child2_domination_chromosome)
        else:
            child1 = Individual(child1_chromosome1)
            child2 = Individual(child2_chromosome1)

        # Calculating initial fitness
        child1.raw_fitness = fitness_function(child1.get_phenotype())
        child1.practical_fitness = child1.raw_fitness
        child2.raw_fitness = fitness_function(child2.get_phenotype())
        child2.practical_fitness = child2.raw_fitness

        offsprings.append(child1)
        offsprings.append(child2)

    # Applying crowding function
    if crowding_function is not None:
        for individual in offsprings:
            individual.practical_fitness = crowding_function(individual, offsprings)

    # Applying elitism
    elites = []
    if elite_size > 0:
        elites = sorted(prev_gen, key=lambda individual: individual.practical_fitness, reverse=True)[:elite_size]

    # Applying generational gap
    if replacement < 1:
        keep = int(pop_size * (1 - replacement))

        survivors = sorted(prev_gen, key=lambda x: x.practical_fitness, reverse=True)[:keep]
        next_gen = survivors + offsprings
        next_gen = next_gen[:pop_size]
    else:
        next_gen = offsprings

    # Ensuring elites are preserved
    if elite_size > 0:
        next_gen = elites + [ind for ind in next_gen if ind not in elites]
        next_gen = next_gen[:pop_size]

    return next_gen

def run_genetic_algorithm(pop_size, chromosome_length, alphabet, p_mutation, crossover, mutation, selection, elite_size, replacement, encoding, is_diploid, domains, fitness_function, crowding_function, end_condition):
    gen_count = 0
    prev_population = initialize_population(chromosome_length, alphabet, domains, encoding, is_diploid, pop_size, fitness_function, crowding_function)

    while end_condition(gen_count, prev_population) is False:
        next_population = next_generation(prev_population, pop_size, p_mutation, fitness_function, crossover, mutation, selection, crowding_function, is_diploid=is_diploid, elite_size=elite_size, replacement=replacement)
        prev_population = next_population
        gen_count += 1

    return prev_population
