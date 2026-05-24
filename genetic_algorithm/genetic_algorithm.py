import math
import random
from individual import Individual
from selection import stochastic_sampling_with_replacement


def initialize_population(chromosome_length, alphabet, domains, encoding, is_haploidal, pop_size):
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

        population.append(new_individual)

    return population

def next_generation(prev_generation, pop_size, p_mutation, p_crossover, mutation,
                    objective_function, custom_transformation=None, constraints=None, scaling="none", scaling_factors=None,
                    selection=stochastic_sampling_with_replacement,
                    sharing=False, elitism=False, gen_gap=False):
    next_generation = []

    while len(next_generation) < pop_size:
        couples = selection(prev_generation, pop_size)
