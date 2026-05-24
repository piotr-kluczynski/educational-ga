import math

def crowding_function(individual, population, sigma_share, alpha, encoding):
    share_number = 0.0
    distance = 0

    for other_individual in population:
        if encoding == "binary":
            distance = distance_discrete(individual.get_phenotype(), other_individual.get_phenotype())
        elif encoding == "ordering":
            distance = distance_ordering(individual.get_phenotype(), other_individual.get_phenotype())
        elif encoding == "real":
            distance = distance_real(individual.get_phenotype(), other_individual.get_phenotype())

        share_number += sharing_function(distance, sigma_share, alpha)

    share_number = max(share_number, 0.00000001)
    return individual.fitness / share_number

def sharing_function(distance, sigma_share, alpha):
    if distance >= sigma_share:
        return 0

    return 1 - (distance / sigma_share) ** alpha

def distance_discrete(chromosome1, chromosome2):
    distance = 0

    for gene1, gene2 in zip(chromosome1, chromosome2):
        if gene1 != gene2:
            distance += 1

    return distance

def distance_ordering(chromosome1, chromosome2):
    distance = 0

    positions = {gene: i for i, gene in enumerate(chromosome2)}

    for i in range(len(chromosome1)):
        for j in range(i + 1, len(chromosome1)):

            gene1 = chromosome1[i]
            gene2 = chromosome1[j]

            if positions[gene1] > positions[gene2]:
                distance += 1

    return distance

def distance_real(chromosome1, chromosome2):
    total = 0

    for gene1, gene2 in zip(chromosome1, chromosome2):
        total += (gene1 - gene2)**2

    return math.sqrt(total)