import random

class Individual:
    def __init__(self, chromosome1, chromosome2 = None, domination_chromosome = None):
        self.chromosome1 = chromosome1
        self.chromosome2 = chromosome2 # None by default
        self.domination_chromosome = domination_chromosome

        self.fitness = -1 # Need to initialize using fitness function

    def get_phenotype(self):
        phenotype = []
        for i in range(len(self.domination_chromosome)):
            if self.domination_chromosome[i] == 0:
                phenotype.append(self.chromosome1[i])
            else:
                phenotype.append(self.chromosome2[i])

        return phenotype

    def __str__(self):
        description = ""

        for gene in self.chromosome1:
            description += str(gene)

        if self.chromosome2 is not None:
            description += "\n"
            for gene in self.chromosome2:
                description += str(gene)

            description += "\n"
            for marker in self.domination_chromosome:
                description += str(marker)

        return description