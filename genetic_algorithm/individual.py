class Individual:
    def __init__(self, chromosome1, chromosome2 = None, domination_chromosome = None):
        self.chromosome1 = chromosome1
        self.chromosome2 = chromosome2 # None by default
        self.domination_chromosome = domination_chromosome # None by default

        self.fitness = -1 # Need to initialize using fitness function

    def get_phenotype(self):
        phenotype = self.chromosome1.copy()

        if self.chromosome2 is not None:
            for i in range(len(self.domination_chromosome)):
                if self.domination_chromosome[i] == 1:
                    phenotype[i] = self.chromosome2[i]

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