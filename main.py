from genetic_algorithm.Individual import Individual
from genetic_algorithm.mutation import inversion

if __name__ == '__main__':
    individual = Individual([0,1,0,1], [1,0,0,1], [1,1,0,1])

    print(inversion(individual.chromosome1, 1))