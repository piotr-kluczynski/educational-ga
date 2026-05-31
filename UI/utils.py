from functools import partial
from genetic_algorithm.selection import deterministic_sampling, stochastic_sampling_with_replacement, tournament, stochastic_tournament
from genetic_algorithm.fitness_function import fitness_function

def get_parameter(message, default_value, param_type):
    data = input(f"{message} [Domyślnie: {default_value}]: ")
    if not data.strip():
        return default_value
    try:
        return param_type(data)
    except ValueError:
        print(f"Błędna wartość. Używam domyślnej: {default_value}")
        return default_value

def choose_selection():
    print("\nWYBÓR METODY SELEKCJI")
    print("1. Turniejowa (Tournament) - Najbezpieczniejsza przy ujemnym fitnessie")
    print("2. Ruletka ze zwracaniem (Stochastic Sampling)")
    print("3. Próbkowanie deterministyczne (Deterministic Sampling)")
    print("4. Turniej stochastyczny (Stochastic Tournament)")

    choice = get_parameter("Wybierz metodę (1-4)", 1, int)

    if choice == 1:
        k_tournament = get_parameter("Podaj rozmiar turnieju (k)", 3, int)
        return partial(tournament, k=k_tournament)
    elif choice == 2:
        return stochastic_sampling_with_replacement
    elif choice == 3:
        return deterministic_sampling
    elif choice == 4:
        return stochastic_tournament
    else:
        print("Niepoprawny wybór. Używam domyślnej: Turniejowa (k=3)")
        return partial(tournament, k=3)

def configure_scaling(default_func):
    print("\nSKALOWANIE FUNKCJI FITNESS")
    print("1. Brak skalowania")
    print("2. Skalowanie liniowe (a * f + b)")
    print("3. Skalowanie potęgowe (f ^ alpha)")
    choice = get_parameter("Wybór (1-3)", 1, int)

    if choice == 2:
        a = get_parameter("Współczynnik 'a'", 1.0, float)
        b = get_parameter("Współczynnik 'b'", 0.0, float)
        return partial(fitness_function, objective_function=default_func, scaling="linear", scaling_factors=(a, b))
    elif choice == 3:
        alpha = get_parameter("Wykładnik 'alpha'", 1.0, float)
        return partial(fitness_function, objective_function=default_func, scaling="power", scaling_factors=(alpha,))
    else:
        return partial(fitness_function, objective_function=default_func, scaling="none")

def find_matching_schemas(population, pattern):
    matched = []
    for individual in population:
        chromosome_str = "".join(map(str, individual.chromosome1))
        if len(chromosome_str) != len(pattern):
            continue
        is_match = True
        for c_gene, p_gene in zip(chromosome_str, pattern):
            if p_gene != '*' and c_gene != p_gene:
                is_match = False
                break
        if is_match:
            matched.append(chromosome_str)
    return matched