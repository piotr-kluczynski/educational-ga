import sys
import csv
from functools import partial
import numpy as np

from UI.utils import get_parameter, choose_selection, configure_scaling, find_matching_schemas
from UI.evaluators import binary_evaluator, continuous_evaluator, tsp_evaluator
from UI.visualizations import show_line_chart, show_tsp_route, run_animation

from genetic_algorithm.genetic_algorithm import initialize_population, next_generation
from genetic_algorithm.sharing import crowding_function
from genetic_algorithm.crossover import kpoint_crossover, uniform, intermediate_recombination, partially_mapped, order, cycle
from genetic_algorithm.mutation import single_point, gauss, rotation, inversion


def run_app():
    print("\n" + "="*50)
    print("APLIKACJA EDUKACYJNA: ALGORYTMY GENETYCZNE")
    print("="*50)
    print("\nWybierz tryb pracy algorytmu:")
    print("1. Optymalizacja funkcji liczb całkowitych (Binarne).")
    print("2. Optymalizacja funkcji liczb zmiennoprzecinkowych (Ciągłe).")
    print("3. Problemy porządkowania (Problem Komiwojażera).")
    print("0. Zakończ program.")

    while True:
        mode = input("Twój wybór (0-3): ")
        if mode == '0':
            sys.exit(0)
        if mode in ['1', '2', '3']:
            break
        print("Nieznany tryb. Spróbuj ponownie.")

    mode_name = ""
    if mode == '1': mode_name = "BINARNY (NATURALNY)"
    elif mode == '2': mode_name = "CIĄGŁY (LICZBY RZECZYWISTE)"
    elif mode == '3': mode_name = "PORZĄDKOWY (KOMIWOJAŻER)"

    state = {
        'population': [],
        'gen': 0,
        'crossover_count': 0,
        'mutation_count': 0,
        'total_crossovers': 0,
        'total_mutations': 0,
        'current_p_mut': 0.1,
        'global_best_individual': None,
        'global_best_fitness': -float('inf'),
        'best_generation': 0
    }

    direction_mult = 1.0
    if mode in ['1', '2']:
        print("\nKIERUNEK OPTYMALIZACJI")
        print("1. Maksymalizacja (Szukaj najwyższego punktu / największej wartości)")
        print("2. Minimalizacja (Szukaj najniższego punktu / najmniejszej wartości)")
        dir_choice = get_parameter("Wybór (1-2)", 1, int, min_val=1, max_val=2)
        if dir_choice == 2:
            direction_mult = -1.0
            print("Uwaga: Wybrano minimalizację. Do selekcji zalecamy użyć Metody Turniejowej (radzi sobie z ujemnym fitnessem).")

    selected_cities = [(0, 0), (0, 3), (4, 0), (4, 3), (2, 2)]

    if mode == '3':
        print("\nKONFIGURACJA MAPY (TSP)")
        custom_cities = input("Czy chcesz wprowadzić własne współrzędne miast? (T/N) [Domyślnie: N]: ").strip().upper()
        if custom_cities == 'T':
            selected_cities = []
            count = get_parameter("Ile miast chcesz dodać na mapę? (minimum 3)", 5, int, min_val=3)
            count = max(3, count)
            print("\nWprowadź współrzędne miast (X i Y) oddzielone spacją (np. '2.5 4.0'):")
            for i in range(count):
                while True:
                    try:
                        coords = input(f"Miasto {i}: ")
                        mx, my = map(float, coords.split())
                        selected_cities.append((mx, my))
                        break
                    except ValueError:
                        print("Błąd formatu. Wpisz dwie liczby oddzielone spacją.")

    print("\nKONFIGURACJA META-PARAMETRÓW ALGORYTMU")
    print("(Wciśnij ENTER, aby użyć wartości domyślnej)")

    chromosome_length = 0
    if mode == '1':
        chromosome_length = get_parameter("Długość chromosomu (np. 8, 16, 32, musi być parzysta do funkcji 2D)", 16, int, min_val=2)
        if chromosome_length % 2 != 0:
            chromosome_length += 1
            print(f"Uwaga: Długość chromosomu zmieniona na {chromosome_length}, aby umożliwić podział na osie X i Y.")

    pop_size = get_parameter("Rozmiar populacji (np. 20, 50, 100)", 20 if mode != '2' else 30, int, min_val=2)
    elite_size = get_parameter("Rozmiar elity (ile najlepszych osobników przechodzi bez zmian)", 0, int, min_val=0)
    replacement_rate = get_parameter("Część zastępowanej populacji (Generational Gap 0.0 - 1.0)", 1.0, float, min_val=0.0, max_val=1.0)

    while True:
        diploid_ans = input("Czy użyć osobników diploidalnych? (T/N) [Domyślnie: N]: ").strip().upper()
        if diploid_ans in ['T', 'N', '']:
            is_diploid = True if diploid_ans == 'T' else False
            break
        else:
            print("Błędna wartość. Wpisz 'T' dla TAK lub 'N' dla NIE (lub wciśnij ENTER).")

    p_cross = get_parameter("Prawdopodobieństwo krzyżowania (0.0 - 1.0)", 0.8, float, min_val=0.0, max_val=1.0)
    k_points = get_parameter("Liczba punktów krzyżowania (k)", 1, int, min_val=1) if mode in ['1', '2'] else 1

    default_mut = 0.1 if mode == '1' else (0.3 if mode == '2' else 0.2)
    p_mut = get_parameter("Prawdopodobieństwo mutacji (0.0 - 1.0)", default_mut, float, min_val=0.0, max_val=1.0)
    state['current_p_mut'] = p_mut

    print("\nMECHANIZM ZAPOBIEGANIA NISZOM (SHARING)")
    while True:
        use_sharing = input("Czy użyć mechanizmu zapobiegania niszom? (T/N) [Domyślnie: N]: ").strip().upper()
        if use_sharing in ['T', 'N', '']:
            break
        print("Błędna wartość. Wpisz 'T' dla TAK lub 'N' dla NIE (lub wciśnij ENTER).")
    
    crowding_func = None
    if use_sharing == 'T':
        sigma_s = get_parameter("Parametr przestrzeni niszy (sigma_share)", 2.0, float, min_val=0.001)
        alpha_s = get_parameter("Parametr skalowania dystansu (alpha)", 1.0, float, min_val=0.001)
        sharing_encoding = "binary" if mode == '1' else ("real" if mode == '2' else "ordering")
        crowding_func = partial(crowding_function, sigma_share=sigma_s, alpha=alpha_s, encoding=sharing_encoding)

    selection_operator = choose_selection()

    crossover_operator = None
    mutation_operator = None
    fitness_evaluator = None
    initialization_func = None

    if mode == '1':
        print("\nALFABET CHROMOSOMU")
        alphabet_input = input("Wprowadź własny alfabet oddzielony spacjami [Domyślnie: 0 1]: ")
        if not alphabet_input.strip(): bin_alphabet = [0, 1]
        else: bin_alphabet = [x if x.isalpha() else int(x) if x.isdigit() else x for x in alphabet_input.split()]

        print("\nDEFINICJA FUNKCJI DOCELOWEJ")
        print(f"1. OneMax (Suma znaków docelowych, domyślnie '{bin_alphabet[-1]}')")
        print(f"2. ZeroMax (Kara za znaki docelowe)")
        print("3. Funkcja Krajobrazu ( x*sin + y*cos )")
        print("4. Funkcja Sferyczna ( x^2 + y^2 )")
        print("5. Rastrigin (Wiele minimów lokalnych)")
        print("6. Zmienna w czasie (Falujące dno, funkcja czasu 't')")
        print("7. Własny wzór matematyczny (zmienne 'x' i 'y' zdekodowane z bitów, 't' to generacja)")
        func_choice = get_parameter("Wybór (1-7)", 1, int, min_val=1, max_val=7)
        custom_formula = input("Podaj wzór (np. x**2 + y**2 + t): ") if func_choice == 7 else None

        objective_func = partial(binary_evaluator, func_type=func_choice, state=state, custom_formula=custom_formula, target_char=str(bin_alphabet[-1]), direction_mult=direction_mult)
        fitness_evaluator = configure_scaling(objective_func)
        initialization_func = partial(initialize_population, alphabet=bin_alphabet, domains=None, encoding="natural", is_diploid=is_diploid, crowding_function=crowding_func)

        print("\nWYBÓR METODY KRZYŻOWANIA")
        print("1. K-punktowe (k-Point Crossover)")
        print("2. Równomierne (Uniform Crossover)")
        cross_choice = get_parameter("Wybór", 1, int, min_val=1, max_val=2)
        crossover_operator = partial(uniform, p_crossover=p_cross) if cross_choice == 2 else partial(kpoint_crossover, k=k_points, p_crossover=p_cross)

        print("\nWYBÓR METODY MUTACJI")
        print("1. Jednopunktowa (Single Point Mutation)")
        mutation_operator = partial(single_point, alphabet=bin_alphabet, p_mutate=p_mut)
        print("\nTRYB BINARNY (NATURALNY)")

    elif mode == '2':
        chromosome_length = 2
        domains = [(-5.0, 5.0), (-5.0, 5.0)]

        print("\nDEFINICJA FUNKCJI DOCELOWEJ")
        print("1. Funkcja Krajobrazu ( x*sin + y*cos )")
        print("2. Funkcja Sferyczna ( x^2 + y^2 )")
        print("3. Rastrigin (Wiele minimów lokalnych)")
        print("4. Zmienna w czasie (Falujące dno, funkcja czasu 't')")
        print("5. Własny wzór (dostępne zmienne: 'x', 'y', 't', operatory: np.sin, math.cos)")
        func_choice = get_parameter("Wybór (1-5)", 1, int, min_val=1, max_val=5)
        custom_formula = input("Podaj wzór (np. x**2 + y**2 + np.sin(t)): ") if func_choice == 5 else None

        objective_func = partial(continuous_evaluator, func_type=func_choice, state=state, custom_formula=custom_formula, direction_mult=direction_mult)
        fitness_evaluator = configure_scaling(objective_func)
        initialization_func = partial(initialize_population, alphabet=None, domains=domains, encoding="real", is_diploid=is_diploid, crowding_function=crowding_func)

        print("\nWYBÓR METODY KRZYŻOWANIA")
        print("1. K-punktowe (k-Point Crossover)")
        print("2. Równomierne (Uniform Crossover)")
        print("3. Rekombinacja pośrednia (Intermediate Recombination)")
        cross_choice = get_parameter("Wybór", 3, int, min_val=1, max_val=3)
        if cross_choice == 1: crossover_operator = partial(kpoint_crossover, k=k_points, p_crossover=p_cross)
        elif cross_choice == 2: crossover_operator = partial(uniform, p_crossover=p_cross)
        else: crossover_operator = partial(intermediate_recombination, p_crossover=p_cross)

        print("\nWYBÓR METODY MUTACJI")
        print("1. Mutacja Gaussa (Gaussian Mutation)")
        mutation_scale = get_parameter("Skala mutacji (odchylenie standardowe)", 0.5, float, min_val=0.001)
        mutation_operator = partial(gauss, p_mutate=p_mut, bounds=domains, mutation_scale=mutation_scale)
        print("\nTRYB CIĄGŁY (LICZBY RZECZYWISTE)")

    elif mode == '3':
        chromosome_length = len(selected_cities)
        bin_alphabet = list(range(chromosome_length))

        print("\nDEFINICJA FUNKCJI DOCELOWEJ")
        print("1. Klasyczny TSP (Stałe odległości)")
        print("2. TSP Dynamiczny (Zmienne warunki drogowe w czasie 't')")
        func_choice = get_parameter("Wybór (1-2)", 1, int, min_val=1, max_val=2)

        objective_func = partial(tsp_evaluator, func_type=func_choice, state=state, selected_cities=selected_cities)
        fitness_evaluator = configure_scaling(objective_func)
        initialization_func = partial(initialize_population, alphabet=bin_alphabet, domains=None, encoding="ordering", is_diploid=is_diploid, crowding_function=crowding_func)

        print("\nWYBÓR METODY KRZYŻOWANIA")
        print("1. Partially Mapped Crossover (PMX)")
        print("2. Order Crossover (OX)")
        print("3. Cycle Crossover (CX)")
        cross_choice = get_parameter("Wybór", 1, int, min_val=1, max_val=3)
        if cross_choice == 2: crossover_operator = partial(order, p_crossover=p_cross)
        elif cross_choice == 3: crossover_operator = partial(cycle, p_crossover=p_cross)
        else: crossover_operator = partial(partially_mapped, p_crossover=p_cross)

        print("\nWYBÓR METODY MUTACJI")
        print("1. Inwersja (Inversion)")
        print("2. Rotacja (Rotation)")
        mut_choice = get_parameter("Wybór", 1, int, min_val=1, max_val=2)
        if mut_choice == 2:
            while True:
                direction = input("Kierunek rotacji (left/right) [Domyślnie: right]: ").strip().lower()
                if direction in ['left', 'right', '']:
                    break
                print("Błędna wartość. Wpisz 'left' lub 'right'.")
            step = get_parameter("Krok rotacji", 1, int, min_val=1)
            mutation_operator = partial(rotation, p_mutate=p_mut, rotation_step=step, direction=direction if direction in ['left', 'right'] else 'right')
        else:
            mutation_operator = partial(inversion, p_mutate=p_mut)
        print("\nTRYB PORZĄDKOWY (KOMIWOJAŻER)")

    def count_crossover(*args, **kwargs):
        state['crossover_count'] += 1
        state['total_crossovers'] += 1
        return crossover_operator(*args, **kwargs)

    def count_mutation(*args, **kwargs):
        state['mutation_count'] += 1
        state['total_mutations'] += 1
        return mutation_operator(*args, **kwargs)

    history_gen = []
    history_best = []
    history_avg = []
    history_worst = []
    history_std = []

    def save_statistics(pop, gen):
        fitness_values = [ind.practical_fitness for ind in pop]
        std_dev = np.std(fitness_values)

        history_gen.append(gen)
        history_best.append(max(fitness_values))
        history_avg.append(sum(fitness_values) / len(fitness_values))
        history_worst.append(min(fitness_values))
        history_std.append(std_dev)

        best_in_current = max(pop, key=lambda ind: ind.practical_fitness)
        if best_in_current.practical_fitness > state['global_best_fitness']:
            state['global_best_fitness'] = best_in_current.practical_fitness
            state['global_best_individual'] = best_in_current
            state['best_generation'] = gen

    def reset_simulation():
        print("\nGenerowanie nowej populacji...")
        state['population'] = initialization_func(pop_size=pop_size, chromosome_length=chromosome_length, fitness_function=fitness_evaluator)
        state['gen'] = state['crossover_count'] = state['mutation_count'] = state['total_crossovers'] = state['total_mutations'] = 0
        state['global_best_fitness'] = -float('inf')
        history_gen.clear(); history_best.clear(); history_avg.clear(); history_worst.clear(); history_std.clear()
        save_statistics(state['population'], 0)

    reset_simulation()

    while True:
        if state['population']:
            fitness_values = [ind.practical_fitness for ind in state['population']]
            std_dev = np.std(fitness_values)
            print(f"\nTRYB: {mode_name} | \nGENERACJA: {state['gen']}")
            print(f"OPERACJE - Liczba krzyżowań: {state['crossover_count']} | Liczba mutacji: {state['mutation_count']}")
            print(f"FITNESS -  Najlepszy: {max(fitness_values):.4f} | Średni: {sum(fitness_values)/len(fitness_values):.4f} | Najgorszy: {min(fitness_values):.4f}")
            print(f"POPULACJA - Odchylenie standardowe: {std_dev:.4f}")

        print("\nDostępne opcje:")
        print("1. Przejdź o N generacji")
        if mode == '1':
            print("2. Wyszukaj schemat w populacji.")
            print("3. Pokaż animację krajobrazu 2D na żywo.")
            print("4. Wykres zbieżności algorytmu.")
            print("5. Wyświetl szczegóły wszystkich osobników.")
            print("6. Pokaż statystyki globalne (Podsumowanie).")
            print("7. Resetuj populację.")
            print("8. Eksportuj statystyki do pliku CSV.")
            print("9. Zmień tryb pracy (Powrót do menu).")
            print("10. Zakończ aplikację.")
            mapping = {'1': '1', '2': 'schema', '3': 'anim', '4': 'chart', '5': 'details', '6': 'stats', '7': 'reset', '8': 'export', '9': 'back', '10': 'exit'}
        else:
            if mode == '2': print("2. Pokaż animację krajobrazu 2D na żywo.")
            elif mode == '3': print("2. Pokaż na mapie najlepszą trasę kuriera.")
            print("3. Wykres zbieżności algorytmu.")
            print("4. Wyświetl szczegóły wszystkich osobników.")
            print("5. Pokaż statystyki globalne (Podsumowanie).")
            print("6. Resetuj populację.")
            print("7. Eksportuj statystyki do pliku CSV.")
            print("8. Zmień tryb pracy (Powrót do menu).")
            print("9. Zakończ aplikację.")
            mapping = {'1': '1', '2': 'anim', '3': 'chart', '4': 'details', '5': 'stats', '6': 'reset', '7': 'export', '8': 'back', '9': 'exit'}

        choice = input("Wybierz opcję: ")
        mapped_choice = mapping.get(choice, 'unknown')

        if mapped_choice == '1':
            steps = get_parameter("Ile generacji obliczyć?", 1, int, min_val=1)
            for i in range(steps):
                state['crossover_count'] = state['mutation_count'] = 0
                state['population'] = next_generation(
                    prev_gen=state['population'], pop_size=pop_size, p_mutation=state['current_p_mut'],
                    fitness_function=fitness_evaluator, crossover=count_crossover, mutation=count_mutation,
                    selection=selection_operator, is_diploid=is_diploid, crowding_function=crowding_func,
                    elite_size=elite_size, replacement=replacement_rate)
                state['gen'] += 1
                save_statistics(state['population'], state['gen'])

        elif mapped_choice == 'schema':
            pattern = input("Podaj schemat do wyszukania (np. **1*0*1*): ")
            found = find_matching_schemas(state['population'], pattern)
            if found:
                print(f"\nZnaleziono {len(found)} osobników:"); [print(f"- {os}") for os in found]
            else: print(f"Brak dopasowań dla schematu: {pattern}")

        elif mapped_choice == 'anim':
            if mode == '1':
                max_val = (1 << (chromosome_length // 2)) - 1
                if max_val <= 0: max_val = 1
                def decode_bin(chrom):
                    half = len(chrom) // 2
                    x_b = ['1' if str(g) == str(bin_alphabet[-1]) else '0' for g in chrom[:half]]
                    y_b = ['1' if str(g) == str(bin_alphabet[-1]) else '0' for g in chrom[half:]]
                    x_int = int("".join(x_b), 2) if x_b else 0
                    y_int = int("".join(y_b), 2) if y_b else 0
                    return -5.0 + 10.0 * (x_int / max_val), -5.0 + 10.0 * (y_int / max_val)
                
                run_animation(state, pop_size, fitness_evaluator, count_crossover, count_mutation, selection_operator, is_diploid, crowding_func, elite_size, replacement_rate, save_statistics, bounds=(-5, 5), decode_func=decode_bin)
            elif mode == '2':
                run_animation(state, pop_size, fitness_evaluator, count_crossover, count_mutation, selection_operator, is_diploid, crowding_func, elite_size, replacement_rate, save_statistics)
            elif mode == '3':
                show_tsp_route(state['population'], selected_cities, state['gen'])

        elif mapped_choice == 'chart': show_line_chart(history_gen, history_best, history_avg, history_worst, history_std)

        elif mapped_choice == 'details':
            print(f"\nPopulacja w Generacji: {state['gen']}")
            for i, ind in enumerate(state['population']):
                phenotype = ind.get_phenotype()
                display_phenotype = "".join(map(str, phenotype)) if mode == '1' else ([round(float(x), 4) for x in phenotype] if mode == '2' else phenotype)
                print(f"Osobnik {i + 1:02d} (Fenotyp): {display_phenotype} | Fitness: {ind.practical_fitness:.4f}")
                if is_diploid and ind.chromosome2 is not None:
                    c1 = "".join(map(str, ind.chromosome1)) if mode == '1' else ([round(float(x), 4) for x in ind.chromosome1] if mode == '2' else ind.chromosome1)
                    c2 = "".join(map(str, ind.chromosome2)) if mode == '1' else ([round(float(x), 4) for x in ind.chromosome2] if mode == '2' else ind.chromosome2)
                    d = "".join(map(str, ind.domination_chromosome)) if mode == '1' else ind.domination_chromosome
                    print(f"   (Chr1: {c1} | Chr2: {c2} | Dom: {d})")

        elif mapped_choice == 'stats':
            if not state['global_best_individual']: print("\nBrak danych."); continue
            print(f"\nSTATYSTYKI GLOBALNE - PODSUMOWANIE DZIAŁANIA\nLiczba krzyżowań: {state['total_crossovers']} | Liczba mutacji: {state['total_mutations']}")
            phenotype = state['global_best_individual'].get_phenotype()
            display_phenotype = "".join(map(str, phenotype)) if mode == '1' else ([round(float(x), 4) for x in phenotype] if mode == '2' else phenotype)
            print(f"Najlepszy osobnik: {display_phenotype} | Fitness: {state['global_best_fitness']:f} | Generacja powstania: {state['best_generation']}")

        elif mapped_choice == 'reset': reset_simulation()

        elif mapped_choice == 'export':
            if not history_gen: print("\nBrak danych do eksportu."); continue
            file_name = input("\nPodaj nazwę pliku (np. wyniki.csv) [Domyślnie: statystyki.csv]: ").strip()
            if not file_name: file_name = "statystyki.csv"
            if not file_name.endswith('.csv'): file_name += '.csv'
            try:
                with open(file_name, mode='w', newline='', encoding='utf-8-sig') as f:
                    writer = csv.writer(f, delimiter=';')
                    writer.writerow(["Generacja", "Najlepszy Fitness", "Sredni Fitness", "Najgorszy Fitness", "Odchylenie Std"])
                    for i in range(len(history_gen)):
                        writer.writerow([history_gen[i], str(history_best[i]).replace('.', ','), str(history_avg[i]).replace('.', ','), str(history_worst[i]).replace('.', ','), str(history_std[i]).replace('.', ',')])
                print(f"Dane zapisane w: {file_name}")
            except Exception as e: print(f"Błąd zapisu: {e}")

        elif mapped_choice == 'back':
            print("\nWracam do menu głównego...\n")
            return

        elif mapped_choice == 'exit': sys.exit(0)
        else: print("\nNieznana opcja. Spróbuj ponownie.")

def main():
    while True:
        run_app()

if __name__ == "__main__":
    main()