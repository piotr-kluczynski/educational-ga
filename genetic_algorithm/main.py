import sys
import csv
from functools import partial
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button, Slider
import numpy as np
import time
import math

from genetic_algorithm import initialize_population, next_generation
from fitness_function import fitness_function
from sharing import crowding_function

from crossover import kpoint_crossover, uniform, intermediate_recombination, partially_mapped, order, cycle
from mutation import single_point, gauss, rotation, inversion
from selection import deterministic_sampling, stochastic_sampling_with_replacement, tournament, stochastic_tournament


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
    print("1. Turniejowa (Tournament) - Najbezpieczniejsza przy ujemnym fitnessie (np. w Minimalizacji)")
    print("2. Ruletka ze zwracaniem (Stochastic Sampling) - UWAGA: Wymaga ściśle dodatniego fitnessu!")
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


def main():
    print("APLIKACJA EDUKACYJNA: ALGORYTMY GENETYCZNE")
    print("\nWybierz tryb pracy algorytmu:")
    print("1. Optymalizacja funkcji liczb całkowitych (Binarne).")
    print("2. Optymalizacja funkcji liczb zmiennoprzecinkowych (Ciągłe).")
    print("3. Problemy porządkowania (Problem Komiwojażera).")

    mode = input("Twój wybór (1-3): ")

    if mode not in ['1', '2', '3']:
        print("Nieznany tryb. Zamykam program.")
        sys.exit(1)

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
        dir_choice = get_parameter("Wybór (1-2)", 1, int)
        if dir_choice == 2:
            direction_mult = -1.0
            print("Uwaga: Wybrano minimalizację. Do selekcji zalecamy użyć Metody Turniejowej (radzi sobie z ujemnym fitnessem).")

    def binary_evaluator(chromosome, func_type, custom_formula=None, target_char='1'):
        if func_type == 1:
            result = sum([1 for gene in chromosome if str(gene) == target_char])
        elif func_type == 2:
            result = sum([1 for gene in chromosome if str(gene) != target_char])
        elif func_type == 3:
            c = [float(x) if str(x).replace('.', '').isdigit() else x for x in chromosome]
            t = state['gen']
            try:
                result = eval(custom_formula, {"math": math, "np": np, "c": c, "t": t, "sum": sum, "len": len})
            except Exception as e:
                result = 0.0
        return result * direction_mult

    def continuous_evaluator(chromosome, func_type, custom_formula=None):
        x, y = chromosome[0], chromosome[1]
        t = state['gen']

        if func_type == 1:
            result = np.sin(x) * np.cos(y) + 2
        elif func_type == 2:
            result = x ** 2 + y ** 2
        elif func_type == 3:
            result = 20 + (x ** 2 - 10 * np.cos(2 * math.pi * x)) + (y ** 2 - 10 * np.cos(2 * math.pi * y))
        elif func_type == 4:
            result = x ** 2 + y ** 2 + 5 * np.sin(t / 5.0)
        elif func_type == 5:
            try:
                result = eval(custom_formula, {"math": math, "np": np, "x": x, "y": y, "t": t})
            except Exception as e:
                result = 0.0

        return result * direction_mult

    selected_cities = [(0, 0), (0, 3), (4, 0), (4, 3), (2, 2)]

    def tsp_evaluator(chromosome, func_type):
        distance = 0
        t = state['gen']
        for i in range(len(chromosome)):
            city1 = selected_cities[chromosome[i]]
            city2 = selected_cities[chromosome[(i + 1) % len(chromosome)]]
            distance += math.dist(city1, city2)

        if func_type == 1:
            return 100.0 / (distance + 0.0001)
        elif func_type == 2:
            weather_factor = 1.0 + 0.5 * abs(np.sin(t / 10.0))
            return 100.0 / ((distance * weather_factor) + 0.0001)

    if mode == '3':
        print("\nKONFIGURACJA MAPY (TSP)")
        custom_cities = input("Czy chcesz wprowadzić własne współrzędne miast? (T/N) [Domyślnie: N]: ").strip().upper()
        if custom_cities == 'T':
            selected_cities = []
            count = get_parameter("Ile miast chcesz dodać na mapę? (minimum 3)", 5, int)
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

    pop_size = get_parameter("Rozmiar populacji (np. 20, 50, 100)", 20 if mode != '2' else 30, int)
    elite_size = get_parameter("Rozmiar elity (ile najlepszych osobników przechodzi bez zmian)", 0, int)
    replacement_rate = get_parameter("Część zastępowanej populacji (Generational Gap 0.0 - 1.0)", 1.0, float)

    while True:
        diploid_ans = input("Czy użyć osobników diploidalnych? (T/N) [Domyślnie: N]: ").strip().upper()
        if diploid_ans in ['T', 'N', '']:
            is_diploid = True if diploid_ans == 'T' else False
            break
        else:
            print("Błędna wartość. Wpisz 'T' dla TAK lub 'N' dla NIE (lub wciśnij ENTER).")

    p_cross = get_parameter("Prawdopodobieństwo krzyżowania (0.0 - 1.0)", 0.8, float)

    k_points = 1
    if mode in ['1', '2']:
        k_points = get_parameter("Liczba punktów krzyżowania (k)", 1, int)

    default_mut = 0.1 if mode == '1' else (0.3 if mode == '2' else 0.2)
    p_mut = get_parameter("Prawdopodobieństwo mutacji (0.0 - 1.0)", default_mut, float)
    state['current_p_mut'] = p_mut

    print("\nMECHANIZM ZAPOBIEGANIA NISZOM (SHARING)")
    use_sharing = input("Czy użyć mechanizmu zapobiegania niszom? (T/N) [Domyślnie: N]: ").strip().upper()
    crowding_func = None
    if use_sharing == 'T':
        sigma_s = get_parameter("Parametr przestrzeni niszy (sigma_share)", 2.0, float)
        alpha_s = get_parameter("Parametr skalowania dystansu (alpha)", 1.0, float)
        sharing_encoding = "binary" if mode == '1' else ("real" if mode == '2' else "ordering")
        crowding_func = partial(crowding_function, sigma_share=sigma_s, alpha=alpha_s, encoding=sharing_encoding)

    selection_operator = choose_selection()

    crossover_operator = None
    mutation_operator = None
    fitness_evaluator = None
    initialization_func = None
    chromosome_length = 0

    if mode == '1':
        chromosome_length = 8

        print("\nALFABET CHROMOSOMU")
        alphabet_input = input("Wprowadź własny alfabet oddzielony spacjami [Domyślnie: 0 1]: ")
        if not alphabet_input.strip():
            bin_alphabet = [0, 1]
        else:
            bin_alphabet = [x if x.isalpha() else int(x) if x.isdigit() else x for x in alphabet_input.split()]

        print("\nDEFINICJA FUNKCJI DOCELOWEJ")
        print(f"1. OneMax (Suma znaków docelowych, domyślnie '{bin_alphabet[-1]}')")
        print(f"2. ZeroMax (Kara za znaki docelowe)")
        print("3. Własny wzór matematyczny (zmienna 'c' to lista genów, 't' to generacja)")
        func_choice = get_parameter("Wybór (1-3)", 1, int)

        custom_formula = None
        target_char = str(bin_alphabet[-1])
        if func_choice == 3:
            custom_formula = input("Podaj wzór (np. sum(c) + t): ")

        objective_func = partial(binary_evaluator, func_type=func_choice, custom_formula=custom_formula, target_char=target_char)
        fitness_evaluator = configure_scaling(objective_func)
        initialization_func = partial(initialize_population, alphabet=bin_alphabet, domains=None, encoding="natural",
                                is_diploid=is_diploid, crowding_function=crowding_func)

        print("\nWYBÓR METODY KRZYŻOWANIA")
        print("1. K-punktowe (k-Point Crossover)")
        print("2. Równomierne (Uniform Crossover)")
        cross_choice = get_parameter("Wybór", 1, int)
        if cross_choice == 2:
            crossover_operator = partial(uniform, p_crossover=p_cross)
        else:
            crossover_operator = partial(kpoint_crossover, k=k_points, p_crossover=p_cross)

        print("\nWYBÓR METODY MUTACJI")
        print("1. Jednopunktowa (Single Point Mutation)")
        mutation_operator = partial(single_point, alphabet=bin_alphabet, p_mutate=p_mut)


    elif mode == '2':
        chromosome_length = 2
        domains = [(-5.0, 5.0), (-5.0, 5.0)]

        print("\nDEFINICJA FUNKCJI DOCELOWEJ")
        print("1. Funkcja Krajobrazu ( x*sin + y*cos )")
        print("2. Funkcja Sferyczna ( x^2 + y^2 )")
        print("3. Rastrigin (Wiele minimów lokalnych)")
        print("4. Zmienna w czasie (Falujące dno, funkcja czasu 't')")
        print("5. Własny wzór (dostępne zmienne: 'x', 'y', 't', operatory: np.sin, math.cos)")
        func_choice = get_parameter("Wybór (1-5)", 1, int)

        custom_formula = None
        if func_choice == 5:
            custom_formula = input("Podaj wzór (np. x**2 + y**2 + np.sin(t)): ")

        objective_func = partial(continuous_evaluator, func_type=func_choice, custom_formula=custom_formula)
        fitness_evaluator = configure_scaling(objective_func)
        initialization_func = partial(initialize_population, alphabet=None, domains=domains, encoding="real",
                                is_diploid=is_diploid, crowding_function=crowding_func)

        print("\nWYBÓR METODY KRZYŻOWANIA")
        print("1. K-punktowe (k-Point Crossover)")
        print("2. Równomierne (Uniform Crossover)")
        print("3. Rekombinacja pośrednia (Intermediate Recombination)")
        cross_choice = get_parameter("Wybór", 3, int)
        if cross_choice == 1:
            crossover_operator = partial(kpoint_crossover, k=k_points, p_crossover=p_cross)
        elif cross_choice == 2:
            crossover_operator = partial(uniform, p_crossover=p_cross)
        else:
            crossover_operator = partial(intermediate_recombination, p_crossover=p_cross)

        print("\nWYBÓR METODY MUTACJI")
        print("1. Mutacja Gaussa (Gaussian Mutation)")
        mutation_scale = get_parameter("Skala mutacji (odchylenie standardowe)", 0.5, float)
        mutation_operator = partial(gauss, p_mutate=p_mut, bounds=domains, mutation_scale=mutation_scale)


    elif mode == '3':
        chromosome_length = len(selected_cities)
        bin_alphabet = list(range(chromosome_length))

        print("\nDEFINICJA FUNKCJI DOCELOWEJ")
        print("1. Klasyczny TSP (Stałe odległości)")
        print("2. TSP Dynamiczny (Zmienne warunki drogowe w czasie 't')")
        func_choice = get_parameter("Wybór (1-2)", 1, int)

        objective_func = partial(tsp_evaluator, func_type=func_choice)
        fitness_evaluator = configure_scaling(objective_func)
        initialization_func = partial(initialize_population, alphabet=bin_alphabet, domains=None, encoding="ordering",
                                is_diploid=is_diploid, crowding_function=crowding_func)

        print("\nWYBÓR METODY KRZYŻOWANIA")
        print("1. Partially Mapped Crossover (PMX)")
        print("2. Order Crossover (OX)")
        print("3. Cycle Crossover (CX)")
        cross_choice = get_parameter("Wybór", 1, int)
        if cross_choice == 2:
            crossover_operator = partial(order, p_crossover=p_cross)
        elif cross_choice == 3:
            crossover_operator = partial(cycle, p_crossover=p_cross)
        else:
            crossover_operator = partial(partially_mapped, p_crossover=p_cross)

        print("\nWYBÓR METODY MUTACJI")
        print("1. Inwersja (Inversion)")
        print("2. Rotacja (Rotation)")
        mut_choice = get_parameter("Wybór", 1, int)
        if mut_choice == 2:
            direction = input("Kierunek rotacji (left/right) [Domyślnie: right]: ").strip().lower()
            if direction not in ['left', 'right']: direction = 'right'
            step = get_parameter("Krok rotacji", 1, int)
            mutation_operator = partial(rotation, p_mutate=p_mut, rotation_step=step, direction=direction)
        else:
            mutation_operator = partial(inversion, p_mutate=p_mut)



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

    def show_global_statistics():
        if state['global_best_individual'] is None:
            print("\nBrak danych. Wykonaj najpierw algorytm.")
            return

        print("\nSTATYSTYKI GLOBALNE - PODSUMOWANIE DZIAŁANIA")
        print(f"Liczba wykonanych operacji krzyżowania: {state['total_crossovers']}")
        print(f"Liczba wykonanych mutacji: {state['total_mutations']}")

        best_ind = state['global_best_individual']
        phenotype = best_ind.get_phenotype()

        if mode == '1':
            display_phenotype = "".join(map(str, phenotype))
        elif mode == '2':
            display_phenotype = [round(float(x), 4) for x in phenotype]
        elif mode == '3':
            display_phenotype = phenotype

        print(f"Informacje o najlepszym znalezionym osobniku: Fenotyp -> {display_phenotype}")
        if is_diploid and best_ind.chromosome2 is not None:
            if mode == '1':
                c1 = "".join(map(str, best_ind.chromosome1))
                c2 = "".join(map(str, best_ind.chromosome2))
                d = "".join(map(str, best_ind.domination_chromosome))
            elif mode == '2':
                c1 = [round(float(x), 4) for x in best_ind.chromosome1]
                c2 = [round(float(x), 4) for x in best_ind.chromosome2]
                d = best_ind.domination_chromosome
            elif mode == '3':
                c1 = best_ind.chromosome1
                c2 = best_ind.chromosome2
                d = best_ind.domination_chromosome
            print(f"   (Szczegóły -> Chr1: {c1} | Chr2: {c2} | Dom: {d})")

        print(f"Wartość funkcji fitness najlepszego rozwiązania: {state['global_best_fitness']:f}")
        print(f"Numer generacji, w której najlepszy osobnik pojawił się po raz pierwszy: {state['best_generation']}")

    def reset_simulation():
        print("\nGenerowanie nowej populacji...")
        state['population'] = initialization_func(
            pop_size=pop_size, chromosome_length=chromosome_length, fitness_function=fitness_evaluator
        )
        state['gen'] = 0
        state['crossover_count'] = 0
        state['mutation_count'] = 0
        state['total_crossovers'] = 0
        state['total_mutations'] = 0
        state['global_best_fitness'] = -float('inf')

        history_gen.clear()
        history_best.clear()
        history_avg.clear()
        history_worst.clear()
        history_std.clear()
        save_statistics(state['population'], 0)

    def export_to_csv():
        if not history_gen:
            print("\nBrak danych do eksportu. Uruchom najpierw algorytm.")
            return

        file_name = input("\nPodaj nazwę pliku (np. wyniki.csv) [Domyślnie: statystyki.csv]: ").strip()
        if not file_name:
            file_name = "statystyki.csv"
        if not file_name.endswith('.csv'):
            file_name += '.csv'

        try:
            with open(file_name, mode='w', newline='', encoding='utf-8-sig') as file:
                writer = csv.writer(file, delimiter=';')
                writer.writerow(
                    ["Generacja", "Najlepszy Fitness", "Sredni Fitness", "Najgorszy Fitness", "Odchylenie Std"])

                for i in range(len(history_gen)):
                    row = [
                        history_gen[i],
                        str(history_best[i]).replace('.', ','),
                        str(history_avg[i]).replace('.', ','),
                        str(history_worst[i]).replace('.', ','),
                        str(history_std[i]).replace('.', ',')
                    ]
                    writer.writerow(row)
            print(f"Dane zostały pomyślnie zapisane do pliku: {file_name}")
        except Exception as e:
            print(f"Wystąpił problem podczas zapisu do pliku: {e}")

    def run_animation():
        fig, ax = plt.subplots(figsize=(8, 9))
        plt.subplots_adjust(bottom=0.25)

        x = np.linspace(-5, 5, 100)
        y = np.linspace(-5, 5, 100)
        X, Y = np.meshgrid(x, y)

        Z = np.zeros_like(X)
        for i in range(100):
            for j in range(100):
                Z[i, j] = fitness_evaluator([X[i, j], Y[i, j]])

        contour = ax.contour(X, Y, Z, levels=10, cmap='Greens')
        x_individuals = [ind.chromosome1[0] for ind in state['population']]
        y_individuals = [ind.chromosome1[1] for ind in state['population']]
        scatter = ax.scatter(x_individuals, y_individuals, color='magenta', edgecolors='black', s=50)

        ax.set_title(f"Generacja {state['gen']}")
        ax.set_xlim(-5, 5)
        ax.set_ylim(-5, 5)

        flags = {'is_playing': True, 'last_update': time.time(), 'contour': contour}

        ax_speed = plt.axes([0.15, 0.07, 0.45, 0.03])
        slider_speed = Slider(ax_speed, 'Czas (sek):', 0.1, 5.0, valinit=0.5, valstep=0.1)

        def update(frame):
            current_time = time.time()
            if flags['is_playing'] and (current_time - flags['last_update'] >= slider_speed.val):
                state['crossover_count'] = 0
                state['mutation_count'] = 0

                state['population'] = next_generation(
                    prev_gen=state['population'], pop_size=pop_size, p_mutation=state['current_p_mut'],
                    fitness_function=fitness_evaluator,
                    crossover=count_crossover,
                    mutation=count_mutation,
                    selection=selection_operator,
                    is_diploid=is_diploid,
                    crowding_function=crowding_func,
                    elite_size=elite_size,
                    replacement=replacement_rate
                )
                state['gen'] += 1
                save_statistics(state['population'], state['gen'])

                new_x = [ind.chromosome1[0] for ind in state['population']]
                new_y = [ind.chromosome1[1] for ind in state['population']]
                scatter.set_offsets(np.c_[new_x, new_y])
                ax.set_title(f"Generacja {state['gen']}")

                try:
                    for c in flags['contour'].collections:
                        c.remove()
                except AttributeError:
                    flags['contour'].remove()

                Z_new = np.zeros_like(X)
                for i in range(100):
                    for j in range(100):
                        Z_new[i, j] = fitness_evaluator([X[i, j], Y[i, j]])
                flags['contour'] = ax.contour(X, Y, Z_new, levels=10, cmap='Greens')

                flags['last_update'] = current_time
            return scatter,

        anim = FuncAnimation(fig, update, interval=50, cache_frame_data=False)

        ax_pause = plt.axes([0.7, 0.05, 0.2, 0.075])
        btn_pause = Button(ax_pause, 'Pauza / Wznów')

        def toggle_pause(event):
            flags['is_playing'] = not flags['is_playing']
            if flags['is_playing']:
                flags['last_update'] = time.time()

        btn_pause.on_clicked(toggle_pause)
        plt.show()

    def show_line_chart():
        if not history_gen:
            print("\nBrak danych. Wykonaj najpierw algorytm.")
            return

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

        ax1.plot(history_gen, history_best, label='Najlepszy', color='green', marker='o')
        ax1.plot(history_gen, history_avg, label='Średni', color='blue', linestyle='--')
        ax1.plot(history_gen, history_worst, label='Najgorszy', color='red', linestyle=':')
        ax1.set_title("Zbieżność Algorytmu (Wartość Fitness)")
        ax1.set_ylabel("Fitness")
        ax1.legend()
        ax1.grid(True)

        ax2.plot(history_gen, history_std, color='orange', linewidth=2)
        ax2.set_title("Różnorodność Populacji (Odchylenie Standardowe)")
        ax2.set_xlabel("Generacja")
        ax2.set_ylabel("Odchylenie Std")
        ax2.grid(True)

        plt.tight_layout()
        plt.show()

    def show_tsp_route():
        if not state['population']:
            print("\nBrak danych. Wykonaj najpierw algorytm.")
            return

        best_ind = max(state['population'], key=lambda ind: ind.practical_fitness)
        route = best_ind.chromosome1

        x = [selected_cities[i][0] for i in route]
        y = [selected_cities[i][1] for i in route]

        x.append(selected_cities[route[0]][0])
        y.append(selected_cities[route[0]][1])

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot(x, y, marker='o', linestyle='-', color='blue', markersize=10, markerfacecolor='red', linewidth=2)

        for i, (mx, my) in enumerate(selected_cities):
            ax.annotate(f"Miasto {i}", (mx, my), textcoords="offset points", xytext=(0, 10), ha='center', fontsize=12)

        distance = (100.0 / best_ind.practical_fitness) - 0.0001 if best_ind.practical_fitness > 0 else 0

        ax.set_title(f"Trasa Komiwojażera | Gen: {state['gen']} | Dystans bazowy: {distance:.2f}")
        ax.grid(True)
        plt.show()

    reset_simulation()

    while True:
        if state['population']:
            fitness_values = [ind.practical_fitness for ind in state['population']]
            std_dev = np.std(fitness_values)
            print(f"\nTRYB {mode_name}")
            print(f"\nGENERACJA: {state['gen']}")
            print(
                f"OPERACJE - Liczba krzyżowań: {state['crossover_count']} | Liczba mutacji: {state['mutation_count']}")
            print(
                f"FITNESS -  Najlepszy: {max(fitness_values):.4f} | Średni: {sum(fitness_values) / len(fitness_values):.4f} | Najgorszy: {min(fitness_values):.4f}")
            print(f"POPULACJA - Odchylenie standardowe: {std_dev:.4f}")

        print("\nDostępne opcje:")
        print("1. Przejdź o N generacji")
        if mode == '1':
            print("2. Wyszukaj schemat w populacji.")
        elif mode == '2':
            print("2. Pokaż animację krajobrazu 2D na żywo.")
        elif mode == '3':
            print("2. Pokaż na mapie najlepszą trasę kuriera.")
        print("3. Wykres zbieżności algorytmu.")
        print("4. Wyświetl szczegóły wszystkich osobników.")
        print("5. Pokaż statystyki globalne (Podsumowanie).")
        print("6. Resetuj populację.")
        print("7. Eksportuj statystyki do pliku CSV.")
        print("8. Zakończ.")

        choice = input("Wybierz opcję: ")

        if choice == '1':
            try:
                steps = int(input("Ile generacji obliczyć? (np. 1): "))

                for i in range(steps):
                    state['crossover_count'] = 0
                    state['mutation_count'] = 0

                    state['population'] = next_generation(
                        prev_gen=state['population'], pop_size=pop_size, p_mutation=state['current_p_mut'],
                        fitness_function=fitness_evaluator,
                        crossover=count_crossover,
                        mutation=count_mutation,
                        selection=selection_operator,
                        is_diploid=is_diploid,
                        crowding_function=crowding_func,
                        elite_size=elite_size,
                        replacement=replacement_rate
                    )
                    state['gen'] += 1
                    save_statistics(state['population'], state['gen'])

            except ValueError:
                print("Błąd: Podaj liczbę całkowitą.")

        elif choice == '2':
            if mode == '1':
                pattern = input("Podaj schemat do wyszukania (np. **1*0*1*): ")
                found = find_matching_schemas(state['population'], pattern)
                if found:
                    print(f"\nZnaleziono {len(found)} osobników:")
                    for os in found: print(f"- {os}")
                else:
                    print(f"Brak dopasowań dla schematu: {pattern}")
            elif mode == '2':
                run_animation()
            elif mode == '3':
                show_tsp_route()
            else:
                print("\nTa opcja jest niedostępna w wybranym trybie.")

        elif choice == '3':
            show_line_chart()

        elif choice == '4':
            print(f"\nPopulacja w Generacji: {state['gen']}")
            for i, ind in enumerate(state['population']):
                phenotype = ind.get_phenotype()

                if mode == '1':
                    display_phenotype = "".join(map(str, phenotype))
                elif mode == '2':
                    display_phenotype = [round(float(x), 4) for x in phenotype]
                elif mode == '3':
                    display_phenotype = phenotype

                print(f"Osobnik {i + 1:02d} (Fenotyp): {display_phenotype} | Fitness: {ind.practical_fitness:.4f}")

                if is_diploid and ind.chromosome2 is not None:
                    if mode == '1':
                        c1 = "".join(map(str, ind.chromosome1))
                        c2 = "".join(map(str, ind.chromosome2))
                        d = "".join(map(str, ind.domination_chromosome))
                    elif mode == '2':
                        c1 = [round(float(x), 4) for x in ind.chromosome1]
                        c2 = [round(float(x), 4) for x in ind.chromosome2]
                        d = ind.domination_chromosome
                    elif mode == '3':
                        c1 = ind.chromosome1
                        c2 = ind.chromosome2
                        d = ind.domination_chromosome

                    print(f"   (Chr1: {c1} | Chr2: {c2} | Dom: {d})")

        elif choice == '5':
            show_global_statistics()

        elif choice == '6':
            reset_simulation()

        elif choice == '7':
            export_to_csv()

        elif choice == '8':
            sys.exit(0)

        else:
            print("\nNieznana opcja. Spróbuj ponownie.")


if __name__ == "__main__":
    main()