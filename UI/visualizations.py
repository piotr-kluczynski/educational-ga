import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button, Slider
import numpy as np
import time
from genetic_algorithm.genetic_algorithm import next_generation

def show_line_chart(history_gen, history_best, history_avg, history_worst, history_std):
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

def show_tsp_route(population, selected_cities, gen):
    if not population:
        print("\nBrak danych. Wykonaj najpierw algorytm.")
        return

    best_ind = max(population, key=lambda ind: ind.practical_fitness)
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
    ax.set_title(f"Trasa Komiwojażera | Gen: {gen} | Dystans bazowy: {distance:.2f}")
    ax.grid(True)
    plt.show()

def run_animation(state, pop_size, fitness_evaluator, count_crossover, count_mutation, selection_operator,
                  is_diploid, crowding_func, elite_size, replacement_rate, save_statistics_callback):
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
                fitness_function=fitness_evaluator, crossover=count_crossover, mutation=count_mutation,
                selection=selection_operator, is_diploid=is_diploid, crowding_function=crowding_func,
                elite_size=elite_size, replacement=replacement_rate
            )
            state['gen'] += 1
            save_statistics_callback(state['population'], state['gen'])

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