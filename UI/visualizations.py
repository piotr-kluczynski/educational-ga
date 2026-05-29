import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button, Slider, TextBox
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
                  is_diploid, crowding_func, elite_size, replacement_rate, save_statistics_callback,
                  bounds=(-5, 5), decode_func=None):
    fig, ax = plt.subplots(figsize=(8, 9))
    plt.subplots_adjust(bottom=0.25)

    min_val, max_val = bounds
    x = np.linspace(min_val, max_val, 100)
    y = np.linspace(min_val, max_val, 100)
    X, Y = np.meshgrid(x, y)

    Z = np.zeros_like(X)
    for i in range(100):
        for j in range(100):
            Z[i, j] = fitness_evaluator([X[i, j], Y[i, j]])

    contour = ax.contour(X, Y, Z, levels=10, cmap='Greens')
    
    if decode_func:
        x_individuals = [decode_func(ind.chromosome1)[0] for ind in state['population']]
        y_individuals = [decode_func(ind.chromosome1)[1] for ind in state['population']]
    else:
        x_individuals = [ind.chromosome1[0] for ind in state['population']]
        y_individuals = [ind.chromosome1[1] for ind in state['population']]
        
    scatter = ax.scatter(x_individuals, y_individuals, color='magenta', edgecolors='black', s=50)

    ax.set_title(f"Generacja {state['gen']}")
    ax.set_xlim(min_val, max_val)
    ax.set_ylim(min_val, max_val)

    flags = {'is_playing': True, 'last_update': time.time(), 'contour': contour}

    ax_speed = plt.axes([0.15, 0.08, 0.45, 0.03])
    slider_speed = Slider(ax_speed, 'Czas (sek):', 0.1, 5.0, valinit=0.5, valstep=0.1)

    show_schema = (decode_func is not None)
    
    if show_schema:
        ax_schema = plt.axes([0.15, 0.02, 0.45, 0.04])
        text_box_schema = TextBox(ax_schema, 'Filtr schematu:', initial='')

    def update_scatter_colors(*args):
        if not show_schema:
            return
            
        schema = text_box_schema.text.strip()
        colors = []
        edges = []
        sizes = []
        for ind in state['population']:
            chromosome_str = "".join(map(str, ind.chromosome1))
            is_match = False
            if schema and len(schema) == len(chromosome_str):
                is_match = True
                for c_gene, p_gene in zip(chromosome_str, schema):
                    if p_gene != '*' and c_gene != p_gene:
                        is_match = False
                        break
            if is_match:
                colors.append('yellow')
                edges.append('red')
                sizes.append(100)
            else:
                colors.append('magenta')
                edges.append('black')
                sizes.append(50)
                
        scatter.set_facecolors(colors)
        scatter.set_edgecolors(edges)
        scatter.set_sizes(sizes)
        fig.canvas.draw_idle()

    if show_schema:
        text_box_schema.on_text_change(update_scatter_colors)
        update_scatter_colors()

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

            if decode_func:
                new_x = [decode_func(ind.chromosome1)[0] for ind in state['population']]
                new_y = [decode_func(ind.chromosome1)[1] for ind in state['population']]
            else:
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
            if show_schema:
                update_scatter_colors()
            flags['last_update'] = current_time
        return scatter,

    anim = FuncAnimation(fig, update, interval=50, cache_frame_data=False)

    ax_pause = plt.axes([0.7, 0.08, 0.2, 0.075])
    btn_pause = Button(ax_pause, 'Pauza / Wznów')

    def toggle_pause(event):
        flags['is_playing'] = not flags['is_playing']
        if flags['is_playing']:
            flags['last_update'] = time.time()

    btn_pause.on_clicked(toggle_pause)
    plt.show()