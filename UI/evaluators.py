import numpy as np
import math

def binary_evaluator(chromosome, func_type, state, direction_mult=1.0, custom_formula=None, target_char='1'):
    is_decoded = len(chromosome) == 2 and isinstance(chromosome[0], (int, float, np.number))
    
    if is_decoded:
        x, y = chromosome[0], chromosome[1]
    else:
        half = len(chromosome) // 2
        x_bits = ['1' if str(g) == target_char else '0' for g in chromosome[:half]]
        y_bits = ['1' if str(g) == target_char else '0' for g in chromosome[half:]]
        x_int = int("".join(x_bits), 2) if x_bits else 0
        y_int = int("".join(y_bits), 2) if y_bits else 0
        
        max_val = (1 << half) - 1
        if max_val <= 0: max_val = 1
        x = -5.0 + 10.0 * (x_int / max_val)
        y = -5.0 + 10.0 * (y_int / max_val)

    if func_type == 1:
        if is_decoded: return (x + y) * direction_mult
        result = sum([1 for gene in chromosome if str(gene) == target_char])
    elif func_type == 2:
        if is_decoded: return -(x + y) * direction_mult
        result = sum([1 for gene in chromosome if str(gene) != target_char])
    elif func_type == 3:
        result = np.sin(x) * np.cos(y) + 2
    elif func_type == 4:
        result = x ** 2 + y ** 2
    elif func_type == 5:
        result = 20 + (x ** 2 - 10 * np.cos(2 * math.pi * x)) + (y ** 2 - 10 * np.cos(2 * math.pi * y))
    elif func_type == 6:
        result = x ** 2 + y ** 2 + 5 * np.sin(state['gen'] / 5.0)
    elif func_type == 7:
        t = state['gen']
        try:
            result = eval(custom_formula, {"math": math, "np": np, "x": x, "y": y, "t": t})
        except Exception:
            result = 0.0
            
    return result * direction_mult

def continuous_evaluator(chromosome, func_type, state, direction_mult=1.0, custom_formula=None):
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
        except Exception:
            result = 0.0

    return result * direction_mult

def tsp_evaluator(chromosome, func_type, state, selected_cities):
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