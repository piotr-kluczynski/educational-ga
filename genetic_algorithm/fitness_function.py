def fitness_function(chromosome, objective_function, custom_transformation=None, constraints=None, scaling="none", scaling_factors=None
):

    if constraints is None:
        constraints = []

    # Objective
    value = objective_function(chromosome)

    # Custom transformation
    if custom_transformation is not None:
        value = custom_transformation(value)

    # Constraints
    for constraint in constraints:
        value -= constraint(chromosome)


    # Scaling
    if scaling == "linear":
        a, b = scaling_factors
        value = a * value + b

    elif scaling == "power":
        alpha = scaling_factors[0]
        if value > 0:
            value = value ** alpha

    return value