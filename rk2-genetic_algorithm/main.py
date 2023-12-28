# Copyright 2020 Alexey Alexandrov <sks2311211@yandex.ru>

import genetic_algorithm as genetic
import numpy as np

# Вариант 1.
# f(x, y) = sin(x) / (1 + x^2 + y^2)

if __name__ == '__main__':
    fit_function = lambda x, y: np.sin(x) / (1 + x ** 2 + y ** 2)
    bounds = [-2., 2., -2., 2.]
    genetic.GeneticAlgorithm(fit_function, bounds).Solve()
