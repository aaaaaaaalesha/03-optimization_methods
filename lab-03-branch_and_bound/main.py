# Copyright 2020 Alexey Alexandrov <sks2311211@yandex.ru>

"""
Лабораторная работа № 3
Целочисленное линейное программирование. Метод ветвей и границ.
Цель работы: изучить постановку задачи ЦЛП; получить навыки
решения задачи ЦЛП методом ветвей и границ.

Вариант 1.
"""

from simplex import *
from branch_and_bound import *
import brute_force_method
import branch_and_bound

if __name__ == '__main__':
    problem = Simplex("input_data.json")

    print(problem)
    # Находим опорное решение задачи ЛП
    problem.reference_solution()
    # Находим оптимальное решение задачи ЛП
    problem.optimal_solution()

    # Находим рещение задачи ЦЛП полным перебором.
    brute_force = brute_force_method.BruteForceMethod("input_data.json")
    print(brute_force)

    # Переходим к методу ветвей и границ.
    bb = branch_and_bound.BranchAndBound(problem)
    print(bb)
