# Copyright 2020 Alexey Alexandrov <sks2311211@yandex.ru>

"""
Лабораторная работа № 5
Матричные игры с нулевой суммой. Смешанные стратегии.
Цель работы: изучить постановку антагонстической игры двух лиц в нормальной форме;
получить навыки нахождения решения игры в смешанных стратегиях (стратегическую седловую точку) за обоих игроков.

Вариант 1.
"""
import dual_problem
from simplex import Simplex
import strategic

if __name__ == '__main__':
    print("\tНайдём смешанные стратегии для игрока А. Сформулируем задачу для решения симплекс-методом:")
    dual_p = dual_problem.DualProblem("input_data.json")

    # Находим опорное решение.
    dual_p.reference_solution()
    # Находим оптимальное решение.
    dual_p.optimal_solution()

    #
    print(strategic.StrategyA(dual_p.simplex_table_))

    print("\tНайдём смешанные стратегии для игрока B. Сформулируем задачу для решения симплекс-методом:")
    problem = Simplex("input_data.json")
    print(problem)

    # Находим опорное решение.
    problem.reference_solution()
    # Находим оптимальное решение.
    problem.optimal_solution()

    print(strategic.StrategyB(problem.simplex_table_))
