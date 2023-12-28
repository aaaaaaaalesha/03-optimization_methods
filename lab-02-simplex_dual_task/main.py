# Copyright 2020 Alexey Alexandrov <sks2311211@yandex.ru>

"""
Лабораторная работа № 2
Двойственность в линейном программировании.
Цель: Изучить постановку двойственной задачи (ДЗ);
ЛП по прямой задаче (ПЗ); получить навыки решения соответствующей
ДЗ по прямой задаче.

Вариант 1.
"""

from simplex import *
import dual_problem

if __name__ == '__main__':
    print('ЛР2 по МО. "ДЗ в линейном программировании.', )

    print("\tПрямая задача ЛП:")
    problem = Simplex("input_data.json")
    print(problem)

    # Находим опорное решение.
    problem.reference_solution()
    # Находим оптимальное решение.
    problem.optimal_solution()

    print("\tДвойственная задача ЛП:")
    dual_p = dual_problem.DualProblem("input_data.json")

    # Находим опорное решение.
    dual_p.reference_solution()
    # Находим оптимальное решение.
    dual_p.optimal_solution()
