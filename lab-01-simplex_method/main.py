# Copyright 2020 Alexey Alexandrov <sks2311211@yandex.ru>

"""
Лабораторная работа № 1
Линейное программирование. Симплекс-метод.
Цель: Изучить постановку задачи линейного программирования (ЛП);
овладеть навыками решения задач ЛП с помощью симплекс-метода.

Вариант 1.
"""

from simplex import *

if __name__ == '__main__':
    problem = Simplex("input_data.json")

    print('ЛР1 по МО. "ЛП. Симплекс-метод.', )
    print(problem)

    problem.reference_solution()

    problem.optimal_solution()
