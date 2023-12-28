# Copyright 2020 Alexey Alexandrov <sks2311211@yandex.ru>

import numpy as np
from prettytable import PrettyTable

# Вероятность "везения" для критерия Гурвица.
ALPHA = 0.5


def OutMatrix(matrix: np.array):
    table = PrettyTable()
    table.field_names = ["Стратегии"] + [f"b{j}" for j in range(1, matrix.shape[1] + 1)]
    for i in range(matrix.shape[0]):
        table.add_row([f"a{i + 1}"] + list(matrix[i]))

    return table


def BernulliCriteria(matrix: np.array):
    """
    Криетрий недостаточного основания (Бернулли).
    :param matrix: матрица стратегий.
    :return полученная оптимальная стратегия.
    """
    table = OutMatrix(matrix)

    col_sum = np.sum(matrix, axis=1)
    table.add_column("Ψᵢ", list(col_sum / matrix.shape[1]))

    optimal_strategy = f"a{col_sum.argmax(axis=0) + 1}"
    print(f"1) Криетрий недостаточного основания Бернулли.\n"
          f"Если пользоваться критетрием Бернулли, то следует руководствоваться стратегией {optimal_strategy}.\n"
          f"Соответствующее математическое ожидание выигрыша при этом"
          f"максимально и равно {col_sum.max(axis=0) / matrix.shape[1]}.\n{table}")
    return optimal_strategy


def ValdCriteria(matrix: np.array):
    """
    Криетрий пессимизма (Вальда).
    :param matrix: матрица стратегий.
    :return полученная оптимальная стратегия.
    """
    table = OutMatrix(matrix)
    col_min = np.min(matrix, axis=1)
    table.add_column("αᵢ", list(col_min))

    optimal_strategy = f"a{col_min.argmax(axis=0) + 1}"
    print(f"2) Криетрий пессимизма Вальда.\n"
          f"Пессимистическая стратегия (Вальда) определяет выбор {optimal_strategy} "
          f"(нижняя цена игры равна {col_min.max(axis=0)}).\n{table}")
    return optimal_strategy


def OptimismCriteria(matrix: np.array):
    """
    Криетрий авантюры (максимума, оптимизма).
    :param matrix: матрица стратегий.
    :return полученная оптимальная стратегия.
    """
    optimal_strategy = f"a{np.max(matrix, axis=1).argmax() + 1}"
    print(f"3) Критерий авантюры(максимума, оптимизма).\n "
          f"Оптимистическая стратегия соответствует выбору {optimal_strategy} "
          f"c максимальным выигрышем в матрице - {matrix.max()}.")
    return optimal_strategy


def GurvitzCriteria(matrix: np.array):
    """
    Криетрий Гурвица.
    :param matrix: матрица стратегий.
    :return полученная оптимальная стратегия.
    """
    table = OutMatrix(matrix)
    psi = ALPHA * np.min(matrix, axis=1) + (1 - ALPHA) * np.max(matrix, axis=1)
    table.add_column("Ψ", psi)

    optimal_strategy = f"a{psi.argmax() + 1}"
    print(f"4) Криетрий Гурвица.\n"
          f"α = {ALPHA}\n{table}\nНаилучшая стратегия {optimal_strategy}\n"
          f"Ожидаемый выигрыш: {psi.max()}")
    return optimal_strategy


def SevigeCriteria(matrix: np.array):
    """
    Криетрий рисков Севиджа.
    :param matrix: матрица стратегий.
    :return полученная оптимальная стратегия.
    """
    risks = np.max(matrix, axis=0) - matrix
    table = OutMatrix(risks)
    max_col = np.max(risks, axis=1)
    table.add_column("αᵢ", max_col)

    optimal_strategy = f"a{max_col.argmin() + 1}"
    print(f"5) Критерий Севиджа.\n"
          f"Составим таблицу рисков стратегий:\n{table}\n"
          f"Таким образом, оптимальная рисковая стратегия - {optimal_strategy}.")
    return optimal_strategy


def ChooseBestStrategy(matrix: np.array):
    """
    Отбирает наилучшую стратегию согласно принципу большинства на основе всех рассмотренных критериев.
    :param matrix: матрица стратегий.
    """
    result_map = {}
    for k in [f"a{i + 1}" for i in range(matrix.shape[0])]:
        result_map[k] = 0

    result_map[BernulliCriteria(matrix)] += 1
    result_map[ValdCriteria(matrix)] += 1
    result_map[OptimismCriteria(matrix)] += 1
    result_map[GurvitzCriteria(matrix)] += 1
    result_map[SevigeCriteria(matrix)] += 1

    strategies = list(result_map.keys())
    ranks = list(result_map.values())

    table = PrettyTable()
    table.field_names = strategies
    table.add_row(ranks)
    print(f"\nВ итоге выберем стратегию, которая оказалась оптимальной в большем числе критериев:\n{table}\n"
          f"По принципу большинства рекомендуем стратегию {strategies[max(enumerate(ranks), key=lambda x: x[1])[0]]}")
