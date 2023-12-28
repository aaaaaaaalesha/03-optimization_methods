# Copyright 2020 Alexey Alexandrov <sks2311211@yandex.ru>

from simplex_table import SimplexTable

ROUND_CONST = 3


def StrategyA(simplex_table: SimplexTable):
    """
    Решение для смешанной стратегии игрока А.
    :param simplex_table: конечная симплекс таблица.
    :return: консольный вывод.
    """
    d = {"x1": 0, "x2": 0, "x3": 0, "x4": 0, "x5": 0, "W": 0, 'g': 0}
    for i in range(len(simplex_table.left_column_)):
        if simplex_table.left_column_[i] in ["x1", "x2", "x3", "x4", "x5"]:
            d[simplex_table.left_column_[i]] = round(simplex_table.main_table_[i][0], ROUND_CONST)

    d["W"] = round(simplex_table.main_table_[-1][0], ROUND_CONST)
    d["g"] = round(1 / d["W"], ROUND_CONST)

    strategy_a = [round(d['x1'] * d['g'], ROUND_CONST), round(d['x2'] * d['g'], ROUND_CONST),
                  round(d['x3'] * d['g'], ROUND_CONST), round(d['x4'] * d['g'], ROUND_CONST),
                  round(d['x5'] * d['g'], ROUND_CONST)]

    output = f"Решение для смешанной стратегии игрока А:\nu1 = {d['x1']}, u2 = {d['x2']}, " \
             f"u3 = {d['x3']}, u4 = {d['x4']}, u5 = {d['x5']};\n" \
             f"W = {d['W']};\ng = 1/W = {d['g']}.\n" \
             f"Частоты выбора стратегий:\n" \
             f"x1 = u1·g = {d['x1']}·{d['g']} = {strategy_a[0]};\nx2 = u2·g = {d['x2']}·{d['g']} = {strategy_a[1]};\n" \
             f"x3 = u3·g = {d['x3']}·{d['g']} = {strategy_a[2]};\nx4 = u4·g = {d['x4']}·{d['g']} = {strategy_a[3]};\n" \
             f"x5 = u5·g = {d['x5']}·{d['g']} = {strategy_a[4]};\n" \
             f"Таким образом, оптимальная смешанная стратегия игрока A имеет вид\n {strategy_a}.\n\n"

    return output


def StrategyB(simplex_table: SimplexTable):
    """
    Решение для смешанной стратегии игрока B.
    :param simplex_table: конечная симплекс таблица.
    :return: консольный вывод.
    """
    d = {"x1": 0, "x2": 0, "x3": 0, "x4": 0, "Z": 0, 'h': 0}
    for i in range(len(simplex_table.left_column_)):
        if simplex_table.left_column_[i] in ["x1", "x2", "x3", "x4"]:
            d[simplex_table.left_column_[i]] = round(simplex_table.main_table_[i][0], ROUND_CONST)

    d["Z"] = round(simplex_table.main_table_[-1][0], ROUND_CONST)
    d["h"] = round(1 / d["Z"], ROUND_CONST)

    strategy_b = [round(d['x1'] * d['h'], ROUND_CONST), round(d['x2'] * d['h'], ROUND_CONST),
                  round(d['x3'] * d['h'], ROUND_CONST), round(d['x4'] * d['h'], ROUND_CONST)]

    output = f"Решение для смешанной стратегии игрока В:\nv1 = {d['x1']}, " \
             f"v2 = {d['x2']}, v3 = {d['x3']}, v4 = {d['x4']};\n" \
             f"Z = {d['Z']};\nh = 1/Z = {d['h']}.\n" \
             f"Частоты выбора стратегий:\n" \
             f"y1 = v1·h = {d['x1']}·{d['h']} = {strategy_b[0]};\ny2 = v2·h = {d['x2']}·{d['h']} = {strategy_b[1]};\n" \
             f"y3 = v3·h = {d['x3']}·{d['h']} = {strategy_b[2]};\ny4 = v4·h = {d['x4']}·{d['h']} = {strategy_b[3]};\n" \
             f"Таким образом, оптимальная смешанная стратегия игрока B имеет вид\n {strategy_b}."

    return output
