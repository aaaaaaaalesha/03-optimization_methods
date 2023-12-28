# Copyright 2020 Alexey Alexandrov <sks2311211@yandex.ru>

"""
Лабораторная работа № 5
Решение многокритериальной оптимизации.
Цель работы: Изучить постановку задачи МКО;
овдладеть навыками решения задач МКО с помощью различных методов,
осуществить сравнительный анализ резульатов, полученных при помощи разных методов.

Вариант 1.
"""
from multicriteria import *

if __name__ == '__main__':
    print("Ход работы:")
    multcrit = Multicriteria("input_data.json")
    print(multcrit.OutWeight())

    multcrit.MainCriteriaMethod()
    multcrit.ParetoMethod()
    multcrit.WeighAndCombineMethod()
    multcrit.HierarchiesAnalysisMethod()
