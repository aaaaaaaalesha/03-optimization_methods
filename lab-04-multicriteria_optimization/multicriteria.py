# Copyright 2020 Alexey Alexandrov <sks2311211@yandex.ru>

import json

import numpy as np
from prettytable import PrettyTable
import matplotlib.pyplot as plt
from scipy.spatial import distance

CRITERIA_COUNT = 4
MAIN_CRITERIA_INDEX = 2  # Главный критерий -- расход бензина.
MAIN_CRITERIA_LIMITS = [0.2, 0.3, 0, 0.5]  # Минимально допустимые доли критериев.

MAX_WEIGHT = 10

SECOND_CRITERIA_INDEX = 1
THIRD_CRITERIA_INDEX = 2

COST_FILL_LIST = [0.25, 0.2, 0.125, 0.8, 0.5, 0.625]
EXPENSES_FILL_LIST = [5, 2, 1, 0.125, 0.2, 2]
CONSUMPTION_FILL_LIST = [0.143, 0.333, 0.125, 2, 0.5, 0.333]
COMFORT_FILL_LIST = [0.286, 0.666, 0.25, 2.333, 0.875, 0.375]
CRITERIA_FILL_LIST = [0.666, 0.5, 2, 0.666, 4, 4]


class Multicriteria:
    """Базовый класс задачи многокритериальной оптимизации"""

    def __init__(self, path_to_file):
        # Парсим JSON-файл с входными данными
        with open(path_to_file, "r") as read_file:
            json_data = json.load(read_file)
            # Задача.
            self.task_name_ = json_data["task_name"]
            # Альтернативы.
            self.alternative_names_ = list(json_data["alternative_names"])
            # Критерии.
            self.criteria_names_ = list(json_data["criteria_names"])

            # Вектор весов критериев.
            self.criteria_weight_ = np.array(json_data["criterias_weight"])
            # Нормализованный вектор весов критериев.
            self.normalized_weight_ = self.NormingVector(self.criteria_weight_)
            # Матрица А оценок для альтернатив.
            self.json_matrix_ = np.array(json_data["alternative_matrix"], dtype=np.float64)
            self.alternative_matrix_ = self.AlternativeMatrix(np.array(json_data["alternative_matrix"],
                                                                       dtype=np.float64),
                                                              list(json_data["criterias_direction"]))
            # Нормализованная матрица.
            self.normalized_matrix_ = self.NormalizeMatrix(self.alternative_matrix_)

    def NormingVector(self, vector):
        """Нормирует вектор."""
        normalize_weight = vector.copy()
        weight_sum = np.sum(normalize_weight)

        normalize_weight = normalize_weight / weight_sum

        return normalize_weight

    def AlternativeMatrix(self, alternative_matrix, criterias_direction):
        """Своидит все криитерии к максимизации."""
        for j in range(len(criterias_direction)):
            if criterias_direction[j] == "min":
                for i in range(alternative_matrix.shape[0]):
                    alternative_matrix[i][j] = MAX_WEIGHT - alternative_matrix[i][j] + 1

        return alternative_matrix

    def NormalizeMatrix(self, matrix):
        """Нормализует матрицу."""
        normalized_matrix = matrix.copy()
        minimums = normalized_matrix.min(axis=0)
        maximums = normalized_matrix.max(axis=0)

        for j in range(normalized_matrix.shape[1]):
            if j != MAIN_CRITERIA_INDEX:
                for i in range(matrix.shape[0]):
                    normalized_matrix[i][j] = (normalized_matrix[i][j] - minimums[j]) / (
                            maximums[j] - minimums[j])

        return normalized_matrix

    def OutMatrix(self, matrix):
        """Выводит матрицу альтернатив."""
        table = PrettyTable()

        table.field_names = ["Альтернативы"] + self.criteria_names_

        for i in range(len(self.alternative_names_)):
            new_row = [self.alternative_names_[i]]
            for j in range(len(self.criteria_names_)):
                new_row.append(round(matrix[i][j], 2))

            table.add_row(new_row)

        return table

    def OutWeight(self):
        """Выводит вектор весов критериев."""
        out = "Составляем веткор весов критериев, используя шкалу 1-10:\n"
        table = PrettyTable()
        table.field_names = self.criteria_names_
        table.add_row(self.criteria_weight_)

        out += table.__str__()

        out += "\nНормализовав, получим вектор " + self.normalized_weight_.__str__()

        return out

    def MainCriteriaMethod(self):
        """Решение методом главного критерия."""
        print("\n1) Метод замены критериев ограничениями (метод главного критерия).\n"
              "Составим матрицу оценок альтернатив.")
        print(self.OutMatrix(self.json_matrix_))

        matrix = self.normalized_matrix_.copy()
        maximums = matrix.max(axis=0)

        print("Ограничения:")
        for j in range(len(self.criteria_names_)):
            if j != MAIN_CRITERIA_INDEX:
                print(f"{self.criteria_names_[j]} не менее {MAIN_CRITERIA_LIMITS[j] * maximums[j]}")

        print(f"\nПроведём нормирование матрицы:\n{self.OutMatrix(self.NormalizeMatrix(self.json_matrix_))}")

        constraints = []
        for j in range(len(self.criteria_names_)):
            if j == MAIN_CRITERIA_INDEX:
                constraints.append(None)
            else:
                constraints.append(MAIN_CRITERIA_LIMITS[j] * maximums[j])

        acceptable_rows = []

        for i in range(len(self.alternative_names_)):
            row = matrix[i]
            if (row < MAIN_CRITERIA_LIMITS).any():
                continue

            acceptable_rows.append(i)

        if len(acceptable_rows):
            print("При заданных ограничениях приемлимыми являются следующие решения:")
            for i in acceptable_rows:
                print(self.alternative_names_[i])

            max_alternative = None
            for i in acceptable_rows:
                curr = self.normalized_matrix_[i][MAIN_CRITERIA_INDEX]
                if max_alternative is None or self.normalized_matrix_[max_alternative][MAIN_CRITERIA_INDEX] < curr:
                    max_alternative = i

            print("Итоговое решение:")
            print(self.alternative_names_[max_alternative])

        else:
            print("При заданных ограничениях не нашлось приемлимых решений.")

    def ParetoMethod(self):
        """Решение формированием и сужением множества Парето."""
        print(f"\n 2) Формирование и сужение множества Парето. \n"
              f"Выберем в качестве критериев для данного метода {self.criteria_names_[SECOND_CRITERIA_INDEX]} и "
              f"{self.criteria_names_[THIRD_CRITERIA_INDEX]}.\n"
              f"{self.criteria_names_[SECOND_CRITERIA_INDEX]} - по оси X, "
              f"{self.criteria_names_[THIRD_CRITERIA_INDEX]} - по оси Y.\n"
              f"Сформируем множество Парето графическим методом. (см. график)")
        plt.title("Графическое решение методом сужения множества Парето.")
        plt.xlabel(f"Критерий: {self.criteria_names_[SECOND_CRITERIA_INDEX]}")
        plt.ylabel(f"Критерий: {self.criteria_names_[THIRD_CRITERIA_INDEX]}")

        xValues = self.json_matrix_[:, SECOND_CRITERIA_INDEX]
        yValues = self.json_matrix_[:, THIRD_CRITERIA_INDEX]
        plt.grid()
        plt.plot(xValues, yValues, "b")

        euclid_length = []
        for i in range(len(self.json_matrix_[:, SECOND_CRITERIA_INDEX])):
            x_i = self.json_matrix_[i, SECOND_CRITERIA_INDEX]
            y_i = self.json_matrix_[i, THIRD_CRITERIA_INDEX]
            plt.plot(x_i, y_i, "bo")
            plt.text(x_i + 0.1, y_i, self.alternative_names_[i][0])

            euclid_distance = distance.euclidean((x_i, y_i), (xValues.min(), yValues.min()))

            euclid_length.append(euclid_distance)

        plt.plot(xValues.min(), yValues.min(), "rD")
        plt.text(xValues.min() + 0.1, yValues.min() + 0.1, "Точка утопии")

        plt.show()
        plt.savefig("pareto.png")

        min_index = min(enumerate(euclid_length), key=lambda x: x[1])[0]

        print(f"Исходя из графика можно сказать, что Евклидово расстояние до "
              f"точки минимально для варианта:\n{self.alternative_names_[min_index]}")

    def NormalizeByColumns(self, current_matrix):
        """Нормализует колонки в матрице."""
        matrix = current_matrix.copy()
        for i in range(len(self.criteria_names_)):
            col_sum = np.sum(matrix[i])
            matrix[i] = matrix[i] / col_sum

        return matrix

    def CriteriaEvaluation(self, y12, y13, y14, y23, y24, y34):
        table = PrettyTable()
        table.field_names = [""] + self.criteria_names_
        table.add_row([self.criteria_names_[0]] + [0, y12, y13, y14])
        table.add_row([self.criteria_names_[1]] + [1 - y12, 0, y23, y24])
        table.add_row([self.criteria_names_[2]] + [1 - y13, 1 - y23, 0, y34])
        table.add_row([self.criteria_names_[3]] + [1 - y14, 1 - y24, 1 - y34, 0])

        return table

    def WeighAndCombineMethod(self):
        """Решение методом взвешивания и объединения критериев."""
        rating_matrix = self.NormalizeByColumns(self.alternative_matrix_)
        rm = self.NormalizeByColumns(self.json_matrix_)

        print("\n 3) Взвешивание и объединение критериев. \n"
              f"Составим матрицу рейтингов альтернатив по критериям, используя шкалу 1-10: \n\n "
              f"{self.OutMatrix(self.json_matrix_)} \n\n Нормализуем её: \n"
              f"{self.OutMatrix(rm)}\n")

        print("Составим экспертную оценку критериев (по методу попарного сравнения):\n")
        y12 = 0.5
        y13 = 0
        y14 = 1
        y23 = 0
        y24 = 1
        y34 = 1
        print(self.CriteriaEvaluation(y12, y13, y14, y23, y24, y34))

        weight_vector = np.array([y12 + y13 + y14, y12 + y14, y14 + y24 + y34, 0])

        weight_vector = self.NormingVector(weight_vector)

        print(f"alpha = {weight_vector}")

        weight_vector.transpose()

        combine_criteria = rating_matrix.dot(weight_vector)

        print(f"Умножив нормализированную матрицу на нормализированный вектор весов критериев, "
              f"получаем значения объединённого критерия альтернатив:\n{combine_criteria}")

        max_index = None
        for i in range(len(combine_criteria) - 1, 0, -1):
            if max_index is None or combine_criteria[i] > combine_criteria[max_index]:
                max_index = i

        print(f"Наиболее приемлемой является альтернатива:\n{self.alternative_names_[max_index]}")

    def PairCompareMatrix(self, fill_list):
        """Заполянет матрицу попарных сравнений."""
        k = 0
        pc_matrix = np.ones((CRITERIA_COUNT, CRITERIA_COUNT))
        # Заполняем верхний треугольник.
        for i in range(CRITERIA_COUNT):
            for j in range(CRITERIA_COUNT):
                if i < j:
                    pc_matrix[i][j] = round(fill_list[k], 3)
                    k += 1

        k = 0
        # Заполняем нижний треугольник.
        for i in range(CRITERIA_COUNT):
            for j in range(CRITERIA_COUNT):
                if i < j:
                    pc_matrix[j][i] = round(1 / fill_list[k], 3)
                    k += 1

        return pc_matrix

    def PairCompareTable(self, names, main_matrix, sum_col, normalize_sum_col):
        """Составляет таблицу с матрицей попарных сравнений"""
        table = PrettyTable()
        table.field_names = [""] + names + ["Сумма по строке", "Нормированная сумма по строке"]
        for i in range(len(self.alternative_names_)):
            row = [names[i]] + list(main_matrix[i])
            row.append(round(sum_col[i], 2))
            row.append(round(normalize_sum_col[i], 2))
            table.add_row(row)

        return table

    def ConsensusDivision(self, main_matrix, normalize_sum_col):
        """Находит отношение согласованности."""
        columns_sum = np.sum(main_matrix, axis=0)
        mult_col = columns_sum * normalize_sum_col
        return (np.sum(mult_col) - CRITERIA_COUNT) / (CRITERIA_COUNT - 1)

    def HierarchiesAnalysisMethod(self):
        """Решение методом анализа иерархий."""
        print("\n4) Меотд анализа иерархий.\nСоставим для каждого из критериев матрицу попарного сравнения альтернатив,"
              " нормализуем ее и матрицу из векторов приоритетов альтернатив:\n")

        fill_lists = [COST_FILL_LIST, EXPENSES_FILL_LIST, CONSUMPTION_FILL_LIST, COMFORT_FILL_LIST]

        hierarchies_matrix = None

        for i in range(len(self.criteria_names_)):
            print(f"• {self.criteria_names_[i]}")
            main_matrix = self.PairCompareMatrix(fill_lists[i])

            sum_col = np.sum(main_matrix, axis=1)

            normalize_sum_col = self.NormingVector(sum_col)
            print(self.PairCompareTable(self.alternative_names_, main_matrix, sum_col, normalize_sum_col))

            print(f"Отношение согласованности: {round(self.ConsensusDivision(main_matrix, normalize_sum_col), 3)}\n")

            if hierarchies_matrix is None:
                hierarchies_matrix = normalize_sum_col.transpose()
            else:
                hierarchies_matrix = np.c_[hierarchies_matrix, normalize_sum_col.transpose()]

        print("Оценка приоритетов:")
        criteria_matrix = self.PairCompareMatrix(CRITERIA_FILL_LIST)
        sum_col = np.sum(criteria_matrix, axis=1)

        normalize_sum_col = self.NormingVector(sum_col)
        print(self.PairCompareTable(self.criteria_names_, criteria_matrix, sum_col, normalize_sum_col))

        print(f"Отношение согласованности: {round(self.ConsensusDivision(criteria_matrix, normalize_sum_col), 3)}\n")

        normalize_sum_col.transpose()

        resulted_vec = hierarchies_matrix.dot(normalize_sum_col.transpose())

        print(resulted_vec)

        print("Умножив матрицу, состваленную из норм. сумм по строкам на вектор-столбец оценки приоритетов, "
              "получим вектор:")

        max_index = np.argmax(resulted_vec)

        print(f"Наиболее приемлемой является альтернатива:\n{self.alternative_names_[max_index]}")
