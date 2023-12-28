# Copyright 2020 Alexey Alexandrov <sks2311211@yandex.ru>

from simplex import *


class DualProblem(Simplex):
    """
    Класс унаследован от Simplex и нужен для переформулирования задачи из ПЗ в ДЗ.
    """

    def __init__(self, path_to_file):
        """
        Переопределённый метод __init__. Регистрирует входные данные из JSON-файла.
        Определяем условие двойственной задачи.
        :param path_to_file: путь до JSON-файла с входными данными.
        """

        # Парсим JSON-файл с входными данными
        with open(path_to_file, "r") as read_file:
            json_data = json.load(read_file)

            # Коэффициенты при ЦФ в ДЗ равны свободным членам ограничений в ПЗ.
            self.obj_func_coffs_ = np.array(json_data["constraint_system_rhs"])

            # Свободные члены ограничений в ДЗ равны коэффициентам при ЦФ в ПЗ.
            self.constraint_system_lhs_ = np.array(
                json_data["constraint_system_lhs"]).transpose()

            # Коэффициенты  любого ограничения ДЗ равны коэффициентам при одной переменной из всех ограничений ПЗ.
            self.constraint_system_rhs_ = np.array(json_data["obj_func_coffs"])

            # Минимизация ЦФ в ПЗ соответвстует максимизации ЦФ в ДЗ.
            self.func_direction_ = "max" if json_data[
                                                "func_direction"] == "min" else "min"

            print(self.__str__())

            # Ограничения вида (<=) ПЗ переходят в ограничения вида (>=) ДЗ.
            self.constraint_system_lhs_ *= -1
            self.constraint_system_rhs_ *= -1

            if len(self.constraint_system_rhs_) != self.constraint_system_rhs_.shape[0]:
                raise Exception("Ошибка при вводе данных. Число строк в матрице и столбце ограничений не совпадает.")

            if len(self.constraint_system_rhs_) > len(self.obj_func_coffs_):
                raise Exception("СЛАУ несовместна! Число уравнений больше числа переменных.")

            # Если задача на max, то меняем знаки ЦФ и направление задачи (в конце возьмем решение со знаком минус и
            # получим искомое).
            if self.func_direction_ == "max":
                self.obj_func_coffs_ *= -1

            # Инициализация симплекс-таблицы.
            self.simplex_table_ = SimplexTable(self.obj_func_coffs_, self.constraint_system_lhs_,
                                               self.constraint_system_rhs_)

    def __str__(self):
        """
        Переопренный метод __str__ для условия двойственной задачи.
        :return: Строка с выводом условия двойственной задачи.
        """

        output = """------------------------------------------------------------"""

        output += f"\nF = cx -> {self.func_direction_},"
        output += "\nAx >= 1,\nx1,x2, ..., xn >= 0"

        if self.func_direction_ == "max":
            output += f"\nC = {-self.obj_func_coffs_},"
        else:
            output += f"\nC = {self.obj_func_coffs_},"

        output += f"\nA =\n{self.constraint_system_lhs_},"
        output += f"\nb^T = {self.constraint_system_rhs_}."
        output += "\n------------------------------------------------------------"

        return output
