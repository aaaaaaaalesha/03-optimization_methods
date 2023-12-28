# Copyright 2020 Alexey Alexandrov <sks2311211@yandex.ru>
import os
import random

import imageio
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.pyplot as plt

import numpy as np
from prettytable import PrettyTable

ITERATIONS_COUNT = 101
SPECIES_COUNT = 4
MUTATE_CHANCE = 0.25
MUTATE_CONSTANT = 5
ROUND_CONSTANT = 6


class GeneticAlgorithm:
    def __init__(self, fit_func, bounds):
        """Инициализация задачи по фит функции и границам диапазона."""
        self.fit_func_ = fit_func
        self.iterations_count_ = ITERATIONS_COUNT
        self.bounds_ = bounds
        self.species_count_ = SPECIES_COUNT
        self.population_number_ = 0
        self.population_ = self.InitPopulation()

        self.max_result_ = np.max(self.population_[:, 2])
        self.mean_result_ = np.mean(self.population_[:, 2])

    def PrintPopulation(self):
        """Вывод таблицы."""
        print(f"Поколение: {self.population_number_}")
        table = PrettyTable()
        table.field_names = ["X", "Y", "FIT"]

        for i in range(self.species_count_):
            table.add_row(list(np.round(self.population_[i], ROUND_CONSTANT)))

        self.population_number_ += 1
        print(table)

        print(f"Максимальный результат: {round(self.max_result_, ROUND_CONSTANT)}")
        print(f"Средний результат: {round(self.mean_result_, ROUND_CONSTANT)}")

    def Solve(self):
        """Решение задачи."""
        for i in range(self.iterations_count_):
            self.VisualizePopulation()
            self.CrossoverAndCalcFit()
            self.PrintPopulation()

            # Mutate in 25% of population.
            if random.uniform(0, 1) <= 0.25:
                self.Mutation()

        self.CreateGifVisualization()

    def InitPopulation(self):
        """Начальная популяция."""
        population = np.random.rand(self.species_count_, 3)

        # Столбец X.
        population[:, 0] = self.bounds_[0] + population[:, 0] * (self.bounds_[1] - self.bounds_[0])
        # Столбец Y.
        population[:, 1] = self.bounds_[2] + population[:, 1] * (self.bounds_[3] - self.bounds_[2])
        # Столбец FIT.
        population[:, 2] = self.fit_func_(population[:, 0], population[:, 1])

        return population

    def Mutation(self):
        """Мутация"""
        delta = (np.random.rand(*self.population_.shape) - 0.5) / MUTATE_CONSTANT
        self.population_ = self.population_ + delta
        x_column = self.population_[:][0]
        y_column = self.population_[:][1]

        satisfying_x = np.where(np.logical_and(x_column > self.bounds_[0], x_column < self.bounds_[1]))
        satisfying_y = np.where(np.logical_and(y_column > self.bounds_[2], y_column < self.bounds_[0]))
        if len(satisfying_x) != len(x_column) or len(satisfying_y) != len(y_column):
            self.population_ = self.population_ - delta

    def CrossoverAndCalcFit(self):
        """Кроссовер и вычисление значений фит функции."""
        curr_population = self.population_[self.population_[:, 2].argsort()].copy()

        x3 = curr_population[3][0]
        y3 = curr_population[3][1]
        x2 = curr_population[2][0]
        y2 = curr_population[2][1]
        x1 = curr_population[1][0]
        y1 = curr_population[1][1]

        new_population = np.array(
            [[x3, y1, self.fit_func_(x3, y1)], [x1, y3, self.fit_func_(x1, y3)],
             [x3, y2, self.fit_func_(x3, y2)], [x2, y3, self.fit_func_(x2, y3)]])

        self.max_result_ = np.max(new_population[:, 2])
        self.mean_result_ = np.mean(new_population[:, 2])

        self.population_ = new_population.copy()

    def VisualizePopulation(self):
        if not os.path.exists('results/'):
            os.makedirs('results/')
        x0, x1 = self.bounds_[:2]
        x = np.arange(x0, x1 + 0.1, 0.1)
        y0, y1 = self.bounds_[2:]
        y = np.arange(y0, y1 + 0.1, 0.1)
        x, y = np.meshgrid(x, y)
        z = self.fit_func_(x, y)

        x_p, y_p = np.transpose(self.population_[:, :2])
        plt.title(f"Поколение {self.population_number_}.")
        plt.xlabel('x')
        plt.ylabel('y')
        plt.pcolormesh(x, y, z, cmap=cm.plasma)
        plt.plot(x_p, y_p, marker='D', linestyle='', color='lime')
        plt.savefig('results/' + str(self.population_number_) + '.png')
        # plt.show()
        plt.clf()

    def CreateGifVisualization(self):
        results = []
        filenames = os.listdir("results/")
        for filename in sorted(filenames, key=lambda x: int(os.path.splitext(x)[0])):
            results.append(imageio.imread("results/" + filename))
        imageio.mimsave('final_result.gif', results, duration=0.1)
