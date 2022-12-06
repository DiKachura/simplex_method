import sys
from tkinter import *
import matplotlib.pyplot as plt
import numpy as np
from numpy import exp, sqrt
import random
from timeit import default_timer as timer
clicks = 0
# Дерево решений
class Node:
    def __init__(self):
        self.value = 0
        self.next = 0
        self.root = 0
        self.way = 0
        self.rowNums = 0
        self.colNums = 0
        self.a = 0 # Координата задается рандмно от 0 до 100
        self.m = 0
        self.ib = 0 # номер первого города
        self.n = 0
        self.X = 0
        self.Y = 0
    def setDefault(self, n, a, m, ib, matrix, X, Y):
        self.a = a # Координата задается рандомно от 0 до 100
        self.m = m
        self.ib = ib # номер первого города
        self.n = n
        self.X = X
        self.Y = Y
        self.matrix = matrix
    def setValue(self, node):
        self.root = node
        self.rowNums = node.rowNums
        self.colNums = node.colNums
        self.matrix = node.matrix[:]
    def setNums(self):
        nums = [(x + 1) for x in range(len(self.matrix))]
        self.rowNums = nums[:]
        self.colNums = nums[:]
class BnP:

    @staticmethod
    def findSmallestElementInRow(array, row):
        """
        Поиск наименьшего значение в СТРОКЕ
        :param array:
        :param row:
        :return: минимальный элемент
        """
        minElement = sys.maxsize
        for element in array[row]:
            if minElement > element:
                minElement = element
        return minElement
    @staticmethod
    def findSmallestElementInCol(array, col):
        """
        Поиск наименьшего значение в СТОЛБЦЕ
        :param array:
        :param col:
        :return: минимальный элемент
        """
        minElement = sys.maxsize
        for element in array[:, col]:
            if minElement > element:
                minElement = element
        return minElement
    @staticmethod
    def rowReduction(array):
        """
        Редукция строк
        :param array:
        :return:
        """
        rowCoefficients = []
        for i in range(len(array)):
            minElement = BnP.findSmallestElementInRow(array, i)
            rowCoefficients.append(minElement)
            for j in range(len(array[i])):
                array[i][j] -= minElement
        return array, rowCoefficients
    @staticmethod
    def colReduction(array):
        """
        Редукция столбцов
        :param array:
        :return:
        """
        colCoefficients = []
        for i in range(len(array)):

            minElement = BnP.findSmallestElementInCol(array, i)
            colCoefficients.append(minElement)
            for j in range(len(array[:, i])):
                array[j][i] -= minElement
        return array, colCoefficients
    @staticmethod
    def evaluationCalculation(array):
        """
        Оценка нулевых точек (коэффициент по строке и столбцу)
        :param array:
        :return:
        """
        maxAssessment = -1 # Максимальная оценка
        maxI = -1 # Строка максимального значения
        maxJ = -1 # Столбец максимального значения
        for i in range(len(array)):
            for j in range(len(array[i])):
                if array[i][j] == 0:
                    minRow = sys.maxsize
                    minCol = sys.maxsize
                    for k in range(len(array[i])):
                        if array[i][k] < minRow and k != j:
                            minRow = array[i][k]
                    for k in range(len(array[:, j])):
                        if array[k][j] < minCol and k != i:
                            minCol = array[k][j]
                    sumOfElements = minCol + minRow # Сумма элементов по столбцу и строке
                    if sumOfElements > maxAssessment:
                        maxAssessment = sumOfElements
                        maxI = i
                        maxJ = j
        return maxI, maxJ
    @staticmethod
    def cutMatrix(array, indexI, indexJ):
        """
        Обрезаем матрицу по индексам i и j
        :param array:
        :param indexI:
        :param indexJ:
        :return:
        """
        newArray = []

        for i in range(len(array)):
            if i != indexI:
                newRow = []
                for j in range(len(array)):
                    if j != indexJ:
                        newRow.append(array[i][j])
                    newArray.append(newRow)
        return np.array(newArray)
    @staticmethod
    def startIteration(node):
        """
        Проводим одну итерацию:
        > находим минимум по строкам и столбцам
        > считаем полный путь, учитывая путь предыдущего
        начения
        > находим ноль с максимальной оценкой
        > задаём все пути к нему равным бесконечности
        > удаляем сам путь
        > заносим в массив путей очередной путь
        :param node:
        :return:
        """
        node.matrix, sumOfRowReduction = BnP.rowReduction(node.matrix)
        node.matrix, sumOfColReduction = BnP.colReduction(node.matrix)
        node.value = sum(sumOfColReduction) + sum(sumOfRowReduction)
        if node.root != 0:
            node.value += node.root.value
        indexI, indexJ = BnP.evaluationCalculation(node.matrix)
        node.way = [node.rowNums[indexI], node.colNums[indexJ]]
        node.next = Node()
        node.next.setValue(node)
        node.next.matrix[indexI][indexJ] = float("inf")
        node.next.matrix = BnP.cutMatrix(node.matrix, indexI, indexJ)
        del node.next.rowNums[indexI]
        del node.next.colNums[indexJ]
        if (node.way[0] in node.next.colNums) and (node.way[1] in node.next.rowNums):
            indexI = node.next.rowNums.index(node.way[1])
            indexJ = node.next.colNums.index(node.way[0])
            node.next.matrix[indexI][indexJ] = float("inf")

        if len(node.matrix) > 1:
            BnP.startIteration(node.next)
    @staticmethod
    def getResult(node):
        """
        Выдать результат всего пути
        :param node:
        :return:
        """
        if node.next != 0 and len(node.next.matrix) != 1:
            return BnP.getResult(node.next)
        else:
            return node.value
    @staticmethod
    def getWayNode(node, way):
        """
        Сбор всего пути
        :param node:
        :param way:
        :return:
        """
        way.append(node.way)
        if node.next != 0 and len(node.next.matrix) != 0:
            way = BnP.getWayNode(node.next, way)
        return way
    @staticmethod
    def setResultWay(array):
        resultWay = array[0]
        for i in range(len(array[1])):
            for j in range(2):
                set = array[1][i][j]
                check = True
                for k in range(len(resultWay)):
                    if resultWay[k] == set:
                        check = False
                if check:
                    resultWay.insert(len(resultWay) - 1, set)
        return resultWay
    @staticmethod
    def getResultWay(node):
        """
        Получить вектор опитмального пути
        :param node:
        :return:
        """

        allWays = BnP.getWayNode(node, [])
        optimalWay = []
        for z in range(len(allWays)):
            fullWay = allWays[:]
            optimalWay = [fullWay[z][0], fullWay[z][1]]
            del fullWay[z]
            isCorrect = True
            while len(fullWay) != 0 and isCorrect:
                flag = False
                for l in range(len(fullWay)):
                    bunch = fullWay[l]
                    if bunch[0] == optimalWay[(len(optimalWay) - 1)]:
                        optimalWay.append(bunch[1])
                        del fullWay[l]
                        flag = True
                        break
                if not flag:
                    isCorrect = False
        return [optimalWay, allWays]
class NearestNeighbor:
    @staticmethod
    def hex_code_colors():
        a = hex(random.randrange(0, 256))
        b = hex(random.randrange(0, 256))
        c = hex(random.randrange(0, 256))
        a = a[2:]
        b = b[2:]
        c = c[2:]
        if len(a) < 2:
            a = "0" + a
        if len(b) < 2:
            b = "0" + b
        if len(c) < 2:
            c = "0" + c
        z = a + b + c
        return "#" + z.upper()
    @staticmethod
    def methodImplementation(self, index):
        w = []
        M = self.matrix.copy()
        w.append(index)
        for i in np.arange(1, self.n, 1):

            s = []
            for j in np.arange(0, self.n, 1):
                s.append(M[w[i-1], j])
            w.append(s.index(min(s)))
            for j in np.arange(0, i, 1):
                M[w[i], w[j]] = float('inf')
                M[w[i], w[j]] = float('inf')
        S = sum([sqrt((self.X[w[i]]-self.X[w[i+1]])**2+(self.Y[w[i]]-self.Y[w[i+1]])**2) for i in np.arange(0, self.n - 1, 1)]) + sqrt((self.X[w[self.n-1]]-self.X[w[0]])**2+(self.Y[w[self.n-1]]-self.Y[w[0]])**2)
        return S, w
    def __init__(self, n, a, m, ib, matrix, X, Y):
        self.a = a # Координата задается рандмно от 0 до 100
        self.m = m
        self.ib = ib # номер первого города
        self.n = n
        self.X = X
        self.Y = Y
        self.matrix = matrix
    @staticmethod
    def runMethod(self):
        way = [] # путь прохода
        Matrix = np.zeros([self.n, self.n]) # Шаблон матрицы относительных расстояний между пунктами
        for i in np.arange(0, self.n, 1):
            for j in np.arange(0, self.n, 1):
                if i != j:
                    Matrix[i, j] = sqrt((self.X[i]-self.X[j])**2+(self.Y[i]-self.Y[j])**2) # Заполнение матрицы
                else:
                    Matrix[i, j] = float('inf') # Заполнение главной диагонали матрицы
        Summs = []
        Ways = []
        for i in range(self.n):
            start_time = timer()
            tempS, tempWay = NearestNeighbor.methodImplementation(self, i)
            end_time = timer() - start_time
            Summs.append(tempS)
            Ways.append(tempWay)
        indexWay = Summs.index(min(Summs))
        way = Ways[indexWay]

        time.config(text="Время работы алгоритма %f" % float(end_time))
        summs.config(text="Сумма маршрута %s" % round(Summs[indexWay], 2))
        plt.title("Метод ближайшего соседа", size=15)
        X1 = [self.X[way[i]] for i in np.arange(0, self.n, 1)]
        Y1 = [self.Y[way[i]] for i in np.arange(0, self.n, 1)]
        # Рандомные цвета
        t = []
        for i in range(self.n):
            plt.plot(self.X[i], self.Y[i], color=NearestNeighbor.hex_code_colors(), linestyle=' ', marker='o')
            t.append(i + 1)
            plt.text(self.X[i] * (1 + 0.01), self.Y[i] * (1 + 0.01), i + 1)
            # Обозначение каждой точки
        if len(t) < 16:
            plt.legend(t, loc='center left', bbox_to_anchor=(1.,0.5), ncol=1)
        plt.plot(X1, Y1, color='b', linewidth=1)
        X2 = [self.X[way[self.n-1]], self.X[way[0]]]
        Y2 = [self.Y[way[self.n-1]], self.Y[way[0]]]
        plt.xlabel("Количество вершин - N")
        plt.ylabel("Время работы")
        plt.plot(X2, Y2, color='g', linewidth=2, linestyle='-')
        plt.grid(True)
        plt.show()
class MyMatrix:
    def __init__(self):
        self.matrix = 0
        self.beforeN = 0
        self.X = 0
        self.Y = 0
    def generateMatrix(self, n):
        a = 0 # Координата задается рандмно от 0 до 100
        m = 100
        ib = 0 # номер первого города
        way = [] # путь прохода
        self.X = np.random.uniform(a, m, n)
        self.Y = np.random.uniform(a, m, n)

        Matrix = np.zeros([n, n]) # Шаблон матрицы относительных расстояний между пунктами
        for i in np.arange(0, n, 1):
            for j in np.arange(0, n, 1):
                if i != j:
                    Matrix[i, j] = sqrt((self.X[i] - self.X[j]) ** 2 + (self.Y[i] - self.Y[j]) ** 2) # Заполнение матрицы
                else:
                    Matrix[i, j] = float('inf') # Заполнение главной диагонали матрицы
        self.matrix = Matrix
        self.beforeN = n

M = MyMatrix()

def runNearestNeighbor():
    n = town.get()
    if not n.isdigit():
        cheak['text'] = " " * 80
        cheak['text'] = "Введите число"
    elif int(n) <= 1:
        cheak['text'] = " " * 80
        cheak['text'] = "Количество точек должно быть больше 1"
    else:
        cheak['text'] = " " * 80
        n = int(n)
        if M.beforeN == 0 or M.beforeN != n:
            M.generateMatrix(n)
        methodNeighbor = NearestNeighbor(n, 0, 100, 0, M.matrix, M.X, M.Y)
        NearestNeighbor.runMethod(methodNeighbor)
def runBnP():
    n = town.get()
    if not n.isdigit():
        cheak['text'] = " " * 80
        cheak['text'] = "Введите число"
    elif int(n) <= 1:
        cheak['text'] = " " * 80
        cheak['text'] = "Количество точек должно быть больше 1"
    else:
        cheak['text'] = " " * 80
        n = int(n)
        if M.beforeN == 0 or M.beforeN != n:
            M.generateMatrix(n)
        rootNode = Node()
        rootNode.setDefault(n, 0, 100, 0, M.matrix, M.X, M.Y)

        rootNode.setNums()
        matrixSize = len(rootNode.matrix)
        start_time = timer()
        BnP.startIteration(rootNode)
        time.config(text="Время работы алгоритма %s" % round((timer() - start_time), 5))
        way = BnP.getResultWay(rootNode)[0]
        if len(way) != matrixSize + 1:
            way = BnP.setResultWay(BnP.getResultWay(rootNode))
        del way[-1]
        way = [x - 1 for x in way]
        summs.config(text="Сумма маршрута %s" % round(BnP.getResult(rootNode), 2))
        plt.title("Метод ветвей и границ", size=15)
        X1 = [rootNode.X[way[i]] for i in np.arange(0, n, 1)]
        Y1 = [rootNode.Y[way[i]] for i in np.arange(0, n, 1)]
        # Рандомные цвета
        t = []
        for i in range(n):
            plt.plot(rootNode.X[i], rootNode.Y[i], color=NearestNeighbor.hex_code_colors(), linestyle=' ', marker='o')
            t.append(i + 1)
            plt.text(rootNode.X[i] * (1 + 0.01), rootNode.Y[i] * (1 + 0.01), i + 1)
            # Обозначение каждой точки
        if len(t) < 16:
            plt.legend(t, loc='center left', bbox_to_anchor=(1., 0.5), ncol=1)
        plt.plot(X1, Y1, color='b', linewidth=1)
        X2 = [rootNode.X[way[rootNode.n - 1]], rootNode.X[way[0]]]
        Y2 = [rootNode.Y[way[rootNode.n - 1]], rootNode.Y[way[0]]]
        plt.plot(X2, Y2, color='g', linewidth=2, linestyle='-')
        plt.xlabel("Количество вершин - N")
        plt.ylabel("Время работы")
        plt.grid(True)
        plt.show()
def creatMatrix():
    n = town.get()
    if not n.isdigit():
        cheak['text'] = " " * 80
        cheak['text'] = "Введите число"

    elif int(n) <= 1:
        cheak['text'] = " " * 80
        cheak['text'] = "Количество точек должно быть больше 1"
    else:
        cheak['text'] = " " * 80
        n = int(n)
        M.generateMatrix(n)
# цвет
bg = '#e7f2e1'
# создание окна интерфейса
root = Tk()
root.configure(background=bg)
root.title("Решение задачи коммивояжера")
root.geometry("720x200")
town = StringVar()
town_label = Label(text="Введите количество городов:",
fg="#000000", bg=bg, font='Times 14')
town_label.grid(row=0, column=0, sticky="w")
town_entry = Entry(textvariable=town)
town_entry.grid(row=0, column=1, padx=5, pady=5)
# кнопка для решения задачи коммивояжера методом ближайшего соседа
message_button = Button(text="Nearest neighbor ", fg="#000000", bg="#c6f2ae", font='Times 14', command=runNearestNeighbor)
message_button.grid(row=0, column=2, padx=5, pady=5, sticky="w")
# кнопка для решения задачи коммивояжера методом ветвей и границ
message_button2 = Button(text="BnP", fg="#000000", bg="#c6f2ae", font='Times 14', command=runBnP)
message_button2.grid(row=0, column=3, padx=5, pady=5, sticky="w")
# кнопка для создания матрциы
matrix_button = Button(text="Create matrix", fg="#000000", bg="#c6f2ae", font='Times 14', command=creatMatrix)
matrix_button.grid(row=0, column=4, padx=5, pady=5, sticky="w")
# Время работы алгоритма
time = Label(padx=0, pady=0, font='Times 14', bg="#e7f2e1")
time.grid(row=1, column=0, sticky=W, columnspan=3)
# Сумма построенного маршрута
summs = Label(padx=0, pady=0, font='Times 14', bg=bg)
summs.grid(row=2, column=0, sticky=W, columnspan=3)
# Проверка на некорректный ввод
stub1 = Label(font='Times 14', bg=bg, fg='red')
stub1.grid(row=3, column=0, sticky="w", columnspan=3)
# Проверка на некорректный ввод
stub2 = Label(font='Times 14', bg=bg, fg='red')
stub2.grid(row=4, column=0, sticky="w", columnspan=3)
# Проверка на некорректный ввод
cheak = Label(font='Times 14', bg=bg, fg='red')
cheak.grid(row=5, column=0, sticky="w", columnspan=3)
root.mainloop()
