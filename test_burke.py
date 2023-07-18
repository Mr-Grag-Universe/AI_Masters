import numpy as np
# import scipy as sp
import sys
from fractions import Fraction
from math import gcd
from random import shuffle, seed, choice
from collections import defaultdict, deque
import heapq
import matplotlib.pyplot as plt
import time
import copy
import bisect
import itertools

def benchmark(func):
    import time

    def wrapper(*args, **kwargs):
        start = time.time()
        x = func(*args, **kwargs)
        end = time.time()
        delta_t = (end - start)
        m = int(delta_t)//60
        s = int(delta_t)%60
        ms = int(1000*(delta_t-m*60-s))
        print(f"[*] Время выполнения: {m}:{s}:{ms}")
        return x
    return wrapper


class Rect():
    __slots__ = ("r", "c1", "c2")

    @staticmethod
    def __find_fraction(x : float) -> Fraction:
        a = 10 ** len(str(x).split('.')[1])
        b = x*a
        g = gcd(a, b)
        return Fraction(b//g, a//g)
        

    def __init__(self, r_):
        self.r = max(r_, 1/r_)
        self.c1 = {'x': 0, 'y': 0}
        f = Fraction.from_float(self.r).limit_denominator(1000)
        self.c2 = {'x' : f.denominator, 'y' : f.numerator}
        # print(f"Created: {self.c1} - {self.c2}, r: {self.r}, id: {self.r_id}")

    def move(self, point):
        self.c1['x'] += point[0]
        self.c2['x'] += point[0]
        self.c1['y'] += point[1]
        self.c2['y'] += point[1]
        # print(f"moved to c1: {self.c1}, c2: {self.c2}")

    def __eq__(self, other):
        return self.r == other.r
    
    def __ne__(self, other):
        return self.r != other.r
    
    def __lt__(self, other):
        return self.r < other.r

    def __gt__(self, other):
        return self.r > other.r

    def __str__(self):
        return f"<Rect object: (r: {self.r}, c1: {self.c1}, c2: {self.c2})>"

    def size(self):
        return (self.c2['y'] - self.c1['y'], self.c2['x'] - self.c1['x']) # (h, w)

    def get_coord_list(self) -> list:
        return [self.c1['x'], self.c1['y'], self.c2['x']-1, self.c2['y']-1]

    def setSize(self, h_w_size):
        self.c2 = {'x' : (self.c1['x'] + h_w_size[1]), 'y' : (self.c1['y'] + h_w_size[0])}

    def turn(self):
        self.c2['x'], self.c2['y'] = self.c1['x'] - self.c1['y'] + self.c2['y'], self.c1['y'] - self.c1['x'] + self.c2['x']
        self.r = 1/self.r

    def getSize(self):
        return (self.c2['y'] - self.c1['y'], self.c2['x'] - self.c1['x'])

    def height(self):
        return self.c2['y']-self.c1['y']
    def width(self):
        return self.c2['x']-self.c1['x']

    def S(self):
        return (self.c2['x']-self.c1['x']) * (self.c2['y']-self.c1['y'])

    def intersects(self, rect):
        xx1, yy1, xx2, yy2 = self.get_coord_list()
        x1, y1, x2, y2 = rect.get_coord_list()
        dx = np.minimum(x2, xx2) - np.maximum(x1, xx1) + 1
        dy = np.minimum(y2, yy2) - np.maximum(y1, yy1) + 1
        
        intersect = (dx > 0) & (dy > 0)
        return not (~intersect)

    def moveToOrigin(self):
        h, w = self.getSize()
        self.c1 = {'x': 0, 'y': 0}
        self.c2 = {'x': w, 'y': h}

'''
    т.к. мы можем вращать прямоугольники - предлагаю закомбить стратегию
    будем хранить не 1, а 2 отсортированных списка
    в первом по самой длинной стороне, во втором по короткой стороне 

    на каждом шаге выбираем точку с минимальной высотой (если не подходит - берём следующую)

    пытаемся заткнуть её как только можем, заняв как можно больше места в ней
    для этого составляем массив всех длин + всех ширин и ищем по нему бинпоиском

'''


def packBurke(rects_ : list, fieldSize):
    rects = copy.deepcopy(rects_)
    for i in range(len(rects)):
        rects[i].moveToOrigin()

    positions = []
    real_indexes = []

    H, W = fieldSize
    hs, ws = zip(*[rect.getSize() for rect in rects])
    sorted_sides = sorted(list(enumerate(hs)) + list(enumerate(ws)), key=lambda x: x[1])

    if sorted_sides[-1][1] > max(W, H):
        return None, None, True

    heights = [0 for _ in range(W)]
    while sorted_sides:
        # пока так. потом можно будет соптимизировать
        heights_zip = sorted(zip(heights, range(W)))
        height, ind = heights_zip[0]
        h = 0
        # для выбранной высоты пытаемся вставить

        i = ind
        while i < W and heights[i] <= height:
            i += 1
        
        j = ind
        while j > -1 and heights[j] <= height:
            j -= 1

        space = i-j-1

        # мы нашли границы свободного пространства
        ss = list(map(lambda x : x[1], sorted_sides))
        r_ind = bisect.bisect_left(ss, space)
        if r_ind < len(ss) and r_ind>0 and ss[r_ind] > space:
            r_ind = bisect.bisect_left(ss, ss[r_ind-1])

        # включительно
        l_bound, r_bound = (0, 0)
        pos = None
        real_ind = None
        if r_ind == 0 and sorted_sides[0][1] > space:
            # у нас нет достаточно узкого прямоугольника
            l_bound, r_bound = (j+1, i-1)
            
            r_b_height, l_b_height = None, None
            if i >= W:
                r_b_height, l_b_height = heights[j], heights[j]
            elif j < 0:
                l_b_height, r_b_height = heights[i], heights[i]
            else:
                l_b_height, r_b_height = heights[j], heights[i]
                

            new_h = min(l_b_height, r_b_height)
            
            h = new_h-height
            # heights[j+1:i] = [new_h for _ in range(j+1, i)]
            # continue
        else:
            if r_ind == len(sorted_sides):
                r_ind = len(sorted_sides)-1
            
            r_b_height, l_b_height = (0, 0)
            if i < W:
                r_b_height = heights[i]
            if j > -1:
                l_b_height = heights[j]
            
            real_ind, width = sorted_sides[r_ind]

            # манипуляции с массивами прямоугольников
            if rects[real_ind].getSize()[1] != width:
                rects[real_ind].setSize(rects[real_ind].getSize()[::-1])
            sorted_sides.pop(r_ind)
            ss.pop(r_ind)

            # ищем и удаляем другую сторону
            x = bisect.bisect_left(ss, rects[real_ind].getSize()[0])
            while sorted_sides[x][0] != real_ind:
                x += 1
            sorted_sides.pop(x)
            ss.pop(x)

            if r_b_height > l_b_height:
                # индекс для заливки
                l_bound = j+1
                r_bound = j+width
                h = rects[real_ind].getSize()[0]
                pos = (j+1, height)
            else:
                # индекс для заливки
                r_bound = i-1
                l_bound = i-width
                h = rects[real_ind].getSize()[0]
                pos = (i-width, height)

        heights[l_bound: r_bound+1] = [height+h for _ in range(l_bound, r_bound+1)]
        if pos:
            positions.append(pos)
            real_indexes.append(real_ind)
            rects[real_ind].move(pos)

    positions = [positions[ind] for ind in real_indexes]
    return rects, positions, False


def visualize_placements(position, max_rectangles, container, order_res):
    print(order_res)
    def __convert(position):
        placements = []
        for i in range(0, len(position)//4):
            ind = 4*(i+1)
            placements.append([position[ind-4:ind-2], position[ind-2:ind]])
        return placements

    all_placements = [__convert(position)]

    # Создаем график
    fig, ax = plt.subplots()

    # Закрашиваем свободное пространство на листе черным цветом
    ax.add_patch(plt.Rectangle((0, 0), container[0], container[1], facecolor='black'))

    # Определяем список цветов для прямоугольников
    colors = ['red', 'green', 'blue', 'orange', 'purple', 'cyan', 'magenta']

    # Проходимся по всем расстановкам прямоугольников
    for _, placements in enumerate(all_placements):
        # Проходимся по каждому прямоугольнику в расстановке
        for i, placement in enumerate(placements):
            rect_coords = placement[0] + placement[1]  # Координаты прямоугольника
            # print(rect_coords)
            color = colors[i % len(colors)]  # Цвет прямоугольника
            rectangle = plt.Rectangle(rect_coords[:2], rect_coords[2]-rect_coords[0]+1, rect_coords[3]-rect_coords[1]+1, facecolor=color, alpha=0.5)
            rx, ry = rectangle.get_xy()
            cx = rx + rectangle.get_width()/2.0
            cy = ry + rectangle.get_height()/2.0
            # Добавляем прямоугольник на график
            ax.add_patch(rectangle)
            ax.annotate(str(order_res[i]), (cx, cy), color='black', weight='bold', fontsize=10, ha='center', va='center')

    # Устанавливаем пределы графика и оси
    ax.set_xlim(0, container[0])
    ax.set_ylim(0, container[1])

    # Отображаем график
    plt.show()

def convert_to_clasic(pos, rects, fieldSize=(100, 100)):
    H, W = fieldSize
    field = set([(H, 0), (0, W), (H, W)])
    rects = []
    cols = set([0, W])
    rows = set([0, H])

    for rect in rects:
        c1, c2 = rect.c1, rect.c2
        x1, y1, x2, y2 = c1['x'], c1['y'], c2['x'], c2['y']
        cols.add(x1)
        cols.add(x2)
        rows.add(y1)
        rows.add(y2)

    rows = sorted(list(rows))
    cols = sorted(list(cols))

    for rect in rects:
        c1, c2 = rect.c1, rect.c2
        x1, y1, x2, y2 = c1['x'], c1['y'], c2['x'], c2['y']
        i1 = bisect.bisect_left(rows, x1)
        i2 = bisect.bisect_left(rows, x2)
        j1 = bisect.bisect_left(cols, y1)
        j2 = bisect.bisect_left(cols, y2)

        field = field.union(itertools.product(rows[i1:i2], cols[j1:j2]))
    field = field.union(itertools.product(rows, [0 for _ in range(H)]))
    field = field.union(itertools.product(cols, [0 for _ in range(W)]))

    return field, cols, rows


def main():
    rects, positions, err = packBurke([Rect(1), Rect(2), Rect(3), Rect(1.2), Rect(1.1), Rect(1.3), Rect(1.7), Rect(1.5), Rect(2), Rect(1)], (25, 15))
    # print(order)
    for r, pos in zip(rects, positions):
        print(r, pos)

    pp = positions.copy()


    positions = []
    for r in rects:
        positions += r.get_coord_list()
    # O = order.copy()
    # for i, o in enumerate(order):
    #     O[o] = i+1
    visualize_placements(positions, Rect(1.), (15, 50), range(10))

    field, cols, rows = convert_to_clasic(positions, rects, (50, 15))


main()