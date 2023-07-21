import numpy as np
import scipy as sp
import sys
from fractions import Fraction
from math import gcd
from random import shuffle, seed, choice
from collections import defaultdict, deque
import matplotlib.pyplot as plt
import time
import copy
import bisect
import itertools
# from tqdm import tqdm

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

def wrapped_func(*args, **kwargs):
    return benchmark(solveCaseFullRand)(*args, **kwargs)

def visualize_placements(position, container):
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
    ind = 0
    for placements in all_placements:
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
            ind += 1
            ax.add_patch(rectangle)
            ax.annotate(str(ind), (cx, cy), color='black', weight='bold', fontsize=10, ha='center', va='center')

    # Устанавливаем пределы графика и оси
    ax.set_xlim(0, container[0])
    ax.set_ylim(0, container[1])

    # Отображаем график
    plt.show()


'''
    новый план действий:
    1) генерируем все возможные целочисленные размеры для r-ок
    2) рекурсивно перебираем их комбинации, ища те, которые подходят
    3) поиск оптимизируем бинарным поиском (мб потом реализую что-то такое. можно посмотреть, как байесовский оптимизатор работает)
        * вместо этого ищем из последних i штук + меняем местами порядок приоритета (рандомно перетасовываем пару раз)
    4) при каждом погружении считаем метрику (свободную/занятую площадь)
    5) можно заранее отсеивать проигрышные варианты слишком большой суммарной площадью
    6) начинаем с максимальных размеров
    7) сортируем по площади фигуры

    Стратегия заполнения простая и описана вот тут : https://www.codeproject.com/Articles/210979/Fast-optimizing-rectangle-packing-algorithm-for-bu
    По сути просто перебираем n-ое колечество кобинаций размеров, пытаясь скомпоновать их этим алгоритмом

    как выяснилось эксперементальным путём:
    * сортировка в порядке убывания площади - плохая идея - NOOOOOO
    * увеличение числа shuffle - перестановок порядков прямоугольников - хорошая идея (2->3 : 2:27->3:50 : 875->798) YESS
    * добавление рандомного выбора размера после нахождения максимального рабочего - хорошая идея? (3:50->3:13 : 798->744) не знаю почему стало работать быстрее)) YESSS
    * 

    можно попробовать вариации алгоритма на тему не сляпывать фигуры, а наоборот разносить
    или ставить так, чтобы оставлять как можно больше свободного места

    долбаное проклятие размерности!!!
    пришлось переделывать. сейчас у меня всё решает великий рандом - отсеиваем минимум признаков

    оказалось, что полный рандом - довольно эффективная стратегия

    подводя итоги - по выдимому я выбрал неверную стратегию, разделяя подбор комбинации размеров и их размещение.
    путём подгона и теста разных алгоритмов вставки/упаковки за разумное время (сильно ограниченное подбором/перебором размеров)
    получить меньше 600 очков я не смог
    по видимому надо было думать над каким нибудь генетическим алгоритмом или ещё что-то комбинированное, 
    но я уже не успею это придумать и реализовать

    посмотрел на википеции и по репозиториям гитхаба решения (не знаю, почему сразу так не сделал).
    глаза на лоб лезут. там задачи все со статическими размерами, но мне страшно. люди пишут НИРы по этим темам
    ну и задачка...

    финальный сабмит. решил оставить немного коментариев
'''


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
        return abs(self.r - other.r) < 0.09 or abs(self.r - 1/other.r) < 0.09
    
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


class Level():
    def __init__(self, b : int, h : int, f : int =0, w : int =0):
        self.bottom = b
        self.height = h
        self.floor = f
        self.initW = w
        self.ceiling = 0

    def put(self, rect : Rect, H, W, f : bool = True, leftJustified : bool = True) -> Rect:
        newRect : Rect
 
        y = 0
        # по хорошему надо проверить работу r и {h, w}
        if f:
            if leftJustified:
                newRect = Rect(rect.r)
                newRect.setSize((rect.height(), rect.width()))
                y = H-(self.bottom + rect.height() + 1)
                newRect.move((self.floor, y))
            else:
                # 'ceiling' is used for right-justified rectangles packed on the floor
                newRect = Rect(1/rect.r)
                newRect.setSize((rect.width(), rect.height()))
                y = H-(self.bottom + rect.height() + 1)
                newRect.move((W-(self.ceiling + rect.width()), y))
                self.ceiling += rect.width()
            
            self.floor += rect.width()
        else:
            newRect = Rect(1/rect.r)
            newRect.setSize((rect.width(), rect.height()))
            y = H-(self.bottom + rect.height() + 1)
            newRect.move((W-(self.ceiling + rect.width()), y))
            self.ceiling += rect.width()
    
        if y >= 0 and y < H:
            return newRect
        return None

    def ceilingFeasible(self, rect : Rect, H, W, existing : list) -> bool:
        testRect : Rect
        testRect = Rect(1/rect.r)
        testRect.setSize((rect.width(), rect.height()))
        testRect.move((W-(self.ceiling + rect.width()), H-(self.bottom + rect.height() + 1)))

        intersected = False
        for i in range(len(existing)):
            if (testRect.intersects(existing[i])):
                intersected = True
                break
        
        fit : bool = rect.width() <= (W - self.ceiling - self.initW)
        return fit and not intersected

    def floorFeasible(self, rect : Rect, W : int) -> bool:
        return rect.width() <= (W - self.floor)

    def getSpace(self, W : int, f : bool=True) -> bool:
        if f:
            return W - self.floor
        else:
            return W - self.ceiling - self.initW


def get_cell(cols, rows, i, j, H, W):
    c_h, c_w = (0, 0) # max(, ) # cells[(cols[i], rows[j])]
    if i == len(cols)-1:
        c_w = W - cols[i]
    else:
        c_w = cols[i+1] - cols[i]
    
    if j == len(rows)-1:
        c_h = H - rows[j]
    else:
        c_h = rows[j+1] - rows[j]

    return c_h, c_w

# старый перепечатанный вариант - не разбирал
def burke(rects, H, W):
    gap = []
    unpacked = copy.deepcopy(rects)
    unpacked.sort(key=(lambda x: x.height()))
    unpacked.sort(key=(lambda x: x.width()))

    gap = [0 for _ in range(W)]

    packed = []
    while unpacked:
        minG = gap[0]
        coordX = 0
        for i in range(len(gap)):
            if gap[i] < minG:
                minG = gap[i]
                coordX = i

        i = coordX+1
        gapWidth = 1
        while i < len(gap) and gap[i] == gap[i - 1]:
            gapWidth += 1
            i += 1

        # find best fitting rectangle
        ind :int  = -1
        fit :float = 0.0
        for j in range(len(unpacked)):
            curFit : float =   unpacked[j].width() / gapWidth
            if curFit < 1 and curFit > fit:
                fit = curFit
                ind = j

        if ind > -1:
            # place best fitting rectangle using placement policy 'Leftmost'
            newRect : Rect
            newRect = Rect(unpacked[ind].width() / unpacked[ind].height())
            y = H - (gap[coordX] + unpacked[ind].height())
            if y < 0:
                return None, True
            newRect.move((coordX, y))

            packed.append(newRect)

            # raise elements of array to appropriate height
            for j in range(coordX, coordX+unpacked[ind].width()):
                gap[j] += unpacked[ind].height()
            
            unpacked.pop(ind)
        
        else:
            # raise gap to height of the lowest neighbour
            lowest : int
            if coordX == 0:
                lowest = gap[gapWidth % len(gap)]
            elif coordX + gapWidth == len(gap):
                lowest = gap[len(gap) - gapWidth - 1]
            elif gap[coordX - 1] < gap[coordX + gapWidth]:
                lowest = gap[coordX - 1]
            else:
                lowest = gap[coordX + gapWidth]
            for j in range(coordX, coordX+gapWidth):
                gap[j] = lowest
        
    return packed, False

# актуальный вариант - писал сам с нуля
def packBurke(rects_ : list, fieldSize):
    '''
        по сути играем в тетрис
    '''
    rects = copy.deepcopy(rects_)
    for i in range(len(rects)):
        rects[i].moveToOrigin()
        # print(rects[i])

    positions = []
    real_indexes = []

    H, W = fieldSize
    hs, ws = zip(*[rect.getSize() for rect in rects])
    sorted_sides = sorted(list(enumerate(hs)) + list(enumerate(ws)), key=lambda x: x[1])
    ss = list(map(lambda x : x[1], sorted_sides))

    if sorted_sides[-1][1] > max(W, H):
        return None, None, True

    heights = [0 for _ in range(W)]
    while sorted_sides:
        # print(heights)
        # пока так. потом можно будет соптимизировать
        # heights_zip = sorted(heights)
        height, ind = heights[0], 0
        for i, h in enumerate(heights):
            if h < height:
                ind = i
                height = h
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
            h_, w = rects[real_ind].getSize()
            if w != width:
                rects[real_ind].setSize((w, h_))
                w, h_ = h_, w
            sorted_sides.pop(r_ind)
            ss.pop(r_ind)

            # ищем и удаляем другую сторону
            x = bisect.bisect_left(ss, h_)
            while sorted_sides[x][0] != real_ind:
                x += 1
            sorted_sides.pop(x)
            ss.pop(x)

            if r_b_height > l_b_height:
                # индекс для заливки
                l_bound = j+1
                r_bound = j+width
                pos = (j+1, height)
            else:
                # индекс для заливки
                r_bound = i-1
                l_bound = i-width
                pos = (i-width, height)
            h = h_

        heights[l_bound: r_bound+1] = [height+h for _ in range(l_bound, r_bound+1)]
        if pos:
            positions.append(pos)
            real_indexes.append(real_ind)
            rects[real_ind].move(pos)

    positions = [positions[ind] for ind in real_indexes]
    return rects, positions, False

def check_packed(packed, H, W) -> bool:
    '''
        возвращает True, если нашёл ошибку
    '''
    for rect in packed:
        coords = rect.get_coord_list()
        if coords[0] < 0 or coords[2] < coords[0] or coords[2] >= W:
            return False
        if coords[1] < 0 or coords[3] < coords[1] or coords[3] >= H:
            return True

    return False


def transpose(packed, H, W):
    for i, rect in enumerate(packed):
        coords = rect.c1
        rect.move((-coords['x'], -coords['y']))
        rect.turn()
        rect.move((coords['y'], coords['x']))
        rect.r = 1/rect.r
        packed[i] = rect
    return packed


def FCNR(rects : list, H, W) -> list :
    flag = "up"
    if H < W:
        flag = "right"
        H, W = W, H

    unpacked : list = copy.deepcopy(rects)
    unpacked.sort(key=(lambda x : x.height()))
 
    levels = []
    level = Level(0, unpacked[0].height(), 0, unpacked[0].width())
    packed = []
 
    packed.append(level.put(unpacked[0], H, W))
    levels.append(level)
 
    for i in range(1, len(unpacked)): # (int i = 1; i < unpacked.size(); i++) {
        found = -1
        minA = W
        for j in range(len(levels)): # (int j = 0; j < levels.size(); j++) {
            if levels[j].floorFeasible(unpacked[i], W):
                if levels[j].getSpace(W) < minA:
                    found = j
                    minA = levels[j].getSpace(W)
        
        if found > -1: # floor-pack on existing level
            packed.append(levels[found].put(unpacked[i], H, W))
        else:
            found = -1
            minA = W
            for j in range(len(levels)): # (int j = 0; j < levels.size(); j++) {
                if levels[j].ceilingFeasible(unpacked[i], H, W, packed):
                    if levels[j].getSpace(W, False) < minA:
                        found = j
                        minA = levels[j].getSpace(W, False)
            
            if found > -1: # ceiling-pack on existing level
                packed.append(levels[found].put(unpacked[i], H, W, False))
            else: # a new level
                newLevel = Level(levels[-1].bottom + levels[-1].height, unpacked[i].height(), 0, unpacked[i].width())
                packed.append(newLevel.put(unpacked[i], H, W))
                levels.append(newLevel)
        
        if packed[-1] is None:
            return None, True

    err = check_packed(packed, H, W)
    if not err:
        if flag == "right":
            pass #packed = transpose(packed, H, W)
        return packed, False
    else:
        return None, True


def append_to_field(rect : Rect, field, cols, rows, H_, W_):
    '''
        классическая вставка
        ищет место для вставки в поле и если находит - вставляет
    '''
    # H_, W_ = len(field), len(field[0])
    h, w = rect.getSize()
    l_r, l_c = len(rows), len(cols)

    for i, col in enumerate(cols):
        for j, row in enumerate(rows):
            # print("init: ", row, col)
            # если клетка cell не занята
            # проверка на случай, если у нас граница прямоугольника наложилась на 
            if col < W_ and row < H_ and not (row, col) in field: # field[row][col]:
                # если она достаточно высокая

                # если недостаточно высокая, то попробуем объеденить с ячейками выше
                k = j
                H = 0
                while k < l_r and not (rows[k],col) in field: # field[rows[k]][col]:
                    # считаем размеры текущей клетки
                    c_h, c_w = get_cell(cols, rows, i, k, H_, W_)
                    
                    # может ли получиться так, что c_h < h - думаю нет
                    H += c_h
                    k += 1
                    if H >= h:
                        break
                
                if k == l_r:
                    H = H_ - row

                # если не получилось собрать высоту
                if H < h:
                    continue

                # если ячейка нормальной высоты
                # собираем ширину
                l = i
                W = 0
                # проверяем все в выбранном диапазоне rows
                while l < l_c and not any((rows[t], cols[l]) in field for t in range(j, min(k, l_r))): # field[rows[t]][cols[l]]
                    c_h, c_w = get_cell(cols, rows, l, j, H, W_)

                    # может ли получиться так, что c_h < h - думаю нет
                    W += c_w
                    l += 1
                    if W >= w:
                        break

                if l == l_c:
                    W = W_ - col
                if W < w:
                    continue

                # собрали клетки нормального размера
                # можно на их место добавлять прямойгольник
                # print(i, j, l, k)
                # print(h, w)
                # print("prep: ", field, rows, cols)
                if (col + w) not in cols and (col + w) < W_:
                    bisect.insort(cols, col + w)
                    for ind, row_ in enumerate(rows):
                        if (row_, cols[l-1]) in field:
                            field.add((row_, cols[l]))
                # print("new1: ", field, rows, cols)
                if (row + h) not in rows and (row + h) < H_:
                    bisect.insort(rows, row + h)
                    for ind, col_ in enumerate(cols):
                        # print((rows[k-1], col_))
                        if (rows[k-1], col_) in field:
                            field.add((rows[k], cols[ind]))
                # print("new2: ", field, rows, cols)

                for ind1 in range(j, k):
                    for ind2 in range(i, l):
                        field.add((rows[ind1], cols[ind2]))
                        # print((rows[ind1], cols[ind2]))
                # print("new: ", field, rows, cols)
                # забиваем поле True
                # for ind in range(row, row+h):
                    # field[ind][col:col+w] = list(map(lambda x: True, range(col,col+w)))
                # raise "dfssdf"

                # там ещё есть отсечённые. надо просмотреть все
                

                # print((col, row))
                rect.move((col, row))
                return rect, False

    return None, True

def convert_to_clasic(pos, rects, fieldSize=(100, 100)):
    '''
        конвертируем результат поля бёрка для работы классического алгоритма
    '''
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


def printField(field):
    for line in field:
        if not (True in line):
            print("  ###  ", end='')
            continue
        print(' '.join(list(map(lambda x: '  |' if not x else '* |', line))))
        print('+'*(4*len(field[0])))


# генерирует все возможные комбинации сторон, удовленворяющие ограничениям
def generateSizes(r : float, H, W):
    '''
        генерирует все возможные комбинации сторон, удовленворяющие ограничениям
        работает за O(n/2), где n ~ H
    '''

    if r < 1:
        r = 1/r

    r += 0.0001
    sizes = set()
    for i in range(1, min(H, W), 2):
        f = Fraction.from_float(r).limit_denominator(i)
        den, num = f.denominator, f.numerator
        d = abs(num/den-r)
        if d < 0.09 and num <= max(H, W) and den <= min(H, W):
            num, den = max(den, num), min(den, num)
            k1 = i // den
            k2 = i // num
            k = max(1, min(k1, k2))
            sizes.add((num * k, den * k))
    return sorted(list(sizes), reverse=True)


def tryCombo(rs, size_combo, H, W, algo="classic"):
    # field = [[False for _ in range(W)] for _ in range(H)]
    field = set([(H, 0), (0, W), (H, W)])
    rects = []
    cols = [0, W]
    rows = [0, H]
    # print(len(rs), len(size_combo))
    for i, combo in enumerate(size_combo):
        # print(combo)
        rect = Rect(rs[i])
        rect.setSize(combo)
        rects.append(rect)


    if algo == "classic":
        rects.sort(key=(lambda x: x.height()), reverse=True)
        
        for i in range(len(rects)):
            err = True
            rect = rects[i]
            # print(rect)
            # rect.setSize((max(rect.getSize()), min(rect.getSize())))
            rr, err = append_to_field(rect, field, cols, rows, H, W)
            # if err == False:
            #     print(field)
            if err:
                rect.turn()
                rr, err = append_to_field(rect, field, cols, rows, H, W)
                # print(field)
            rects[i] = rr
            # print(rect)

            if err == True:
                return None, True
        # print(cols, rows)
    elif algo == "burke":
        '''
            пихаем бёрком как получится, а потом невпихнутое довпихиваем классикой
            работает дольше классики, качество +- то же, если не хуже
        '''

        rects1, positions, err = packBurke(rects, (H, W))
        if err == True:
            return None, True
        
        bad_rects = list(filter(lambda rect : rect.c2['y'] > H, rects1))
        good_rects = list(filter(lambda rect : rect.c2['y'] <= H, rects1))
        field, cols, rows = convert_to_clasic(positions, good_rects, (H, W))

        for i, rect in enumerate(bad_rects):
            err = True
            rect = bad_rects[i]
            rr, err = append_to_field(rect, field, cols, rows, H, W)
            if err:
                rect.turn()
                rr, err = append_to_field(rect, field, cols, rows, H, W)
            bad_rects[i] = rr

            if err == True:
                return None, True

        rects = good_rects + bad_rects
            
    elif algo == "FCNR":
        rects, err = FCNR(rects, H, W)
        if err == True:
            return None, True

    line = []
    placements = []

    # rects_r = [max(rect.r, 1/rect.r) for rect in rects]
    # бин поиск не сильно ускорил дело
    for r in rs:
        # print(rs, rects_r)
        for i, rect in enumerate(rects):
            if abs(r - rect.r) < 0.09 or abs(r - 1/rect.r) < 0.09:
                line += rect.get_coord_list()
                rects.pop(i)
                break
        # i = bisect.bisect_left(rects_r, r)
        # if i == len(rects):
        #     i -= 1
        # line += rects[i].get_coord_list()
        # rects.pop(i)
        # rects_r.pop(i)



    # если не все прямоугольники получилось распределить
    if len(rects) != 0:
        print("!!! there is some bad rect !!!")
        for rect in rects:
            print(rect)
        raise "rect determination problem"
    
    return line, False

# тестил бинпоиск, но он не дал прирост производительности, ибо слишком мало элементов
def bisect_right(arr, x, key=None):
    if key is None:
        key = lambda value: value
    
    lo = 0
    hi = len(arr)
    
    while lo < hi:
        mid = (lo + hi) // 2
        if key(arr[mid]) > x:
            hi = mid
        else:
            lo = mid + 1
    
    return lo


def findOpt(rs, sizes, size_combo, H, W, selfS):
    '''
        рекурсивно перебирает все варианты размеров, начиная с самых больших
        прерывается ПОЧТИ сразу, как найдёт подходящий вариант
        технически работает за O((2*k)^n) или около того, но на самом деле куда быстрее
    '''

    if sizes == []:
        positions, err = tryCombo(rs, size_combo, H, W, algo="classic")
        if err == False:
            return positions, selfS, err
        else:
            return None, None, err

    # можно добавить бин поиск
    best_combo_positions = None
    minS = 0
    i = 0
    # print(sizes[0])
    
    '''
    lo = 0
    hi = len(sizes[0])
    
    while lo < hi:
        mid = (lo + hi) // 2
        # print(mid, lo, hi)
        if sizes[0][mid][0]*sizes[0][mid][1] > H*W-selfS:
            hi = mid
            continue

        size_combo.append(sizes[0][mid])
        # print(size_combo)
        positions, S, err = findOpt(rs, sizes[1:], size_combo, H, W, selfS+size_combo[-1][0]*size_combo[-1][1])
        size_combo.pop(-1)

        if err == False:
            hi = mid
        else:
            lo = mid + 1

    if lo == 0:
        # print(sizes[0])
        return None, None, True

    # lo - нужный индекс
    size_combo.append(sizes[0][lo-1])
    # print("nice combo: ", size_combo)
    best_combo_positions, minS, err = findOpt(rs, sizes[1:], size_combo, H, W, selfS+size_combo[-1][0]*size_combo[-1][1])
    size_combo.pop(-1)
    '''

    
    for ind, size in enumerate(sizes[0]):
        # если мы берём слишком большой размер - даже не пробуем подставлять
        if size[0]*size[1] > H*W-selfS:
            continue

        size_combo.append(size)
        if i > 0:
            size_combo[-1] = choice(sizes[0][ind:])
        
        positions, S, err = findOpt(rs, sizes[1:], size_combo, H, W, selfS+size_combo[-1][0]*size_combo[-1][1])
        size_combo.pop(-1)
        # нужно добавить сравниние площадей
        if err == False:
            i += 1
            if S > minS:
                best_combo_positions = positions
                minS = S

        if i > 1:
            return best_combo_positions, minS, False
        # else:
        #     continue
    

    if best_combo_positions:
        return best_combo_positions, minS, False
    else:
        return None, None, True

# пробовать несколько выриантов - слишком дорого и не выгодно. возвращаем первый подошедший
def findOptFullRand(rs, sizes, size_combo, H, W, selfS):
    if sizes == []:
        positions, err = tryCombo(rs, size_combo, H, W, algo="classic")
        if err == False:
            return positions, selfS, err
        else:
            return None, None, err

    # можно добавить бин поиск
    best_combo_positions = None
    minS = 0
    
    for ind, size in enumerate(sizes[0]):
        # если мы берём слишком большой размер - даже не пробуем подставлять
        if size[0]*size[1] > H*W-selfS:
            continue

        size_combo.append(size)
        positions, S, err = findOptFullRand(rs, sizes[1:], size_combo, H, W, selfS+size_combo[-1][0]*size_combo[-1][1])
        size_combo.pop(-1)
        # нужно добавить сравниние площадей
        if err == False:
            # i += 1
            if S > minS:
                best_combo_positions = positions
                minS = S
                return best_combo_positions, minS, False

    if best_combo_positions:
        return best_combo_positions, minS, False
    else:
        return None, None, True


def unshuffle(pos, sh_r, order):
    '''
        восстанавливает изначальный порядок
        O(n), где n - количество прямоугольников
    '''
    undone_pos = pos.copy()
    for i, s in enumerate(sh_r):
        ind = order[s][-1]
        order[s].pop(-1)
        undone_pos[ind*4 : (ind+1)*4] = pos[i*4 : (i+1)*4]

    return undone_pos

@benchmark
def solveCase(case) -> np.array:
    print(case)
    H, W = int(case[0]), int(case[1])
    cols = [0]
    rows = [0]
    rs = case[2:]

    # генерируем все возможные размеры для rects
    sizes = []
    for r in rs:
        sizes.append(generateSizes(r, H, W))
    
    Smax = 0
    position = None

    r_sizes = list(zip(rs, sizes))
    order = defaultdict(list)
    for i in range(len(r_sizes)):
        order[r_sizes[i][0]].append(i)
    
    seed(0)
    for _ in range(1): 
        print("try: ")
        sh_r_sizes = r_sizes.copy()
        shuffle(sh_r_sizes)
        sh_r_sizes.sort(key = (lambda x: x[0]))
        sh_r = [r for r, _ in sh_r_sizes]
        sh_sizes = [sizes for _, sizes in sh_r_sizes]

        size_combo = []
        pos, S, err = findOpt(sh_r, sh_sizes, size_combo, H, W, 0)

        if S > Smax:
            position = unshuffle(pos, sh_r, copy.deepcopy(order))
            Smax = S

        assert err == False

    # print(position)
    # visualize_placements(position, Rect(1.), (W, H), range(5))
    return position

import random
# @benchmark
def solveCaseRand(case) -> np.array:
    # print(case)
    H, W = int(case[0]), int(case[1])
    rs = case[2:]

    # генерируем все возможные размеры для rects
    sizes = []
    for r in rs:
        sizes.append(generateSizes(r, H, W))
    

    SM = 0
    P = []

    Smax = 0
    position = None
    seed(42)
    for _ in range(30):
        rand_sizes = [random.sample(line[:max(1, (len(line)*9)//10)], max(1, (len(line)*1)//10)) + line[(len(line)*9)//10:] for line in sizes]
            
        r_sizes = list(zip(rs, rand_sizes))
        order = defaultdict(list)
        for i in range(len(r_sizes)):
            order[r_sizes[i][0]].append(i)
        
        for _ in range(1):
            sh_r_sizes = r_sizes.copy()
            shuffle(sh_r_sizes)
            sh_r_sizes.sort(key = (lambda x: x[0]))
            sh_r = [r for r, _ in sh_r_sizes]
            sh_sizes = [sizes for _, sizes in sh_r_sizes]

            size_combo = []
            # print("find Opt")
            pos, S, err = findOpt(sh_r, sh_sizes, size_combo, H, W, 0)

            if not err and S > Smax:
                position = unshuffle(pos, sh_r, copy.deepcopy(order))
                Smax = S

    return position


# использую это - всё пологается на рандомный выбор набора размеров
def solveCaseFullRand(case) -> np.array:
    # print(case)
    H, W = int(case[0]), int(case[1])
    rs = case[2:]

    # генерируем все возможные размеры для rects
    sizes = []
    for r in rs:
        sizes.append(generateSizes(r, H, W))

    Smax = 0
    position = None
    seed(42)
    x1 = [(line[:max(1, (len(line)*9)//10)], 1) for line in sizes]
    x2 = [[line[-1]] for line in sizes]
    # print(x1, x2)
    for _ in range(300):
        # делаем рандомное подмножество, не забывая про минимум, чтоб точно нашлось решение
        rand_sizes = [random.sample(x1[0], x1[1]) + x2 for x1, x2 in zip(x1, x2)]

        for _ in range(2):
            sh_r_sizes = list(zip(rs, rand_sizes))
            order = defaultdict(list)
            for i, s in enumerate(sh_r_sizes):
                order[s[0]].append(i)
            
            # sh_r_sizes = r_sizes.copy()

            # пробуем 2 варианта : перемешку и сортированный
            if i % 2 == 0:
                shuffle(sh_r_sizes)
            elif i % 2 == 1:
                # sh_r_sizes.sort()
                sh_r_sizes.sort(key = (lambda x: x[0]))

            sh_r = [r for r, _ in sh_r_sizes]
            sh_sizes = [sizes for _, sizes in sh_r_sizes]

            pos, S, err = findOptFullRand(sh_r, sh_sizes, [], H, W, 0)

            if not err and S > Smax:
                position = unshuffle(pos, sh_r, copy.deepcopy(order))
                Smax = S
            if err:
                raise "error"

    return position

def funcSciPy(A, *args):
    ss = args[0]
    rs = args[1]
    H, W = args[2]
    # print(A, args)
    # A, fieldSize = A[:-2], A[-2:]
    # H, W = fieldSize

    # rs =    A[ : len(A) // 2]
    sizes = A # [len(A) // 2 : ]
    # print(sizes)
    # print(ss)

    sizes = list(map(lambda x : ss[x[0]][bisect.bisect_left(ss[x[0]], round(x[1]))] , enumerate(sizes)))
    sizes = [(size, round(size/r)) for r, size in zip(rs, sizes)]

    # print(rs, sizes, (H, W))

    pos, err = tryCombo(rs, sizes, H, W, algo='classic') # findOptFullRand(rs, sizes, [], H, W, 0)
    if err:
        return H*W
    pos = np.asarray(pos, dtype=int).reshape(len(pos)//4, 4)
    S = sum([(coord[2]-coord[0]+1)*(coord[3]-coord[1]+1) for coord in pos])
    # print(S)
    return H*W-S

# не использую, ибо условия
@benchmark
def solveCaseSciPy(case):
    # print(case)
    H, W = int(case[0]), int(case[1])
    rs = case[2:]

    # генерируем все возможные размеры для rects
    sizes = []
    for r in rs:
        sizes.append(generateSizes(r, H, W))
        for i, size in enumerate(sizes[-1]):
            if size[0] < size[1]:
                size[0], size[1] = size[1], size[0]

            sizes[-1][i] = size
        sizes[-1].sort()

    Smax = 0
    position = None
    seed(42)
    
    position = None
    i = -1
    while position is None or i < 2:
        i += 1
        sh_r_sizes = list(zip(rs, sizes))
        order = defaultdict(list)
        for i, s in enumerate(sh_r_sizes):
            order[s[0]].append(i)
        
        # sh_r_sizes = r_sizes.copy()

        # пробуем 2 варианта : перемешку и сортированный
        if i % 2 == 0:
            shuffle(sh_r_sizes)
        elif i % 2 == 1:
            # sh_r_sizes.sort()
            sh_r_sizes.sort(key = (lambda x: x[0]))

        sh_r = [r for r, _ in sh_r_sizes]
        sh_sizes = [sizes for _, sizes in sh_r_sizes]
        
        grid = []
        # for r in sh_r:
        #     grid.append([r, r])
        for line in sh_sizes:
            grid.append([line[0][0], line[-1][0]])
        # grid += [[H, H], [W, W]]

        # for line in grid:
        #     print(line)
        ss = [sorted([s[0] for s in line]) for line in sh_sizes]
        # print("ss: ", ss)
        # print(sp.optimize.rosen)
        res = sp.optimize.differential_evolution(func=funcSciPy, bounds=grid, args=(ss, sh_r, (H, W), ), strategy="best1bin", maxiter=100)
        # print(res)
        opt_sizes = res.x # [len(rs) : -2]
        opt_sizes = list(map(lambda x : ss[x[0]][bisect.bisect_left(ss[x[0]], round(x[1]))] , enumerate(opt_sizes)))
        opt_sizes = [(size, round(size/r)) for r, size in zip(sh_r, opt_sizes)]
        # print("opt sizes:", opt_sizes)

        # pos, S, err = findOptFullRand(rs, sizes, [], H, W, 0)
        pos, err = tryCombo(sh_r, opt_sizes, H, W, algo='classic')
        # if err:
        #     print("errr")
        #     continue

        pos_1 = np.asarray(pos, dtype=int).reshape(len(pos)//4, 4)
        # for p in pos_1:
        #     print(p)
        S = sum([(coord[2]-coord[0]+1)*(coord[3]-coord[1]+1) for coord in pos_1])
        # print("S: ", S)
        # print(pos)
       #  pos, S, err = findOptFullRand(sh_r, opt_sizes, [], H, W, 0)

        if not err and S > Smax:
            # print("updated S rest: ", H*W - S, S)
            position = unshuffle(pos, sh_r, copy.deepcopy(order))
            Smax = S
        
        # raise "fine"

    # visualize_placements(position, (W, H))
    # print(position)
    return position


def solution(task) -> np.array:
    data_frame = []

    # немного параллелизма в студию ))
    with multiprocessing.Pool(2) as pool:
        data_frame = pool.map(wrapped_func, task)

    # for case in task:
    #     positions = solveCaseSciPy(case) # solveCaseRand(case)
    #     data_frame.append(positions)
    
    return np.asarray(data_frame, dtype=int)


import multiprocessing

def main():
    '''sys.argv[1]'''
    # для теста производительности на task2
    # task = np.concatenate((np.genfromtxt(sys.argv[1] , delimiter=",", skip_header=1)[:,:], np.genfromtxt(sys.argv[1] , delimiter=",", skip_header=1)[:,2:]), axis=1)
    task = np.genfromtxt(sys.argv[1] , delimiter=",", skip_header=1)[:10,:]
    print(task.shape)

    start_time = time.time()

    sol = solution(task)
    delta_t = (time.time() - start_time)
    m = int(delta_t)//60
    s = int(delta_t)%60
    ms = int(1000*(delta_t-m*60-s))
    print(f"{m}:{s}:{ms}")

    sol = np.asarray(sol, dtype=str)
    # print(sol)

    header = (', '.join([f'X{i+1}min, Y{i+1}min, X{i+1}max, Y{i+1}max' for i in range(len(sol[0]) // 4)])).split(', ')
    # print(header)
    sol = np.insert(sol, 0, np.asarray(header, dtype=str), axis=0)
    # print(sol)
    np.savetxt("solution.csv", sol, delimiter=",", fmt="%s")

if __name__ == '__main__':
    main()