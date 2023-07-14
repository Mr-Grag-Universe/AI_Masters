import numpy as np
# import scipy as sp
import sys
from fractions import Fraction
from math import gcd
from random import shuffle, seed, choice
from collections import defaultdict
# import matplotlib.pyplot as plt
import time
import copy

def benchmark(func):
    import time

    def wrapper(*args, **kwargs):
        start = time.time()
        func(*args, **kwargs)
        end = time.time()
        delta_t = (end - start)
        m = int(delta_t)//60
        s = int(delta_t)%60
        ms = int(1000*(delta_t-m*60-s))
        print(f"[*] Время выполнения: {m}:{s}:{ms}")
    return wrapper

# def visualize_placements(all_placements, max_rectangles, container, order_res):
#     # Создаем график
#     fig, ax = plt.subplots()

#     # Закрашиваем свободное пространство на листе черным цветом
#     ax.add_patch(plt.Rectangle((0, 0), container[0], container[1], facecolor='black'))

#     # Определяем список цветов для прямоугольников
#     colors = ['red', 'green', 'blue', 'orange', 'purple', 'cyan', 'magenta']

#     # Проходимся по всем расстановкам прямоугольников
#     ind = 0
#     for placements in all_placements:
#         # Проходимся по каждому прямоугольнику в расстановке
#         for i, placement in enumerate(placements):
#             rect_coords = placement[0] + placement[1]  # Координаты прямоугольника
#             # print(rect_coords)
#             color = colors[i % len(colors)]  # Цвет прямоугольника
#             rectangle = plt.Rectangle(rect_coords[:2], rect_coords[2]-rect_coords[0]+1, rect_coords[3]-rect_coords[1]+1, facecolor=color, alpha=0.5)
#             rx, ry = rectangle.get_xy()
#             cx = rx + rectangle.get_width()/2.0
#             cy = ry + rectangle.get_height()/2.0
#             # Добавляем прямоугольник на график
#             ind += 1
#             ax.add_patch(rectangle)
#             ax.annotate(str(ind), (cx, cy), color='black', weight='bold', fontsize=10, ha='center', va='center')

#     # Устанавливаем пределы графика и оси
#     ax.set_xlim(0, container[0])
#     ax.set_ylim(0, container[1])

#     # Отображаем график
#     plt.show()


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
'''


class Rect():
    c1 : dict
    c2 : dict
    # r : Fraction
    count = 0
    r_id : int

    @staticmethod
    def __find_fraction(x : float) -> Fraction:
        a = 10 ** len(str(x).split('.')[1])
        b = x*a
        g = gcd(a, b)
        return Fraction(b//g, a//g)
        

    def __init__(self, r_):
        self.r = max(r_, 1/r_)
        self.c1 = {'x': 0, 'y': 0}
        self.r_id = Rect.count
        Rect.count += 1
        f = Fraction.from_float(self.r).limit_denominator(1000)
        self.c2 = {'x' : f.numerator, 'y' : f.denominator}
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

    def setSize(self, size):
        self.c2 = {'x' : (self.c1['x'] + size[0]), 'y' : (self.c1['y'] + size[1])}

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


def append_to_field(rect : Rect, field, cols, rows):
    '''
        ищет место для вставки в поле и если находит - вставляет
    '''
    h, w = rect.getSize()

    for i in range(len(cols)):
        for j in range(len(rows)):
            # если клетка cell не занята
            # проверка на случай, если у нас граница прямоугольника наложилась на 
            if cols[i] < len(field[0]) and rows[j] < len(field) and not field[rows[j]][cols[i]]:
                # если она достаточно высокая

                # если недостаточно высокая, то попробуем объеденить с ячейками выше
                k = j
                H = 0
                while k < len(rows) and not field[rows[k]][cols[i]]:
                    # считаем размеры текущей клетки
                    c_h, c_w = get_cell(cols, rows, i, k, len(field), len(field[0]))
                    
                    # может ли получиться так, что c_h < h - думаю нет
                    H += c_h
                    if H >= h:
                        break
                    k += 1
                
                if k == len(rows):
                    H = len(field) - rows[j]

                # если не получилось собрать высоту
                if H < h:
                    continue

                # если ячейка нормальной высоты
                # собираем ширину
                l = i + 1
                W = c_w
                # проверяем все в выбранном диапазоне rows
                while l < len(cols) and not any(field[rows[t]][cols[l]] for t in range(j, min(k+1, len(rows)))):
                    c_h, c_w = get_cell(cols, rows, l, j, len(field), len(field[0]))

                    # может ли получиться так, что c_h < h - думаю нет
                    W += c_w
                    if W >= w:
                        break
                    l += 1

                if l == len(cols):
                    W = len(field[0]) - cols[i]
                if W < w:
                    continue

                # собрали клетки нормального размера
                # можно на их место добавлять прямойгольник
                if (cols[i] + w) not in cols and (cols[i] + w) < len(field[0]):
                    cols.append(cols[i] + w)
                    cols.sort()
                if (rows[j] + h) not in rows and (rows[j] + h) < len(field):
                    rows.append(rows[j] + h)
                    rows.sort()

                # забиваем поле True
                for ind in range(rows[j], rows[j]+h):
                    field[ind][cols[i]:cols[i]+w] = list(map(lambda x: True, range(cols[i],cols[i]+w)))

                rect.move((cols[i], rows[j]))
                return rect, False

    return None, True


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
    field = [[False for _ in range(W)] for _ in range(H)]
    rects = []
    cols = [0]
    rows = [0]
    # print(len(rs), len(size_combo))
    for i in range(len(size_combo)):
        rect = Rect(rs[i])
        rect.setSize((max(size_combo[i]), min(size_combo[i])))
        rects.append(rect)


    if algo == "classic":
        rects.sort(key=(lambda x: x.height()), reverse=True)
        
        for i in range(len(rects)):
            err = True
            rect = rects[i]
            # rect.setSize((max(rect.getSize()), min(rect.getSize())))
            rr, err = append_to_field(rect, field, cols, rows)
            if err:
                rect.turn()
                rr, err = append_to_field(rect, field, cols, rows)
            rects[i] = rr
            # print(rect)

            if err == True:
                return None, True
    elif algo == "burke":
        rects, err = burke(rects, H, W)
        if err == True:
            return None, True
    elif algo == "FCNR":
        rects, err = FCNR(rects, H, W)
        if err == True:
            return None, True

    line = []
    placements = []
    rr = rects.copy()
    for r in rs:
        for i in range(len(rr)):
            if abs(r - rr[i].r) < 0.09 or abs(r - 1/rr[i].r) < 0.09:
                line += rr[i].get_coord_list()
                # print(rects[i])
                rr.pop(i)
                break

    # print("line:", line)
    # если не все прямоугольники получилось распределить
    if len(rr) != 0:
        print("!!! there is some bad rect !!!")
        for rect in rr:
            print(rect)
        raise "rect determination problem"
    
    return line, False


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


def unshuffle(pos, sh_r, order):
    '''
        восстанавливает изначальный порядок
        O(n), где n - количество прямоугольников
    '''
    undone_pos = pos.copy()
    for i in range(len(sh_r)):
        r = sh_r[i]
        ind = order[r][-1]
        order[r].pop(-1)
        undone_pos[ind*4 : (ind+1)*4] = pos[i*4 : (i+1)*4]

    return undone_pos

@benchmark
def solveCase(case) -> np.array:
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
    for _ in range(3): 
        sh_r_sizes = r_sizes.copy()
        shuffle(sh_r_sizes)
        sh_r = [r for r, _ in sh_r_sizes]
        sh_sizes = [sizes for _, sizes in sh_r_sizes]

        size_combo = []
        pos, S, err = findOpt(sh_r, sh_sizes, size_combo, H, W, 0)

        if S > Smax:
            position = unshuffle(pos, sh_r, copy.deepcopy(order))
            Smax = S

        assert err == False

    return position


def solution(task) -> np.array:
    data_frame = []

    for case in task:
        # print(case)
        positions = solveCase(case)
        data_frame.append(positions)
        # break
        continue

        H, W = int(case[0]), int(case[1])
        field = [[False for _ in range(W)] for _ in range(H)]
        columns = [Column(0, W)]
        rows = [Row(0, H)]

        cols = [0]
        rows = [0]

        cells = {(0, 0) : (H, W)}

        rects = []
        for r in case[2:]:
            print(f"r: {r}")
            rects.append(Rect(r))
            size = chose_min_rect_size(rects[-1])
            rects[-1].setSize(size)
            print(rects[-1])
        
        rects.sort(reverse=False)

        # r in rects - r копия из списка - не влияет на список
        for i in range(len(rects)):
            rect = rects[i]
            # size = chose_rect_size(rect, field, cols, rows)
            # assert size is not None
            # rect.setSize(size)
            print(rect)

            # пытаемся добавить максимального размера
            rect, err = append_to_field(rect, field, cols, rows, cells)
            if err:
                rect.turn()
                rect, err = append_to_field(rect, field, cols, rows, cells)

            # если не получилось - пробуем с минимальным размером
            if err == False:
                rects[i] = rect
            else:
                rects[i], err = append_to_field(rects[i], field, cols, rows, cells)
                if err:
                    rects[i].turn()
                    rects[i], err = append_to_field(rects[i], field, cols, rows, cells)
            
            assert err == False

            printField(field)
        
        line = []
        print(case[2:])
        for r in case[2:]:
            for i in range(len(rects)):
                if abs(r - rects[i].r) < 0.09 or abs(r - 1/rects[i].r) < 0.09:
                    print(rects[i], r)
                    line += rects[i].get_coord_list()
                    rects.pop(i)
                    break

        if len(rects) != 0:
            print("!!! there is some bad rect !!!")
            for rect in rects:
                print(rect)
            raise "dfddf"
        data_frame.append(line)

    # print(data_frame)
    return data_frame

def main():
    '''sys.argv[1]'''
    task = np.genfromtxt(sys.argv[1] , delimiter=",", skip_header=1)[:10, :]
    # print(task)


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


main()