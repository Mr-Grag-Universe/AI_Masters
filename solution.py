import numpy as np
import scipy as sp
import sys
from fractions import gcd, Fraction


# для начала попробуем простенький жадный алгоритм
# 1) прямоугольники делаем целосторонними
# 2) проверяем, что суммарная площадь <= площади поля
#   * если это не так - крутим как получится - пока не придумал
#   * если всё норм продолжаем
# 3) максимизируем размеры прямоугольника самого большого отношения r (так чтобы площади хватило под остальные)
# 4) если места хватает - продолжаем. для оставшихся прямоугольников

class Rect():
    c1 : dict
    c2 : dict
    r : Fraction
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
        print(f"Created: {self.c1} - {self.c2}, r: {self.r}, id: {self.r_id}")

    def move(self, point):
        self.c1['x'] += point[0]
        self.c2['x'] += point[0]
        self.c1['y'] += point[1]
        self.c2['y'] += point[1]
        print("moved to ")

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
        return [self.c1['x'], self.c1['y'], self.c2['x'], self.c2['y']]


class Column():
    isEmpty : bool

    def __init__(self, ind=0, size=1):
        self.ind = ind
        self.size = size
    
    def resize(self, new_size=1):
        self.size = new_size

    def setIndex(self, new_ind=0):
        self.ind = new_ind

    def empty(self):
        return self.isEmpty

class Row():
    isEmpty : bool

    def __init__(self, empty=True, ind=0, size=1):
        self.isEmpty = empty
        self.ind = ind
        self.size = size
    
    def resize(self, new_size=1):
        self.size = new_size

    def setIndex(self, new_ind=0):
        self.ind = new_ind

    def empty(self):
        return self.isEmpty


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


def append_to_field(rect : Rect, field, cols, rows, cells : dict):
    h, w = rect.size()
    for i in range(len(cols)):
        for j in range(len(rows)):
            # если клетка cell не занята
            # проверка на случай, если у нас граница прямоугольника наложилась на 
            if cols[i] < len(field) and rows[j] < len(field[0]) and not field[rows[j]][cols[i]]:
                print("cols and rows : ", cols[i], rows[j])
                # если она достаточно высокая
                c_h, c_w = get_cell(cols, rows, i, j, len(field), len(field[0]))
                print(f"c_w: {c_w}, c_h: {c_h}")


                if c_h >= h:
                    if c_w >= w:
                        # ячейка целиком поглощает наш прямоугольник
                        print(f"Вставляем {rect} в ({cols[i]}, {rows[j]}) - полностью поместился")
                        # можно соптимизировать бинарной вставкой
                        if (cols[i] + w) not in cols and (cols[i] + w) < len(field):
                            cols.append(cols[i] + w)
                            cols.sort()
                        if (rows[j] + h) not in rows and (rows[j] + h) < len(field[0]):
                            rows.append(rows[j] + h)
                            rows.sort()

                        # забиваем поле True
                        print(f"h: {h}, w: {w}")
                        for ind in range(rows[j], rows[j]+h):
                            print("ind: ", ind)
                            field[ind][cols[i]:cols[i]+w] = [True for _ in range(cols[i],cols[i]+w)]
                        
                        rect.move((cols[i], rows[j]))
                        return rect, False

                    # если справа больше нет ячеек, чтобы нарастить ширину скипаем
                    if i == len(cols)-1:
                        print("Пробуем дальше")
                        continue

                    # начинаем сляпывать столбцы
                    # row у нас одна и та же
                    k = i + 1
                    W = c_w
                    while k < len(cols) and not field[cols[k]][rows[j]]:
                        c_h, c_w = get_cell(cols, rows, k, j, len(field), len(field[0]))
                        # может ли получиться так, что c_h < h - думаю нет
                        W += c_w
                        if W >= w:
                            break
                    # если не получилось собрать ширину
                    if W < w:
                        print("Пробуем дальше")
                        continue

                    # если ячейка нормального размера
                    # всталяем прямоугольник
                    print(f"Вставляем {rect} в ({cols[i]}, {rows[j]})")
                    if (cols[i] + w) not in cols and (cols[i] + w) < len(field):
                        cols.append(cols[i] + w)
                        cols.sort()
                    if (rows[j] + h) not in rows and (rows[j] + h) < len(field[0]):
                        rows.append(rows[j] + h)
                        rows.sort()

                    # забиваем поле True
                    for ind in range(rows[j], rows[j]+h):
                        field[ind][cols[i]:cols[i]+w] = list(map(lambda x: True, range(cols[i],cols[i]+w)))
                    rect.move((cols[i], rows[j]))
                    return rect, False

                else:
                    # если недостаточно высокая, то попробуем объеденить с ячейками выше
                    k = j + 1
                    H = c_h
                    while k < len(rows) and not field[cols[i]][rows[k]]:
                        # c_h, c_w = cells[(cols[i], rows[k])]
                        c_h, c_w = get_cell(cols, rows, i, k, len(field), len(field[0]))
                        
                        # может ли получиться так, что c_h < h - думаю нет
                        H += c_h
                        if H >= h:
                            break
                    
                    # если не получилось собрать высоту
                    if H < h:
                        continue

                    # если ячейка нормальной высоты
                    # собираем ширину
                    l = i + 1
                    W = c_w
                    # проверяем все в выбранном диапазоне rows
                    while l < len(cols) and not any(field[cols[l]][rows[t]] for t in range(j, k+1)):
                        # c_h, c_w = cells[(cols[l], rows[j])]
                        c_h, c_w = get_cell(cols, rows, l, j, len(field), len(field[0]))

                        # может ли получиться так, что c_h < h - думаю нет
                        W += c_w
                        if W >= w:
                            break

                    if W < w:
                        continue

                    # собрали клетки нормального размера
                    # можно на их место добавлять прямойгольник
                    print(f"Вставляем {rect} в ({cols[i]}, {rows[j]})")
                    if (cols[i] + w) not in cols and (cols[i] + w) < len(field):
                        cols.append(cols[i] + w)
                        cols.sort()
                    if (rows[j] + h) not in rows and (rows[j] + h) < len(field[0]):
                        rows.append(rows[j] + h)
                        rows.sort()

                    # забиваем поле True
                    for ind in range(rows[j], rows[j]+h):
                        field[ind][cols[i]:cols[i]+w] = list(map(lambda x: True, range(cols[i],cols[i]+w)))

                    rect.move((cols[i], rows[j]))
                    return rect, False
    return rect, True



def solution(task) -> np.array:
    data_frame = []
    # for i in range(len(task[0])):
    #     data_frame[0] += [f'X{i+1}min, Y{i+1}min, X{i+1}max, Y{i+1}max']

    for case in task:
        print(case)
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
            print(rects[-1])
        
        rects.sort(reverse=True)

        # r in rects - r копия из списка - не влияет на список
        for i in range(len(rects)):
            rects[i], err = append_to_field(rects[i], field, cols, rows, cells)
            assert err == False

            for line in field:
                if not (True in line):
                    print("  ###  ")
                    print("+++++++")
                    continue
                print(' '.join(list(map(lambda x: '  |' if not x else '* |', line))))
                print('+'*(4*W))
        
        line = []
        print(case[2:])
        for r in case[2:]:
            for i in range(len(rects)):
                if r == rects[i].r:
                    line += rects[i].get_coord_list()
                    rects.pop(i)
                    break

        data_frame.append(line)

    print(data_frame)
    return data_frame

task = np.genfromtxt(sys.argv[1], delimiter=",", skip_header=1)
print(task)


sol = np.asarray(solution(task), dtype=str)
header = (', '.join([f'X{i+1}min, Y{i+1}min, X{i+1}max, Y{i+1}max' for i in range(len(sol[0]) // 4)])).split(', ')
print(header)
sol = np.insert(sol, 0, np.asarray(header, dtype=str), axis=0)
print(sol)
np.savetxt(sys.argv[2], sol, delimiter=",", fmt="%s")

