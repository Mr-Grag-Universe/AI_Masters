class Rect:
    def __init__(self, width, height):
        self.width = width
        self.height = height

def pack_rectangles(rectangles, container_width, container_height):
    # Сортируем прямоугольники по убыванию площади
    rectangles.sort(key=lambda r: r.width * r.height, reverse=True)

    # Создаем список для хранения координат прямоугольников
    coordinates = []

    # Создаем пустую матрицу для представления контейнера
    container_matrix = [[0] * container_width for _ in range(container_height)]

    for rectangle in rectangles:
        # Пробегаемся по каждому ряду контейнера
        for y in range(container_height - rectangle.height + 1):
            # Пробегаемся по каждому столбцу контейнера
            for x in range(container_width - rectangle.width + 1):
                # Проверяем, можно ли разместить прямоугольник в текущей позиции
                if check_placement(container_matrix, x, y, rectangle.width, rectangle.height):
                    # Если можно, добавляем координаты прямоугольника в список
                    coordinates.append((x, y))
                    # Обновляем матрицу контейнера
                    update_container(container_matrix, x, y, rectangle.width, rectangle.height)
                    # Выходим из цикла
                    break
            # Если удалось разместить прямоугольник, выходим из цикла
            if len(coordinates) >= len(rectangles):
                break

    return coordinates

def check_placement(container_matrix, x, y, width, height):
    # Проверяем, свободна ли область для размещения прямоугольника
    for i in range(y, y + height):
        for j in range(x, x + width):
            # Если область занята, возвращаем False
            if container_matrix[i][j] != 0:
                return False
    return True

def update_container(container_matrix, x, y, width, height):
    # Заполняем область прямоугольника в матрице контейнера
    for i in range(y, y + height):
        for j in range(x, x + width):
            container_matrix[i][j] = 1


# Пример использования
rectangles = [Rect(2, 3), Rect(3, 4), Rect(4, 5)]
container_width = 10
container_height = 10
coordinates = pack_rectangles(rectangles, container_width, container_height)
for i, rectangle in enumerate(rectangles):
    x, y = coordinates[i]
    print(f"Прямоугольник {i+1}: координаты ({x}, {y})")
