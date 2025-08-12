from queue import PriorityQueue
import math


def a_star_search(Grid, Sx, Sy, Dx, Dy):
    N = len(Grid)

    def heuristic(x, y):
        return math.sqrt((Dx - x) ** 2 + (Dy - y) ** 2)

    g_score = [[float("inf") for _ in range(N)] for _ in range(N)]
    f_score = [[float("inf") for _ in range(N)] for _ in range(N)]

    g_score[Sx][Sy] = Grid[Sx][Sy]
    f_score[Sx][Sy] = g_score[Sx][Sy] + heuristic(Sx, Sy)

    came_from = {}
    pq = PriorityQueue()
    pq.put((f_score[Sx][Sy], Sx, Sy))
    visited = set()

    while not pq.empty():
        current_f, x, y = pq.get()

        if x == Dx and y == Dy:
            path = []
            total_cost = g_score[x][y]
            while (x, y) != (Sx, Sy):
                path.append((x, y))
                x, y = came_from[(x, y)]
            path.append((Sx, Sy))
            path.reverse()
            return total_cost, path

        if (x, y) in visited:
            continue
        visited.add((x, y))

        # Up
        if x > 0 and Grid[x - 1][y] != -1:
            temp_g = g_score[x][y] + Grid[x - 1][y]
            temp_f = temp_g + heuristic(x - 1, y)
            if temp_f < f_score[x - 1][y]:
                came_from[(x - 1, y)] = (x, y)
                g_score[x - 1][y] = temp_g
                f_score[x - 1][y] = temp_f
                pq.put((temp_f, x - 1, y))

        # Down
        if x < N - 1 and Grid[x + 1][y] != -1:
            temp_g = g_score[x][y] + Grid[x + 1][y]
            temp_f = temp_g + heuristic(x + 1, y)
            if temp_f < f_score[x + 1][y]:
                came_from[(x + 1, y)] = (x, y)
                g_score[x + 1][y] = temp_g
                f_score[x + 1][y] = temp_f
                pq.put((temp_f, x + 1, y))

        # Left
        if y > 0 and Grid[x][y - 1] != -1:
            temp_g = g_score[x][y] + Grid[x][y - 1]
            temp_f = temp_g + heuristic(x, y - 1)
            if temp_f < f_score[x][y - 1]:
                came_from[(x, y - 1)] = (x, y)
                g_score[x][y - 1] = temp_g
                f_score[x][y - 1] = temp_f
                pq.put((temp_f, x, y - 1))

        # Right
        if y < N - 1 and Grid[x][y + 1] != -1:
            temp_g = g_score[x][y] + Grid[x][y + 1]
            temp_f = temp_g + heuristic(x, y + 1)
            if temp_f < f_score[x][y + 1]:
                came_from[(x, y + 1)] = (x, y)
                g_score[x][y + 1] = temp_g
                f_score[x][y + 1] = temp_f
                pq.put((temp_f, x, y + 1))

        # Up-Left (Diagonal)
        if x > 0 and y > 0 and Grid[x - 1][y - 1] != -1:
            temp_g = g_score[x][y] + Grid[x - 1][y - 1]
            temp_f = temp_g + heuristic(x - 1, y - 1)
            if temp_f < f_score[x - 1][y - 1]:
                came_from[(x - 1, y - 1)] = (x, y)
                g_score[x - 1][y - 1] = temp_g
                f_score[x - 1][y - 1] = temp_f
                pq.put((temp_f, x - 1, y - 1))

        # Up-Right (Diagonal)
        if x > 0 and y < N - 1 and Grid[x - 1][y + 1] != -1:
            temp_g = g_score[x][y] + Grid[x - 1][y + 1]
            temp_f = temp_g + heuristic(x - 1, y + 1)
            if temp_f < f_score[x - 1][y + 1]:
                came_from[(x - 1, y + 1)] = (x, y)
                g_score[x - 1][y + 1] = temp_g
                f_score[x - 1][y + 1] = temp_f
                pq.put((temp_f, x - 1, y + 1))

        # Down-Left (Diagonal)
        if x < N - 1 and y > 0 and Grid[x + 1][y - 1] != -1:
            temp_g = g_score[x][y] + Grid[x + 1][y - 1]
            temp_f = temp_g + heuristic(x + 1, y - 1)
            if temp_f < f_score[x + 1][y - 1]:
                came_from[(x + 1, y - 1)] = (x, y)
                g_score[x + 1][y - 1] = temp_g
                f_score[x + 1][y - 1] = temp_f
                pq.put((temp_f, x + 1, y - 1))

        # Down-Right (Diagonal)
        if x < N - 1 and y < N - 1 and Grid[x + 1][y + 1] != -1:
            temp_g = g_score[x][y] + Grid[x + 1][y + 1]
            temp_f = temp_g + heuristic(x + 1, y + 1)
            if temp_f < f_score[x + 1][y + 1]:
                came_from[(x + 1, y + 1)] = (x, y)
                g_score[x + 1][y + 1] = temp_g
                f_score[x + 1][y + 1] = temp_f
                pq.put((temp_f, x + 1, y + 1))

    return float("inf"), []


Grid = [[2, 3, 1, -1], [1, -1, 4, 2], [1, 2, 3, 1], [3, -1, 2, 1]]
Sx2, Sy2 = (0, 0)
Dx2, Dy2 = (3, 3)


cost, path = a_star_search(Grid, Sx2, Sy2, Dx2, Dy2)
print(f"Optimal Cost: {cost}")
print(f"Optimal Path: {' → '.join(str(p) for p in path)}")
