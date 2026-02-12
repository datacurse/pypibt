from typing import TypeAlias

import numpy as np

# y, x
Grid: TypeAlias = np.ndarray
Coord: TypeAlias = tuple[int, int]
Config: TypeAlias = list[Coord]
Configs: TypeAlias = list[Config]


def get_grid(height: int, width: int, obstacles: list[Coord] | None = None) -> Grid:
    # grid[y, x] -> True: available, False: obstacle
    grid = np.ones((height, width), dtype=bool)
    if obstacles:
        for y, x in obstacles:
            grid[y, x] = False
    return grid


def expand_areas(areas: list[Coord | tuple[Coord, Coord]]) -> list[Coord]:
    """Expand a mix of single coordinates and rectangular areas into a flat coordinate list.

    Each element is either:
    - A single Coord: (y, x) -- passed through as-is
    - A rectangular area: ((y1, x1), (y2, x2)) -- expanded to all cells in the
      rectangle (inclusive on both corners)
    """
    result: list[Coord] = []
    for item in areas:
        if isinstance(item[0], tuple):
            (y1, x1), (y2, x2) = item  # type: ignore[misc]
            for y in range(y1, y2 + 1):
                for x in range(x1, x2 + 1):
                    result.append((y, x))
        else:
            result.append(item)  # type: ignore[arg-type]
    return result


def is_valid_coord(grid: Grid, coord: Coord) -> bool:
    y, x = coord
    if y < 0 or y >= grid.shape[0] or x < 0 or x >= grid.shape[1] or not grid[coord]:
        return False
    return True


def get_neighbors(grid: Grid, coord: Coord) -> list[Coord]:
    # coord: y, x
    neigh: list[Coord] = []

    # check valid input
    if not is_valid_coord(grid, coord):
        return neigh

    y, x = coord

    if x > 0 and grid[y, x - 1]:
        neigh.append((y, x - 1))

    if x < grid.shape[1] - 1 and grid[y, x + 1]:
        neigh.append((y, x + 1))

    if y > 0 and grid[y - 1, x]:
        neigh.append((y - 1, x))

    if y < grid.shape[0] - 1 and grid[y + 1, x]:
        neigh.append((y + 1, x))

    return neigh
