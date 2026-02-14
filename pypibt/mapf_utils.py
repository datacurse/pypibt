from typing import TypeAlias

import numpy as np

# y, x
Grid: TypeAlias = np.ndarray
Coord: TypeAlias = tuple[int, int]
Config: TypeAlias = list[Coord]
Configs: TypeAlias = list[Config]

# Orientation: 0=North, 1=East, 2=South, 3=West
# rotate CW: (o + 1) % 4, rotate CCW: (o - 1) % 4
Orientation: TypeAlias = int
Operation: TypeAlias = tuple[str, ...]  # e.g. ("R", "W", "F")

# Direction vectors (dy, dx) for each orientation, matching (y, x) coords
DIRECTION_VECTORS: list[Coord] = [
    (-1, 0),  # 0: North (up)
    (0, 1),   # 1: East (right)
    (1, 0),   # 2: South (down)
    (0, -1),  # 3: West (left)
]


def get_grid(height: int, width: int, obstacles: list[Coord] | None = None) -> Grid:
    """Create a grid where True = traversable, False = obstacle."""
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
            corners: tuple[Coord, Coord] = item  # type: ignore[assignment]
            (y1, x1), (y2, x2) = corners
            for y in range(y1, y2 + 1):
                for x in range(x1, x2 + 1):
                    result.append((y, x))
        else:
            result.append(item)  # type: ignore[arg-type]
    return result


def is_valid_coord(grid: Grid, coord: Coord) -> bool:
    y, x = coord
    return 0 <= y < grid.shape[0] and 0 <= x < grid.shape[1] and bool(grid[y, x])


def get_neighbors(grid: Grid, coord: Coord) -> list[Coord]:
    """Return traversable 4-connected neighbors of coord."""
    if not is_valid_coord(grid, coord):
        return []

    y, x = coord
    neigh: list[Coord] = []
    for dy, dx in DIRECTION_VECTORS:
        ny, nx = y + dy, x + dx
        if 0 <= ny < grid.shape[0] and 0 <= nx < grid.shape[1] and grid[ny, nx]:
            neigh.append((ny, nx))
    return neigh