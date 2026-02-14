from collections import deque
from dataclasses import dataclass, field

import numpy as np

from .mapf_utils import Coord, Grid, get_neighbors, is_valid_coord

_UNREACHED: int = np.iinfo(np.int32).max  # safe sentinel — no real distance hits this


@dataclass
class DistTable:
    """Lazy BFS distance table from a goal cell.

    Distances are computed on demand: calling get(target) runs BFS only
    as far as needed to reach target, then caches the result.
    """

    grid: Grid
    goal: Coord
    queue: deque[Coord] = field(init=False)
    table: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.queue = deque([self.goal])
        self.table = np.full(self.grid.shape, _UNREACHED, dtype=np.int32)
        self.table[self.goal] = 0

    def get(self, target: Coord) -> int:
        """Return shortest-path distance from target to self.goal.

        Returns _UNREACHED if target is invalid or unreachable.
        """
        if not is_valid_coord(self.grid, target):
            return _UNREACHED

        # Already computed
        if self.table[target] != _UNREACHED:
            return int(self.table[target])

        # Continue BFS until we reach target or exhaust the queue
        while self.queue:
            u = self.queue.popleft()
            d = int(self.table[u])
            for v in get_neighbors(self.grid, u):
                if self.table[v] == _UNREACHED:
                    self.table[v] = d + 1
                    self.queue.append(v)
            if u == target:
                return d

        return _UNREACHED