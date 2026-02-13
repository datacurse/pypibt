from itertools import product

import numpy as np

from .dist_table import DistTable
from .mapf_utils import (
    DIRECTION_VECTORS,
    Config,
    Configs,
    Coord,
    Grid,
    Operation,
    Orientation,
    is_valid_coord,
)


def _generate_operations(op_len: int) -> list[Operation]:
    """Generate pruned operations of given length.

    Pruning rules:
    - Remove consecutive R-C or C-R (they cancel out)
    - Remove ops ending in R or C (merge with W-ending equivalent)
    """
    result: list[Operation] = []
    for op in product("FRCW", repeat=op_len):
        # Check consecutive R-C / C-R
        bad = False
        for i in range(len(op) - 1):
            if (op[i] == "R" and op[i + 1] == "C") or (
                op[i] == "C" and op[i + 1] == "R"
            ):
                bad = True
                break
        if bad:
            continue
        # Remove ops ending in R or C
        if op[-1] in ("R", "C"):
            continue
        result.append(op)
    return result


def _compute_cell_path(
    grid: Grid, coord: Coord, orientation: Orientation, operation: Operation
) -> tuple[tuple[Coord, ...], int] | None:
    """Compute cell sequence and final orientation for an operation.

    Returns (cell_path, final_orientation) or None if any forward hits obstacle.
    cell_path has op_len+1 entries: positions at t=0 through t=op_len.
    """
    cells = [coord]
    cur = coord
    ori = orientation
    for action in operation:
        if action == "F":
            dy, dx = DIRECTION_VECTORS[ori]
            nxt = (cur[0] + dy, cur[1] + dx)
            if not is_valid_coord(grid, nxt):
                return None
            cur = nxt
        elif action == "R":
            ori = (ori + 1) % 4
        elif action == "C":
            ori = (ori - 1) % 4
        cells.append(cur)
    return tuple(cells), ori


class PIBT:
    """EPIBT solver with rotation action model.

    Implements Enhanced PIBT with:
    - Multi-action operations (length op_len, default 3)
    - Agent revisiting (up to max_revisits per timestep)
    - Operation inheritance from previous timestep
    """

    def __init__(
        self,
        grid: Grid,
        starts: Config,
        goals: Config,
        seed: int = 0,
        op_len: int = 3,
        max_revisits: int = 10,
    ):
        self.grid = grid
        self.starts = starts
        self.goals = goals
        self.N = len(self.starts)
        self.op_len = op_len
        self.max_revisits = max_revisits

        self.dist_tables = [DistTable(grid, goal) for goal in goals]

        self.NIL = self.N
        self.rng = np.random.default_rng(seed)

        # Agent orientations (persists across steps)
        self.orientations: list[int] = [
            int(self.rng.integers(4)) for _ in range(self.N)
        ]

        # Inherited operations from previous step (initially all-wait)
        wait_op: Operation = ("W",) * op_len
        self.inherited_ops: list[Operation] = [wait_op] * self.N

        # 3D reservation table: reserved[t][y][x] = agent_id or NIL
        self.reserved = np.full((op_len + 1, *grid.shape), self.NIL, dtype=int)

        # Pre-generate valid operations
        self.all_operations = _generate_operations(op_len)

        # Per-step mutable state (set in step())
        self._visit_count: list[int] = []
        self._hit: list[int] = []
        self._agent_ops: list[Operation] = []
        self._agent_paths: list[tuple[Coord, ...]] = []
        self._Q_from: Config = []
        self._priorities: list[float] = []

    def _reserve_path(self, agent_id: int, path: tuple[Coord, ...]) -> None:
        for t, cell in enumerate(path):
            self.reserved[t][cell] = agent_id

    def _unreserve_path(self, agent_id: int, path: tuple[Coord, ...]) -> None:
        for t, cell in enumerate(path):
            if self.reserved[t][cell] == agent_id:
                self.reserved[t][cell] = self.NIL

    def _get_conflicts(self, agent_id: int, path: tuple[Coord, ...]) -> set[int]:
        """Find agents whose reserved paths conflict with the given path."""
        conflicts: set[int] = set()
        for t, cell in enumerate(path):
            # Vertex conflict
            occ = int(self.reserved[t][cell])
            if occ != self.NIL and occ != agent_id:
                conflicts.add(occ)
            # Edge conflict: agent moves from path[t-1] to path[t],
            # check if someone at path[t] at t-1 moves to path[t-1] at t
            if t > 0 and path[t] != path[t - 1]:
                prev_occ = int(self.reserved[t - 1][cell])
                if prev_occ != self.NIL and prev_occ != agent_id:
                    if int(self.reserved[t][path[t - 1]]) == prev_occ:
                        conflicts.add(prev_occ)
        return conflicts

    def _get_sorted_candidates(
        self, agent_id: int, coord: Coord, orientation: int
    ) -> list[tuple[Operation, tuple[Coord, ...]]]:
        """Generate, deduplicate by cell path, and sort candidate operations."""
        alpha = self.grid.size * 10

        raw: list[tuple[float, Operation, tuple[Coord, ...]]] = []
        for op in self.all_operations:
            result = _compute_cell_path(self.grid, coord, orientation, op)
            if result is None:
                continue
            cell_path, _final_ori = result
            h = self.dist_tables[agent_id].get(cell_path[-1])
            # Tie-breaking: F=0, R/C=1, W=2
            beta = sum(2 if a == "W" else (1 if a in ("R", "C") else 0) for a in op)
            weight = h * alpha + beta
            raw.append((weight, op, cell_path))

        # Deduplicate by cell path (keep best weight per path)
        best_by_path: dict[tuple[Coord, ...], tuple[float, Operation]] = {}
        for weight, op, cell_path in raw:
            if cell_path not in best_by_path or weight < best_by_path[cell_path][0]:
                best_by_path[cell_path] = (weight, op)

        deduped = [(weight, op, path) for path, (weight, op) in best_by_path.items()]

        # Shuffle for random tie-breaking, then stable sort by weight
        self.rng.shuffle(deduped)
        deduped.sort(key=lambda x: x[0])

        return [(op, path) for _, op, path in deduped]

    def _epibt_select(self, k: int, p: float) -> bool:
        """Try to select an operation for agent k. Returns True on success."""
        coord = self._Q_from[k]
        orientation = self.orientations[k]

        candidates = self._get_sorted_candidates(k, coord, orientation)

        self._visit_count[k] += 1
        self._hit[k] = 1

        for op, cell_path in candidates:
            conflicts = self._get_conflicts(k, cell_path)

            if len(conflicts) == 0:
                self._agent_ops[k] = op
                self._agent_paths[k] = cell_path
                self._reserve_path(k, cell_path)
                self._hit[k] = 0
                return True

            if len(conflicts) > 1:
                continue

            l = next(iter(conflicts))

            if (
                self._hit[l] == 1
                or self._visit_count[l] >= self.max_revisits
                or self._priorities[l] >= p
            ):
                continue

            # Save l's current state
            old_op_l = self._agent_ops[l]
            old_path_l = self._agent_paths[l]

            # Remove l's path, add k's
            self._unreserve_path(l, old_path_l)
            self._agent_ops[k] = op
            self._agent_paths[k] = cell_path
            self._reserve_path(k, cell_path)

            # Recursively ask l to find a new operation
            if self._epibt_select(l, p):
                self._hit[k] = 0
                return True

            # Failed — restore l, remove k
            self._unreserve_path(k, cell_path)
            self._agent_ops[l] = old_op_l
            self._agent_paths[l] = old_path_l
            self._reserve_path(l, old_path_l)

        # All failed — fall back to inherited
        self._agent_ops[k] = self.inherited_ops[k]
        inh_result = _compute_cell_path(
            self.grid, coord, orientation, self.inherited_ops[k]
        )
        if inh_result is not None:
            self._agent_paths[k] = inh_result[0]
        else:
            self._agent_paths[k] = tuple([coord] * (self.op_len + 1))
        self._hit[k] = 0
        return False

    def update_goal(self, agent_id: int, new_goal: Coord) -> None:
        self.goals[agent_id] = new_goal
        self.dist_tables[agent_id] = DistTable(self.grid, new_goal)

    def step(self, Q_from: Config, priorities: list[float]) -> Config:
        N = self.N
        self._Q_from = Q_from
        self._priorities = priorities
        self._visit_count = [0] * N
        self._hit = [0] * N
        self._agent_ops = list(self.inherited_ops)
        self._agent_paths = [tuple()] * N

        # Clear and populate reservation table with inherited operations
        self.reserved[:] = self.NIL
        for i in range(N):
            result = _compute_cell_path(
                self.grid, Q_from[i], self.orientations[i], self.inherited_ops[i]
            )
            if result is not None:
                self._agent_paths[i] = result[0]
            else:
                # Fallback: stay in place
                self._agent_paths[i] = tuple([Q_from[i]] * (self.op_len + 1))
                self._agent_ops[i] = ("W",) * self.op_len
            self._reserve_path(i, self._agent_paths[i])

        # Sort agents by priority (descending)
        agent_order = sorted(range(N), key=lambda i: priorities[i], reverse=True)

        # Main EPIBT loop (Algorithm 1)
        for k in agent_order:
            if self._visit_count[k] != 0:
                continue

            # Remove k's current path before re-selecting
            self._unreserve_path(k, self._agent_paths[k])

            if not self._epibt_select(k, priorities[k]):
                # Failed — inherited was set inside _epibt_select, reserve it
                self._reserve_path(k, self._agent_paths[k])

        # Execute first action of each agent's operation
        Q_to: Config = []
        new_inherited: list[Operation] = []
        for i in range(N):
            op = self._agent_ops[i]
            first_action = op[0]
            pos = Q_from[i]
            ori = self.orientations[i]

            if first_action == "F":
                dy, dx = DIRECTION_VECTORS[ori]
                new_pos = (pos[0] + dy, pos[1] + dx)
                if is_valid_coord(self.grid, new_pos):
                    pos = new_pos
            elif first_action == "R":
                ori = (ori + 1) % 4
            elif first_action == "C":
                ori = (ori - 1) % 4

            Q_to.append(pos)
            self.orientations[i] = ori
            new_inherited.append(op[1:] + ("W",))

        self.inherited_ops = new_inherited
        return Q_to

    def run(self, max_timestep: int = 1000) -> Configs:
        priorities: list[float] = []
        for i in range(self.N):
            priorities.append(self.dist_tables[i].get(self.starts[i]) / self.grid.size)

        configs = [self.starts]
        while len(configs) <= max_timestep:
            Q = self.step(configs[-1], priorities)
            configs.append(Q)

            flg_fin = True
            for i in range(self.N):
                if Q[i] != self.goals[i]:
                    flg_fin = False
                    priorities[i] += 1
                else:
                    priorities[i] -= np.floor(priorities[i])
            if flg_fin:
                break

        return configs
