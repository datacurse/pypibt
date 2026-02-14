from itertools import product

import numpy as np

from .dist_table import DistTable, _UNREACHED
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


def _net_rotation(op: tuple[str, ...]) -> int:
    """Net rotation of an operation in quarter-turns (mod 4). R=+1, C=-1."""
    rot = 0
    for a in op:
        if a == "R":
            rot += 1
        elif a == "C":
            rot -= 1
    return rot % 4


def _generate_operations(op_len: int) -> list[Operation]:
    """Generate pruned operations of given length.

    Pruning rules (per the paper, Section "Multi-action Operations"):
    1. Remove consecutive R-C or C-R pairs (they cancel out)
    2. Remove consecutive same-direction rotations that are redundant:
       - RRR ≡ C (3 quarter-turns CW = 1 quarter-turn CCW)
       - CCC ≡ R
       - RRRR, CCCC ≡ identity
       More generally: any prefix with net rotation magnitude > 2 is wasteful,
       since it could be achieved with fewer rotation actions.
    3. Remove ops ending in R or C (merge with W-ending equivalent,
       since final orientation doesn't change the cell path)
    """
    result: list[Operation] = []
    for op in product("FRCW", repeat=op_len):
        # Rule 3: skip ops ending with rotation
        if op[-1] in ("R", "C"):
            continue

        bad = False
        rot_count = 0  # running count of consecutive rotations
        net_rot = 0    # net quarter-turns in current rotation sequence

        for i in range(len(op)):
            a = op[i]

            if a in ("R", "C"):
                rot_count += 1
                net_rot += 1 if a == "R" else -1

                # Rule 1: adjacent R-C or C-R cancel
                if i > 0:
                    prev = op[i - 1]
                    if (prev == "R" and a == "C") or (prev == "C" and a == "R"):
                        bad = True
                        break

                # Rule 2: redundant rotation sequences
                # If we've used more rotation actions than needed for the net effect,
                # this sequence is suboptimal (e.g., RRR when C suffices)
                abs_net = abs(net_rot)
                # The minimum actions needed for this net rotation is min(abs_net, 4 - abs_net)
                if rot_count > min(abs_net % 4, 4 - (abs_net % 4)):
                    # Exception: rot_count == 0 net_rot == 0 means full circle — always redundant
                    if abs_net % 4 == 0 and rot_count > 0:
                        bad = True
                        break
                    elif abs_net % 4 != 0 and rot_count > min(abs_net % 4, 4 - (abs_net % 4)):
                        bad = True
                        break
            else:
                # Reset rotation tracking after a non-rotation action
                rot_count = 0
                net_rot = 0

        if bad:
            continue
        result.append(op)
    return result


def _compute_cell_path(
    grid: Grid, coord: Coord, orientation: Orientation, operation: Operation
) -> tuple[tuple[Coord, ...], int] | None:
    """Compute cell sequence and final orientation for an operation.

    Returns (cell_path, final_orientation) or None if any forward move
    goes out of bounds or hits an obstacle.
    cell_path has len(operation)+1 entries: positions at t=0 through t=len(operation).
    """
    cells: list[Coord] = [coord]
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
        # "W" does nothing
        cells.append(cur)
    return tuple(cells), ori


class PIBT:
    """EPIBT solver with rotation action model.

    Implements Enhanced PIBT (Algorithm 1 + Algorithm 2 from the paper) with:
    - Multi-action operations (length op_len, default 3)
    - Agent revisiting (up to max_revisits per timestep)
    - Operation inheritance from previous timestep

    Key design decisions matching the paper:
    - Operations that collide with >1 agent are skipped (Alg 2, line 10)
    - Tie-breaking: forward > rotation > wait (FRW order)
    - Failed agents fall back to inherited operations
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
        self.goals = list(goals)
        self.N = len(self.starts)
        self.op_len = op_len
        self.max_revisits = max_revisits

        self.dist_tables = [DistTable(grid, goal) for goal in goals]

        self.NIL = self.N  # sentinel for "no agent"
        self.rng = np.random.default_rng(seed)

        # Agent orientations (persists across steps)
        self.orientations: list[int] = [
            int(self.rng.integers(4)) for _ in range(self.N)
        ]

        # Inherited operations from previous step (initially all-wait)
        wait_op: Operation = ("W",) * op_len
        self.inherited_ops: list[Operation] = [wait_op] * self.N

        # 3D reservation table: reserved[t][y][x] = agent_id or NIL
        self.reserved = np.full((op_len + 1, *grid.shape), self.NIL, dtype=np.int32)

        # Pre-generate valid operations
        self.all_operations = _generate_operations(op_len)

        # Per-step mutable state (set in step())
        self._visit_count: list[int] = []
        self._hit: list[int] = []
        self._agent_ops: list[Operation] = []
        self._agent_paths: list[tuple[Coord, ...]] = []
        self._Q_from: Config = []
        self._priorities: list[float] = []

    # ------------------------------------------------------------------
    # Reservation table helpers
    # ------------------------------------------------------------------

    def _reserve_path(self, agent_id: int, path: tuple[Coord, ...]) -> None:
        for t, cell in enumerate(path):
            self.reserved[t][cell] = agent_id

    def _unreserve_path(self, agent_id: int, path: tuple[Coord, ...]) -> None:
        for t, cell in enumerate(path):
            if self.reserved[t][cell] == agent_id:
                self.reserved[t][cell] = self.NIL

    def _get_conflicts(self, agent_id: int, path: tuple[Coord, ...]) -> set[int]:
        """Find agents whose reserved paths conflict with the given path.

        Checks both vertex conflicts (same cell at same time) and
        edge conflicts (two agents swapping cells between t-1 and t).
        """
        conflicts: set[int] = set()
        for t, cell in enumerate(path):
            # Vertex conflict
            occ = int(self.reserved[t][cell])
            if occ != self.NIL and occ != agent_id:
                conflicts.add(occ)

            # Edge conflict: if agent moves from path[t-1]→path[t],
            # check if someone at path[t] at t-1 moves to path[t-1] at t
            if t > 0 and path[t] != path[t - 1]:
                prev_occ = int(self.reserved[t - 1][cell])
                if (
                    prev_occ != self.NIL
                    and prev_occ != agent_id
                    and int(self.reserved[t][path[t - 1]]) == prev_occ
                ):
                    conflicts.add(prev_occ)
        return conflicts

    # ------------------------------------------------------------------
    # Operation candidate generation and sorting
    # ------------------------------------------------------------------

    def _get_sorted_candidates(
        self, agent_id: int, coord: Coord, orientation: int
    ) -> list[tuple[Operation, tuple[Coord, ...], int]]:
        """Generate, deduplicate by cell path, and sort candidate operations.

        Returns list of (operation, cell_path, final_orientation) sorted by
        weight = h_value * alpha + beta, with:
        - alpha large enough to make h dominant
        - beta: FRW tie-breaking (F=0, R/C=1, W=2 per action)

        Deduplication: operations with the same cell sequence are merged,
        keeping the one with the best final orientation (lowest h after rotation)
        and then best beta.
        """
        alpha = self.grid.size * 10

        # Collect all valid operations with their metrics
        raw: list[tuple[float, Operation, tuple[Coord, ...], int]] = []
        for op in self.all_operations:
            result = _compute_cell_path(self.grid, coord, orientation, op)
            if result is None:
                continue
            cell_path, final_ori = result

            h = self.dist_tables[agent_id].get(cell_path[-1])
            if h == _UNREACHED:
                continue  # skip unreachable destinations

            beta = sum(2 if a == "W" else (1 if a in ("R", "C") else 0) for a in op)
            weight = h * alpha + beta
            raw.append((weight, op, cell_path, final_ori))

        # Deduplicate by cell path: keep the op with the lowest weight
        # (which accounts for both h-value and FRW tie-breaking)
        best_by_path: dict[tuple[Coord, ...], tuple[float, Operation, int]] = {}
        for weight, op, cell_path, final_ori in raw:
            if cell_path not in best_by_path or weight < best_by_path[cell_path][0]:
                best_by_path[cell_path] = (weight, op, final_ori)

        deduped = [
            (weight, op, path, ori)
            for path, (weight, op, ori) in best_by_path.items()
        ]

        # Shuffle first for random tie-breaking among equal weights, then sort
        self.rng.shuffle(deduped)
        deduped.sort(key=lambda x: x[0])

        return [(op, path, ori) for _, op, path, ori in deduped]

    # ------------------------------------------------------------------
    # Core EPIBT recursive selection (Algorithm 2)
    # ------------------------------------------------------------------

    def _epibt_select(self, k: int, p: float) -> bool:
        """Try to select an operation for agent k with inherited priority p.

        Returns True on success (collision-free operation found or pushed
        conflicting agent successfully).
        """
        coord = self._Q_from[k]
        orientation = self.orientations[k]

        candidates = self._get_sorted_candidates(k, coord, orientation)

        self._visit_count[k] += 1
        self._hit[k] = 1  # prevent re-entry in same recursion branch

        for op, cell_path, final_ori in candidates:
            conflicts = self._get_conflicts(k, cell_path)

            # No conflict → adopt this operation (Alg 2, lines 6-9)
            if len(conflicts) == 0:
                self._agent_ops[k] = op
                self._agent_paths[k] = cell_path
                self._reserve_path(k, cell_path)
                self._hit[k] = 0
                return True

            # Multi-agent collision → skip (Alg 2, line 10)
            if len(conflicts) > 1:
                continue

            # Single conflict → try to push agent l
            l = next(iter(conflicts))

            # Skip if l is in current recursion branch, exceeded revisit limit,
            # or has higher/equal priority (Alg 2, line 12)
            if (
                self._hit[l] == 1
                or self._visit_count[l] >= self.max_revisits
                or self._priorities[l] >= p
            ):
                continue

            # Save l's current state
            old_op_l = self._agent_ops[l]
            old_path_l = self._agent_paths[l]

            # Tentatively: remove l's reservation, adopt k's operation (Alg 2, lines 14-15)
            self._unreserve_path(l, old_path_l)
            self._agent_ops[k] = op
            self._agent_paths[k] = cell_path
            self._reserve_path(k, cell_path)

            # Recursively ask l to find a new operation (Alg 2, line 16)
            if self._epibt_select(l, p):
                self._hit[k] = 0
                return True

            # Failed — restore everything (Alg 2, line 18)
            self._unreserve_path(k, cell_path)
            self._agent_ops[l] = old_op_l
            self._agent_paths[l] = old_path_l
            self._reserve_path(l, old_path_l)

        # All operations failed — fall back to inherited (Alg 2, line 19)
        self._agent_ops[k] = self.inherited_ops[k]
        inh_result = _compute_cell_path(
            self.grid, coord, orientation, self.inherited_ops[k]
        )
        if inh_result is not None:
            self._agent_paths[k] = inh_result[0]
        else:
            # Inherited op is invalid (e.g., orientation changed) → stay in place
            self._agent_paths[k] = tuple([coord] * (self.op_len + 1))
        self._hit[k] = 0
        return False

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def update_goal(self, agent_id: int, new_goal: Coord) -> None:
        """Update goal for a single agent (for lifelong/MAPD settings)."""
        self.goals[agent_id] = new_goal
        self.dist_tables[agent_id] = DistTable(self.grid, new_goal)

    def step(self, Q_from: Config, priorities: list[float]) -> Config:
        """Execute one EPIBT timestep (Algorithm 1).

        Args:
            Q_from: current positions of all agents
            priorities: priority values (higher = more important)

        Returns:
            Q_to: next positions of all agents after executing the first
                  action of each agent's chosen operation
        """
        N = self.N
        self._Q_from = Q_from
        self._priorities = priorities
        self._visit_count = [0] * N
        self._hit = [0] * N
        self._agent_ops = list(self.inherited_ops)
        self._agent_paths = [tuple()] * N

        # --- Initialize reservation table with inherited operations (Alg 1, lines 2-4) ---
        self.reserved[:] = self.NIL
        for i in range(N):
            result = _compute_cell_path(
                self.grid, Q_from[i], self.orientations[i], self.inherited_ops[i]
            )
            if result is not None:
                self._agent_paths[i] = result[0]
            else:
                # Inherited op is no longer valid → stay in place
                self._agent_paths[i] = tuple([Q_from[i]] * (self.op_len + 1))
                self._agent_ops[i] = ("W",) * self.op_len
            self._reserve_path(i, self._agent_paths[i])

        # --- Sort agents by priority descending (Alg 1, lines 5-6) ---
        agent_order = sorted(range(N), key=lambda i: priorities[i], reverse=True)

        # --- Main loop: each unvisited agent tries to select an operation (Alg 1, lines 7-11) ---
        for k in agent_order:
            if self._visit_count[k] != 0:
                continue

            # Remove k's inherited reservation before re-selecting
            self._unreserve_path(k, self._agent_paths[k])

            if not self._epibt_select(k, priorities[k]):
                # Failed — inherited was set inside _epibt_select; reserve it
                self._reserve_path(k, self._agent_paths[k])

        # --- Execute first action of each agent's chosen operation ---
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
            # "W" → no change

            Q_to.append(pos)
            self.orientations[i] = ori

            # Inherit: remove executed first action, append W to keep length
            new_inherited.append(op[1:] + ("W",))

        self.inherited_ops = new_inherited
        return Q_to

    def run(self, max_timestep: int = 1000) -> Configs:
        """Run EPIBT for one-shot MAPF until all agents reach goals or timeout."""
        priorities: list[float] = []
        for i in range(self.N):
            d = self.dist_tables[i].get(self.starts[i])
            priorities.append(d / self.grid.size if d != _UNREACHED else 0.0)

        configs = [self.starts]
        while len(configs) <= max_timestep:
            Q = self.step(configs[-1], priorities)
            configs.append(Q)

            all_done = True
            for i in range(self.N):
                if Q[i] != self.goals[i]:
                    all_done = False
                    priorities[i] += 1
                else:
                    priorities[i] -= np.floor(priorities[i])
            if all_done:
                break

        return configs