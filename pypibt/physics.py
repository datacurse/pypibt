import math

from .mapf_utils import Config


class PhysicsLayer:
    def __init__(
        self,
        initial_config: Config,
        cell_size_m: float = 1.0,
        speed: float = 1.0,
    ):
        self.cell_size_m = cell_size_m
        self.speed = speed

        # convert grid coords to metric centers
        self.positions: list[tuple[float, float]] = [
            self._cell_to_meters(r, c) for r, c in initial_config
        ]
        self.targets: list[tuple[float, float]] = list(self.positions)

    def _cell_to_meters(self, r: int, c: int) -> tuple[float, float]:
        return (
            r * self.cell_size_m + self.cell_size_m / 2,
            c * self.cell_size_m + self.cell_size_m / 2,
        )

    def set_targets(self, config: Config) -> None:
        self.targets = [self._cell_to_meters(r, c) for r, c in config]

    def snap_to_targets(self) -> None:
        self.positions = list(self.targets)

    def update(self, dt: float) -> None:
        max_dist = self.speed * dt
        for i, ((py, px), (ty, tx)) in enumerate(
            zip(self.positions, self.targets)
        ):
            dy = ty - py
            dx = tx - px
            dist = math.hypot(dy, dx)
            if dist <= max_dist:
                self.positions[i] = (ty, tx)
            else:
                ratio = max_dist / dist
                self.positions[i] = (py + dy * ratio, px + dx * ratio)

    def all_settled(self) -> bool:
        eps = 1e-6
        for (py, px), (ty, tx) in zip(self.positions, self.targets):
            if abs(py - ty) > eps or abs(px - tx) > eps:
                return False
        return True
