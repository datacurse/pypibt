import math

from .mapf_utils import Config


class PhysicsLayer:
    def __init__(
        self,
        initial_config: Config,
        cell_size_m: float = 1.0,
        speed: float = 1.0,
        initial_orientations: list[int] | None = None,
    ):
        self.cell_size_m = cell_size_m
        self.speed = speed

        # convert grid coords to metric centers
        self.positions: list[tuple[float, float]] = [
            self._cell_to_meters(r, c) for r, c in initial_config
        ]
        self.targets: list[tuple[float, float]] = list(self.positions)

        # orientation angles in radians (screen coords: 0=East/right, pi/2=South/down)
        # N=-pi/2, E=0, S=pi/2, W=pi
        n = len(initial_config)
        if initial_orientations is not None:
            self.angles: list[float] = [
                self._orient_to_angle(o) for o in initial_orientations
            ]
        else:
            self.angles = [self._orient_to_angle(2)] * n  # default: South
        self.target_angles: list[float] = list(self.angles)

    @staticmethod
    def _orient_to_angle(orientation: int) -> float:
        """Convert discrete orientation (0=N,1=E,2=S,3=W) to radians."""
        return [-(math.pi / 2), 0.0, math.pi / 2, math.pi][orientation]

    def _cell_to_meters(self, r: int, c: int) -> tuple[float, float]:
        return (
            r * self.cell_size_m + self.cell_size_m / 2,
            c * self.cell_size_m + self.cell_size_m / 2,
        )

    def set_targets(
        self, config: Config, orientations: list[int] | None = None
    ) -> None:
        self.targets = [self._cell_to_meters(r, c) for r, c in config]
        if orientations is not None:
            self.target_angles = [self._orient_to_angle(o) for o in orientations]

    def snap_to_targets(self) -> None:
        self.positions = list(self.targets)
        self.angles = list(self.target_angles)

    def update(self, dt: float) -> None:
        max_dist = self.speed * dt
        for i, ((py, px), (ty, tx)) in enumerate(zip(self.positions, self.targets)):
            dy = ty - py
            dx = tx - px
            dist = math.hypot(dy, dx)
            if dist <= max_dist:
                self.positions[i] = (ty, tx)
            else:
                ratio = max_dist / dist
                self.positions[i] = (py + dy * ratio, px + dx * ratio)

        # angle interpolation (shortest arc)
        max_rot = self.speed * math.pi * 2 * dt  # ~full turn per cell travel
        for i, (ca, ta) in enumerate(zip(self.angles, self.target_angles)):
            diff = (ta - ca + math.pi) % (2 * math.pi) - math.pi
            if abs(diff) <= max_rot:
                self.angles[i] = ta
            else:
                self.angles[i] = ca + math.copysign(max_rot, diff)

    def all_settled(self) -> bool:
        eps = 1e-6
        for (py, px), (ty, tx) in zip(self.positions, self.targets):
            if abs(py - ty) > eps or abs(px - tx) > eps:
                return False
        for ca, ta in zip(self.angles, self.target_angles):
            diff = abs((ta - ca + math.pi) % (2 * math.pi) - math.pi)
            if diff > eps:
                return False
        return True
