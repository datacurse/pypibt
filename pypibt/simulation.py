from dataclasses import dataclass
from enum import Enum, auto

import numpy as np

from .dist_table import DistTable, _UNREACHED
from .mapf_utils import Config, Coord, Grid
from .pibt import PIBT


class AgentState(Enum):
    IDLE = auto()
    MOVING_TO_PICKUP = auto()
    MOVING_TO_DELIVERY = auto()


@dataclass
class Task:
    task_id: int
    pickup: Coord
    delivery: Coord
    created_at: int
    assigned_to: int | None = None
    picked_up_at: int | None = None
    delivered_at: int | None = None


@dataclass
class AgentInfo:
    agent_id: int
    state: AgentState = AgentState.IDLE
    current_task: Task | None = None


class MAPDSimulation:
    """Multi-Agent Pickup and Delivery simulation using EPIBT.

    Agents are assigned pickup-delivery tasks dynamically. Tasks arrive
    according to a Poisson process. Idle agents are greedily matched to
    pending tasks using actual BFS distances (via DistTable) rather than
    Manhattan distance, which accounts for obstacles.
    """

    def __init__(
        self,
        grid: Grid,
        num_agents: int,
        pickup_locations: list[Coord],
        delivery_locations: list[Coord],
        task_frequency: float = 0.3,
        seed: int = 0,
    ):
        self.grid = grid
        self.rng = np.random.default_rng(seed)
        self.pickup_locations = pickup_locations
        self.delivery_locations = delivery_locations
        self.task_frequency = task_frequency

        # Place agents at random walkable positions (not on stations)
        walkable = [(int(y), int(x)) for y, x in zip(*np.where(grid))]
        reserved = set(pickup_locations + delivery_locations)
        available = [c for c in walkable if c not in reserved]
        start_indices = self.rng.choice(len(available), size=num_agents, replace=False)
        starts: Config = [available[int(i)] for i in start_indices]

        # Initial goals = current positions (all idle)
        goals: Config = list(starts)

        # Init PIBT solver
        self.pibt = PIBT(grid, starts, goals, seed=seed)

        # Agent info
        self.agents = [AgentInfo(i) for i in range(num_agents)]

        # Priorities
        self.priorities: list[float] = [0.0] * num_agents

        # Task tracking
        self.task_counter = 0
        self.pending_tasks: list[Task] = []
        self.active_tasks: list[Task] = []
        self.completed_tasks: list[Task] = []

        # Pre-compute dist tables for pickup locations (for task assignment)
        # Maps pickup coord → DistTable (so BFS from that pickup is reusable)
        self._pickup_dist_tables: dict[Coord, DistTable] = {}
        for loc in pickup_locations:
            if loc not in self._pickup_dist_tables:
                self._pickup_dist_tables[loc] = DistTable(grid, loc)

        # State
        self.current_config: Config = list(starts)
        self.timestep: int = 0

    def _generate_tasks(self) -> None:
        """Generate new tasks according to Poisson arrival process."""
        num_new = int(self.rng.poisson(self.task_frequency))
        for _ in range(num_new):
            pickup = self.pickup_locations[
                int(self.rng.integers(len(self.pickup_locations)))
            ]
            delivery = self.delivery_locations[
                int(self.rng.integers(len(self.delivery_locations)))
            ]
            task = Task(
                task_id=self.task_counter,
                pickup=pickup,
                delivery=delivery,
                created_at=self.timestep,
            )
            self.task_counter += 1
            self.pending_tasks.append(task)

    def _check_arrivals(self) -> None:
        """Check if any agents have reached their current goal (pickup or delivery)."""
        for agent in self.agents:
            if agent.current_task is None:
                continue
            pos = self.current_config[agent.agent_id]
            task = agent.current_task

            if agent.state == AgentState.MOVING_TO_PICKUP and pos == task.pickup:
                # Picked up → change goal to delivery
                task.picked_up_at = self.timestep
                agent.state = AgentState.MOVING_TO_DELIVERY
                self.pibt.update_goal(agent.agent_id, task.delivery)

            elif agent.state == AgentState.MOVING_TO_DELIVERY and pos == task.delivery:
                # Delivered → agent becomes idle
                task.delivered_at = self.timestep
                self.active_tasks.remove(task)
                self.completed_tasks.append(task)
                agent.state = AgentState.IDLE
                agent.current_task = None
                # Goal = current position (stay put until next assignment)
                self.pibt.update_goal(agent.agent_id, pos)

    def _assign_tasks(self) -> None:
        """Greedy task assignment using actual BFS distances.

        Uses pre-computed DistTables from pickup locations so that
        distances account for obstacles in the grid (unlike Manhattan).
        """
        idle_agents = [a for a in self.agents if a.state == AgentState.IDLE]
        assigned: list[Task] = []

        for task in self.pending_tasks:
            if not idle_agents:
                break

            # Use BFS distance from the task's pickup to each idle agent
            pickup_dt = self._pickup_dist_tables[task.pickup]

            best = min(
                idle_agents,
                key=lambda a: pickup_dt.get(self.current_config[a.agent_id]),
            )

            # Check that the agent can actually reach the pickup
            dist = pickup_dt.get(self.current_config[best.agent_id])
            if dist == _UNREACHED:
                continue  # no reachable idle agent for this task

            task.assigned_to = best.agent_id
            best.state = AgentState.MOVING_TO_PICKUP
            best.current_task = task
            self.active_tasks.append(task)
            idle_agents.remove(best)
            assigned.append(task)
            self.pibt.update_goal(best.agent_id, task.pickup)

        for task in assigned:
            self.pending_tasks.remove(task)

    def tick(self) -> Config:
        """Advance the simulation by one timestep.

        Returns the new configuration (positions) of all agents.
        """
        self.timestep += 1

        self._generate_tasks()
        self._check_arrivals()
        self._assign_tasks()

        # Update priorities: agents not at goal get increasing priority
        for i in range(len(self.agents)):
            if self.current_config[i] != self.pibt.goals[i]:
                self.priorities[i] += 1
            else:
                self.priorities[i] -= np.floor(self.priorities[i])

        # PIBT step
        new_config = self.pibt.step(self.current_config, self.priorities)
        self.current_config = new_config
        return new_config