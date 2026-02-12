from pypibt import get_grid
from pypibt.simulation import MAPDSimulation
from pypibt.visualizer import run_visualizer

if __name__ == "__main__":
    # 20x20 warehouse grid with shelf-like obstacle blocks and aisles
    height, width = 20, 20
    obstacles = [
        # shelf block rows (2 rows tall, 3 cols wide), spaced by aisles
        (3, 3), (3, 4), (3, 5),   (3, 8), (3, 9), (3, 10),   (3, 13), (3, 14), (3, 15),
        (4, 3), (4, 4), (4, 5),   (4, 8), (4, 9), (4, 10),   (4, 13), (4, 14), (4, 15),
        (7, 3), (7, 4), (7, 5),   (7, 8), (7, 9), (7, 10),   (7, 13), (7, 14), (7, 15),
        (8, 3), (8, 4), (8, 5),   (8, 8), (8, 9), (8, 10),   (8, 13), (8, 14), (8, 15),
        (11, 3), (11, 4), (11, 5), (11, 8), (11, 9), (11, 10), (11, 13), (11, 14), (11, 15),
        (12, 3), (12, 4), (12, 5), (12, 8), (12, 9), (12, 10), (12, 13), (12, 14), (12, 15),
        (15, 3), (15, 4), (15, 5), (15, 8), (15, 9), (15, 10), (15, 13), (15, 14), (15, 15),
        (16, 3), (16, 4), (16, 5), (16, 8), (16, 9), (16, 10), (16, 13), (16, 14), (16, 15),
    ]

    # pickup stations along left wall (incoming goods)
    pickup_locations = [(1, 0), (5, 0), (9, 0), (13, 0), (17, 0)]

    # delivery stations along right wall (outgoing goods)
    delivery_locations = [(1, 19), (5, 19), (9, 19), (13, 19), (17, 19)]

    grid = get_grid(height, width, obstacles)

    sim = MAPDSimulation(
        grid=grid,
        num_agents=8,
        pickup_locations=pickup_locations,
        delivery_locations=delivery_locations,
        task_frequency=0.2,
        seed=42,
    )

    run_visualizer(sim, cell_size=32, tick_ms=400)
