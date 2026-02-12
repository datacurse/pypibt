from pypibt import expand_areas, get_grid
from pypibt.simulation import MAPDSimulation
from pypibt.visualizer import run_visualizer

if __name__ == "__main__":
    # 20x20 warehouse grid with shelf-like obstacle blocks and aisles
    height, width = 20, 20
    # obstacles = expand_areas([
    #     ((3, 3), (4, 5)),    ((3, 8), (4, 10)),   ((3, 13), (4, 15)),
    #     ((7, 3), (8, 5)),    ((7, 8), (8, 10)),   ((7, 13), (8, 15)),
    #     ((11, 3), (12, 5)),  ((11, 8), (12, 10)), ((11, 13), (12, 15)),
    #     ((15, 3), (16, 5)),  ((15, 8), (16, 10)), ((15, 13), (16, 15)),
    # ])

    # pickup stations along left wall (incoming goods)
    pickup_locations = [(1, 0), (5, 0), (9, 0), (13, 0), (17, 0)]

    # delivery stations: shelf blocks (2 rows x 3 cols), spaced by aisles
    delivery_locations = expand_areas([
        ((2, 3), (2, 15)),
        ((5, 3), (5, 15)),
        ((8, 3), (8, 15)),
        ((11, 3), (11, 15)),
        ((14, 3), (14, 15)),
        ((17, 3), (17, 15)),
    ])

    grid = get_grid(height, width, obstacles=[])

    sim = MAPDSimulation(
        grid=grid,
        num_agents=100,
        pickup_locations=pickup_locations,
        delivery_locations=delivery_locations,
        task_frequency=2,
        seed=42,
    )

    run_visualizer(sim, cell_size_m=1.0, speed=1.0, pixels_per_meter=32.0)
