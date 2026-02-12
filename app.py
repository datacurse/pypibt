import os

from pypibt import (
    PIBT,
    get_grid,
    get_scenario,
    is_valid_mapf_solution,
    save_configs_for_visualizer,
)

if __name__ == "__main__":
    # grid parameters
    height = 32
    width = 32
    obstacles = [
        (0, 7), (0, 17), (0, 18), (0, 26), (1, 21), (1, 25), (1, 31),
        (2, 7), (2, 8), (2, 15), (2, 18), (2, 19), (2, 26), (3, 15),
        (3, 29), (4, 0), (4, 15), (4, 24), (5, 9), (5, 20), (5, 22),
        (6, 0), (6, 4), (6, 6), (7, 8), (7, 23), (7, 24), (8, 2),
        (8, 6), (8, 12), (8, 14), (8, 15), (8, 19), (8, 26), (9, 8),
        (9, 12), (9, 26), (10, 17), (11, 7), (11, 8), (11, 13), (11, 20),
        (11, 31), (12, 0), (12, 5), (12, 7), (12, 11), (12, 13), (12, 27),
        (12, 31), (13, 0), (13, 9), (13, 11), (13, 30), (14, 4), (14, 7),
        (14, 8), (15, 8), (15, 15), (15, 20), (16, 0), (16, 18), (16, 31),
        (17, 28), (18, 0), (18, 1), (18, 6), (18, 15), (18, 17), (19, 3),
        (19, 26), (19, 28), (20, 4), (21, 4), (21, 30), (21, 31), (22, 0),
        (22, 14), (22, 22), (22, 25), (23, 20), (23, 23), (23, 25), (23, 28),
        (24, 5), (24, 6), (24, 21), (26, 1), (26, 14), (26, 21), (26, 28),
        (26, 31), (27, 7), (28, 3), (28, 21), (28, 22), (29, 9), (29, 22),
        (30, 4), (30, 15), (31, 3), (31, 23),
    ]

    # scenario parameters
    scen_file = os.path.join(
        os.path.dirname(__file__), "assets", "random-32-32-10-random-1.scen"
    )
    num_agents = 200
    seed = 0
    max_timestep = 1000
    output_file = "output.txt"

    # define problem instance
    grid = get_grid(height, width, obstacles)
    starts, goals = get_scenario(scen_file, num_agents)

    # solve MAPF
    pibt = PIBT(grid, starts, goals, seed=seed)
    plan = pibt.run(max_timestep=max_timestep)

    # validation: True -> feasible solution
    print(f"solved: {is_valid_mapf_solution(grid, starts, goals, plan)}")

    # save result
    save_configs_for_visualizer(plan, output_file)
