import pygame
import numpy as np

from .simulation import AgentState, MAPDSimulation

# colors
COLOR_BG = (30, 30, 30)
COLOR_WALKABLE = (240, 240, 240)
COLOR_OBSTACLE = (60, 60, 60)
COLOR_PICKUP = (144, 220, 144)
COLOR_DELIVERY = (144, 144, 220)
COLOR_GRID_LINE = (200, 200, 200)
COLOR_AGENT_IDLE = (160, 160, 160)
COLOR_AGENT_TO_PICKUP = (40, 180, 40)
COLOR_AGENT_CARRYING = (230, 130, 20)
COLOR_AGENT_BORDER = (20, 20, 20)
COLOR_PENDING_TASK = (220, 50, 50)
COLOR_STATUS_BG = (40, 40, 40)
COLOR_STATUS_TEXT = (220, 220, 220)


def run_visualizer(sim: MAPDSimulation, cell_size: int = 32, tick_ms: int = 400):
    h, w = sim.grid.shape
    status_height = 36
    win_w = w * cell_size
    win_h = h * cell_size + status_height

    pygame.init()
    screen = pygame.display.set_mode((win_w, win_h))
    pygame.display.set_caption("MAPD Simulation (PIBT)")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("consolas", cell_size // 3)
    status_font = pygame.font.SysFont("consolas", 16)

    # pre-render static grid surface
    grid_surface = pygame.Surface((win_w, h * cell_size))
    grid_surface.fill(COLOR_BG)
    for y in range(h):
        for x in range(w):
            rect = pygame.Rect(x * cell_size, y * cell_size, cell_size, cell_size)
            if sim.grid[y, x]:
                pygame.draw.rect(grid_surface, COLOR_WALKABLE, rect)
            else:
                pygame.draw.rect(grid_surface, COLOR_OBSTACLE, rect)
            pygame.draw.rect(grid_surface, COLOR_GRID_LINE, rect, 1)

    # mark pickup/delivery stations on grid surface
    for loc in sim.pickup_locations:
        rect = pygame.Rect(loc[1] * cell_size, loc[0] * cell_size, cell_size, cell_size)
        pygame.draw.rect(grid_surface, COLOR_PICKUP, rect)
        pygame.draw.rect(grid_surface, COLOR_GRID_LINE, rect, 1)
    for loc in sim.delivery_locations:
        rect = pygame.Rect(loc[1] * cell_size, loc[0] * cell_size, cell_size, cell_size)
        pygame.draw.rect(grid_surface, COLOR_DELIVERY, rect)
        pygame.draw.rect(grid_surface, COLOR_GRID_LINE, rect, 1)

    running = True
    last_tick = pygame.time.get_ticks()

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False

        now = pygame.time.get_ticks()
        if now - last_tick >= tick_ms:
            sim.tick()
            last_tick = now

        # draw grid
        screen.blit(grid_surface, (0, 0))

        # draw pending tasks as red diamonds at pickup locations
        for task in sim.pending_tasks:
            cx = task.pickup[1] * cell_size + cell_size // 2
            cy = task.pickup[0] * cell_size + cell_size // 2
            s = cell_size // 5
            points = [(cx, cy - s), (cx + s, cy), (cx, cy + s), (cx - s, cy)]
            pygame.draw.polygon(screen, COLOR_PENDING_TASK, points)

        # draw agents
        radius = cell_size // 3
        for agent in sim.agents:
            pos = sim.current_config[agent.agent_id]
            cx = pos[1] * cell_size + cell_size // 2
            cy = pos[0] * cell_size + cell_size // 2

            if agent.state == AgentState.IDLE:
                color = COLOR_AGENT_IDLE
            elif agent.state == AgentState.MOVING_TO_PICKUP:
                color = COLOR_AGENT_TO_PICKUP
            else:
                color = COLOR_AGENT_CARRYING

            pygame.draw.circle(screen, color, (cx, cy), radius)
            pygame.draw.circle(screen, COLOR_AGENT_BORDER, (cx, cy), radius, 2)

            # agent id label
            label = font.render(str(agent.agent_id), True, (0, 0, 0))
            label_rect = label.get_rect(center=(cx, cy))
            screen.blit(label, label_rect)

        # status bar
        status_rect = pygame.Rect(0, h * cell_size, win_w, status_height)
        pygame.draw.rect(screen, COLOR_STATUS_BG, status_rect)
        status_text = (
            f"t={sim.timestep}  |  "
            f"Delivered: {len(sim.completed_tasks)}  |  "
            f"Pending: {len(sim.pending_tasks)}  |  "
            f"Active: {len(sim.active_tasks)}"
        )
        text_surface = status_font.render(status_text, True, COLOR_STATUS_TEXT)
        screen.blit(text_surface, (10, h * cell_size + 8))

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()
