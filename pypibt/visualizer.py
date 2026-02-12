import pygame

from .physics import PhysicsLayer
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


def run_visualizer(
    sim: MAPDSimulation,
    cell_size_m: float = 1.0,
    speed: float = 1.0,
    pixels_per_meter: float = 32.0,
):
    physics = PhysicsLayer(sim.current_config, cell_size_m, speed)
    step_interval = cell_size_m / speed  # seconds between logical steps

    h, w = sim.grid.shape
    ppm = pixels_per_meter
    cell_px = cell_size_m * ppm
    status_height = 36
    win_w = int(w * cell_px)
    win_h = int(h * cell_px) + status_height

    pygame.init()
    screen = pygame.display.set_mode((win_w, win_h))
    pygame.display.set_caption("MAPD Simulation (PIBT)")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("consolas", max(10, int(cell_px // 3)))
    status_font = pygame.font.SysFont("consolas", 16)

    # pre-render static grid surface
    grid_h_px = int(h * cell_px)
    grid_surface = pygame.Surface((win_w, grid_h_px))
    grid_surface.fill(COLOR_BG)
    for y in range(h):
        for x in range(w):
            rect = pygame.Rect(
                int(x * cell_px), int(y * cell_px), int(cell_px), int(cell_px)
            )
            if sim.grid[y, x]:
                pygame.draw.rect(grid_surface, COLOR_WALKABLE, rect)
            else:
                pygame.draw.rect(grid_surface, COLOR_OBSTACLE, rect)
            pygame.draw.rect(grid_surface, COLOR_GRID_LINE, rect, 1)

    for loc in sim.pickup_locations:
        rect = pygame.Rect(
            int(loc[1] * cell_px), int(loc[0] * cell_px), int(cell_px), int(cell_px)
        )
        pygame.draw.rect(grid_surface, COLOR_PICKUP, rect)
        pygame.draw.rect(grid_surface, COLOR_GRID_LINE, rect, 1)
    for loc in sim.delivery_locations:
        rect = pygame.Rect(
            int(loc[1] * cell_px), int(loc[0] * cell_px), int(cell_px), int(cell_px)
        )
        pygame.draw.rect(grid_surface, COLOR_DELIVERY, rect)
        pygame.draw.rect(grid_surface, COLOR_GRID_LINE, rect, 1)

    running = True
    elapsed = 0.0  # seconds since last logical step

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False

        dt = clock.tick(60) / 1000.0  # seconds this frame

        # advance physics
        physics.update(dt)
        elapsed += dt

        # trigger logical step when enough time passed and robots settled
        if elapsed >= step_interval and physics.all_settled():
            sim.tick()
            physics.set_targets(sim.current_config)
            elapsed = 0.0

        # draw grid
        screen.blit(grid_surface, (0, 0))

        # pending tasks
        for task in sim.pending_tasks:
            cx = task.pickup[1] * cell_px + cell_px / 2
            cy = task.pickup[0] * cell_px + cell_px / 2
            s = cell_px / 5
            points = [(cx, cy - s), (cx + s, cy), (cx, cy + s), (cx - s, cy)]
            pygame.draw.polygon(screen, COLOR_PENDING_TASK, points)

        # draw agents at their physical (metric) positions
        radius = int(cell_px // 3)
        for agent in sim.agents:
            ym, xm = physics.positions[agent.agent_id]
            px_x = xm * ppm
            px_y = ym * ppm

            if agent.state == AgentState.IDLE:
                color = COLOR_AGENT_IDLE
            elif agent.state == AgentState.MOVING_TO_PICKUP:
                color = COLOR_AGENT_TO_PICKUP
            else:
                color = COLOR_AGENT_CARRYING

            pygame.draw.circle(screen, color, (int(px_x), int(px_y)), radius)
            pygame.draw.circle(
                screen, COLOR_AGENT_BORDER, (int(px_x), int(px_y)), radius, 2
            )

            label = font.render(str(agent.agent_id), True, (0, 0, 0))
            label_rect = label.get_rect(center=(int(px_x), int(px_y)))
            screen.blit(label, label_rect)

        # status bar
        status_rect = pygame.Rect(0, grid_h_px, win_w, status_height)
        pygame.draw.rect(screen, COLOR_STATUS_BG, status_rect)
        status_text = (
            f"t={sim.timestep}  |  "
            f"Delivered: {len(sim.completed_tasks)}  |  "
            f"Pending: {len(sim.pending_tasks)}  |  "
            f"Active: {len(sim.active_tasks)}"
        )
        text_surface = status_font.render(status_text, True, COLOR_STATUS_TEXT)
        screen.blit(text_surface, (10, grid_h_px + 8))

        pygame.display.flip()

    pygame.quit()
