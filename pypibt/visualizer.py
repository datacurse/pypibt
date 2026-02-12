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
COLOR_BUTTON_BG = (70, 70, 70)
COLOR_BUTTON_ACTIVE = (50, 130, 200)
COLOR_BUTTON_HOVER = (90, 90, 90)
COLOR_BUTTON_TEXT = (220, 220, 220)
COLOR_BUTTON_BORDER = (100, 100, 100)
COLOR_GOAL_LINE = (200, 80, 80)


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
    status_height = 82
    win_w = int(w * cell_px)
    win_h = int(h * cell_px) + status_height

    pygame.init()
    screen = pygame.display.set_mode((win_w, win_h))
    pygame.display.set_caption("MAPD Simulation (PIBT)")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("consolas", max(10, int(cell_px // 3)))
    status_font = pygame.font.SysFont("consolas", 16)

    # speed control
    SPEED_MULTIPLIERS = [1, 2, 5, 10, 100, 1000]
    MAX_TICKS_PER_FRAME = 50
    current_speed_mult = 1

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

    # pre-compute speed button rects
    speed_label_surf = status_font.render("Speed:", True, COLOR_STATUS_TEXT)
    _btn_x = 10 + speed_label_surf.get_width() + 8
    speed_buttons: list[tuple[pygame.Rect, int, pygame.Surface]] = []
    for _mult in SPEED_MULTIPLIERS:
        _label_surf = status_font.render(f"x{_mult}", True, COLOR_BUTTON_TEXT)
        _btn_w = _label_surf.get_width() + 16
        _btn_rect = pygame.Rect(_btn_x, grid_h_px + 50, _btn_w, 22)
        speed_buttons.append((_btn_rect, _mult, _label_surf))
        _btn_x += _btn_w + 4

    # checkbox: show goal lines
    show_goals = False
    cb_label_surf = status_font.render("Goals", True, COLOR_BUTTON_TEXT)
    cb_size = 14
    cb_x = _btn_x + 12
    cb_y = grid_h_px + 54
    checkbox_rect = pygame.Rect(cb_x, cb_y, cb_size, cb_size)
    cb_label_x = cb_x + cb_size + 5

    running = True
    elapsed = 0.0  # seconds since last logical step

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if checkbox_rect.collidepoint(event.pos):
                    show_goals = not show_goals
                else:
                    for btn_rect, mult, _ in speed_buttons:
                        if btn_rect.collidepoint(event.pos):
                            current_speed_mult = mult
                            step_interval = cell_size_m / (speed * current_speed_mult)
                            physics.speed = speed * current_speed_mult
                            break

        dt = clock.tick(100) / 1000.0  # seconds this frame

        # advance physics
        physics.update(dt)
        elapsed += dt

        # trigger logical steps -- may need multiple per frame at high speeds
        ticks_this_frame = 0
        while elapsed >= step_interval and ticks_this_frame < MAX_TICKS_PER_FRAME:
            if current_speed_mult >= 100 or physics.all_settled():
                sim.tick()
                physics.set_targets(sim.current_config)
                if current_speed_mult >= 100:
                    physics.snap_to_targets()
                elapsed -= step_interval
                ticks_this_frame += 1
            else:
                break

        if ticks_this_frame >= MAX_TICKS_PER_FRAME:
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

        # goal lines
        if show_goals:
            for agent in sim.agents:
                goal = sim.pibt.goals[agent.agent_id]
                ym, xm = physics.positions[agent.agent_id]
                ax = int(xm * ppm)
                ay = int(ym * ppm)
                gx = int(goal[1] * cell_px + cell_px / 2)
                gy = int(goal[0] * cell_px + cell_px / 2)
                if ax != gx or ay != gy:
                    pygame.draw.line(screen, COLOR_GOAL_LINE, (ax, ay), (gx, gy), 2)

        # status bar
        status_rect = pygame.Rect(0, grid_h_px, win_w, status_height)
        pygame.draw.rect(screen, COLOR_STATUS_BG, status_rect)
        t = sim.timestep
        hrs, rem = divmod(t, 3600)
        mins, secs = divmod(rem, 60)
        time_str = f"{hrs}:{mins:02d}:{secs:02d}"
        status_text = (
            f"{time_str}  |  "
            f"Delivered: {len(sim.completed_tasks)}  |  "
            f"Pending: {len(sim.pending_tasks)}  |  "
            f"Active: {len(sim.active_tasks)}"
        )
        text_surface = status_font.render(status_text, True, COLOR_STATUS_TEXT)
        screen.blit(text_surface, (10, grid_h_px + 8))

        # row 2: throughput and agent utilization
        if sim.timestep > 0:
            del_per_hour = len(sim.completed_tasks) / sim.timestep * 3600
        else:
            del_per_hour = 0.0
        num_active = sum(1 for a in sim.agents if a.state != AgentState.IDLE)
        active_pct = num_active / len(sim.agents) * 100 if sim.agents else 0.0
        stats2_text = (
            f"Throughput: {del_per_hour:.1f} del/hr  |  "
            f"Busy: {num_active}/{len(sim.agents)} ({active_pct:.0f}%)"
        )
        stats2_surface = status_font.render(stats2_text, True, COLOR_STATUS_TEXT)
        screen.blit(stats2_surface, (10, grid_h_px + 26))

        # speed buttons (row 3)
        screen.blit(speed_label_surf, (10, grid_h_px + 54))
        mouse_pos = pygame.mouse.get_pos()
        for btn_rect, mult, label_surf in speed_buttons:
            if mult == current_speed_mult:
                bg = COLOR_BUTTON_ACTIVE
            elif btn_rect.collidepoint(mouse_pos):
                bg = COLOR_BUTTON_HOVER
            else:
                bg = COLOR_BUTTON_BG
            pygame.draw.rect(screen, bg, btn_rect, border_radius=3)
            pygame.draw.rect(screen, COLOR_BUTTON_BORDER, btn_rect, 1, border_radius=3)
            screen.blit(label_surf, (btn_rect.x + 8, btn_rect.y + 3))

        # checkbox: show goals
        if show_goals:
            pygame.draw.rect(screen, COLOR_BUTTON_ACTIVE, checkbox_rect)
        else:
            pygame.draw.rect(screen, COLOR_BUTTON_BG, checkbox_rect)
        pygame.draw.rect(screen, COLOR_BUTTON_BORDER, checkbox_rect, 1)
        screen.blit(cb_label_surf, (cb_label_x, cb_y))

        pygame.display.flip()

    pygame.quit()
