import math

import pygame
import pygame.gfxdraw

from .physics import PhysicsLayer
from .simulation import AgentState, MAPDSimulation

_STATE_NAMES_RU = {
    "IDLE": "Свободен",
    "MOVING_TO_PICKUP": "К загрузке",
    "MOVING_TO_DELIVERY": "Доставляет",
}

# Catppuccin Mocha palette
COLOR_BG = (30, 30, 46)  # Base
COLOR_WALKABLE = (69, 71, 90)  # Surface 1
COLOR_OBSTACLE = (24, 24, 37)  # Mantle
COLOR_PICKUP = (98, 129, 104)  # Green (dimmed)
COLOR_DELIVERY = (84, 105, 148)  # Blue (dimmed)
COLOR_GRID_LINE = (49, 50, 68)  # Surface 0
COLOR_AGENT_IDLE = (166, 173, 200)  # Subtext 0
COLOR_AGENT_TO_PICKUP = (166, 227, 161)  # Green (matches Pickup stations)
COLOR_AGENT_CARRYING = (137, 180, 250)  # Blue (matches Delivery stations)
COLOR_AGENT_BORDER = (17, 17, 27)  # Crust
COLOR_PENDING_TASK = (243, 139, 168)  # Red
COLOR_STATUS_BG = (24, 24, 37)  # Mantle
COLOR_STATUS_TEXT = (205, 214, 244)  # Text
COLOR_BUTTON_BG = (49, 50, 68)  # Surface 0
COLOR_BUTTON_ACTIVE = (203, 166, 247)  # Mauve
COLOR_BUTTON_HOVER = (88, 91, 112)  # Surface 2
COLOR_BUTTON_TEXT = (205, 214, 244)  # Text
COLOR_BUTTON_BORDER = (108, 112, 134)  # Overlay 0


def run_visualizer(
    sim: MAPDSimulation,
    cell_size_m: float = 1.0,
    speed: float = 1.0,
    pixels_per_meter: float = 32.0,
):
    physics = PhysicsLayer(
        sim.current_config,
        cell_size_m,
        speed,
        initial_orientations=sim.pibt.orientations,
    )
    step_interval = cell_size_m / speed  # seconds between logical steps

    h, w = sim.grid.shape
    ppm = pixels_per_meter
    cell_px = cell_size_m * ppm
    status_height = 82
    win_w = int(w * cell_px)
    win_h = int(h * cell_px) + status_height

    pygame.init()
    screen = pygame.display.set_mode((win_w, win_h))
    pygame.display.set_caption("MAPD Симуляция (EPIBT)")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("consolas", max(10, int(cell_px // 3)))
    status_font = pygame.font.SysFont("consolas", 16)

    # speed control
    SPEED_MULTIPLIERS = [1, 2, 5, 10, 25, 50, 100, 1000]
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
    speed_label_surf = status_font.render("Скорость:", True, COLOR_STATUS_TEXT)
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
    cb_label_surf = status_font.render("Цели", True, COLOR_BUTTON_TEXT)
    cb_size = 14
    cb_x = _btn_x + 12
    cb_y = grid_h_px + 54
    checkbox_rect = pygame.Rect(cb_x, cb_y, cb_size, cb_size)
    cb_label_x = cb_x + cb_size + 5

    # pre-render legend surface
    legend_font = pygame.font.SysFont("consolas", 13)
    legend_entries: list[tuple[tuple[int, int, int], str, str]] = [
        (COLOR_AGENT_IDLE, "circle", "Свободен"),
        (COLOR_AGENT_TO_PICKUP, "circle", "К загрузке"),
        (COLOR_AGENT_CARRYING, "circle", "Несёт груз"),
        (COLOR_PICKUP, "square", "Загрузка"),
        (COLOR_DELIVERY, "square", "Доставка"),
        (COLOR_PENDING_TASK, "diamond", "В очереди"),
    ]
    _leg_pad = 8
    _leg_entry_h = 18
    _leg_swatch = 12
    _leg_w = 160
    _leg_h = _leg_pad * 2 + len(legend_entries) * _leg_entry_h
    legend_surface = pygame.Surface((_leg_w, _leg_h), pygame.SRCALPHA)
    legend_surface.fill((30, 30, 46, 180))
    for _i, (_color, _shape, _label) in enumerate(legend_entries):
        _ey = _leg_pad + _i * _leg_entry_h
        _sx = _leg_pad
        _sy = _ey + _leg_entry_h // 2
        if _shape == "circle":
            pygame.draw.circle(
                legend_surface, _color, (_sx + _leg_swatch // 2, _sy), _leg_swatch // 2
            )
            pygame.draw.circle(
                legend_surface,
                COLOR_AGENT_BORDER,
                (_sx + _leg_swatch // 2, _sy),
                _leg_swatch // 2,
                1,
            )
        elif _shape == "square":
            pygame.draw.rect(
                legend_surface,
                _color,
                (_sx, _sy - _leg_swatch // 2, _leg_swatch, _leg_swatch),
            )
        elif _shape == "diamond":
            _ds = _leg_swatch // 2
            _cx, _cy = _sx + _ds, _sy
            pygame.draw.polygon(
                legend_surface,
                _color,
                [
                    (_cx, _cy - _ds),
                    (_cx + _ds, _cy),
                    (_cx, _cy + _ds),
                    (_cx - _ds, _cy),
                ],
            )
        _txt = legend_font.render(_label, True, COLOR_STATUS_TEXT)
        legend_surface.blit(_txt, (_sx + _leg_swatch + 6, _ey + 1))

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
                physics.set_targets(sim.current_config, sim.pibt.orientations)
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

        # pending tasks -- aggregate by pickup location
        task_counts: dict[tuple[int, int], int] = {}
        for task in sim.pending_tasks:
            task_counts[task.pickup] = task_counts.get(task.pickup, 0) + 1
        for loc, count in task_counts.items():
            cx = loc[1] * cell_px + cell_px / 2
            cy = loc[0] * cell_px + cell_px / 2
            s = cell_px / 3
            points = [(cx, cy - s), (cx + s, cy), (cx, cy + s), (cx - s, cy)]
            pygame.draw.polygon(screen, COLOR_PENDING_TASK, points)
            pygame.draw.polygon(screen, (205, 214, 244), points, 2)
            if count > 1:
                count_surf = font.render(str(count), True, (205, 214, 244))
                count_rect = count_surf.get_rect(center=(int(cx), int(cy)))
                screen.blit(count_surf, count_rect)

        # draw agents at their physical (metric) positions
        radius = int(cell_px * 0.4)
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

            ix, iy = int(px_x), int(px_y)
            pygame.gfxdraw.filled_circle(screen, ix, iy, radius, color)
            pygame.gfxdraw.aacircle(screen, ix, iy, radius, color)

            # direction arrow (triangle pointing in facing direction)
            angle = physics.angles[agent.agent_id]
            arrow_len = radius * 0.75
            arrow_hw = radius * 0.35
            tip_x = ix + math.cos(angle) * arrow_len
            tip_y = iy + math.sin(angle) * arrow_len
            perp = angle + math.pi / 2
            b1x = ix + math.cos(perp) * arrow_hw
            b1y = iy + math.sin(perp) * arrow_hw
            b2x = ix - math.cos(perp) * arrow_hw
            b2y = iy - math.sin(perp) * arrow_hw
            pygame.draw.polygon(
                screen,
                COLOR_AGENT_BORDER,
                [
                    (int(tip_x), int(tip_y)),
                    (int(b1x), int(b1y)),
                    (int(b2x), int(b2y)),
                ],
            )

        # legend
        screen.blit(legend_surface, (win_w - _leg_w - 8, 8))

        # goal lines -- color-coded by agent state with alpha
        if show_goals:
            goal_surface = pygame.Surface((win_w, grid_h_px), pygame.SRCALPHA)
            for agent in sim.agents:
                if agent.state == AgentState.IDLE:
                    continue
                goal = sim.pibt.goals[agent.agent_id]
                ym, xm = physics.positions[agent.agent_id]
                ax = int(xm * ppm)
                ay = int(ym * ppm)
                gx = int(goal[1] * cell_px + cell_px / 2)
                gy = int(goal[0] * cell_px + cell_px / 2)
                if ax != gx or ay != gy:
                    if agent.state == AgentState.MOVING_TO_PICKUP:
                        line_color = (*COLOR_AGENT_TO_PICKUP, 80)
                    else:
                        line_color = (*COLOR_AGENT_CARRYING, 80)
                    pygame.draw.line(goal_surface, line_color, (ax, ay), (gx, gy), 2)
            screen.blit(goal_surface, (0, 0))

        # hover tooltip for agents
        mouse_pos = pygame.mouse.get_pos()
        for agent in sim.agents:
            ym, xm = physics.positions[agent.agent_id]
            ax, ay = int(xm * ppm), int(ym * ppm)
            if math.hypot(mouse_pos[0] - ax, mouse_pos[1] - ay) <= radius:
                state_ru = _STATE_NAMES_RU.get(agent.state.name, agent.state.name)
                ori_name = ["N", "E", "S", "W"][sim.pibt.orientations[agent.agent_id]]
                lines = [f"Агент {agent.agent_id}  ({state_ru})  [{ori_name}]"]
                if agent.current_task is not None:
                    t = agent.current_task
                    lines.append(f"Задача #{t.task_id}")
                    lines.append(f"Загрузка: {t.pickup}")
                    lines.append(f"Доставка: {t.delivery}")
                tip_surfs = [
                    legend_font.render(ln, True, COLOR_STATUS_TEXT) for ln in lines
                ]
                tip_w = max(s.get_width() for s in tip_surfs) + 12
                tip_h = len(tip_surfs) * 16 + 8
                tip_x = min(mouse_pos[0] + 14, win_w - tip_w - 4)
                tip_y = min(mouse_pos[1] + 14, win_h - tip_h - 4)
                tip_bg = pygame.Surface((tip_w, tip_h), pygame.SRCALPHA)
                tip_bg.fill((24, 24, 37, 220))
                screen.blit(tip_bg, (tip_x, tip_y))
                for j, s in enumerate(tip_surfs):
                    screen.blit(s, (tip_x + 6, tip_y + 4 + j * 16))
                break

        # status bar
        status_rect = pygame.Rect(0, grid_h_px, win_w, status_height)
        pygame.draw.rect(screen, COLOR_STATUS_BG, status_rect)
        t = sim.timestep
        hrs, rem = divmod(t, 3600)
        mins, secs = divmod(rem, 60)
        time_str = f"{hrs}:{mins:02d}:{secs:02d}"
        status_text = (
            f"{time_str}  |  "
            f"Доставлено: {len(sim.completed_tasks)}  |  "
            f"В очереди: {len(sim.pending_tasks)}  |  "
            f"В работе: {len(sim.active_tasks)}"
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
            f"Пропускная сп.: {del_per_hour:.1f} дост/ч  |  "
            f"Заняты: {num_active}/{len(sim.agents)} ({active_pct:.0f}%)"
        )
        stats2_surface = status_font.render(stats2_text, True, COLOR_STATUS_TEXT)
        screen.blit(stats2_surface, (10, grid_h_px + 26))

        # speed buttons (row 3)
        screen.blit(speed_label_surf, (10, grid_h_px + 54))
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
