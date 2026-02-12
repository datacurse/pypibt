# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

pypibt is a minimal Python implementation of Priority Inheritance with Backtracking (PIBT) for Multi-Agent Path Finding (MAPF). The core algorithm is intentionally compact (~110 lines) for educational clarity.

Reference: Okumura, K., Machida, M., Défago, X., & Tamura, Y. "Priority inheritance with backtracking for iterative multi-agent path finding." AIJ. 2022.

## Commands

```sh
# Install dependencies (Poetry-based project, but pyproject.toml was removed)
pip install -r requirements.txt

# Run the demo (grid params and scenario are hardcoded in app.py)
python app.py

# Formatting (enforced via pre-commit hooks)
black .
isort --profile black .

# Pre-commit hooks
pre-commit run --all-files
```

Note: Tests were removed from the repository. CI (`.github/workflows/ci.yml`) references `poetry run pytest` but no test directory exists.

## Architecture

All source code lives in the `pypibt/` package (3 modules) with `app.py` as the CLI entry point.

**Coordinates are (y, x)** — row-first indexing aligned with numpy array shapes. The grid is a boolean numpy array where `True` = walkable, `False` = obstacle.

### Module responsibilities

- **`pibt.py`** — `PIBT` class: the solver. `run()` drives the main loop, `step()` computes one timestep, `funcPIBT()` is the recursive core with priority inheritance and backtracking. Uses `occupied_now`/`occupied_nxt` numpy arrays for O(1) collision checking. `NIL` sentinel = `N` (agent count), `NIL_COORD` sentinel = `grid.shape`.
- **`dist_table.py`** — `DistTable` dataclass: lazy BFS distance computation from each agent's goal. Distances are computed on-demand and cached in a numpy array. Unreachable/unknown cells default to `grid.size`.
- **`mapf_utils.py`** — Grid construction, I/O, and validation: `get_grid(height, width, obstacles)` builds the grid from dimensions and obstacle coordinates, validates solutions (vertex/edge collisions, connectivity), saves output for the external `mapf-visualizer` tool.

### Type aliases (defined in `mapf_utils.py`)

- `Grid = np.ndarray` (2D bool), `Coord = tuple[int, int]`, `Config = list[Coord]`, `Configs = list[Config]`

### Data flow

`app.py` → `get_grid()` → `PIBT(grid, starts, goals)` → `pibt.run()` → `is_valid_mapf_solution()` → `save_configs_for_visualizer()`

## Key conventions

- Only runtime dependency is `numpy`. Dev tools: `black`, `isort`, `pre-commit`.
- isort uses `--profile black` for compatibility.
- Grid, starts, and goals are all hardcoded in `app.py` as lists of `(y, x)` tuples — no external config files.
