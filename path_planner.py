"""
In this file, you should implement your own path planning class or function.
Within your implementation, you may call `env.is_collide()` and `env.is_outside()`
to verify whether candidate path points collide with obstacles or exceed the
environment boundaries.

You are required to write the path planning algorithm by yourself. Copying or calling 
any existing path planning algorithms from others is strictly
prohibited. Please avoid using external packages beyond common Python libraries
such as `numpy`, `math`, or `scipy`. If you must use additional packages, you
must clearly explain the reason in your report.
"""
            




import heapq
from typing import Tuple, List, Dict, Optional

import numpy as np


class PathPlanner:
    """
    Simple 3D grid-based A* path planner.

    - The continuous environment [0, W] x [0, L] x [0, H] is discretized
      into a regular 3D grid with spacing = resolution.
    - Each grid cell center is treated as a point and checked for collision
      using env.is_collide() and env.is_outside().
    """

    def __init__(self, env, resolution: float = 0.5, epsilon: float = 0.2):
        """
        Parameters
        ----------
        env : FlightEnvironment
            The flight environment instance.
        resolution : float
            Grid spacing (in meters). Smaller -> finer grid but slower.
        epsilon : float
            Safety margin passed to env.is_collide().
        """
        self.env = env
        self.resolution = float(resolution)
        self.epsilon = float(epsilon)

        # Pre-build occupancy grid
        self._build_occupancy_grid()

    # ------------------------------------------------------------------ #
    #   Public API
    # ------------------------------------------------------------------ #
    def plan(self, start: Tuple[float, float, float],
             goal: Tuple[float, float, float]) -> np.ndarray:
        """
        Plan a collision-free path from start to goal using A*.

        Parameters
        ----------
        start : (x, y, z)
        goal  : (x, y, z)

        Returns
        -------
        path : (N, 3) numpy.ndarray
            Sequence of 3D points (in world coordinates) from start to goal.
        """
        # Check basic validity
        if self.env.is_outside(start) or self.env.is_collide(start, self.epsilon):
            raise ValueError(f"Start point {start} is invalid (outside or in collision).")
        if self.env.is_outside(goal) or self.env.is_collide(goal, self.epsilon):
            raise ValueError(f"Goal point {goal} is invalid (outside or in collision).")

        start_idx = self._world_to_grid(start)
        goal_idx = self._world_to_grid(goal)

        if not self._is_free_index(start_idx):
            raise ValueError("Start cell is occupied.")
        if not self._is_free_index(goal_idx):
            raise ValueError("Goal cell is occupied.")

        # Run A*
        grid_path = self._a_star_search(start_idx, goal_idx)

        if grid_path is None or len(grid_path) == 0:
            raise RuntimeError("A* failed to find a path.")

        # Convert grid indices to world coordinates
        path_world = np.array([self._grid_to_world(idx) for idx in grid_path],
                              dtype=np.float32)

        # Optionally ensure exact start/goal at the ends
        path_world[0, :] = np.array(start, dtype=np.float32)
        path_world[-1, :] = np.array(goal, dtype=np.float32)

        return path_world

    # ------------------------------------------------------------------ #
    #   Occupancy grid construction
    # ------------------------------------------------------------------ #
    def _build_occupancy_grid(self):
        """
        Create a 3D occupancy grid:
            True  -> free cell
            False -> occupied (outside env or in collision)
        """
        W = self.env.env_width
        L = self.env.env_length
        H = self.env.env_height

        res = self.resolution

        # Number of grid points along each axis (inclusive of both ends)
        self.nx = int(W / res) + 1
        self.ny = int(L / res) + 1
        self.nz = int(H / res) + 1

        self.free = np.ones((self.nx, self.ny, self.nz), dtype=bool)

        for ix in range(self.nx):
            x = ix * res
            for iy in range(self.ny):
                y = iy * res
                for iz in range(self.nz):
                    z = iz * res
                    p = (x, y, z)
                    # Outside or colliding -> mark as occupied
                    if self.env.is_outside(p) or self.env.is_collide(p, self.epsilon):
                        self.free[ix, iy, iz] = False

    # ------------------------------------------------------------------ #
    #   A* core
    # ------------------------------------------------------------------ #
    def _a_star_search(self, start_idx: Tuple[int, int, int],
                       goal_idx: Tuple[int, int, int]) -> Optional[List[Tuple[int, int, int]]]:
        """
        A* on 3D grid.

        Returns a list of grid indices from start to goal, or None if no path.
        """
        start = start_idx
        goal = goal_idx

        # Open set is a priority queue of (f_score, g_score, node)
        open_heap: List[Tuple[float, float, Tuple[int, int, int]]] = []

        # g_score: cost from start to this node
        g_score: Dict[Tuple[int, int, int], float] = {start: 0.0}
        # f_score: g + heuristic
        f_start = self._heuristic(start, goal)
        heapq.heappush(open_heap, (f_start, 0.0, start))

        came_from: Dict[Tuple[int, int, int], Tuple[int, int, int]] = {}

        closed_set = set()

        while open_heap:
            _, g_curr, current = heapq.heappop(open_heap)

            if current in closed_set:
                continue
            closed_set.add(current)

            # Goal check
            if current == goal:
                return self._reconstruct_path(came_from, current)

            # Expand neighbors
            for neighbor in self._neighbors(current):
                if neighbor in closed_set:
                    continue

                tentative_g = g_curr + self._cost(current, neighbor)

                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    g_score[neighbor] = tentative_g
                    f_neighbor = tentative_g + self._heuristic(neighbor, goal)
                    came_from[neighbor] = current
                    heapq.heappush(open_heap, (f_neighbor, tentative_g, neighbor))

        # No path
        return None

    def _neighbors(self, idx: Tuple[int, int, int]) -> List[Tuple[int, int, int]]:
        """
        26-connected neighborhood in 3D (all ±1 steps, excluding (0,0,0)).
        Only returns free (unoccupied) grid cells.
        """
        ix, iy, iz = idx
        res = []

        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    if dx == 0 and dy == 0 and dz == 0:
                        continue
                    nx = ix + dx
                    ny = iy + dy
                    nz = iz + dz
                    if 0 <= nx < self.nx and 0 <= ny < self.ny and 0 <= nz < self.nz:
                        if self.free[nx, ny, nz]:
                            res.append((nx, ny, nz))
        return res

    def _heuristic(self, a: Tuple[int, int, int],
                   b: Tuple[int, int, int]) -> float:
        """
        Euclidean distance in world coordinates as heuristic.
        """
        wa = self._grid_to_world(a)
        wb = self._grid_to_world(b)
        return float(np.linalg.norm(wa - wb))

    def _cost(self, a: Tuple[int, int, int],
              b: Tuple[int, int, int]) -> float:
        """
        Step cost between two neighboring cells: Euclidean distance in world space.
        """
        wa = self._grid_to_world(a)
        wb = self._grid_to_world(b)
        return float(np.linalg.norm(wa - wb))

    def _reconstruct_path(self, came_from: Dict[Tuple[int, int, int], Tuple[int, int, int]],
                          current: Tuple[int, int, int]) -> List[Tuple[int, int, int]]:
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path

    # ------------------------------------------------------------------ #
    #   Index <-> world coordinate conversions
    # ------------------------------------------------------------------ #
    def _world_to_grid(self, point: Tuple[float, float, float]) -> Tuple[int, int, int]:
        """
        Map world coordinate (x, y, z) to grid indices (ix, iy, iz).
        """
        x, y, z = point
        ix = int(round(x / self.resolution))
        iy = int(round(y / self.resolution))
        iz = int(round(z / self.resolution))

        # Clamp to valid range just in case of round-off
        ix = min(max(ix, 0), self.nx - 1)
        iy = min(max(iy, 0), self.ny - 1)
        iz = min(max(iz, 0), self.nz - 1)
        return ix, iy, iz

    def _grid_to_world(self, idx: Tuple[int, int, int]) -> np.ndarray:
        """
        Map grid indices (ix, iy, iz) to world coordinate (x, y, z).
        Uses the grid point position at ix * resolution, etc.
        """
        ix, iy, iz = idx
        x = ix * self.resolution
        y = iy * self.resolution
        z = iz * self.resolution
        return np.array([x, y, z], dtype=np.float32)

    def _is_free_index(self, idx: Tuple[int, int, int]) -> bool:
        ix, iy, iz = idx
        if 0 <= ix < self.nx and 0 <= iy < self.ny and 0 <= iz < self.nz:
            return bool(self.free[ix, iy, iz])
        return False




