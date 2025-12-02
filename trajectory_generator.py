"""
In this file, you should implement your trajectory generation class or function.
Your method must generate a smooth 3-axis trajectory (x(t), y(t), z(t)) that 
passes through all the previously computed path points. A positional deviation 
up to 0.1 m from each path point is allowed.

You should output the generated trajectory and visualize it. The figure must
contain three subplots showing x, y, and z, respectively, with time t (in seconds)
as the horizontal axis. Additionally, you must plot the original discrete path 
points on the same figure for comparison.

You are expected to write the implementation yourself. Do NOT copy or reuse any 
existing trajectory generation code from others. Avoid using external packages 
beyond general scientific libraries such as numpy, math, or scipy. If you decide 
to use additional packages, you must clearly explain the reason in your report.
"""

import numpy as np
import matplotlib.pyplot as plt


class TrajectoryGenerator:
    """
    Generate a smooth 3D trajectory that passes through a given set of path points.

    Implementation:
        - Use Catmull–Rom spline (piecewise cubic, C1 continuous).
        - Parameterize trajectory by time assuming each path segment
          has equal duration.
    """

    def __init__(self):
        pass

    # ------------------------------------------------------------------ #
    #   Public API
    # ------------------------------------------------------------------ #
    def generate_trajectory(self,
                            path: np.ndarray,
                            total_time: float = 10.0,
                            samples_per_segment: int = 20):
        """
        Generate a smooth trajectory (x(t), y(t), z(t)) passing through path points.

        Parameters
        ----------
        path : (N, 3) numpy.ndarray
            Discrete path points in 3D, each row is [x, y, z].
        total_time : float
            Total duration of the trajectory (seconds).
        samples_per_segment : int
            Number of trajectory sample points per path segment.

        Returns
        -------
        t   : (M,) numpy.ndarray
            Time stamps (seconds) from 0 to total_time.
        traj : (M, 3) numpy.ndarray
            Smooth trajectory positions corresponding to t.
        """
        path = np.asarray(path, dtype=np.float32)
        if path.ndim != 2 or path.shape[1] != 3:
            raise ValueError("path must be an (N, 3) array.")

        N = path.shape[0]
        if N < 2:
            raise ValueError("Need at least 2 points to generate a trajectory.")

        # Number of segments = N - 1
        num_segments = N - 1
        dt = total_time / num_segments

        time_list = []
        traj_list = []

        # For each segment, construct a local Catmull–Rom spline
        for i in range(num_segments):
            # Four control points P0, P1, P2, P3
            P1 = path[i]
            P2 = path[i + 1]

            if i == 0:
                # Extrapolate for P0 at the beginning
                P0 = P1 - (P2 - P1)
            else:
                P0 = path[i - 1]

            if i + 2 < N:
                P3 = path[i + 2]
            else:
                # Extrapolate for P3 at the end
                P3 = P2 + (P2 - P1)

            # Local parameter u in [0, 1) for this segment
            # We use endpoint=False to avoid duplicating the knot at segment boundary
            u_values = np.linspace(0.0, 1.0, samples_per_segment, endpoint=False)
            t_values = i * dt + u_values * dt

            # Catmull–Rom spline formula (uniform)
            # 0.5 * [ 2P1 + (-P0+P2)u + (2P0-5P1+4P2-P3)u^2 + (-P0+3P1-3P2+P3)u^3 ]
            # We can vectorize this computation
            u = u_values[:, None]            # (M, 1)
            u2 = u * u
            u3 = u2 * u

            term1 = 2.0 * P1
            term2 = (-P0 + P2) * u
            term3 = (2 * P0 - 5 * P1 + 4 * P2 - P3) * u2
            term4 = (-P0 + 3 * P1 - 3 * P2 + P3) * u3

            segment_points = 0.5 * (term1 + term2 + term3 + term4)  # (M, 3)

            time_list.append(t_values)
            traj_list.append(segment_points)

        # Append the final endpoint exactly
        time_list.append(np.array([total_time], dtype=np.float32))
        traj_list.append(path[-1][None, :])

        t = np.concatenate(time_list, axis=0)
        traj = np.concatenate(traj_list, axis=0)

        return t, traj

    # ------------------------------------------------------------------ #
    #   Plotting helper
    # ------------------------------------------------------------------ #
    def plot_trajectory(self, t: np.ndarray, traj: np.ndarray, path: np.ndarray):
        """
        Plot x(t), y(t), z(t) in three subplots, and overlay discrete path points.

        Parameters
        ----------
        t    : (M,) numpy.ndarray
            Time stamps (seconds).
        traj : (M, 3) numpy.ndarray
            Continuous trajectory [x(t), y(t), z(t)].
        path : (N, 3) numpy.ndarray
            Discrete path points for reference.
        """
        t = np.asarray(t)
        traj = np.asarray(traj)
        path = np.asarray(path)

        if traj.shape[1] != 3 or path.shape[1] != 3:
            raise ValueError("traj and path must both be (N, 3)-like arrays.")

        fig, axs = plt.subplots(3, 1, sharex=True, figsize=(8, 8))
        fig.suptitle("Trajectory Time Histories (x(t), y(t), z(t))")

        # x(t)
        axs[0].plot(t, traj[:, 0], label="smooth trajectory")
        # 将离散路径点按照均匀时间分布投影到时间轴上
        path_times = np.linspace(0.0, t[-1], path.shape[0])
        axs[0].scatter(path_times, path[:, 0], marker='o', s=20, c="red", label="path points")
        axs[0].set_ylabel("x (m)")
        axs[0].legend()
        axs[0].grid(True)

        # y(t)
        axs[1].plot(t, traj[:, 1], label="smooth trajectory")
        axs[1].scatter(path_times, path[:, 1], marker='o', s=20, c="red",label="path points")
        axs[1].set_ylabel("y (m)")
        axs[1].legend()
        axs[1].grid(True)

        # z(t)
        axs[2].plot(t, traj[:, 2], label="smooth trajectory")
        axs[2].scatter(path_times, path[:, 2], marker='o', s=20, c="red", label="path points")
        axs[2].set_ylabel("z (m)")
        axs[2].set_xlabel("time (s)")
        axs[2].legend()
        axs[2].grid(True)

        plt.tight_layout()
        plt.show()
