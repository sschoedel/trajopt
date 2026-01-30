"""
Trajectory optimization solver using direct collocation with CasADi.

Uses the Opti stack from CasADi to formulate and solve trajectory optimization
problems with the IPOPT solver.
"""

import casadi as ca
import numpy as np


class TrajectoryOptimizer:
    """
    Trajectory optimizer using direct collocation.

    Solves optimal control problems by discretizing the trajectory into N points
    and enforcing dynamics constraints via collocation.
    """

    def __init__(self, model, N, T):
        """
        Initialize the trajectory optimizer.

        Args:
            model: Dynamics model (e.g., BicycleModel instance)
            N: Number of collocation points
            T: Time horizon (seconds)
        """
        self.model = model
        self.N = N
        self.T = T
        self.dt = T / N

        # Storage for solution
        self.solution = None
        self.times = None
        self.states = None
        self.controls = None

    def solve(self, x_start, x_goal,
              W=None,
              u_lower=None,
              u_upper=None,
              initial_guess=None):
        """
        Solve the trajectory optimization problem.

        Args:
            x_start: Initial state vector (state_dim,)
            x_goal: Goal state vector (state_dim,)
            W: Cost matrix for control effort (control_dim x control_dim).
               If None, uses identity matrix.
            u_lower: Lower bounds on controls (control_dim,). If None, no lower bounds.
            u_upper: Upper bounds on controls (control_dim,). If None, no upper bounds.
            initial_guess: Dict with 'states' and 'controls' for warm start

        Returns:
            success: Boolean indicating if optimization succeeded
        """
        # Create Opti instance
        opti = ca.Opti()

        # Decision variables
        X = opti.variable(self.model.state_dim, self.N + 1)  # States at each point
        U = opti.variable(self.model.control_dim, self.N)     # Controls at each interval

        # Default cost matrix: identity
        if W is None:
            W = np.eye(self.model.control_dim)

        # Convert W to CasADi matrix
        W_ca = ca.DM(W)

        # Objective: minimize control effort (u^T * W * u)
        cost = 0
        for k in range(self.N):
            u_k = U[:, k]
            cost += self.dt * ca.mtimes([u_k.T, W_ca, u_k])

        opti.minimize(cost)

        # Initial condition constraint
        opti.subject_to(X[:, 0] == x_start)

        # Final condition constraint
        opti.subject_to(X[:, -1] == x_goal)

        # Dynamics constraints using RK4 integration
        for k in range(self.N):
            # Current state and control
            x_k = X[:, k]
            u_k = U[:, k]

            # Next state
            x_next = X[:, k + 1]

            # RK4 integration (4th order Runge-Kutta)
            # Assumes control is constant over the interval
            k1 = self.model.dynamics(x_k, u_k)
            k2 = self.model.dynamics(x_k + self.dt/2 * k1, u_k)
            k3 = self.model.dynamics(x_k + self.dt/2 * k2, u_k)
            k4 = self.model.dynamics(x_k + self.dt * k3, u_k)

            # RK4 update formula
            x_rk4 = x_k + self.dt / 6 * (k1 + 2*k2 + 2*k3 + k4)
            opti.subject_to(x_next == x_rk4)

        # Control limits (optional)
        if u_lower is not None and u_upper is not None:
            for i in range(self.model.control_dim):
                opti.subject_to(U[i, :] >= u_lower[i])
                opti.subject_to(U[i, :] <= u_upper[i])
        elif u_lower is not None:
            for i in range(self.model.control_dim):
                opti.subject_to(U[i, :] >= u_lower[i])
        elif u_upper is not None:
            for i in range(self.model.control_dim):
                opti.subject_to(U[i, :] <= u_upper[i])

        # Initial guess (warm start)
        if initial_guess is not None:
            opti.set_initial(X, initial_guess['states'])
            opti.set_initial(U, initial_guess['controls'])
        else:
            # Simple linear interpolation for states
            for i in range(self.model.state_dim):
                opti.set_initial(X[i, :], np.linspace(x_start[i], x_goal[i], self.N + 1))
            # Zero control as initial guess
            opti.set_initial(U, np.zeros((self.model.control_dim, self.N)))

        # Solver options
        opts = {
            'ipopt.print_level': 5,
            'print_time': True,
            'ipopt.max_iter': 1000,
            'ipopt.tol': 1e-6,
            'ipopt.acceptable_tol': 1e-4,
        }
        opti.solver('ipopt', opts)

        # Solve
        try:
            sol = opti.solve()

            # Extract solution
            self.solution = sol
            self.times = np.linspace(0, self.T, self.N + 1)
            self.states = sol.value(X)
            self.controls = sol.value(U)

            return True

        except RuntimeError as e:
            print(f"Optimization failed: {e}")

            # Try to extract debug solution
            try:
                self.solution = opti.debug
                self.times = np.linspace(0, self.T, self.N + 1)
                self.states = opti.debug.value(X)
                self.controls = opti.debug.value(U)
                print("Extracted debug solution (may not satisfy constraints)")
            except:
                pass

            return False

    def get_trajectory(self):
        """
        Get the optimized trajectory.

        Returns:
            times: Array of time points
            states: State trajectory (state_dim x N+1)
            controls: Control trajectory (control_dim x N)
        """
        if self.solution is None:
            raise ValueError("No solution available. Call solve() first.")

        return self.times, self.states, self.controls
