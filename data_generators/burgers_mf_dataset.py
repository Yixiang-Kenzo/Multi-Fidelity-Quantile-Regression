"""
Multi-fidelity Burgers equation dataset generator.

Burgers equation:  du/dt + u * du/dx = nu * d²u/dx²

Physics: A velocity profile evolves over time. The nonlinear term (u * du/dx)
steepens the profile; viscosity (nu) smooths it. When nu is small, sharp
shocks form — coarse meshes can't resolve these, creating non-stationary
LF-HF gaps that challenge GP models.

Input X (6D):
  x0: A      - amplitude of initial Gaussian bump        [0.5, 2.0]
  x1: sigma  - width of initial bump                     [0.05, 0.2]
  x2: x0     - center position of bump                   [0.3, 0.7]
  x3: nu     - viscosity (small = shocks)                [0.001, 0.1]
  x4: T      - observation time                          [0.1, 0.5]
  x5: x_obs  - observation point (where we read Y)       [0.3, 0.9]

Output Y (scalar): velocity u(x_obs, T)

LF: Solve on coarse mesh (N_x = 64 grid points)
HF: Solve on fine mesh   (N_x = 512 grid points)

The LF-HF gap is non-stationary:
  - Large nu (smooth solution): LF ≈ HF, gap ≈ 0
  - Small nu + shock near x_obs: LF badly smears shock, gap is large
  - Small nu + shock far from x_obs: gap small again
"""

import numpy as np
from scipy.integrate import solve_ivp


def burgers_rhs(t, u, dx, nu):
    """
    Right-hand side for method of lines.
    Spatial discretization of: -u * du/dx + nu * d²u/dx²

    Uses conservative form for advection: -d(u²/2)/dx with upwind bias,
    and central difference for diffusion.
    Boundary: u[0] = u[-1] = 0 (Dirichlet).
    """
    n = len(u)
    dudt = np.zeros(n)

    # Interior points (1 to n-2)
    # Diffusion: central difference
    diffusion = nu * (u[2:] - 2*u[1:-1] + u[:-2]) / dx**2

    # Advection: upwind scheme (depends on sign of u)
    # For u > 0: backward difference  du/dx ≈ (u[i] - u[i-1]) / dx
    # For u < 0: forward difference   du/dx ≈ (u[i+1] - u[i]) / dx
    u_mid = u[1:-1]
    dudx_backward = (u[1:-1] - u[:-2]) / dx
    dudx_forward  = (u[2:] - u[1:-1]) / dx

    advection = np.where(u_mid >= 0,
                         u_mid * dudx_backward,
                         u_mid * dudx_forward)

    dudt[1:-1] = -advection + diffusion
    # Boundaries stay 0

    return dudt


def solve_burgers(A, sigma, x0, nu, T, n_x):
    """
    Solve Burgers equation on [0, 1] with n_x interior grid points.

    Initial condition: u(x, 0) = A * exp(-(x - x0)^2 / (2 * sigma^2))

    Returns: x_grid, u_final (solution at time T)
    """
    # Grid: n_x points including boundaries
    x_grid = np.linspace(0, 1, n_x)
    dx = x_grid[1] - x_grid[0]

    # Initial condition
    u0 = A * np.exp(-(x_grid - x0)**2 / (2 * sigma**2))
    # Enforce boundary conditions
    u0[0] = 0.0
    u0[-1] = 0.0

    # Solve ODE system
    # Use BDF for potentially stiff problems (low viscosity)
    sol = solve_ivp(
        burgers_rhs,
        t_span=(0, T),
        y0=u0,
        method='BDF',
        args=(dx, nu),
        rtol=1e-6,
        atol=1e-8,
        max_step=dx / 5,   # CFL-like constraint for safety
    )

    if not sol.success:
        # Fallback: try with tighter tolerances
        sol = solve_ivp(
            burgers_rhs,
            t_span=(0, T),
            y0=u0,
            method='Radau',
            args=(dx, nu),
            rtol=1e-8,
            atol=1e-10,
        )

    return x_grid, sol.y[:, -1]  # final time solution


def evaluate_single(params, n_x):
    """
    Given params = [A, sigma, x0, nu, T, x_obs], solve Burgers on n_x grid
    and return u(x_obs, T) by interpolation.
    """
    A, sigma, x0, nu, T, x_obs = params

    x_grid, u_final = solve_burgers(A, sigma, x0, nu, T, n_x)

    # Interpolate to get value at x_obs
    y = np.interp(x_obs, x_grid, u_final)
    return y


def generate_dataset(N=3000, seed=42, n_x_lf=16, n_x_hf=128):
    """
    Generate N samples of (X, Y_lf, Y_hf).

    X: (N, 6) input parameters
    Y_lf: (N,) low-fidelity outputs (coarse mesh)
    Y_hf: (N,) high-fidelity outputs (fine mesh)
    """
    rng = np.random.RandomState(seed)

    # Sample input parameters uniformly
    A     = rng.uniform(0.5, 2.0, N)
    sigma = rng.uniform(0.05, 0.2, N)
    x0    = rng.uniform(0.3, 0.7, N)
    nu    = 10**rng.uniform(-3, -1, N)  # log-uniform: 0.001 to 0.1
    T     = rng.uniform(0.1, 0.5, N)
    x_obs = rng.uniform(0.3, 0.9, N)

    X = np.column_stack([A, sigma, x0, nu, T, x_obs])

    Y_lf = np.zeros(N)
    Y_hf = np.zeros(N)

    for i in range(N):
        if i % 500 == 0:
            print(f"  Solving {i}/{N}...")

        params = X[i]
        Y_lf[i] = evaluate_single(params, n_x_lf)
        Y_hf[i] = evaluate_single(params, n_x_hf)

    return X, Y_lf, Y_hf


# ============================================================
# Quick test: solve one example and print diagnostics
# ============================================================
if __name__ == '__main__':
    import time

    # Test single solve at two fidelities
    test_params = [1.5, 0.1, 0.5, 0.005, 0.3, 0.7]  # low viscosity → shock
    print("Test params: A=1.5, sigma=0.1, x0=0.5, nu=0.005, T=0.3, x_obs=0.7")

    t0 = time.time()
    y_lf = evaluate_single(test_params, n_x=32)
    t_lf = time.time() - t0

    t0 = time.time()
    y_hf = evaluate_single(test_params, n_x=256)
    t_hf = time.time() - t0

    print(f"  LF (32 pts):  y = {y_lf:.6f}  [{t_lf:.3f}s]")
    print(f"  HF (256 pts): y = {y_hf:.6f}  [{t_hf:.3f}s]")
    print(f"  Gap: {y_hf - y_lf:.6f}")
    print(f"  Speedup: {t_hf/t_lf:.1f}x")

    # Test smooth case (high viscosity)
    test_smooth = [1.0, 0.1, 0.5, 0.05, 0.3, 0.7]
    y_lf_s = evaluate_single(test_smooth, n_x=32)
    y_hf_s = evaluate_single(test_smooth, n_x=256)
    print(f"\nSmooth case (nu=0.05):")
    print(f"  LF: {y_lf_s:.6f}, HF: {y_hf_s:.6f}, Gap: {y_hf_s - y_lf_s:.6f}")

    # Small dataset for diagnostics
    print("\n--- Generating small test dataset (N=200) ---")
    t0 = time.time()
    X, Y_lf, Y_hf = generate_dataset(N=200, seed=42)
    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s")
    print(f"  r(LF, HF) = {np.corrcoef(Y_lf, Y_hf)[0,1]:.4f}")
    print(f"  Y_lf: mean={Y_lf.mean():.4f}, std={Y_lf.std():.4f}")
    print(f"  Y_hf: mean={Y_hf.mean():.4f}, std={Y_hf.std():.4f}")
    print(f"  Gap:  mean={np.mean(Y_hf-Y_lf):.4f}, std={np.std(Y_hf-Y_lf):.4f}")
    print(f"  |Gap| max: {np.max(np.abs(Y_hf-Y_lf)):.4f}")
    print(f"  Estimated time for N=5000: {elapsed/200*5000/60:.0f} min")
