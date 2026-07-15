import numpy as np
from numba import njit
import h5py
import time

# PDE SOLVER
@njit
def klausmeier(W, N, a_current, dt, dw, dn, m, dx):

    nx, ny = W.shape

    W_new = np.empty_like(W)
    N_new = np.empty_like(N)

    dx2 = dx * dx

    for i in range(nx):

        ip = (i + 1) % nx
        im = (i - 1) % nx

        for j in range(ny):

            jp = (j + 1) % ny
            jm = (j - 1) % ny

            lapW = (
                W[ip, j] + W[im, j] +
                W[i, jp] + W[i, jm] -
                4.0 * W[i, j]
            ) / dx2

            lapN = (
                N[ip, j] + N[im, j] +
                N[i, jp] + N[i, jm] -
                4.0 * N[i, j]
            ) / dx2

            growth = W[i, j] * N[i, j] * N[i, j]

            W_new[i, j] = (
                W[i, j]
                + dt * (
                    a_current
                    - W[i, j]
                    - growth
                    + dw * lapW
                )
            )

            N_new[i, j] = (
                N[i, j]
                + dt * (
                    growth
                    - m * N[i, j]
                    + dn * lapN
                )
            )

    return W_new, N_new

# COARSE GRAIN
def coarse_grain(field, block_size):

    nx, ny = field.shape
    nx2 = nx // block_size
    ny2 = ny // block_size

    field = field[:nx2 * block_size, :ny2 * block_size]

    return field.reshape(
        nx2, block_size,
        ny2, block_size
    ).mean(axis=(1, 3))

# CONTINUATION SIMULATION
def run_continuous_simulation(a_values):

    corr_matrices = []
    biomass_values = []

    t0 = time.time()
    avg_time = None

    W = np.random.rand(Ngrid, Ngrid)
    Nfield = np.random.rand(Ngrid, Ngrid)

    for ia, a in enumerate(a_values):

        step_start = time.time()

        print(f"Running for a = {a:.3f} ({ia+1}/{len(a_values)})")

        snapshots = []

        # evolve system under current a
        for i in range(nsteps):

            W, Nfield = klausmeier(
                W, Nfield, a, dt, dw, dn, m, dx
            )

            # noise
            W += sigma * np.random.randn(*W.shape)

            # sampling for functional network
            if i >= burn_in and i % functional_steps == 0:
                snapshots.append(coarse_grain(Nfield, block_size))

        snapshots = np.array(snapshots)

        # functional network
        C = np.corrcoef(snapshots.reshape(snapshots.shape[0], -1).T)

        biomass = Nfield.mean()

        corr_matrices.append(C)
        biomass_values.append(biomass)

        # timing
        elapsed = time.time() - step_start

        if avg_time is None:
            avg_time = elapsed
        else:
            avg_time = 0.9 * avg_time + 0.1 * elapsed

        remaining = len(a_values) - ia - 1
        eta = remaining * avg_time

        print(f"step time: {elapsed:.2f} s")
        print(f"ETA: {eta/60:.2f} min\n")

    print(f"Total time: { (time.time()-t0)/60:.2f} min")

    return corr_matrices, np.array(biomass_values)

# PARAMETERS
L = 15
Ngrid = 210
dx = L / Ngrid

dt = 0.001
nsteps = 500000

dw = 1.0
dn = 0.001
m = 0.08
sigma = 0.05

functional_steps = 100
burn_in = nsteps // 2

pattern_size = 0.5
block_size = int(pattern_size / dx)

# HIGH → LOW (continuation sweep)
a_values = np.linspace(0.32, 0.036, 100)

# RUN + SAVE
with h5py.File("functional_networks_continuous.h5", "w") as f:

    corr_matrices, biomass_values = run_continuous_simulation(a_values)

    f.create_dataset("a_values", data=a_values)
    f.create_dataset("biomass_final", data=biomass_values)

    for i, (a, C) in enumerate(zip(a_values, corr_matrices)):

        grp = f.create_group(f"a_{i:03d}")
        grp.attrs["a"] = a

        grp.create_dataset(
            "corr_matrix",
            data=C,
            compression="gzip"
        )