import numpy as np
from numba import njit
import matplotlib.colors as mcolors
import h5py
import time

# LAPLACIAN
@njit
def laplacian(Z):

    nx, ny = Z.shape
    lap = np.empty_like(Z)

    for i in range(nx):

        ip = (i + 1) % nx
        im = (i - 1) % nx

        for j in range(ny):

            jp = (j + 1) % ny
            jm = (j - 1) % ny

            lap[i, j] = (
                Z[ip, j] +
                Z[im, j] +
                Z[i, jp] +
                Z[i, jm] -
                4 * Z[i, j]
            ) / (dx * dx)

    return lap

# STEP
@njit
def step(W, N, a_current, dt, dw, dn, m):

    lapW = laplacian(W)
    lapN = laplacian(N)

    W_new = W + dt * (
        a_current
        - W
        - W * N * N
        + dw * lapW
    )

    N_new = N + dt * (
        W * N * N
        - m * N
        + dn * lapN
    )

    return W_new, N_new

# SIMULATION
def run_simulation(a):

    print(f"\nSimulation running for a = {a:.3f}")

    np.random.seed(1)
    W = np.random.rand(Ngrid, Ngrid)
    Nfield = np.random.rand(Ngrid, Ngrid)

    functional_snapshots = []

    t0 = time.time()
    avg_step_time = None

    for i in range(nsteps):

        step_start = time.time()

        W, Nfield = step(W, Nfield, a, dt, dw, dn, m)
        W = W + sigma * np.random.randn(*W.shape)

        if i >= burn_in and i % functional_steps == 0:
            functional_snapshots.append(Nfield.copy())

        elapsed = time.time() - step_start

        if avg_step_time is None:
            avg_step_time = elapsed

        else:
            avg_step_time = 0.95 * avg_step_time + 0.05 * elapsed

    total_time = time.time() - t0
    print(f"Finished a = {a:.3f} | time = {total_time/60:.2f} min")

    return np.array(functional_snapshots), W, Nfield

# COARSE GRAIN
def coarse_grain(field, block_size):

    nx, ny = field.shape
    nx2 = nx // block_size
    ny2 = ny // block_size

    coarse = np.zeros((nx2, ny2))

    for i in range(nx2):
        for j in range(ny2):
            block = field[
                i*block_size:(i+1)*block_size,
                j*block_size:(j+1)*block_size
            ]
            coarse[i, j] = np.mean(block)

    return coarse

# NETWORK
def build_network(functional_snapshots):

    coarse_snapshots = []

    for field in functional_snapshots:
        coarse_snapshots.append(
            coarse_grain(field, block_size)
        )

    coarse_array = np.array(coarse_snapshots)
    time_series = np.transpose(coarse_array, (1, 2, 0))
    Nx, Ny, T = time_series.shape
    data = time_series.reshape(Nx * Ny, T)

    return np.corrcoef(data)

# PARAMETERS
L = 15
Ngrid = 210
dx = L / Ngrid
dt = 0.001
nsteps = 500000
dw = 1.0
dn = 0.001
m = 0.08
sigma = 0.0
functional_steps = 100
burn_in = nsteps // 2
pattern_size = 0.5
block_size = int(pattern_size / dx)

# sweep
a_values = np.linspace(0.32,0.036,100)
N_a = len(a_values)

# VALUES OF a TO SAVE FIELDS
save_field_values = np.array([

    0.32,
    0.30852525,
    0.27123232,
    0.24254545,
    0.19377778,
    0.15074747,
    0.09624242,
    0.06181818

])

# RUN + SAVE
with h5py.File("functional_networks_wn.h5", "w") as f:
    biomass_values = []
    max_biomass_values = []
    saved_fields = {}
    t_total = time.time()
    avg_time = None

    for i, a in enumerate(a_values):
        step_start = time.time()
        print(f"\n[{i+1}/{N_a}] Running a = {a:.8f}")
        snapshots, final_W, final_state = run_simulation(a)
        C = build_network(snapshots)
        biomass = final_state.mean()
        max_biomass = final_state.max()
        biomass_values.append(biomass)
        max_biomass_values.append(max_biomass)

        if np.any(np.isclose(a,save_field_values,atol=1e-4)):
            saved_fields[f"a_{a:.8f}"] = {
                "W": final_W.copy(),
                "N": final_state.copy()
            }

            print(f"Field saved for a={a:.8f}")

        grp = f.create_group(f"a_{i:03d}")
        grp.attrs["a"] = a
        grp.create_dataset("corr_matrix",data=C,compression="gzip")

        elapsed = time.time() - step_start

        if avg_time is None:
            avg_time = elapsed

        else:
            avg_time = (0.9 * avg_time + 0.1 * elapsed)

        remaining = N_a - i - 1
        eta = remaining * avg_time

        print(f"step time: {elapsed:.2f} s")
        print(f"ETA: {eta/60:.2f} min\n")

    f.create_dataset("a_values",data=a_values)
    f.create_dataset("biomass_final",data=np.array(biomass_values))
    f.create_dataset("biomass_max",data=np.array(max_biomass_values))

    # SAVE FIELDS
    fields_grp = f.create_group("saved_fields")

    for key, fields in saved_fields.items():
        grp = fields_grp.create_group(key)
        grp.create_dataset("W_field",data=fields["W"],compression="gzip")
        grp.create_dataset("N_field",data=fields["N"],compression="gzip")

print(f"\nTOTAL TIME: {(time.time()-t_total)/60:.2f} min")
print("Simulation finished and data saved.")