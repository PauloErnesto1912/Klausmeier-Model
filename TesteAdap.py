import numpy as np
from numba import njit
import h5py
import time
from collections import deque

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
    field = field[:nx2 * block_size,:ny2 * block_size]

    return field.reshape(nx2,block_size,ny2,block_size).mean(axis=(1, 3))

# CONTINUATION SIMULATION
def run_continuous_simulation(a_values, save_field_values):

    corr_matrices = []
    biomass_values = []
    max_biomass_values = []
    saved_fields = {}
    t0 = time.time()
    avg_time = None

    # initial condition
    W = np.random.rand(Ngrid, Ngrid)
    Nfield = np.random.rand(Ngrid, Ngrid)

    for ia, a in enumerate(a_values):
        step_start = time.time()
        print(f"Running for a = {a:.8f} "f"({ia+1}/{len(a_values)})")

        # DYNAMIC BURN-IN
        biomass_history = []
        pattern_history = []
        stable_count = 0
        biomass_std = np.nan
        pattern_delta_mean = np.nan
        pattern_delta_std = np.nan
        field_buffer = deque(maxlen=delta_delay)

        for i in range(max_burn_steps):
            field_buffer.append(Nfield.copy())

            W, Nfield = klausmeier(W,Nfield,a,dt,dw,dn,m,dx)
            W += sigma * np.random.randn(*W.shape)

            # PATTERN DELTA
            if len(field_buffer) == delta_delay:
                old_field = field_buffer[0]
                pattern_delta = np.linalg.norm(Nfield - old_field) / (np.linalg.norm(Nfield)+ 1e-12)

            # CHECKS
            if (i % check_interval == 0 and i >= minimum_burn):

                biomass_history.append(Nfield.mean())
                pattern_history.append(pattern_delta)

                if len(biomass_history) > window_size:
                    biomass_history.pop(0)

                if len(pattern_history) > window_size:
                    pattern_history.pop(0)

                if len(biomass_history) == window_size:
                    biomass_std = np.std(biomass_history)
                    pattern_delta_mean = np.mean(pattern_history)
                    pattern_delta_std = np.std(pattern_history)

                    if (
                        biomass_std < biomass_tol
                        and pattern_delta_mean < pattern_mean_tol
                        and pattern_delta_std < pattern_std_tol):
                        stable_count += 1

                    else:
                        stable_count = 0

                if stable_count >= required_stable_checks:
                    print(f"Burn-in finished at step {i}")
                    print(f"biomass std = {biomass_std:.3e}")
                    print(f"pattern delta mean = {pattern_delta_mean:.3e}")
                    print(f"pattern delta std = {pattern_delta_std:.3e}")

                    break

        else:
            print(f"Warning: maximum burn-in reached for a={a:.5f}")
            print(f"Last biomass std = {biomass_std:.3e}")
            print(f"Last pattern delta mean = {pattern_delta_mean:.3e}")
            print(f"Last pattern delta std = {pattern_delta_std:.3e}")

        # SNAPSHOT COLLECTION
        snapshots = []

        for i in range(collection_steps):

            W, Nfield = klausmeier(W,Nfield,a,dt,dw,dn,m,dx)

            W += sigma * np.random.randn(*W.shape)

            if i % functional_steps == 0:
                snapshots.append(coarse_grain(Nfield,block_size))

        snapshots = np.array(snapshots)

        print(f"Snapshots collected: {len(snapshots)}")

        # SAVE FIELD IF REQUIRED
        if np.any(np.isclose(a,save_field_values,atol=1e-4)):

            saved_fields[f"a_{a:.8f}"] = {"W": W.copy(),"N": Nfield.copy()}

            print(f"Field saved for a={a:.8f}")

        # FUNCTIONAL NETWORK
        C = np.corrcoef(snapshots.reshape(snapshots.shape[0],-1).T)

        # BIOMASS
        biomass = Nfield.mean()
        max_biomass = Nfield.max()
        corr_matrices.append(C)
        biomass_values.append(biomass)
        max_biomass_values.append(max_biomass)

        # TIMING
        elapsed = time.time() - step_start

        if avg_time is None:
            avg_time = elapsed

        else:
            avg_time = (0.9 * avg_time + 0.1 * elapsed)

        remaining = len(a_values) - ia - 1
        eta = remaining * avg_time

        print(f"step time: {elapsed:.2f} s")
        print(f"ETA: {eta/60:.2f} min\n")

    print(f"Total time: {(time.time()-t0)/60:.2f} min")

    return (corr_matrices,np.array(biomass_values),np.array(max_biomass_values),saved_fields)

# PARAMETERS
L = 15
Ngrid = 210
dx = L / Ngrid
dt = 0.001
dw = 1.0
dn = 0.001
m = 0.08
sigma = 0.05

# DYNAMIC BURN-IN PARAMETERS
max_burn_steps = 500000
minimum_burn = 100000
check_interval = 200

# pattern comparison
delta_delay = 500
window_size = 50
required_stable_checks = 5

# tolerances
biomass_tol = 5e-3
pattern_mean_tol = 5e-2
pattern_std_tol = 1e-3

# SNAPSHOTS PARAMETERS
n_snapshots = 2000
functional_steps = 100
collection_steps = (n_snapshots * functional_steps)
pattern_size = 0.5
block_size = int(pattern_size / dx)
a_values = np.linspace(0.32,0.036,100)

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
with h5py.File(
    "functional_networks_continuous_dynamic.h5",
    "w"
) as f:
    (corr_matrices,biomass_values,max_biomass_values,saved_fields) = run_continuous_simulation(a_values,save_field_values)

    f.create_dataset("a_values",data=a_values)
    f.create_dataset("biomass_final",data=biomass_values)
    f.create_dataset("biomass_max",data=max_biomass_values)

    # SAVE CORRELATION MATRICES
    for i, (a, C) in enumerate(zip(a_values,corr_matrices)):
        
        grp = f.create_group(f"a_{i:03d}")
        grp.attrs["a"] = a
        grp.create_dataset("corr_matrix",data=C,compression="gzip")

    # SAVE FIELDS
    fields_grp = f.create_group("saved_fields")

    for key, fields in saved_fields.items():
        grp = fields_grp.create_group(key)
        grp.create_dataset("W_field",data=fields["W"],compression="gzip")
        grp.create_dataset("N_field",data=fields["N"],compression="gzip")

print("Simulation finished and data saved.")