import numpy as np
from numba import njit
import matplotlib.colors as mcolors
import h5py
import time
from scipy.ndimage import label

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

# LOCALIZED STRUCTURE TEST
def remove_half_spots(Nfield):

    threshold = np.mean(Nfield) + 0.5*np.std(Nfield)
    mask = Nfield > threshold
    labels, num_spots = label(mask)

    if num_spots < 2:
        return Nfield.copy(), num_spots

    spot_ids = np.arange(1,num_spots+1)
    remove_ids = np.random.choice(
        spot_ids,
        size=num_spots//2,
        replace=False
    )

    N_removed = Nfield.copy()
    low_value = np.min(Nfield)

    for sid in remove_ids:
        N_removed[labels==sid] = low_value

    return N_removed, num_spots

def count_spots(Nfield):

    threshold = np.mean(Nfield) + 0.5*np.std(Nfield)
    mask = Nfield > threshold
    _, num = label(mask)

    return num



def localized_structure_test(W,Nfield,a):

    N_before = Nfield.copy()
    B_before = np.sum(N_before)

    spots_original = count_spots(N_before)

    N_removed, spots_before = remove_half_spots(N_before)
    B_removed = np.sum(N_removed)

    spots_removed = count_spots(N_removed)

    W_test = W.copy()
    N_test = N_removed.copy()

    for i in range(recovery_steps):

        W_test,N_test = step(
            W_test,
            N_test,
            a,
            dt,
            dw,
            dn,
            m
        )

        W_test += sigma*np.random.randn(*W_test.shape)

    B_after = np.sum(N_test)

    Recovery_B = ((B_after-B_removed)/(B_before-B_removed+1e-12))

    spots_after = count_spots(N_test)

    Recovery_S = ((spots_after-spots_removed)/(spots_original-spots_removed+1e-12))

    spatial_corr = np.corrcoef(N_before.flatten(),N_test.flatten())[0,1]

    return {
        "a":a,
        "B_before":B_before,
        "B_removed":B_removed,
        "B_after":B_after,
        "Recovery_B":Recovery_B,
        "spots_original":spots_original,
        "spots_removed":spots_removed,
        "spots_after":spots_after,
        "spots_before":spots_before,
        "Recovery_S":Recovery_S,
        "spatial_corr":spatial_corr

    }

# SIMULATION
def run_simulation(a):

    print(f"\nSimulation running for a = {a:.3f}")

    np.random.seed(1)
    W = np.random.rand(Ngrid,Ngrid)
    Nfield = np.random.rand(Ngrid,Ngrid)
    functional_snapshots = []
    t0 = time.time()
    avg_step_time = None

    for i in range(nsteps):

        step_start = time.time()

        W,Nfield = step(
            W,
            Nfield,
            a,
            dt,
            dw,
            dn,
            m
        )

        W = W + sigma*np.random.randn(*W.shape)

        if i >= burn_in and i % functional_steps == 0:

            functional_snapshots.append(Nfield.copy())

        elapsed = time.time()-step_start

        if avg_step_time is None:

            avg_step_time = elapsed

        else:

            avg_step_time = (0.95*avg_step_time+0.05*elapsed)

    total_time = time.time()-t0
    print(f"Finished a = {a:.3f} | "f"time = {total_time/60:.2f} min")

    return np.array(functional_snapshots), W, Nfield

# COARSE GRAIN
def coarse_grain(field,block_size):

    nx,ny = field.shape
    nx2 = nx//block_size
    ny2 = ny//block_size
    coarse = np.zeros((nx2,ny2))

    for i in range(nx2):
        for j in range(ny2):

            block = field[
                i*block_size:(i+1)*block_size,
                j*block_size:(j+1)*block_size
            ]

            coarse[i,j] = np.mean(block)

    return coarse

# NETWORK
def build_network(functional_snapshots):

    coarse_snapshots = []

    for field in functional_snapshots:

        coarse_snapshots.append(
            coarse_grain(
                field,
                block_size
            )
        )

    coarse_array = np.array(coarse_snapshots)
    time_series = np.transpose(coarse_array,(1,2,0))
    Nx,Ny,T = time_series.shape
    data = time_series.reshape(Nx*Ny,T)

    return np.corrcoef(data)

# PARAMETERS
L = 15
Ngrid = 210
dx = L / Ngrid
dt = 0.001
nsteps = 600000
dw = 1.0
dn = 0.001
m = 0.08
sigma = 0.0

functional_steps = 100
burn_in = nsteps // 2

pattern_size = 0.5
block_size = int(pattern_size / dx)

recovery_steps = 400000

# SWEEP
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
0.06181818])

# RUN + SAVE
with h5py.File("functional_networks_wn_LS_test_wn1.h5","w") as f:

    biomass_values = []
    max_biomass_values = []
    saved_fields = {}
    localized_results = []
    t_total = time.time()
    avg_time = None

    for i,a in enumerate(a_values):

        step_start = time.time()
        print(f"\n[{i+1}/{N_a}] Running a = {a:.8f}")

        snapshots, final_W, final_state = run_simulation(a)

        if a <= 0.19:

            print("Running localized structure test")

            ls = localized_structure_test(
                final_W,
                final_state,
                a
            )

            localized_results.append(ls)

            print(
                f"Recovery={ls['Recovery_B']:.3f} | "
                f"Spot recovery={ls['Recovery_S']:.3f} | "
                f"Corr={ls['spatial_corr']:.3f}"
            )

        C = build_network(snapshots)
        biomass = final_state.mean()
        max_biomass = final_state.max()
        biomass_values.append(biomass)
        max_biomass_values.append(max_biomass)

        if np.any(
            np.isclose(
                a,
                save_field_values,
                atol=1e-4
            )
        ):

            saved_fields[f"a_{a:.8f}"] = {"W": final_W.copy(),"N": final_state.copy()}

            print(f"Field saved for a={a:.8f}")

        grp = f.create_group(f"a_{i:03d}")
        grp.attrs["a"] = a
        grp.create_dataset("corr_matrix",data=C,compression="gzip")

        elapsed = time.time()-step_start

        if avg_time is None:
            avg_time = elapsed

        else:

            avg_time = (0.9*avg_time+0.1*elapsed)

        remaining = N_a-i-1
        eta = remaining*avg_time

        print(f"step time: {elapsed:.2f} s")

        print(f"ETA: {eta/60:.2f} min\n")

    f.create_dataset("a_values",data=a_values)
    f.create_dataset("biomass_final",data=np.array(biomass_values))
    f.create_dataset("biomass_max",data=np.array(max_biomass_values))

    # SAVE FIELDS
    fields_grp = f.create_group("saved_fields")

    for key,fields in saved_fields.items():

        grp = fields_grp.create_group(key)
        grp.create_dataset("W_field",data=fields["W"],compression="gzip")
        grp.create_dataset("N_field",data=fields["N"],compression="gzip")

    # SAVE LOCALIZED STRUCTURE TEST
    ls_grp = f.create_group("localized_structure_test")

    for i,res in enumerate(localized_results):

        grp = ls_grp.create_group(f"test_{i:03d}")

        for key,value in res.items():

            grp.attrs[key] = value

print(f"\nTOTAL TIME: {(time.time()-t_total)/60:.2f} min")
print("Simulation finished and data saved.")