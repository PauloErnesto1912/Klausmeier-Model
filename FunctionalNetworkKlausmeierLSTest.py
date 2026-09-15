import numpy as np
from numba import njit
import h5py
import time
from collections import deque
from scipy.ndimage import label

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
                W[ip,j] +
                W[im,j] +
                W[i,jp] +
                W[i,jm] -
                4.0*W[i,j]
            ) / dx2

            lapN = (
                N[ip,j] +
                N[im,j] +
                N[i,jp] +
                N[i,jm] -
                4.0*N[i,j]
            ) / dx2

            growth = W[i,j]*N[i,j]*N[i,j]

            W_new[i,j] = (
                W[i,j]
                +
                dt*(
                    a_current
                    - W[i,j]
                    - growth
                    + dw*lapW
                )
            )

            N_new[i,j] = (N[i,j]+dt*(growth- m*N[i,j]+ dn*lapN))

    return W_new, N_new

# COARSE GRAIN
def coarse_grain(field, block_size):

    nx,ny = field.shape
    nx2 = nx//block_size
    ny2 = ny//block_size

    field = field[:nx2*block_size,:ny2*block_size]

    return field.reshape(
        nx2,
        block_size,
        ny2,
        block_size
    ).mean(axis=(1,3))

# LOCALIZED STRUCTURE TEST
def remove_half_spots(Nfield):

    threshold = (
        np.mean(Nfield)
        +
        0.5*np.std(Nfield)
    )

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
        N_removed[labels == sid] = low_value

    return N_removed, num_spots

def count_spots(Nfield):

    threshold = (
        np.mean(Nfield)
        +
        0.5*np.std(Nfield)
    )

    mask = Nfield > threshold
    _, num = label(mask)

    return num

def localized_structure_test(W,Nfield,a):

    results = {}

    # original state
    N_before = Nfield.copy()
    B_before = np.sum(N_before)

    # number of spots before perturbation
    spots_original = count_spots(N_before)

    # remove spots
    N_removed, spots_before = remove_half_spots(N_before)
    B_removed = np.sum(N_removed)

    # number of spots after perturbation
    spots_removed = count_spots(N_removed)

    # recovery simulation
    W_test = W.copy()
    N_test = N_removed.copy()

    for i in range(recovery_steps):

        W_test, N_test = klausmeier(
            W_test,
            N_test,
            a,
            dt,
            dw,
            dn,
            m,
            dx
        )

        W_test += (
            sigma*np.random.randn(
                *W_test.shape
            )
        )

    B_after = np.sum(N_test)

    # biomass recovery
    Recovery_B = ((B_after-B_removed)/(B_before-B_removed+1e-12))

    # spot recovery
    spots_after = count_spots(N_test)

    Recovery_S = ((spots_after-spots_removed)/(spots_original-spots_removed+1e-12))

    # spatial correlation
    spatial_corr = np.corrcoef(
        N_before.flatten(),
        N_test.flatten()
    )[0,1]

    results["a"] = a
    results["B_before"] = B_before
    results["B_removed"] = B_removed
    results["B_after"] = B_after
    results["Recovery_B"] = Recovery_B
    results["spots_original"] = spots_original
    results["spots_removed"] = spots_removed
    results["spots_after"] = spots_after
    results["spots_before"] = spots_before
    results["Recovery_S"] = Recovery_S
    results["spatial_corr"] = spatial_corr

    return results

# CONTINUATION SIMULATION

def run_continuous_simulation(a_values,save_field_values):

    corr_matrices = []
    biomass_values = []
    max_biomass_values = []
    saved_fields = {}
    localized_results = []
    t0 = time.time()
    avg_time = None

    W = np.random.rand(Ngrid,Ngrid)
    Nfield = np.random.rand(Ngrid,Ngrid)

    for ia,a in enumerate(a_values):

        step_start = time.time()

        print(
            f"Running a={a:.8f} "
            f"({ia+1}/{len(a_values)})"
        )

        biomass_history = []
        pattern_history = []
        stable_count = 0

        field_buffer = deque(maxlen=delta_delay)

        # BURN IN
        for i in range(max_burn_steps):

            field_buffer.append(Nfield.copy())

            W,Nfield = klausmeier(
                W,
                Nfield,
                a,
                dt,
                dw,
                dn,
                m,
                dx
            )

            W += sigma*np.random.randn(
                *W.shape
            )

            if len(field_buffer)==delta_delay:
                old_field = field_buffer[0]
                pattern_delta = (
                    np.linalg.norm(Nfield-old_field)/(np.linalg.norm(Nfield)+1e-12))

            if (
                i % check_interval == 0
                and i >= minimum_burn
            ):

                biomass_history.append(Nfield.mean())
                pattern_history.append(pattern_delta)

                if len(biomass_history)>window_size:
                    biomass_history.pop(0)
                    pattern_history.pop(0)

                if len(biomass_history)==window_size:
                    biomass_std = np.std(biomass_history)

                    pattern_delta_mean = np.mean(pattern_history)

                    pattern_delta_std = np.std(pattern_history)

                    if (biomass_std < biomass_tol and pattern_delta_mean < pattern_mean_tol and pattern_delta_std < pattern_std_tol):
                        stable_count +=1

                    else:
                        stable_count=0

                if stable_count>=required_stable_checks:
                    print(f"Burn-in finished at {i}")
                    break

        # LS TEST
        if a <= 0.19:
            print("Running LS perturbation test")

            ls = localized_structure_test(W,Nfield,a)
            localized_results.append(ls)

            print(f"Recovery={ls['Recovery_B']:.3f}, "
                  f"Spot Recovery={ls['Recovery_S']:.3f}, "
                  f"Corr={ls['spatial_corr']:.3f}")

        # SNAPSHOT COLLECTION
        snapshots=[]

        for i in range(collection_steps):

            W,Nfield = klausmeier(W,Nfield,a,dt,dw,dn,m,dx)

            W += sigma*np.random.randn(*W.shape)

            if i % functional_steps ==0:

                snapshots.append(coarse_grain(Nfield,block_size))

        snapshots=np.array(snapshots)

        C=np.corrcoef(snapshots.reshape(snapshots.shape[0],-1).T)

        biomass_values.append(Nfield.mean())
        max_biomass_values.append(Nfield.max())
        corr_matrices.append(C)

        if np.any(
            np.isclose(
                a,
                save_field_values,
                atol=1e-4)):

            saved_fields[f"a_{a:.8f}"] = {
                "W": W.copy(),
                "N": Nfield.copy()}

            print(f"Field saved for a={a:.8f}")

        elapsed=time.time()-step_start

        print(f"time {elapsed:.2f}s\n")

    return (
        corr_matrices,
        np.array(biomass_values),
        np.array(max_biomass_values),
        saved_fields,
        localized_results
    )

# PARAMETERS
L = 15
Ngrid = 210
dx = L / Ngrid
dt = 0.001
dw = 1.0
dn = 0.001
m = 0.08
sigma = 0.0

# DYNAMIC BURN-IN PARAMETERS
max_burn_steps = 500000
minimum_burn = 100000
check_interval = 200
delta_delay = 500
window_size = 50
required_stable_checks = 5
biomass_tol = 5e-3
pattern_mean_tol = 5e-2
pattern_std_tol = 1e-3

# SNAPSHOT PARAMETERS
n_snapshots = 2000
functional_steps = 100
collection_steps = (n_snapshots*functional_steps)
pattern_size = 0.5
block_size = int(pattern_size/dx)

# LOCALIZED STRUCTURE TEST
recovery_steps = 400000

# CONTINUATION PARAMETER
a_values = np.linspace(0.32,0.036,100)

# SAVE FIELDS
save_field_values = np.array([
0.32,
0.30852525,
0.27123232,
0.24254545,
0.19377778,
0.15074747,
0.09624242,
0.06181818])

# RUN SIMULATION + SAVE
with h5py.File(
"functional_networks_continuous_dynamic_LS_test_wn1.h5",
"w"
) as f:

    (
        corr_matrices,
        biomass_values,
        max_biomass_values,
        saved_fields,
        localized_results

    ) = run_continuous_simulation(
        a_values,
        save_field_values
    )

    # BASIC DATA
    f.create_dataset("a_values",data=a_values)
    f.create_dataset("biomass_final",data=biomass_values)
    f.create_dataset("biomass_max",data=max_biomass_values)

    # CORRELATION MATRICES
    corr_grp = f.create_group("correlation_matrices")

    for i,(a,C) in enumerate(
        zip(a_values,corr_matrices)
    ):

        grp = corr_grp.create_group(f"a_{i:03d}")
        grp.attrs["a"] = a
        grp.create_dataset("corr_matrix",data=C,compression="gzip")

    # SAVED FIELDS
    fields_grp = f.create_group("saved_fields")

    for key,fields in saved_fields.items():

        grp = fields_grp.create_group(key)
        grp.create_dataset("W_field",data=fields["W"],compression="gzip")
        grp.create_dataset("N_field",data=fields["N"],compression="gzip")

    # LOCALIZED STRUCTURE TEST

    ls_grp = f.create_group("localized_structure_test")

    for i,res in enumerate(
        localized_results
    ):

        grp = ls_grp.create_group(f"test_{i:03d}")

        for key,value in res.items():
            grp.attrs[key] = value

print("Simulation finished and data saved.")