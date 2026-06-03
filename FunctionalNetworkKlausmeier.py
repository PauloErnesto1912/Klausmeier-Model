import numpy as np
from numba import njit
import matplotlib.colors as mcolors
import h5py

# Calculate the Laplacian using finite differences
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

# Solver for one time step of the Klausmeier model
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

# Simulation loop
def run_simulation(a):
    print(f"Simulation running for a = {a}...")
    np.random.seed(1) # evaluate the same noise for each a
    W = np.random.rand(Ngrid, Ngrid)
    Nfield = np.random.rand(Ngrid, Ngrid)

    functional_snapshots = []

    for i in range(nsteps):

        W, Nfield = step(W, Nfield, a, dt, dw, dn, m)
        W = W + sigma * np.random.randn(*W.shape)

        # Passing the transient phase
        if i >= burn_in:
            if i % functional_steps == 0:
                functional_snapshots.append(Nfield.copy())

    return np.array(functional_snapshots)

# Defining coarse-grain function to calculate average vegetation density in blocks
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

# Building the functional network
def build_network(functional_snapshots):
    coarse_snapshots = []

    for field in functional_snapshots:
        coarse_snapshots.append(
            coarse_grain(field, block_size)
        )

    coarse_array = np.array(coarse_snapshots) # shape: (T, Nx, Ny)

    time_series = np.transpose(coarse_array, (1, 2, 0)) # shape: (Nx, Ny, T)

    Nx, Ny, T = time_series.shape

    Nnodes = Nx * Ny
    data = time_series.reshape(Nnodes, T)

    corr_matrix = np.corrcoef(data)
    return corr_matrix

# Parameters
L = 15 #5.0
Ngrid = 210 #70
dx = L / Ngrid

dt = 0.001
nsteps = 500000

dw = 1.0
dn = 0.001
m = 0.08
a = 0.27

sigma = 0.05  #noise amplitude 

#Simulation colors
colors = [
    "#d95f02",  
    "#fdae61",  
    "#fee08b",  
    "#d9ec94",  
    "#a5d271",
    "#60b53e",
    "#4ba973",
    "#218D43",
]

desert_green = mcolors.LinearSegmentedColormap.from_list(
    "desert_green",
    colors,
    N=256
)

# Data for funcitonal network
functional_steps = 100
burn_in = nsteps // 2

# Defining size of nodes in the functional network
pattern_size = 0.5  # meters
block_size = int(pattern_size / dx)

# Generating coarse-snapshots
a_min = 0.01
a_max = 0.35
N_a = 50
a_values = np.linspace(a_min, a_max, N_a)

correlation_matrices = []

with h5py.File("functional_networks.h5", "w") as f:

    for i, a in enumerate(a_values):

        functional_snapshots = run_simulation(a)
        C = build_network(functional_snapshots)

        grp = f.create_group(f"a_{i:03d}")
        grp.attrs["a"] = a

        grp.create_dataset(
            "corr_matrix",
            data=C,
            compression="gzip"
        )