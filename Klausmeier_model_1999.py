import numpy as np
import matplotlib.pyplot as plt

# ----------------------
# Parâmetros adimensionais
# ----------------------
a = 2.0
m = 0.45
v = 182.5
Nx, Ny = 100, 100
dx, dy = 1.0, 1.0
T = 100.0

# ----------------------
# Inicialização
# ----------------------
w = np.ones((Nx, Ny)) * a
n = np.random.rand(Nx, Ny) * 0.1

# ----------------------
# Funções de derivadas espaciais
# ----------------------
def laplacian(Z, dx, dy):
    Zxx = (np.roll(Z, -1, axis=0) - 2*Z + np.roll(Z, 1, axis=0)) / dx**2
    Zyy = (np.roll(Z, -1, axis=1) - 2*Z + np.roll(Z, 1, axis=1)) / dy**2
    return Zxx + Zyy

def advect_x(Z, dx):
    return (Z - np.roll(Z, 1, axis=0)) / dx

def rhs(w, n, dx, dy):
    dw = a - w - w*n**2 + v*advect_x(w, dx)
    dn = w*n**2 - m*n + laplacian(n, dx, dy)
    return dw, dn

# ----------------------
# Função RK4 para um passo
# ----------------------
def rk4_step(w, n, dt, dx, dy):
    dw1, dn1 = rhs(w, n, dx, dy)
    dw2, dn2 = rhs(w + 0.5*dt*dw1, n + 0.5*dt*dn1, dx, dy)
    dw3, dn3 = rhs(w + 0.5*dt*dw2, n + 0.5*dt*dn2, dx, dy)
    dw4, dn4 = rhs(w + dt*dw3, n + dt*dn3, dx, dy)
    
    w_new = w + dt*(dw1 + 2*dw2 + 2*dw3 + dw4)/6
    n_new = n + dt*(dn1 + 2*dn2 + 2*dn3 + dn4)/6
    
    return w_new, n_new

# ----------------------
# Loop temporal com passo adaptativo
# ----------------------
t = 0.0
dt = 0.01
dt_max = 0.05
dt_min = 1e-5
tol = 1e-3  # tolerância de erro para ajuste de dt

while t < T:
    # passo duplo
    w_full, n_full = rk4_step(w, n, dt, dx, dy)
    w_half1, n_half1 = rk4_step(w, n, dt/2, dx, dy)
    w_half2, n_half2 = rk4_step(w_half1, n_half1, dt/2, dx, dy)
    
    # estimativa de erro
    error_w = np.max(np.abs(w_full - w_half2))
    error_n = np.max(np.abs(n_full - n_half2))
    error = max(error_w, error_n)
    
    # ajustar dt
    if error > tol:
        dt = max(dt/2, dt_min)  # reduzir passo
        continue  # refaz o passo com dt menor
    else:
        t += dt
        w, n = w_half2, n_half2
        # aumentar dt se erro muito pequeno
        if error < tol/10:
            dt = min(dt*2, dt_max)
    
    # visualização periódica
    if int(t/dt) % 100 == 0:
        plt.clf()
        plt.imshow(n, origin='lower', cmap='Greens')
        plt.colorbar(label='Biomassa n')
        plt.title(f'Tempo τ = {t:.2f}, dt = {dt:.5f}')
        plt.pause(0.01)

plt.show()