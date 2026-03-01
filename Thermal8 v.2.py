# -*- coding: utf-8 -*-
"""
Thermal8_v1_eclipse.py

Adds hard-coded eclipse functionality to your original multi-layer,
1D semi-implicit conduction model, then plots temperature vs. depth
and time in a 3D surface plot.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # For 3D plotting

# =============================================================================
# 1) Define layers (unchanged)
# =============================================================================
layers = [
    {"name": "Silicon solar cells", "thickness": 0.02, "rho": 2320.0, "cp": 800.0, "k": 150.0, "Q": 0.0},
    {"name": "Thermal compound 1", "thickness": 0.02, "rho": 2100.0, "cp": 1000.0, "k": 1.5, "Q": 0.0},
    {"name": "FR4 circuit board",  "thickness": 0.063, "rho": 1850.0, "cp": 820.0, "k": 0.3,   "Q": 200.0/0.063},
    {"name": "Thermal compound 2", "thickness": 0.02, "rho": 2100.0, "cp": 1000.0, "k": 1.5, "Q": 0.0},
    {"name": "Aluminum radiator",  "thickness": 0.032, "rho": 2700.0, "cp": 877.0, "k": 205.0, "Q": 0.0}
]
L_total = sum(layer["thickness"] for layer in layers)

# =============================================================================
# 2) Time & space discretization (unchanged)
# =============================================================================
t_total = 64000.0        # s
dt      = 0.1           # s
n_steps = int(t_total / dt)

N       = 31
dx      = L_total / (N - 1)
x       = np.linspace(0, L_total, N)

# =============================================================================
# 3) Material arrays (unchanged)
# =============================================================================
rho_arr = np.zeros(N)
cp_arr  = np.zeros(N)
k_arr   = np.zeros(N)
Q_arr   = np.zeros(N)

# build boundaries for layer assignment
boundaries = [0.0]
for layer in layers:
    boundaries.append(boundaries[-1] + layer["thickness"])
boundaries = np.array(boundaries)

for i, xi in enumerate(x):
    # assign each grid point to its layer
    for j in range(len(layers)):
        if boundaries[j] <= xi < boundaries[j+1]:
            rho_arr[i] = layers[j]["rho"]
            cp_arr[i]  = layers[j]["cp"]
            k_arr[i]   = layers[j]["k"]
            Q_arr[i]   = layers[j]["Q"]
            break
    else:
        # last boundary point
        rho_arr[i] = layers[-1]["rho"]
        cp_arr[i]  = layers[-1]["cp"]
        k_arr[i]   = layers[-1]["k"]
        Q_arr[i]   = layers[-1]["Q"]

# precompute interface conductivities
def k_interface(k1, k2):
    return 0.0 if (k1 + k2) == 0 else 2*k1*k2/(k1+k2)
k_half = np.array([k_interface(k_arr[i], k_arr[i+1]) for i in range(N-1)])

# =============================================================================
# 4) Boundary and material constants (mostly unchanged)
# =============================================================================
sigma      = 5.670374419e-8  # W/m²·K⁴
solar_flux = 1361.0          # W/m²
T_env      = 2.7             # K

alpha_top  = 0.9  # solar absorptivity
eps_top    = 0.9  # top emissivity
eps_bot    = 0.85 # bottom emissivity
A_top      = 1.0  # m²
A_bot      = 1.5  # m²

# =============================================================================
# 5) HARD-CODED ECLIPSE PROFILE
# =============================================================================
# Example: 90-min orbit with 35-min eclipse
orbit_period_s     = 90 * 60
eclipse_duration_s = 35 * 60
cycle_steps        = int(orbit_period_s / dt)
eclipse_steps      = int(eclipse_duration_s / dt)

eclipse_mask = np.ones(n_steps, dtype=int)
for start in range(0, n_steps, cycle_steps):
    eclipse_mask[start:start + eclipse_steps] = 0

# =============================================================================
# 6) Initialize temperature history
# =============================================================================
T = np.full(N, 290.0)           # K
T_hist = np.zeros((n_steps, N))
T_hist[0] = T

# =============================================================================
# 7) Time integration loop (add eclipse to solar term)
# =============================================================================
for n in range(n_steps - 1):
    A = np.zeros((N, N))
    r = np.zeros(N)

    # determine sunlit or eclipse
    in_sun = eclipse_mask[n]

    # top boundary
    q_solar = alpha_top * A_top * solar_flux * in_sun
    q_rad   = eps_top * A_top * sigma * (T[0]**4 - T_env**4)
    q_top   = q_solar - q_rad

    A[0,0] = rho_arr[0]*cp_arr[0]/dt + k_half[0]/dx**2
    A[0,1] = -k_half[0]/dx**2
    r[0]   = rho_arr[0]*cp_arr[0]/dt*T[0] + q_top/dx - Q_arr[0]

    # interior nodes
    for i in range(1, N-1):
        A[i,i-1] = -k_half[i-1]/dx**2
        A[i,i]   = rho_arr[i]*cp_arr[i]/dt + (k_half[i-1]+k_half[i])/dx**2
        A[i,i+1] = -k_half[i]/dx**2
        r[i]     = rho_arr[i]*cp_arr[i]/dt*T[i] + Q_arr[i]

    # bottom boundary
    q_bot = eps_bot * A_bot * sigma * (T[-1]**4 - T_env**4)
    A[-1,-2] = -k_half[-1]/dx**2
    A[-1,-1] = rho_arr[-1]*cp_arr[-1]/dt + k_half[-1]/dx**2
    r[-1]    = rho_arr[-1]*cp_arr[-1]/dt*T[-1] - q_bot/dx - Q_arr[-1]

    # solve
    T = np.linalg.solve(A, r)
    T_hist[n+1] = T

# =============================================================================
# 8) 3D Surface Plot (unchanged)
# =============================================================================
time_array   = np.linspace(0, t_total, n_steps) / 3600.0  # hours
X, Y         = np.meshgrid(x, time_array)

fig = plt.figure(figsize=(10, 6))
ax  = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(
    X,        # depth (m)
    Y,        # time (hr)
    T_hist,   # temperature (K)
    cmap='viridis',
    linewidth=0,
    antialiased=True
)
ax.set_xlabel("Position (m)")
ax.set_ylabel("Time (hours)")
ax.set_zlabel("Temperature (K)")
ax.set_title("Temperature Evolution with Eclipse Cycles")
fig.colorbar(surf, shrink=0.5, aspect=10, label="Temperature (K)")
plt.tight_layout()
plt.show()

# =============================================================================
# 9) Layer Temperature Statistics Over Entire Simulation
# =============================================================================
print("Layer Temperature Statistics (°C) Over Entire Simulation:\n")

for j, layer in enumerate(layers):
    # build mask for this layer's nodes
    if j < len(layers) - 1:
        mask = (x >= boundaries[j]) & (x < boundaries[j+1])
    else:
        mask = (x >= boundaries[j]) & (x <= boundaries[j+1])

    # extract temperature history (K) for those nodes, flatten to 1D
    temps_K = T_hist[:, mask].flatten()
    max_C   = temps_K.max() - 273.15
    min_C   = temps_K.min() - 273.15
    avg_C   = temps_K.mean() - 273.15

    print(f"{layer['name']}:")
    print(f"   🔥 Max: {max_C:6.2f} °C")
    print(f"   ❄️  Min: {min_C:6.2f} °C")
    print(f"   📊 Avg: {avg_C:6.2f} °C\n")
