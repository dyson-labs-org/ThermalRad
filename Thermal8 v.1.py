# -*- coding: utf-8 -*-
"""
Modular Multi-Layer 1D Thermal Conduction Model Using a Semi-Implicit (Backward Euler) Scheme

This model simulates conduction through an arbitrary number of layers. For each layer,
the following properties are defined in a dictionary:
  - 'name': A descriptive name of the layer.
  - 'thickness': The layer thickness in meters.
  - 'rho': Density in kg/m³.
  - 'cp': Specific heat in J/(kg·K).
  - 'k': Thermal conductivity in W/(m·K).
  - 'Q': Volumetric internal heating in W/m³.

Boundary conditions:
  - Top (x = 0): Net flux = (solar absorption) − (radiative loss).
  - Bottom (x = L_total): Radiative loss to deep space.

Other notes:
  - The simulation time is in seconds, but the 3D plot displays time in hours.
  - The printed average temperatures are converted to °C.
  
Users can easily change the number of layers and adjust material properties by modifying the list `layers`.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # For 3D plotting

# =============================================================================
# 1) Simulation Setup (Time and Spatial Discretization)
# =============================================================================
# Define the layers as a list of dictionaries.
# (Edit this list to change the number of layers and their properties.)
# The volumetric heating 'Q' should be specified in W/m³.
layers = [
    {
        "name": "Silicon solar cells",
        "thickness": 0.002,      # in meters
        "rho": 2320.0,           # kg/m³
        "cp": 800.0,             # J/(kg·K)
        "k": 150.0,              # W/(m·K)
        "Q": 0.0                 # W/m³
    },
    {
        "name": "FR4 circuit board",
        "thickness": 0.003,      # in meters
        "rho": 1850.0,           # kg/m³
        "cp": 820.0,             # J/(kg·K)
        "k": 0.3,                # W/(m·K)
        # Total power of 200 W is distributed over this layer.
        "Q": 200.0 / 0.003       # W/m³ (adjust if thickness is changed)
    },
    {
        "name": "Aluminum radiator",
        "thickness": 0.003,      # in meters
        "rho": 2700.0,           # kg/m³
        "cp": 877.0,             # J/(kg·K)
        "k": 205.0,              # W/(m·K)
        "Q": 0.0                 # W/m³
    }
]

# Compute total thickness from the layers.
L_total = sum(layer["thickness"] for layer in layers)

# Time discretization (in seconds)
t_total = 6000.0         # Total simulation time [s]
dt = 0.1                 # Time step [s]
n_steps = int(t_total / dt)

# Spatial discretization
N = 31                                # Number of spatial grid points
dx = L_total / (N - 1)                # Uniform grid spacing [m]
x = np.linspace(0, L_total, N)        # Spatial grid positions

# Effective surface areas (in m²)
A_top = 1.0   # Effective area for solar absorption at the top (layer 1)
A_bot = 1.0   # Effective area for radiation at the bottom (last layer)

# =============================================================================
# 2) Build Material Property Arrays Using Layer Dictionaries
# =============================================================================
# Preallocate arrays for the material properties at each grid cell.
rho_arr = np.zeros(N)
cp_arr  = np.zeros(N)
k_arr   = np.zeros(N)
Q_arr   = np.zeros(N)
# Each cell's effective thickness is assumed to be dx.
cell_length = np.full(N, dx)

# Compute cumulative boundaries for the layers.
boundaries = [0.0]
for layer in layers:
    boundaries.append(boundaries[-1] + layer["thickness"])
boundaries = np.array(boundaries)  # Now boundaries has length (number of layers + 1)

# Loop over the spatial grid and assign properties from the appropriate layer.
for i, xi in enumerate(x):
    # Find the layer index j such that boundaries[j] <= xi < boundaries[j+1]
    for j in range(len(layers)):
        if boundaries[j] <= xi < boundaries[j+1]:
            rho_arr[i] = layers[j]["rho"]
            cp_arr[i]  = layers[j]["cp"]
            k_arr[i]   = layers[j]["k"]
            Q_arr[i]   = layers[j]["Q"]
            break
    else:
        # If xi equals the last boundary, assign the properties of the last layer.
        rho_arr[i] = layers[-1]["rho"]
        cp_arr[i]  = layers[-1]["cp"]
        k_arr[i]   = layers[-1]["k"]
        Q_arr[i]   = layers[-1]["Q"]

# =============================================================================
# 3) Boundary Conditions and Radiative Parameters
# =============================================================================
sigma = 5.670374419e-8  # Stefan-Boltzmann constant [W/(m²·K⁴)]
solar_flux = 1361.0     # Solar flux [W/m²]
T_env = 2.7             # Deep space temperature [K]

# Surface properties (same for all models)
alpha_top = 0.9  # Solar absorptivity at top
eps_top   = 0.9  # Emissivity at top
eps_bot   = 0.85 # Emissivity at bottom

# =============================================================================
# 4) Initial Conditions and Storage for Time History
# =============================================================================
T = np.ones(N) * 290.0  # Initial temperature (in K) everywhere
T_hist = np.zeros((n_steps, N))
T_hist[0, :] = T

# =============================================================================
# 5) Helper Function: Harmonic Mean for Interface Conductivity
# =============================================================================
def k_interface(k1, k2):
    """Return the harmonic mean of two thermal conductivities."""
    if (k1 + k2) == 0:
        return 0.0
    return 2.0 * k1 * k2 / (k1 + k2)

# Precompute interface conductivities between adjacent grid cells.
k_half = np.zeros(N - 1)
for i in range(N - 1):
    k_half[i] = k_interface(k_arr[i], k_arr[i+1])

# =============================================================================
# 6) Time Integration Using a Semi-Implicit (Backward Euler) Scheme
# =============================================================================
# At each time step, assemble and solve a tridiagonal system.
# Radiative fluxes at the boundaries are evaluated explicitly (using the previous T).
for n in range(0, n_steps - 1):
    # Assemble coefficient matrix A and right-hand side vector r.
    A = np.zeros((N, N))
    r = np.zeros(N)
    
    # --- Top Boundary (i = 0) ---
    q_top = alpha_top * A_top * solar_flux - eps_top * A_top * sigma * (T[0]**4 - T_env**4)
    A[0, 0] = rho_arr[0] * cp_arr[0] / dt + k_half[0] / (dx**2)
    A[0, 1] = -k_half[0] / (dx**2)
    r[0] = (rho_arr[0] * cp_arr[0] / dt) * T[0] + q_top / dx - Q_arr[0]
    
    # --- Interior Nodes (i = 1 to N-2) ---
    for i in range(1, N - 1):
        A[i, i - 1] = -k_half[i - 1] / (dx**2)
        A[i, i] = rho_arr[i] * cp_arr[i] / dt + (k_half[i - 1] + k_half[i]) / (dx**2)
        A[i, i + 1] = -k_half[i] / (dx**2)
        r[i] = (rho_arr[i] * cp_arr[i] / dt) * T[i] + Q_arr[i]
    
    # --- Bottom Boundary (i = N-1) ---
    q_bot = eps_bot * A_bot * sigma * (T[N - 1]**4 - T_env**4)
    A[N - 1, N - 2] = -k_half[N - 2] / (dx**2)
    A[N - 1, N - 1] = rho_arr[N - 1] * cp_arr[N - 1] / dt + k_half[N - 2] / (dx**2)
    r[N - 1] = (rho_arr[N - 1] * cp_arr[N - 1] / dt) * T[N - 1] - q_bot / dx - Q_arr[N - 1]
    
    # Solve the linear system for the new temperature profile.
    T_new = np.linalg.solve(A, r)
    T = T_new.copy()
    T_hist[n + 1, :] = T

# =============================================================================
# 7) 3D Plot of Temperature vs. Position and Time
# =============================================================================
# Create a time array (in seconds) and convert to hours for plotting.
time_array = np.linspace(0, t_total, n_steps)
time_array_hours = time_array / 3600.0

X, TimeGrid = np.meshgrid(x, time_array_hours)  # x-axis in meters, time in hours

fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(X, TimeGrid, T_hist, cmap='viridis', linewidth=0, antialiased=True)
ax.set_xlabel("Position (m)")
ax.set_ylabel("Time (hours)")
ax.set_zlabel("Temperature (K)")
ax.set_title("Temperature Evolution (Semi-Implicit Scheme)")
fig.colorbar(surf, shrink=0.5, aspect=10, label="Temperature (K)")
plt.show()

# =============================================================================
# 8) Print Average Temperature (in °C) of Each Layer at Selected Times
# =============================================================================
# Define a dictionary of selected times (with indices) for printing.
# (Times are given in seconds.)
times_to_print = {
    "t = 0 s": 0,
    "t = 0.5 * t_total": n_steps // 2,
    "t = t_total": n_steps - 1
}

print("Average Temperatures (in °C) for Each Layer at Selected Times:")
for time_label, idx in times_to_print.items():
    print(f"\nAt {time_label}:")
    # Loop over the layers using the boundaries array.
    for j, layer in enumerate(layers):
        # Create a mask for grid points belonging to layer j:
        mask = (x >= boundaries[j]) & (x < boundaries[j+1])
        # (Include the right boundary for the last layer.)
        if j == len(layers) - 1:
            mask = (x >= boundaries[j]) & (x <= boundaries[j+1])
        avg_temp_K = np.mean(T_hist[idx, mask])
        avg_temp_C = avg_temp_K - 273.15
        print(f"  {layer['name']}: {avg_temp_C:.2f} °C")
