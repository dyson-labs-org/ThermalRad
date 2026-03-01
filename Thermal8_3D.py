# -*- coding: utf-8 -*-
"""
Thermal8_3D.py

3D transient heat conduction model for a rectangular panel with a layered stack in z.

Assumptions and notes
---------------------
1) Geometry is a box: x in [0, width_x], y in [0, height_y], z through stacked layers.
2) Material properties (rho, cp, k, optional base Q) are piecewise constant by z-layer.
3) Side walls (x/y edges) are adiabatic by default; optional side radiation can be enabled.
4) Top and bottom radiative boundaries:
   - Top (z=0): q_top = alpha_top*solar_flux*in_sun - area_scale_top*eps_top*sigma*(T^4 - T_env^4)
   - Bottom (z=L): q_bot = view_factor_bot*area_scale_bot*eps_bot*sigma*(T^4 - T_env^4)
5) Internal ASIC heating is localized in x-y rectangles and deposited into a chosen z-slab.
6) Time integrator is a directional semi-implicit split (x implicit, y implicit, z implicit).
   Radiation is treated explicitly per full step using current temperature.
7) A conservative explicit stability limit is printed as a runtime guard/reference.

Only numpy and matplotlib are used.
"""

import numpy as np
import os
import matplotlib


DEFAULT_ENV_PRESETS = {
    # Rear/radiator sees deep space only.
    "DEEP_SPACE": {
        "F_earth_top": 0.0,
        "F_earth_bot": 0.0,
        "solar_on_bottom": False,
        "F_earth_value_earth_facing": 0.7,
        "cycled_use_eclipse_lock": True,
        "cycled_duty": 0.5,
        "cycled_F_earth_high": 0.7,
        "cycled_F_earth_low": 0.0,
        "cycled_period_s": None,
    },
    # Rear/radiator sees Earth strongly.
    "EARTH_FACING": {
        "F_earth_top": 0.0,
        "F_earth_bot": 0.7,
        "solar_on_bottom": False,
        "F_earth_value_earth_facing": 0.7,
        "cycled_use_eclipse_lock": True,
        "cycled_duty": 0.5,
        "cycled_F_earth_high": 0.7,
        "cycled_F_earth_low": 0.0,
        "cycled_period_s": None,
    },
    # Rear/radiator alternates between deep-space-facing and Earth-facing.
    "CYCLED": {
        "F_earth_top": 0.0,
        "F_earth_bot": 0.0,
        "solar_on_bottom": False,
        "F_earth_value_earth_facing": 0.7,
        "cycled_use_eclipse_lock": True,
        "cycled_duty": 0.5,
        "cycled_F_earth_high": 0.7,
        "cycled_F_earth_low": 0.0,
        "cycled_period_s": None,
    },
}


def apply_environment_preset(bc_params, env_mode):
    """
    Merge environment preset with user bc_params (user keys win).
    """
    mode = str(env_mode).strip().upper()
    if mode not in DEFAULT_ENV_PRESETS:
        valid = ", ".join(DEFAULT_ENV_PRESETS.keys())
        raise ValueError(f"Unsupported env_mode '{env_mode}'. Use one of: {valid}.")

    merged = dict(DEFAULT_ENV_PRESETS[mode])
    merged.update(dict(bc_params))
    merged["env_mode"] = mode

    # Keep backward compatibility with older single-F_earth field.
    if "F_earth" in merged:
        if "F_earth_top" not in merged:
            merged["F_earth_top"] = float(merged["F_earth"])
        if "F_earth_bot" not in merged:
            merged["F_earth_bot"] = float(merged["F_earth"])

    merged.setdefault("F_earth_top", 0.0)
    merged.setdefault("F_earth_bot", 0.0)
    merged.setdefault("solar_on_bottom", False)
    merged.setdefault("F_earth_value_earth_facing", 0.7)
    merged.setdefault("cycled_use_eclipse_lock", True)
    merged.setdefault("cycled_duty", 0.5)
    merged.setdefault("cycled_F_earth_high", float(merged["F_earth_value_earth_facing"]))
    merged.setdefault("cycled_F_earth_low", 0.0)
    merged.setdefault("cycled_period_s", None)
    return merged


def get_f_earth_bot_at_time(t_s, in_sun_top, orbit_period_s, bc_params):
    """
    Return F_earth_bot(t) based on env_mode and cycle settings.
    """
    mode = str(bc_params.get("env_mode", "DEEP_SPACE")).strip().upper()
    if mode != "CYCLED":
        return float(bc_params.get("F_earth_bot", 0.0))

    use_eclipse_lock = bool(bc_params.get("cycled_use_eclipse_lock", True))
    f_high = float(bc_params.get("cycled_F_earth_high", bc_params.get("F_earth_value_earth_facing", 0.7)))
    f_low = float(bc_params.get("cycled_F_earth_low", 0.0))

    if use_eclipse_lock:
        return f_high if in_sun_top < 0.5 else f_low

    duty = float(bc_params.get("cycled_duty", 0.5))
    duty = float(np.clip(duty, 0.0, 1.0))
    period = bc_params.get("cycled_period_s", None)
    if period is None:
        period = orbit_period_s
    period = float(period)
    if period <= 0.0:
        return f_low

    phase = (t_s % period) / period
    return f_high if phase < duty else f_low


def harmonic_mean(a, b):
    """Harmonic mean for conductivity interfaces."""
    a_arr = np.asarray(a)
    b_arr = np.asarray(b)
    denom = a_arr + b_arr
    out = np.zeros_like(denom, dtype=float)
    mask = denom > 0.0
    out[mask] = 2.0 * a_arr[mask] * b_arr[mask] / denom[mask]
    return out


def nearest_index(arr, value):
    """Index of array entry nearest to value."""
    return int(np.argmin(np.abs(np.asarray(arr) - value)))


def thomas_solve_batched(lower, diag, upper, rhs):
    """
    Solve tridiagonal system Ax=rhs with Thomas algorithm.
    - lower: (n-1,)
    - diag : (n,)
    - upper: (n-1,)
    - rhs  : (n, m) or (n,)
    """
    rhs_arr = np.asarray(rhs, dtype=float)
    squeeze_out = False
    if rhs_arr.ndim == 1:
        rhs_arr = rhs_arr[:, None]
        squeeze_out = True

    n = diag.size
    m = rhs_arr.shape[1]
    c_prime = np.zeros(n - 1, dtype=float) if n > 1 else np.zeros(0, dtype=float)
    d_prime = np.zeros((n, m), dtype=float)

    b0 = diag[0]
    if abs(b0) < 1e-30:
        raise ZeroDivisionError("Encountered near-zero pivot in Thomas solver at row 0.")
    if n > 1:
        c_prime[0] = upper[0] / b0
    d_prime[0, :] = rhs_arr[0, :] / b0

    for i in range(1, n):
        denom = diag[i] - lower[i - 1] * c_prime[i - 1] if n > 1 else diag[i]
        if abs(denom) < 1e-30:
            raise ZeroDivisionError(f"Encountered near-zero pivot in Thomas solver at row {i}.")
        if i < n - 1:
            c_prime[i] = upper[i] / denom
        d_prime[i, :] = (rhs_arr[i, :] - lower[i - 1] * d_prime[i - 1, :]) / denom

    x = np.zeros((n, m), dtype=float)
    x[-1, :] = d_prime[-1, :]
    for i in range(n - 2, -1, -1):
        x[i, :] = d_prime[i, :] - c_prime[i] * x[i + 1, :]

    if squeeze_out:
        return x[:, 0]
    return x


def build_grid_and_materials(
    layers,
    width_x,
    height_y,
    Nx,
    Ny,
    dz_target=None,
    Nz_per_layer=None,
    min_cells_per_layer=1,
):
    """
    Build structured grid and material fields.

    Returns
    -------
    grid : dict
    materials : dict
    resolution_warnings : list[str]
    """
    if Nx < 1 or Ny < 1:
        raise ValueError("Nx and Ny must be >= 1.")
    if len(layers) == 0:
        raise ValueError("layers list cannot be empty.")

    if Nz_per_layer is not None:
        if len(Nz_per_layer) != len(layers):
            raise ValueError("Nz_per_layer length must match len(layers).")
        nz_layer = [max(1, int(v)) for v in Nz_per_layer]
    else:
        if dz_target is None:
            dz_target = 2.0e-4  # 0.2 mm default target
        nz_layer = []
        for layer in layers:
            n_this = int(np.ceil(layer["thickness"] / dz_target))
            n_this = max(min_cells_per_layer, n_this, 1)
            nz_layer.append(n_this)

    dx = width_x / float(Nx)
    dy = height_y / float(Ny)
    x_centers = (np.arange(Nx) + 0.5) * dx
    y_centers = (np.arange(Ny) + 0.5) * dy

    dz_list = []
    layer_id_z = []
    rho_z = []
    cp_z = []
    kx_z = []
    ky_z = []
    kz_z = []
    qvol_z = []
    layer_names = []

    for lid, (layer, n_cells) in enumerate(zip(layers, nz_layer)):
        layer_names.append(layer["name"])
        dz_cell = layer["thickness"] / float(n_cells)
        q_base = float(layer.get("Q", 0.0))
        k_fallback = float(layer.get("k", 0.0))
        use_copper_fraction = bool(layer.get("use_copper_fraction", False))
        if use_copper_fraction:
            # Optional in-plane mixing heuristic (simple rule of mixtures).
            # kz is kept independent so through-thickness resistance is not artificially boosted.
            f = float(layer.get("copper_fraction", 0.0))
            if f < 0.0 or f > 1.0:
                print(
                    f"WARNING: layer '{layer['name']}' has copper_fraction={f:.3f}; clipping to [0, 1]."
                )
                f = float(np.clip(f, 0.0, 1.0))
            k_fr4 = float(layer.get("k_fr4", 0.3))
            k_cu = float(layer.get("k_cu", 385.0))
            kxy_mix = k_fr4 * (1.0 - f) + k_cu * f
            kx_val = kxy_mix
            ky_val = kxy_mix
            kz_val = float(layer.get("kz", k_fallback))
        else:
            has_aniso = any(key in layer for key in ("kx", "ky", "kz"))
            if has_aniso:
                kx_val = float(layer.get("kx", k_fallback))
                ky_val = float(layer.get("ky", k_fallback))
                kz_val = float(layer.get("kz", k_fallback))
            else:
                lname = layer["name"].lower()
                if "pcb" in lname:
                    # Default anisotropic PCB behavior.
                    kx_val = 10.0
                    ky_val = 10.0
                    kz_val = 0.6
                else:
                    kx_val = k_fallback
                    ky_val = k_fallback
                    kz_val = k_fallback
        for _ in range(n_cells):
            dz_list.append(dz_cell)
            layer_id_z.append(lid)
            rho_z.append(float(layer["rho"]))
            cp_z.append(float(layer["cp"]))
            kx_z.append(kx_val)
            ky_z.append(ky_val)
            kz_z.append(kz_val)
            qvol_z.append(q_base)

    dz = np.array(dz_list, dtype=float)
    layer_id_z = np.array(layer_id_z, dtype=int)
    rho_z = np.array(rho_z, dtype=float)
    cp_z = np.array(cp_z, dtype=float)
    kx_z = np.array(kx_z, dtype=float)
    ky_z = np.array(ky_z, dtype=float)
    kz_z = np.array(kz_z, dtype=float)
    qvol_z = np.array(qvol_z, dtype=float)

    Nz = dz.size
    z_faces = np.zeros(Nz + 1, dtype=float)
    z_faces[1:] = np.cumsum(dz)
    z_centers = 0.5 * (z_faces[:-1] + z_faces[1:])

    A_xy = dx * dy
    volumes_z = A_xy * dz
    rho_cp_z = rho_z * cp_z
    C_z = rho_cp_z * volumes_z

    k_face_x = harmonic_mean(kx_z, kx_z)
    k_face_y = harmonic_mean(ky_z, ky_z)
    Gx_z = k_face_x * (dy * dz) / dx if Nx > 1 else np.zeros_like(kx_z)
    Gy_z = k_face_y * (dx * dz) / dy if Ny > 1 else np.zeros_like(ky_z)

    if Nz > 1:
        k_int_z = harmonic_mean(kz_z[:-1], kz_z[1:])
        dz_int = 0.5 * dz[:-1] + 0.5 * dz[1:]
        Gz = k_int_z * A_xy / dz_int
    else:
        Gz = np.zeros(0, dtype=float)

    Qvol = np.broadcast_to(qvol_z, (Nx, Ny, Nz)).astype(float).copy()

    layer_to_k = {lid: np.where(layer_id_z == lid)[0] for lid in range(len(layers))}
    layer_name_to_id = {layer["name"]: i for i, layer in enumerate(layers)}

    boundaries = [0.0]
    for layer in layers:
        boundaries.append(boundaries[-1] + layer["thickness"])
    boundaries = np.array(boundaries, dtype=float)

    resolution_warnings = []
    thin_threshold = 1.0e-4  # 0.1 mm
    for lid, layer in enumerate(layers):
        t = float(layer["thickness"])
        n_cells = nz_layer[lid]
        if t <= thin_threshold and n_cells < 2:
            resolution_warnings.append(
                f"Thin layer '{layer['name']}' has thickness {t:.3e} m but only {n_cells} z-cell(s). "
                "Consider increasing Nz_per_layer or reducing dz_target."
            )

    grid = {
        "Nx": Nx,
        "Ny": Ny,
        "Nz": Nz,
        "dx": dx,
        "dy": dy,
        "dz": dz,
        "x": x_centers,
        "y": y_centers,
        "z": z_centers,
        "z_faces": z_faces,
        "width_x": width_x,
        "height_y": height_y,
        "A_xy": A_xy,
        "volumes_z": volumes_z,
        "layer_id_z": layer_id_z,
        "layer_names": layer_names,
        "layer_to_k": layer_to_k,
        "layer_name_to_id": layer_name_to_id,
        "boundaries": boundaries,
        "nz_per_layer": nz_layer,
    }

    materials = {
        "rho_z": rho_z,
        "cp_z": cp_z,
        "rho_cp_z": rho_cp_z,
        "k_z": kz_z,  # backward-compatible alias (through-thickness conductivity)
        "kx_z": kx_z,
        "ky_z": ky_z,
        "kz_z": kz_z,
        "Qvol": Qvol,
        "C_z": C_z,
        "Gx_z": Gx_z,
        "Gy_z": Gy_z,
        "Gz": Gz,
        "grid": grid,
        "_cache": {},
    }

    return grid, materials, resolution_warnings


def build_eclipse_mask(t_total, dt, orbit_period_s, eclipse_duration_s):
    """Build 0/1 eclipse mask (0=eclipse, 1=sunlit) for each time sample."""
    times = np.arange(0.0, t_total + 0.5 * dt, dt)
    phase = np.mod(times, orbit_period_s)
    mask = np.ones_like(times, dtype=float)
    mask[phase < eclipse_duration_s] = 0.0
    return times, mask


def apply_heat_sources(
    Qvol,
    sources,
    grid,
    layer_ids,
    target_layer_name,
    source_z_mode="top_cells",
    n_top_cells=1,
    power_scale=1.0,
    source_meta_out=None,
):
    """
    Deposit localized source power (W) into selected x-y rectangles and z slab.

    Each source dict must include:
      x_center, y_center, width, height, power_W
    Optional per-source overrides:
      target_layer_name, source_z_mode, n_top_cells
    """
    if sources is None or len(sources) == 0:
        return Qvol

    x = grid["x"]
    y = grid["y"]
    volumes_z = grid["volumes_z"]
    name_to_id = grid["layer_name_to_id"]

    if target_layer_name not in name_to_id:
        raise ValueError(f"target_layer_name '{target_layer_name}' not found in layer list.")

    for src in sources:
        layer_name_this = src.get("target_layer_name", target_layer_name)
        if layer_name_this not in name_to_id:
            raise ValueError(f"Source target layer '{layer_name_this}' not found.")
        target_lid = name_to_id[layer_name_this]

        x0 = float(src["x_center"])
        y0 = float(src["y_center"])
        w = float(src["width"])
        h = float(src["height"])
        pwr = float(src["power_W"]) * float(power_scale)
        z_mode = str(src.get("source_z_mode", source_z_mode)).lower()
        n_top_this = int(src.get("n_top_cells", n_top_cells))
        if n_top_this < 1:
            n_top_this = 1

        x_mask = (x >= x0 - 0.5 * w) & (x <= x0 + 0.5 * w)
        y_mask = (y >= y0 - 0.5 * h) & (y <= y0 + 0.5 * h)
        if not np.any(x_mask):
            ix = nearest_index(x, x0)
            x_mask[ix] = True
            print(
                f"WARNING: source footprint in x under-resolved; snapping source at x={x0:.4f} m to nearest cell index {ix}."
            )
        if not np.any(y_mask):
            iy = nearest_index(y, y0)
            y_mask[iy] = True
            print(
                f"WARNING: source footprint in y under-resolved; snapping source at y={y0:.4f} m to nearest cell index {iy}."
            )
        kz_target = np.where(layer_ids == target_lid)[0]
        if kz_target.size == 0:
            print("WARNING: target layer has no z-cells; source skipped:", src)
            continue
        if z_mode == "full_layer":
            z_selected = kz_target
        elif z_mode == "top_cells":
            z_selected = kz_target[: min(n_top_this, kz_target.size)]
        else:
            raise ValueError(
                f"Unsupported source_z_mode '{z_mode}'. Use 'full_layer' or 'top_cells'."
            )
        z_mask = np.zeros_like(layer_ids, dtype=bool)
        z_mask[z_selected] = True

        if not np.any(z_mask):
            print(
                "WARNING: heat source footprint/layer has no covered cells; source skipped:",
                src,
            )
            continue

        region_mask = x_mask[:, None, None] & y_mask[None, :, None] & z_mask[None, None, :]
        region_volume = np.sum(volumes_z[None, None, :] * region_mask)
        if region_volume <= 0.0:
            print("WARNING: computed source region volume <= 0; source skipped:", src)
            continue

        q_add = pwr / region_volume  # W/m^3
        Qvol[region_mask] += q_add

        if source_meta_out is not None:
            source_meta_out.append(
                {
                    "ix_center": nearest_index(x, x0),
                    "iy_center": nearest_index(y, y0),
                    "kz_deposit_top": int(z_selected[0]),
                    "kz_deposit_bot": int(z_selected[-1]),
                }
            )

    return Qvol


def build_default_asic_sources(
    layout,
    panel_center,
    row_axis="x",
    pitch_asic=None,
    asic_size=None,
    asic_power_W=None,
    n_asics=None,
    power_mode="typical",
    total_power_W=None,
    asic_size_2=(0.03, 0.02),
    asic_size_4=(0.012, 0.012),
    pitch_2asic=0.05,
    pitch_4asic=0.025,
):
    """
    Convenience heat source layouts.
    - layout='2ASIC_ROW', '2ASIC_ROW_X', '2ASIC_ROW_Y'
    - layout='4ASIC_ROW', '4ASIC_ROW_X', '4ASIC_ROW_Y'

    row_axis in {'x','y'} controls row direction for axis-unspecified layout keys.
    pitch_asic and asic_size optionally override layout-specific defaults.
    Power model:
      total power = n_asics * asic_power_W
      power_mode='typical' -> 12 W/ASIC, 'max' -> 15 W/ASIC (unless asic_power_W set)
    """
    cx, cy = panel_center
    sources = []

    layout_key = layout.strip().upper()
    axis_from_layout = None
    base_layout = layout_key
    if layout_key.endswith("_X"):
        axis_from_layout = "x"
        base_layout = layout_key[:-2]
    elif layout_key.endswith("_Y"):
        axis_from_layout = "y"
        base_layout = layout_key[:-2]

    axis = axis_from_layout if axis_from_layout is not None else row_axis.strip().lower()
    if axis not in {"x", "y"}:
        raise ValueError("row_axis must be 'x' or 'y'.")

    if base_layout == "2ASIC_ROW":
        n_layout = 2
        pitch = float(pitch_2asic if pitch_asic is None else pitch_asic)
        size = asic_size_2 if asic_size is None else asic_size
    elif base_layout == "4ASIC_ROW":
        n_layout = 4
        pitch = float(pitch_4asic if pitch_asic is None else pitch_asic)
        size = asic_size_4 if asic_size is None else asic_size
    elif base_layout == "2ASIC":
        # Backward-compatible alias.
        n_layout = 2
        pitch = float(pitch_2asic if pitch_asic is None else pitch_asic)
        size = asic_size_2 if asic_size is None else asic_size
    elif base_layout == "4ASIC":
        # Backward-compatible alias; now interpreted as a 4x1 row.
        n_layout = 4
        pitch = float(pitch_4asic if pitch_asic is None else pitch_asic)
        size = asic_size_4 if asic_size is None else asic_size
    else:
        raise ValueError(
            "Unsupported layout. Use '2ASIC_ROW_X', '2ASIC_ROW_Y', '4ASIC_ROW_X', or '4ASIC_ROW_Y'."
        )

    n_src = n_layout if n_asics is None else int(n_asics)
    if n_src < 1:
        raise ValueError("n_asics must be >= 1.")

    power_mode_key = str(power_mode).strip().lower()
    if power_mode_key == "typical":
        p_each_default = 12.0
    elif power_mode_key == "max":
        p_each_default = 15.0
    else:
        raise ValueError("power_mode must be 'typical' or 'max'.")

    if asic_power_W is not None:
        p_each = float(asic_power_W)
    elif total_power_W is not None:
        p_each = float(total_power_W) / float(n_src)
    else:
        p_each = p_each_default

    offsets = (np.arange(n_src) - 0.5 * (n_src - 1)) * pitch
    for off in offsets:
        if axis == "x":
            xc = cx + off
            yc = cy
        else:
            xc = cx
            yc = cy + off
        sources.append(
            {
                "x_center": float(xc),
                "y_center": float(yc),
                "width": float(size[0]),
                "height": float(size[1]),
                "power_W": float(p_each),
            }
        )

    return sources


def compute_explicit_dt_limit(grid, materials):
    """
    Conservative explicit 3D stability limit reference:
      dt <= 1 / (2*(alpha_x_max/dx^2 + alpha_y_max/dy^2 + alpha_z_max/dz_min^2))
    """
    rho_cp = materials["rho_cp_z"]
    alpha_x_max = float(np.max(materials["kx_z"] / rho_cp))
    alpha_y_max = float(np.max(materials["ky_z"] / rho_cp))
    alpha_z_max = float(np.max(materials["kz_z"] / rho_cp))
    inv_dx2 = (1.0 / grid["dx"] ** 2) if grid["Nx"] > 1 else 0.0
    inv_dy2 = (1.0 / grid["dy"] ** 2) if grid["Ny"] > 1 else 0.0
    inv_dz2 = (1.0 / np.min(grid["dz"]) ** 2) if grid["Nz"] > 1 else 0.0
    denom = 2.0 * (alpha_x_max * inv_dx2 + alpha_y_max * inv_dy2 + alpha_z_max * inv_dz2)
    if denom <= 0.0:
        return np.inf
    return 1.0 / denom


def resolve_effective_areas(bc_params, panel_area):
    """
    Resolve top/bottom effective radiating areas with strong gating.

    By default, radiating areas are tied to the modeled panel area. Larger effective
    areas are only used when thermally_bonded_to_larger_radiator=True.
    """
    out = dict(bc_params)
    bonded = bool(out.get("thermally_bonded_to_larger_radiator", False))
    A_top_req = out.get("A_top_effective_m2", panel_area)
    A_bot_req = out.get("A_bot_effective_m2", out.get("A_rad_effective_m2", panel_area))

    A_top_req = panel_area if A_top_req is None else float(A_top_req)
    A_bot_req = panel_area if A_bot_req is None else float(A_bot_req)

    warnings = []
    if not bonded:
        if A_top_req > panel_area * (1.0 + 1e-12):
            warnings.append(
                "A_top_effective_m2 exceeds panel area but thermally_bonded_to_larger_radiator is False; clamping to panel area."
            )
        if A_bot_req > panel_area * (1.0 + 1e-12):
            warnings.append(
                "A_bot_effective_m2 exceeds panel area but thermally_bonded_to_larger_radiator is False; clamping to panel area."
            )
        A_top_eff = min(A_top_req, panel_area)
        A_bot_eff = min(A_bot_req, panel_area)
    else:
        A_top_eff = A_top_req
        A_bot_eff = A_bot_req

    if A_top_eff <= 0.0:
        warnings.append("A_top_effective_m2 <= 0 detected; resetting to panel area.")
        A_top_eff = panel_area
    if A_bot_eff <= 0.0:
        warnings.append("A_bot_effective_m2 <= 0 detected; resetting to panel area.")
        A_bot_eff = panel_area

    area_scale_top = A_top_eff / panel_area if panel_area > 0.0 else 1.0
    area_scale_bot = A_bot_eff / panel_area if panel_area > 0.0 else 1.0
    out["panel_area_m2"] = panel_area
    out["A_top_effective_m2"] = A_top_eff
    out["A_bot_effective_m2"] = A_bot_eff
    out["area_scale_top"] = area_scale_top
    out["area_scale_bot"] = area_scale_bot
    return out, warnings


def _compute_boundary_power_terms(T, grid, bc_params, in_sun_top):
    """
    Compute boundary/source power terms and return both flux fields and integrated powers.

    Sign convention for integrated powers:
      - Incoming to thermal mass: positive (solar, albedo, Earth IR, internal ASIC power)
      - Leaving thermal mass: positive for radiative losses (top, bottom, side)
    """
    sigma = float(bc_params["sigma"])
    solar_flux = float(bc_params["solar_flux"])
    T_env = float(bc_params["T_env"])
    alpha_top = float(bc_params["alpha_top"])
    eps_top = float(bc_params["eps_top"])
    eps_bot = float(bc_params["eps_bot"])
    eps_side = float(bc_params.get("eps_side", eps_bot))
    view_factor_bot = float(bc_params.get("view_factor_bot", 1.0))
    F_earth_top = float(bc_params.get("F_earth_top", 0.0))
    F_earth_bot = float(bc_params.get("F_earth_bot", 0.0))
    albedo = float(bc_params.get("albedo", 0.3))
    T_earth = float(bc_params.get("T_earth", 255.0))
    solar_on_bottom = bool(bc_params.get("solar_on_bottom", False))
    alpha_bot = float(bc_params.get("alpha_bot", alpha_top))
    in_sun_bot_override = bc_params.get("in_sun_bot_override", None)
    if in_sun_bot_override is None:
        in_sun_bot = float(in_sun_top)
    else:
        in_sun_bot = float(in_sun_bot_override)
    area_scale_top = float(bc_params.get("area_scale_top", 1.0))
    area_scale_bot = float(bc_params.get("area_scale_bot", 1.0))
    radiate_sides = bool(bc_params.get("radiate_sides", False))

    Nx, Ny, Nz = grid["Nx"], grid["Ny"], grid["Nz"]
    A_xy = grid["A_xy"]

    q_solar_top_in = alpha_top * solar_flux * in_sun_top * np.ones((Nx, Ny), dtype=float)
    q_solar_bottom_in = (
        alpha_bot * solar_flux * in_sun_bot * np.ones((Nx, Ny), dtype=float)
        if solar_on_bottom
        else np.zeros((Nx, Ny), dtype=float)
    )
    q_rad_top_out = area_scale_top * eps_top * sigma * (T[:, :, 0] ** 4 - T_env ** 4)
    q_rad_bottom_out = view_factor_bot * area_scale_bot * eps_bot * sigma * (
        T[:, :, -1] ** 4 - T_env ** 4
    )

    # Simple LEO additions at surfaces that can see Earth.
    q_ir_earth_top_in = eps_top * sigma * (T_earth ** 4 - T[:, :, 0] ** 4) * F_earth_top
    q_ir_earth_bot_in = eps_bot * sigma * (T_earth ** 4 - T[:, :, -1] ** 4) * F_earth_bot
    q_albedo_top_in = alpha_top * solar_flux * albedo * F_earth_top * in_sun_top * np.ones((Nx, Ny), dtype=float)
    q_albedo_bot_in = alpha_bot * solar_flux * albedo * F_earth_bot * in_sun_top * np.ones((Nx, Ny), dtype=float)

    q_top_net_in = q_solar_top_in + q_ir_earth_top_in + q_albedo_top_in - q_rad_top_out
    q_bottom_net_in = q_solar_bottom_in + q_ir_earth_bot_in + q_albedo_bot_in - q_rad_bottom_out

    q_side_x0_out = None
    q_side_xn_out = None
    q_side_y0_out = None
    q_side_yn_out = None
    P_rad_side_W = 0.0

    if radiate_sides:
        area_x_face = grid["dy"] * grid["dz"]  # shape (Nz,)
        area_y_face = grid["dx"] * grid["dz"]  # shape (Nz,)

        q_side_x0_out = eps_side * sigma * (T[0, :, :] ** 4 - T_env ** 4)
        P_rad_side_W += float(np.sum(q_side_x0_out * area_x_face[None, :]))
        if Nx > 1:
            q_side_xn_out = eps_side * sigma * (T[-1, :, :] ** 4 - T_env ** 4)
            P_rad_side_W += float(np.sum(q_side_xn_out * area_x_face[None, :]))
        else:
            q_side_xn_out = q_side_x0_out
            P_rad_side_W += float(np.sum(q_side_x0_out * area_x_face[None, :]))

        q_side_y0_out = eps_side * sigma * (T[:, 0, :] ** 4 - T_env ** 4)
        P_rad_side_W += float(np.sum(q_side_y0_out * area_y_face[None, :]))
        if Ny > 1:
            q_side_yn_out = eps_side * sigma * (T[:, -1, :] ** 4 - T_env ** 4)
            P_rad_side_W += float(np.sum(q_side_yn_out * area_y_face[None, :]))
        else:
            q_side_yn_out = q_side_y0_out
            P_rad_side_W += float(np.sum(q_side_y0_out * area_y_face[None, :]))

    return {
        "q_top_net_in": q_top_net_in,
        "q_bottom_net_in": q_bottom_net_in,
        "q_side_x0_out": q_side_x0_out,
        "q_side_xn_out": q_side_xn_out,
        "q_side_y0_out": q_side_y0_out,
        "q_side_yn_out": q_side_yn_out,
        "radiate_sides": radiate_sides,
        "P_solar_absorbed_top_W": float(np.sum(q_solar_top_in) * A_xy),
        "P_solar_absorbed_bottom_W": float(np.sum(q_solar_bottom_in) * A_xy),
        "P_rad_top_W": float(np.sum(q_rad_top_out) * A_xy),
        "P_rad_bottom_W": float(np.sum(q_rad_bottom_out) * A_xy),
        "P_IR_earth_W": float(np.sum(q_ir_earth_top_in + q_ir_earth_bot_in) * A_xy),
        "P_albedo_W": float(np.sum(q_albedo_top_in + q_albedo_bot_in) * A_xy),
        "F_earth_top": F_earth_top,
        "F_earth_bot": F_earth_bot,
        "P_rad_side_W": P_rad_side_W,
    }


def _get_or_build_cache(materials, dt_sub):
    """Precompute tridiagonal coefficients for each directional implicit sweep."""
    cache = materials.get("_cache", {})
    if cache.get("dt_sub") == dt_sub:
        return cache

    grid = materials["grid"]
    Nx, Ny, Nz = grid["Nx"], grid["Ny"], grid["Nz"]
    C_z = materials["C_z"]
    Gx_z = materials["Gx_z"]
    Gy_z = materials["Gy_z"]
    Gz = materials["Gz"]

    new_cache = {"dt_sub": dt_sub, "x": [], "y": [], "z": None}

    for k in range(Nz):
        C = C_z[k]
        c_over_dt = C / dt_sub

        if Nx > 1:
            Gx = Gx_z[k]
            lower = -Gx * np.ones(Nx - 1, dtype=float)
            upper = -Gx * np.ones(Nx - 1, dtype=float)
            diag = (c_over_dt + 2.0 * Gx) * np.ones(Nx, dtype=float)
            diag[0] = c_over_dt + Gx
            diag[-1] = c_over_dt + Gx
            new_cache["x"].append((lower, diag, upper))
        else:
            new_cache["x"].append(None)

        if Ny > 1:
            Gy = Gy_z[k]
            lower = -Gy * np.ones(Ny - 1, dtype=float)
            upper = -Gy * np.ones(Ny - 1, dtype=float)
            diag = (c_over_dt + 2.0 * Gy) * np.ones(Ny, dtype=float)
            diag[0] = c_over_dt + Gy
            diag[-1] = c_over_dt + Gy
            new_cache["y"].append((lower, diag, upper))
        else:
            new_cache["y"].append(None)

    if Nz > 1:
        lower_z = -Gz.copy()
        upper_z = -Gz.copy()
        diag_z = np.zeros(Nz, dtype=float)
        diag_z[0] = C_z[0] / dt_sub + Gz[0]
        diag_z[-1] = C_z[-1] / dt_sub + Gz[-1]
        if Nz > 2:
            diag_z[1:-1] = C_z[1:-1] / dt_sub + Gz[:-1] + Gz[1:]
        new_cache["z"] = (lower_z, diag_z, upper_z)

    materials["_cache"] = new_cache
    return new_cache


def _implicit_sweep_x(T_in, Qpow, materials, dt_sub, cache):
    """Implicit sweep in x (adiabatic side walls)."""
    grid = materials["grid"]
    Nx, Ny, Nz = grid["Nx"], grid["Ny"], grid["Nz"]
    C_z = materials["C_z"]

    T_out = np.empty_like(T_in)
    for k in range(Nz):
        C = C_z[k]
        rhs = (C / dt_sub) * T_in[:, :, k] + Qpow[:, :, k]
        if Nx == 1:
            T_out[:, :, k] = rhs / (C / dt_sub)
        else:
            lower, diag, upper = cache["x"][k]
            T_out[:, :, k] = thomas_solve_batched(lower, diag, upper, rhs)
    return T_out


def _implicit_sweep_y(T_in, Qpow, materials, dt_sub, cache):
    """Implicit sweep in y (adiabatic side walls)."""
    grid = materials["grid"]
    Nx, Ny, Nz = grid["Nx"], grid["Ny"], grid["Nz"]
    C_z = materials["C_z"]

    T_out = np.empty_like(T_in)
    for k in range(Nz):
        C = C_z[k]
        rhs = (C / dt_sub) * T_in[:, :, k] + Qpow[:, :, k]
        if Ny == 1:
            T_out[:, :, k] = rhs / (C / dt_sub)
        else:
            lower, diag, upper = cache["y"][k]
            sol = thomas_solve_batched(lower, diag, upper, rhs.T).T
            T_out[:, :, k] = sol
    return T_out


def _implicit_sweep_z(T_in, Qpow, materials, dt_sub, cache):
    """Implicit sweep in z including layer interfaces."""
    grid = materials["grid"]
    Nx, Ny, Nz = grid["Nx"], grid["Ny"], grid["Nz"]
    C_z = materials["C_z"]

    if Nz == 1:
        C = C_z[0]
        rhs = (C / dt_sub) * T_in[:, :, 0] + Qpow[:, :, 0]
        T_out = np.empty_like(T_in)
        T_out[:, :, 0] = rhs / (C / dt_sub)
        return T_out

    rhs = (C_z[:, None] / dt_sub) * T_in.reshape(Nx * Ny, Nz).T
    rhs += Qpow.reshape(Nx * Ny, Nz).T
    lower_z, diag_z, upper_z = cache["z"]
    sol = thomas_solve_batched(lower_z, diag_z, upper_z, rhs)
    return sol.T.reshape(Nx, Ny, Nz)


def step_temperature(
    T,
    materials,
    dt,
    bc_params,
    in_sun,
    sources=None,
    return_diagnostics=False,
):
    """
    Advance one time step.

    Signature kept as requested:
      step_temperature(T, materials, dt, bc_params, in_sun, sources)
    """
    del sources  # source fields are pre-built into materials["Qvol"] externally

    grid = materials["grid"]
    Nx, Ny, Nz = grid["Nx"], grid["Ny"], grid["Nz"]
    A_xy = grid["A_xy"]
    volumes_z = grid["volumes_z"]
    flux_terms = _compute_boundary_power_terms(T, grid, bc_params, in_sun)

    Qpow = materials["Qvol"] * volumes_z[None, None, :]
    Qpow[:, :, 0] += flux_terms["q_top_net_in"] * A_xy
    Qpow[:, :, -1] += flux_terms["q_bottom_net_in"] * A_xy

    if flux_terms["radiate_sides"]:
        area_x_face = grid["dy"] * grid["dz"]  # per (y,k) boundary face
        area_y_face = grid["dx"] * grid["dz"]  # per (x,k) boundary face

        Qpow[0, :, :] -= flux_terms["q_side_x0_out"] * area_x_face[None, :]
        if Nx > 1:
            Qpow[-1, :, :] -= flux_terms["q_side_xn_out"] * area_x_face[None, :]
        else:
            Qpow[0, :, :] -= flux_terms["q_side_x0_out"] * area_x_face[None, :]

        Qpow[:, 0, :] -= flux_terms["q_side_y0_out"] * area_y_face[None, :]
        if Ny > 1:
            Qpow[:, -1, :] -= flux_terms["q_side_yn_out"] * area_y_face[None, :]
        else:
            Qpow[:, 0, :] -= flux_terms["q_side_y0_out"] * area_y_face[None, :]

    if Nx == 1 and Ny == 1:
        cache = _get_or_build_cache(materials, dt)
        T_next = _implicit_sweep_z(T, Qpow, materials, dt, cache)
        if return_diagnostics:
            return T_next, flux_terms
        return T_next

    dt_sub = dt / 3.0
    cache = _get_or_build_cache(materials, dt_sub)
    Qpow_sub = Qpow / 3.0

    T1 = _implicit_sweep_x(T, Qpow_sub, materials, dt_sub, cache)
    T2 = _implicit_sweep_y(T1, Qpow_sub, materials, dt_sub, cache)
    T3 = _implicit_sweep_z(T2, Qpow_sub, materials, dt_sub, cache)
    if return_diagnostics:
        return T3, flux_terms
    return T3


def simulate_1d_reference_from_same_grid(T0_z, materials, times, eclipse_mask, dt, bc_params):
    """1D reference using same z-grid and physics, for Nx=Ny=1 sanity checks."""
    T = T0_z.copy()
    Nz = T.size
    C_z = materials["C_z"]
    Qpow_z = materials["Qvol"][0, 0, :] * materials["grid"]["volumes_z"]
    A_xy = materials["grid"]["A_xy"]
    sigma = bc_params["sigma"]
    solar_flux = bc_params["solar_flux"]
    T_env = bc_params["T_env"]
    alpha_top = bc_params["alpha_top"]
    eps_top = bc_params["eps_top"]
    eps_bot = bc_params["eps_bot"]
    view_factor_bot = float(bc_params.get("view_factor_bot", 1.0))
    area_scale_top = float(bc_params.get("area_scale_top", 1.0))
    area_scale_bot = float(bc_params.get("area_scale_bot", 1.0))
    F_earth_top = float(bc_params.get("F_earth_top", 0.0))
    F_earth_bot = float(bc_params.get("F_earth_bot", 0.0))
    albedo = float(bc_params.get("albedo", 0.3))
    T_earth = float(bc_params.get("T_earth", 255.0))
    solar_on_bottom = bool(bc_params.get("solar_on_bottom", False))
    alpha_bot = float(bc_params.get("alpha_bot", alpha_top))
    in_sun_bot_override = bc_params.get("in_sun_bot_override", None)

    cache = _get_or_build_cache(materials, dt)
    if Nz > 1:
        lower_z, diag_z, upper_z = cache["z"]

    for n in range(len(times)):
        if n == len(times) - 1:
            break
        in_sun = eclipse_mask[n]
        if in_sun_bot_override is None:
            in_sun_bot = in_sun
        else:
            in_sun_bot = float(in_sun_bot_override)
        q_solar_top = alpha_top * solar_flux * in_sun
        q_solar_bot = alpha_bot * solar_flux * in_sun_bot if solar_on_bottom else 0.0
        q_rad_top = area_scale_top * eps_top * sigma * (T[0] ** 4 - T_env ** 4)
        q_rad_bot = view_factor_bot * area_scale_bot * eps_bot * sigma * (T[-1] ** 4 - T_env ** 4)
        q_ir_earth_top = eps_top * sigma * (T_earth ** 4 - T[0] ** 4) * F_earth_top
        q_ir_earth_bot = eps_bot * sigma * (T_earth ** 4 - T[-1] ** 4) * F_earth_bot
        q_albedo_top = alpha_top * solar_flux * albedo * F_earth_top * in_sun
        q_albedo_bot = alpha_bot * solar_flux * albedo * F_earth_bot * in_sun

        q_top_net_in = q_solar_top + q_ir_earth_top + q_albedo_top - q_rad_top
        q_bot_net_in = q_solar_bot + q_ir_earth_bot + q_albedo_bot - q_rad_bot
        rhs = (C_z / dt) * T + Qpow_z
        rhs[0] += q_top_net_in * A_xy
        rhs[-1] += q_bot_net_in * A_xy

        if Nz == 1:
            T = rhs / (C_z / dt)
        else:
            T = thomas_solve_batched(lower_z, diag_z, upper_z, rhs)
    return T


def run_simulation(
    layers,
    width_x,
    height_y,
    Nx,
    Ny,
    dt,
    t_total,
    T_init,
    dz_target,
    Nz_per_layer,
    orbit_period_s,
    eclipse_duration_s,
    bc_params,
    heat_sources,
    heat_source_target_layer,
    source_z_mode="top_cells",
    n_top_cells=1,
    throttle_params=None,
    energy_diag_every_n=1,
    env_mode="DEEP_SPACE",
):
    grid, materials, resolution_warnings = build_grid_and_materials(
        layers=layers,
        width_x=width_x,
        height_y=height_y,
        Nx=Nx,
        Ny=Ny,
        dz_target=dz_target,
        Nz_per_layer=Nz_per_layer,
        min_cells_per_layer=1,
    )

    bc_with_preset = apply_environment_preset(bc_params, env_mode)
    panel_area = width_x * height_y
    bc_resolved, area_warnings = resolve_effective_areas(bc_with_preset, panel_area)

    times, eclipse_mask = build_eclipse_mask(
        t_total=t_total,
        dt=dt,
        orbit_period_s=orbit_period_s,
        eclipse_duration_s=eclipse_duration_s,
    )
    n_steps = len(times)
    energy_diag_every_n = max(1, int(energy_diag_every_n))

    qvol_base = materials["Qvol"].copy()
    source_qvol_nominal = np.zeros_like(materials["Qvol"])
    source_meta = []
    apply_heat_sources(
        Qvol=source_qvol_nominal,
        sources=heat_sources,
        grid=grid,
        layer_ids=grid["layer_id_z"],
        target_layer_name=heat_source_target_layer,
        source_z_mode=source_z_mode,
        n_top_cells=n_top_cells,
        power_scale=1.0,
        source_meta_out=source_meta,
    )

    dt_explicit_max = compute_explicit_dt_limit(grid, materials)
    nominal_total_power_W = float(np.sum([src.get("power_W", 0.0) for src in (heat_sources or [])]))
    n_asics = len(heat_sources) if heat_sources is not None else 0
    nominal_per_asic_W = nominal_total_power_W / n_asics if n_asics > 0 else 0.0

    if throttle_params is None:
        throttle_params = {}
    throttle_enabled = bool(throttle_params.get("enabled", True))
    T_high_C = float(throttle_params.get("T_high_C", 110.0))
    T_low_C = float(throttle_params.get("T_low_C", 90.0))
    T_shutdown_C = float(throttle_params.get("T_shutdown_C", 125.0))
    ramp_down_per_s = float(throttle_params.get("ramp_down_per_s", 0.10))
    ramp_up_per_s = float(throttle_params.get("ramp_up_per_s", 0.05))
    control_mode = str(throttle_params.get("control_mode", "hottest")).lower()
    if control_mode not in {"hottest", "average"}:
        raise ValueError("throttle control_mode must be 'hottest' or 'average'.")
    if T_low_C >= T_high_C:
        raise ValueError("Throttle thresholds must satisfy T_low_C < T_high_C.")

    print("\n=== Grid Summary ===")
    print(f"Nx, Ny, Nz = {grid['Nx']}, {grid['Ny']}, {grid['Nz']}")
    print(f"dx, dy = {grid['dx']:.4e}, {grid['dy']:.4e} m")
    print(f"z thickness total = {grid['z_faces'][-1]:.6e} m")
    print("Nz per layer:")
    for layer, nz_i in zip(layers, grid["nz_per_layer"]):
        print(f"  {layer['name']}: {nz_i}")

    print("\n=== Effective Radiating Areas ===")
    print(f"panel_area:      {bc_resolved['panel_area_m2']:.6f} m^2")
    print(f"A_top_effective: {bc_resolved['A_top_effective_m2']:.6f} m^2")
    print(f"A_bot_effective: {bc_resolved['A_bot_effective_m2']:.6f} m^2")
    print(f"scale_top:       {bc_resolved['area_scale_top']:.6f}")
    print(f"scale_bot:       {bc_resolved['area_scale_bot']:.6f}")
    for msg in area_warnings:
        print("WARNING:", msg)

    print("\n=== Environment Preset ===")
    print(f"env_mode: {bc_resolved.get('env_mode', 'DEEP_SPACE')}")
    print(f"F_earth_top (default): {float(bc_resolved.get('F_earth_top', 0.0)):.3f}")
    mode_key = str(bc_resolved.get("env_mode", "DEEP_SPACE")).upper()
    if mode_key == "CYCLED":
        print(
            "F_earth_bot cycled: "
            f"use_eclipse_lock={bool(bc_resolved.get('cycled_use_eclipse_lock', True))}, "
            f"duty={float(bc_resolved.get('cycled_duty', 0.5)):.3f}, "
            f"high={float(bc_resolved.get('cycled_F_earth_high', 0.7)):.3f}, "
            f"low={float(bc_resolved.get('cycled_F_earth_low', 0.0)):.3f}"
        )
    else:
        print(f"F_earth_bot (default): {float(bc_resolved.get('F_earth_bot', 0.0)):.3f}")
    print(f"solar_on_bottom: {bool(bc_resolved.get('solar_on_bottom', False))}")

    print("\n=== Runtime Guards ===")
    print(f"Explicit-scheme dt stability estimate: dt <= {dt_explicit_max:.3e} s")
    if dt > dt_explicit_max:
        print(
            "WARNING: chosen dt exceeds explicit limit. "
            "Current solver uses implicit directional sweeps, so explicit would be unstable."
        )
    if len(resolution_warnings) > 0:
        for msg in resolution_warnings:
            print("WARNING:", msg)

    print("\n=== ASIC Power Model ===")
    print(f"Nominal ASIC count: {n_asics}")
    print(f"Nominal per-ASIC power: {nominal_per_asic_W:.2f} W")
    print(f"Nominal total ASIC power: {nominal_total_power_W:.2f} W")
    print("Compute duty: sunlit-only (gated by eclipse mask)")
    print(
        f"Throttle: {'ON' if throttle_enabled else 'OFF'} "
        f"(T_low={T_low_C:.1f} C, T_high={T_high_C:.1f} C, shutdown={T_shutdown_C:.1f} C, mode={control_mode})"
    )

    T = np.full((grid["Nx"], grid["Ny"], grid["Nz"]), T_init, dtype=float)
    C_total = float(np.sum(materials["C_z"]) * grid["Nx"] * grid["Ny"])

    selected_times_s = [0.0, 0.5 * t_total, t_total]
    snapshot_indices = sorted({nearest_index(times, ts) for ts in selected_times_s})
    snapshots = {}

    if source_meta:
        ix_hot = source_meta[0]["ix_center"]
        iy_hot = source_meta[0]["iy_center"]
    else:
        hot_x = heat_sources[0]["x_center"] if heat_sources else 0.5 * width_x
        hot_y = heat_sources[0]["y_center"] if heat_sources else 0.5 * height_y
        ix_hot = nearest_index(grid["x"], hot_x)
        iy_hot = nearest_index(grid["y"], hot_y)

    if heat_source_target_layer in grid["layer_name_to_id"]:
        lid_target = grid["layer_name_to_id"][heat_source_target_layer]
        k_target = grid["layer_to_k"][lid_target]
        k_pcb_mid = int(k_target[len(k_target) // 2])
    else:
        k_pcb_mid = grid["Nz"] // 2

    if source_meta:
        asic_control_points = [(m["ix_center"], m["iy_center"], k_pcb_mid) for m in source_meta]
    elif heat_sources:
        asic_control_points = [
            (
                nearest_index(grid["x"], float(src["x_center"])),
                nearest_index(grid["y"], float(src["y_center"])),
                k_pcb_mid,
            )
            for src in heat_sources
        ]
    else:
        asic_control_points = [(ix_hot, iy_hot, k_pcb_mid)]

    radiator_lid = None
    for lid, layer in enumerate(layers):
        if "radiator" in layer["name"].lower():
            radiator_lid = lid
            break
    if radiator_lid is not None:
        k_radiator = grid["layer_to_k"][radiator_lid]
        k_bottom_radiator = int(k_radiator[-1])
    else:
        k_bottom_radiator = grid["Nz"] - 1

    point_defs = {
        "Top surface above ASIC": (ix_hot, iy_hot, 0),
        "PCB center under ASIC": (ix_hot, iy_hot, k_pcb_mid),
        "Bottom radiator under ASIC": (ix_hot, iy_hot, k_bottom_radiator),
        "Cold corner": (0, 0, k_bottom_radiator),
    }
    point_hist = {name: np.zeros(n_steps, dtype=float) for name in point_defs}
    power_total_hist_W = np.zeros(n_steps, dtype=float)
    power_per_asic_hist_W = np.zeros(n_steps, dtype=float)
    power_scale_hist = np.zeros(n_steps, dtype=float)
    compute_enable_hist = np.zeros(n_steps, dtype=float)
    T_control_hist = np.zeros(n_steps, dtype=float)

    P_solar_absorbed_top_W = np.full(n_steps, np.nan, dtype=float)
    P_rad_top_W = np.full(n_steps, np.nan, dtype=float)
    P_rad_bottom_W = np.full(n_steps, np.nan, dtype=float)
    P_rad_side_W = np.full(n_steps, np.nan, dtype=float)
    P_internal_ASIC_W = np.full(n_steps, np.nan, dtype=float)
    P_IR_earth_W = np.full(n_steps, np.nan, dtype=float)
    P_albedo_W = np.full(n_steps, np.nan, dtype=float)
    P_solar_absorbed_bottom_W = np.full(n_steps, np.nan, dtype=float)
    dUdt_est_W = np.full(n_steps, np.nan, dtype=float)
    P_net_in_W = np.full(n_steps, np.nan, dtype=float)
    P_net_residual_W = np.full(n_steps, np.nan, dtype=float)
    F_earth_bot_hist = np.zeros(n_steps, dtype=float)

    n_layers = len(layers)
    layer_min = np.full(n_layers, np.inf, dtype=float)
    layer_max = np.full(n_layers, -np.inf, dtype=float)
    layer_sum = np.zeros(n_layers, dtype=float)
    layer_count = np.zeros(n_layers, dtype=np.int64)

    power_scale = 1.0  # throttle-only power scale (before sunlit gating)
    hard_shutdown_triggered = False
    for n in range(n_steps):
        in_sun_top = float(eclipse_mask[n])
        compute_enable = 1.0 if in_sun_top > 0.5 else 0.0
        compute_enable_hist[n] = compute_enable

        f_earth_bot_now = get_f_earth_bot_at_time(
            t_s=float(times[n]),
            in_sun_top=in_sun_top,
            orbit_period_s=float(orbit_period_s),
            bc_params=bc_resolved,
        )
        F_earth_bot_hist[n] = f_earth_bot_now
        bc_step = dict(bc_resolved)
        bc_step["F_earth_bot"] = f_earth_bot_now

        T_ctrl_vals = np.array([T[ix, iy, iz] for (ix, iy, iz) in asic_control_points], dtype=float)
        if control_mode == "average":
            T_control = float(np.mean(T_ctrl_vals))
        else:
            T_control = float(np.max(T_ctrl_vals))
        T_control_hist[n] = T_control
        T_control_C = T_control - 273.15

        if (not hard_shutdown_triggered) and (T_control_C > T_shutdown_C):
            hard_shutdown_triggered = True
            power_scale = 0.0
            print(
                f"WARNING: hard ASIC shutdown triggered at t={times[n]:.1f} s "
                f"(control temperature {T_control_C:.2f} C > {T_shutdown_C:.2f} C)."
            )
        elif hard_shutdown_triggered:
            power_scale = 0.0
        elif throttle_enabled:
            if T_control_C > T_high_C:
                power_scale -= ramp_down_per_s * dt
            elif T_control_C < T_low_C:
                power_scale += ramp_up_per_s * dt
            power_scale = float(np.clip(power_scale, 0.0, 1.0))
        else:
            power_scale = 1.0

        power_scale_effective = power_scale * compute_enable
        np.copyto(materials["Qvol"], qvol_base)
        materials["Qvol"] += power_scale_effective * source_qvol_nominal
        power_scale_hist[n] = power_scale_effective
        power_total_hist_W[n] = nominal_total_power_W * power_scale_effective
        power_per_asic_hist_W[n] = nominal_per_asic_W * power_scale_effective

        for name, (ix, iy, iz) in point_defs.items():
            point_hist[name][n] = T[ix, iy, iz]

        if n in snapshot_indices:
            snapshots[n] = T.copy()

        for lid in range(n_layers):
            kz = grid["layer_to_k"][lid]
            vals = T[:, :, kz]
            layer_min[lid] = min(layer_min[lid], float(np.min(vals)))
            layer_max[lid] = max(layer_max[lid], float(np.max(vals)))
            layer_sum[lid] += float(np.sum(vals))
            layer_count[lid] += vals.size

        if n == n_steps - 1:
            if (n % energy_diag_every_n) == 0:
                flux_diag = _compute_boundary_power_terms(T, grid, bc_step, eclipse_mask[n])
                P_solar_absorbed_top_W[n] = flux_diag["P_solar_absorbed_top_W"]
                P_solar_absorbed_bottom_W[n] = flux_diag["P_solar_absorbed_bottom_W"]
                P_rad_top_W[n] = flux_diag["P_rad_top_W"]
                P_rad_bottom_W[n] = flux_diag["P_rad_bottom_W"]
                P_rad_side_W[n] = flux_diag["P_rad_side_W"]
                P_IR_earth_W[n] = flux_diag["P_IR_earth_W"]
                P_albedo_W[n] = flux_diag["P_albedo_W"]
                P_internal_ASIC_W[n] = power_total_hist_W[n]
            break

        T_next, flux_diag = step_temperature(
            T=T,
            materials=materials,
            dt=dt,
            bc_params=bc_step,
            in_sun=eclipse_mask[n],
            sources=heat_sources,
            return_diagnostics=True,
        )

        if (n % energy_diag_every_n) == 0:
            P_solar_absorbed_top_W[n] = flux_diag["P_solar_absorbed_top_W"]
            P_solar_absorbed_bottom_W[n] = flux_diag["P_solar_absorbed_bottom_W"]
            P_rad_top_W[n] = flux_diag["P_rad_top_W"]
            P_rad_bottom_W[n] = flux_diag["P_rad_bottom_W"]
            P_rad_side_W[n] = flux_diag["P_rad_side_W"]
            P_IR_earth_W[n] = flux_diag["P_IR_earth_W"]
            P_albedo_W[n] = flux_diag["P_albedo_W"]
            P_internal_ASIC_W[n] = power_total_hist_W[n]

            dTavg_dt = (float(np.mean(T_next)) - float(np.mean(T))) / dt
            dUdt = C_total * dTavg_dt
            dUdt_est_W[n] = dUdt
            P_net_in = (
                P_solar_absorbed_top_W[n]
                + P_solar_absorbed_bottom_W[n]
                + P_internal_ASIC_W[n]
                + P_IR_earth_W[n]
                + P_albedo_W[n]
                - P_rad_top_W[n]
                - P_rad_bottom_W[n]
                - P_rad_side_W[n]
            )
            P_net_in_W[n] = P_net_in
            P_net_residual_W[n] = P_net_in - dUdt

        T = T_next

    layer_avg = layer_sum / np.maximum(layer_count, 1)

    print("\n=== Layer Temperature Statistics Over Entire Simulation ===")
    for lid, layer in enumerate(layers):
        print(f"{layer['name']}:")
        print(f"  Max: {layer_max[lid] - 273.15:8.2f} C")
        print(f"  Min: {layer_min[lid] - 273.15:8.2f} C")
        print(f"  Avg: {layer_avg[lid] - 273.15:8.2f} C")

    print("\n=== Power Statistics Over Entire Simulation ===")
    print(f"Per-ASIC power (W): min={np.min(power_per_asic_hist_W):.2f}, max={np.max(power_per_asic_hist_W):.2f}")
    print(f"Total ASIC power (W): min={np.min(power_total_hist_W):.2f}, max={np.max(power_total_hist_W):.2f}")
    compute_enable_frac = float(np.mean(compute_enable_hist))
    eclipse_sunlit_frac = float(np.mean(eclipse_mask))
    effective_compute_duty = float(np.mean(power_scale_hist > 1e-12))
    print(f"Compute-enable fraction (mask-based): {compute_enable_frac:.3f}")
    print(f"Eclipse-mask average (sunlit fraction): {eclipse_sunlit_frac:.3f}")
    print(f"Effective compute duty (power>0): {effective_compute_duty:.3f}")

    if (
        grid["Nx"] == 1
        and grid["Ny"] == 1
        and not throttle_enabled
        and str(bc_resolved.get("env_mode", "DEEP_SPACE")).upper() != "CYCLED"
    ):
        np.copyto(materials["Qvol"], qvol_base + source_qvol_nominal)
        T_ref = simulate_1d_reference_from_same_grid(
            T0_z=np.full(grid["Nz"], T_init, dtype=float),
            materials=materials,
            times=times,
            eclipse_mask=eclipse_mask,
            dt=dt,
            bc_params=bc_resolved,
        )
        max_abs_diff = float(np.max(np.abs(T[:, :, :].reshape(-1) - T_ref)))
        print("\n=== Nx=Ny=1 Sanity Check ===")
        print(f"Max |T_3D - T_1D_reference| at final time: {max_abs_diff:.4e} K")
    elif grid["Nx"] == 1 and grid["Ny"] == 1 and throttle_enabled:
        print("\n=== Nx=Ny=1 Sanity Check ===")
        print("Skipped direct 1D comparison because throttle makes source power time-varying.")
    elif grid["Nx"] == 1 and grid["Ny"] == 1:
        print("\n=== Nx=Ny=1 Sanity Check ===")
        print("Skipped direct 1D comparison for env_mode='CYCLED' (time-varying F_earth_bot).")

    return {
        "times": times,
        "eclipse_mask": eclipse_mask,
        "T_final": T,
        "snapshots": snapshots,
        "snapshot_indices": snapshot_indices,
        "point_hist": point_hist,
        "point_defs": point_defs,
        "power_total_hist_W": power_total_hist_W,
        "power_per_asic_hist_W": power_per_asic_hist_W,
        "power_scale_hist": power_scale_hist,
        "compute_enable_hist": compute_enable_hist,
        "T_control_hist": T_control_hist,
        "nominal_total_power_W": nominal_total_power_W,
        "nominal_per_asic_W": nominal_per_asic_W,
        "n_asics": n_asics,
        "P_solar_absorbed_top_W": P_solar_absorbed_top_W,
        "P_solar_absorbed_bottom_W": P_solar_absorbed_bottom_W,
        "P_rad_top_W": P_rad_top_W,
        "P_rad_bottom_W": P_rad_bottom_W,
        "P_rad_side_W": P_rad_side_W,
        "P_internal_ASIC_W": P_internal_ASIC_W,
        "P_IR_earth_W": P_IR_earth_W,
        "P_albedo_W": P_albedo_W,
        "dUdt_est_W": dUdt_est_W,
        "P_net_in_W": P_net_in_W,
        "P_net_residual_W": P_net_residual_W,
        "F_earth_bot_hist": F_earth_bot_hist,
        "bc_params_resolved": bc_resolved,
        "energy_diag_every_n": energy_diag_every_n,
        "grid": grid,
    }


def plot_results(results, pcb_layer_name_for_plot=None, plot_mode="interactive", output_dir="."):
    """Generate temperature, heatmap, and energy-balance plots."""
    mode = str(plot_mode).strip().lower()
    if mode not in {"interactive", "save"}:
        raise ValueError("plot_mode must be 'interactive' or 'save'.")

    if mode == "save":
        # Headless mode for unattended runs.
        matplotlib.use("Agg", force=True)

    import matplotlib.pyplot as plt

    times = results["times"]
    grid = results["grid"]
    point_hist = results["point_hist"]
    snapshots = results["snapshots"]
    snapshot_indices = results["snapshot_indices"]

    power_total_hist_W = results.get("power_total_hist_W", None)
    power_per_asic_hist_W = results.get("power_per_asic_hist_W", None)
    T_control_hist = results.get("T_control_hist", None)
    F_earth_bot_hist = results.get("F_earth_bot_hist", None)

    fig_temp, ax_temp = plt.subplots(figsize=(10, 5))
    for name, series in point_hist.items():
        ax_temp.plot(times / 60.0, series - 273.15, label=name, linewidth=1.8)
    if T_control_hist is not None:
        ax_temp.plot(
            times / 60.0,
            T_control_hist - 273.15,
            linestyle="--",
            linewidth=1.5,
            color="tab:red",
            label="Throttle control temperature",
        )
    ax_temp.set_xlabel("Time (min)")
    ax_temp.set_ylabel("Temperature (C)")
    ax_temp.set_title("Temperature vs Time at Key Points")
    ax_temp.grid(True, alpha=0.3)

    if power_total_hist_W is not None and power_per_asic_hist_W is not None:
        ax_pow = ax_temp.twinx()
        ax_pow.plot(
            times / 60.0,
            power_total_hist_W,
            color="black",
            linewidth=1.6,
            linestyle="-.",
            label="Total ASIC power (W)",
        )
        ax_pow.plot(
            times / 60.0,
            power_per_asic_hist_W,
            color="dimgray",
            linewidth=1.4,
            linestyle=":",
            label="Per-ASIC power (W)",
        )
        ax_pow.set_ylabel("ASIC Power (W)")
        h1, l1 = ax_temp.get_legend_handles_labels()
        h2, l2 = ax_pow.get_legend_handles_labels()
        ax_temp.legend(h1 + h2, l1 + l2, loc="best")
    else:
        ax_temp.legend(loc="best")
    fig_temp.tight_layout()

    if pcb_layer_name_for_plot and pcb_layer_name_for_plot in grid["layer_name_to_id"]:
        lid = grid["layer_name_to_id"][pcb_layer_name_for_plot]
        kz = grid["layer_to_k"][lid]
        k_pcb = int(kz[len(kz) // 2])
    else:
        k_pcb = grid["Nz"] // 2

    k_radiator_bottom = grid["Nz"] - 1
    for lid, lname in enumerate(grid["layer_names"]):
        if "radiator" in lname.lower():
            kz = grid["layer_to_k"][lid]
            k_radiator_bottom = int(kz[-1])
            break

    plane_defs = {
        "PCB midplane": k_pcb,
        "Radiator bottom": k_radiator_bottom,
    }

    nrows = len(snapshot_indices)
    ncols = len(plane_defs)
    fig_heat, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(5.5 * ncols, 3.6 * nrows),
        constrained_layout=True,
        squeeze=False,
    )

    extent = (0.0, grid["width_x"], 0.0, grid["height_y"])
    plane_items = list(plane_defs.items())
    for r, idx in enumerate(snapshot_indices):
        T_snap = snapshots[idx]
        for c, (plane_name, kz) in enumerate(plane_items):
            ax = axes[r, c]
            im = ax.imshow(
                (T_snap[:, :, kz].T - 273.15),
                origin="lower",
                extent=extent,
                aspect="equal",
                cmap="inferno",
            )
            ax.set_xlabel("x (m)")
            ax.set_ylabel("y (m)")
            ax.set_title(f"{plane_name} at t={times[idx] / 60.0:.1f} min")
            fig_heat.colorbar(im, ax=ax, shrink=0.9, label="Temperature (C)")

    # Energy-balance debug plot (critical for sign checks and runaway detection).
    fig_energy, ax_energy = plt.subplots(figsize=(11, 5.5))
    t_min = times / 60.0
    energy_series = [("P_solar_absorbed_top_W", "Solar absorbed top (W)")]
    if bool(results.get("bc_params_resolved", {}).get("solar_on_bottom", False)):
        energy_series.append(("P_solar_absorbed_bottom_W", "Solar absorbed bottom (W)"))
    energy_series.extend(
        [
            ("P_internal_ASIC_W", "ASIC internal power (W)"),
            ("P_IR_earth_W", "Earth IR (W)"),
            ("P_albedo_W", "Albedo (W)"),
            ("P_rad_top_W", "Rad top loss (W)"),
            ("P_rad_bottom_W", "Rad bottom loss (W)"),
            ("P_rad_side_W", "Rad side loss (W)"),
            ("dUdt_est_W", "dU/dt estimate (W)"),
            ("P_net_in_W", "Net input (W)"),
            ("P_net_residual_W", "Net residual (W)"),
        ]
    )
    for key, label in energy_series:
        if key in results:
            ax_energy.plot(t_min, results[key], linewidth=1.3, label=label)
    ax_energy.axhline(0.0, color="black", linewidth=1.0, alpha=0.5)
    ax_energy.set_xlabel("Time (min)")
    ax_energy.set_ylabel("Power (W)")
    ax_energy.set_title("Energy Balance vs Time")
    ax_energy.grid(True, alpha=0.3)
    if F_earth_bot_hist is not None:
        ax_f = ax_energy.twinx()
        ax_f.plot(
            t_min,
            F_earth_bot_hist,
            color="tab:purple",
            linewidth=1.4,
            linestyle="--",
            label="F_earth_bot(t)",
        )
        ax_f.set_ylabel("F_earth_bot (-)")
        ax_f.set_ylim(-0.05, 1.05)
        h1, l1 = ax_energy.get_legend_handles_labels()
        h2, l2 = ax_f.get_legend_handles_labels()
        ax_energy.legend(h1 + h2, l1 + l2, loc="best", ncol=2)
    else:
        ax_energy.legend(loc="best", ncol=2)
    fig_energy.tight_layout()

    if mode == "save":
        os.makedirs(output_dir, exist_ok=True)
        path_temp = os.path.join(output_dir, "temperature_timeseries.png")
        path_heat = os.path.join(output_dir, "heatmaps.png")
        path_energy = os.path.join(output_dir, "energy_balance.png")
        fig_temp.savefig(path_temp, dpi=180)
        fig_heat.savefig(path_heat, dpi=180)
        fig_energy.savefig(path_energy, dpi=180)
        plt.close(fig_temp)
        plt.close(fig_heat)
        plt.close(fig_energy)
        print("\nSaved plots:")
        print(f"  {path_temp}")
        print(f"  {path_heat}")
        print(f"  {path_energy}")
    else:
        plt.show()


if __name__ == "__main__":
    # -------------------------------------------------------------------------
    # USER-EDITABLE: Geometry and grid
    # -------------------------------------------------------------------------
    width_x = 0.20
    height_y = 0.20
    Nx = 41
    Ny = 41
    dz_target = 2.0e-4      # used if Nz_per_layer is None
    Nz_per_layer = None      # e.g. [2,1,2,...] to force per-layer z cells

    # -------------------------------------------------------------------------
    # USER-EDITABLE: Layer stack (same dictionary interface as 1D model)
    # -------------------------------------------------------------------------
    layers = [
        {"name": "Coverglass (front)", "thickness": 235e-6, "rho": 2200.0, "cp": 800.0, "k": 1.2, "Q": 0.0},
        {"name": "Encapsulant/adhesive (front)", "thickness": 75e-6, "rho": 1100.0, "cp": 1000.0, "k": 0.25, "Q": 0.0},
        {"name": "Silicon cell", "thickness": 200e-6, "rho": 2330.0, "cp": 700.0, "k": 130.0, "Q": 0.0},
        {"name": "Encapsulant/adhesive (back)", "thickness": 75e-6, "rho": 1100.0, "cp": 1000.0, "k": 0.25, "Q": 0.0},
        {"name": "Substrate glass (back)", "thickness": 235e-6, "rho": 2200.0, "cp": 800.0, "k": 1.2, "Q": 0.0},
        {"name": "SARCON PG130A (TIM 1)", "thickness": 1.0e-3, "rho": 3000.0, "cp": 1000.0, "k": 13.0, "Q": 0.0},
        {"name": "6-layer FR4+Cu PCB", "thickness": 1.6e-3, "rho": 1850.0, "cp": 900.0, "kx": 10.0, "ky": 10.0, "kz": 0.6, "Q": 0.0},
        {"name": "SARCON PG130A (TIM 2)", "thickness": 1.0e-3, "rho": 3000.0, "cp": 1000.0, "k": 13.0, "Q": 0.0},
        {"name": "Aluminum radiator", "thickness": 2.0e-3, "rho": 2700.0, "cp": 900.0, "k": 205.0, "Q": 0.0},
    ]

    # -------------------------------------------------------------------------
    # USER-EDITABLE: Orbit/eclipse and time integration
    # -------------------------------------------------------------------------
    orbit_period_s = 90.0 * 60.0
    eclipse_duration_s = 35.0 * 60.0
    t_total = 2.0 * orbit_period_s   # simulate 2 orbits by default
    dt = 1.0
    T_init = 290.0

    # -------------------------------------------------------------------------
    # USER-EDITABLE: Mission environment preset
    # -------------------------------------------------------------------------
    # "DEEP_SPACE", "EARTH_FACING", or "CYCLED"
    env_mode = "DEEP_SPACE"

    # -------------------------------------------------------------------------
    # USER-EDITABLE: Thermal optics / radiative boundaries
    # -------------------------------------------------------------------------
    bc_params = {
        "sigma": 5.670374419e-8,
        "solar_flux": 1361.0,
        "T_env": 2.7,
        "alpha_top": 0.9,
        "eps_top": 0.9,
        "eps_bot": 0.85,
        "eps_side": 0.85,
        "radiate_sides": False,
        # Effective radiative areas are tied to modeled panel area unless this is True.
        "thermally_bonded_to_larger_radiator": False,
        "A_top_effective_m2": width_x * height_y,
        "A_bot_effective_m2": width_x * height_y,
        # Simple LEO environment (set F_earth=0 to recover deep-space-only behavior).
        # These are user overrides on top of env preset defaults.
        "F_earth_top": 0.0,
        "F_earth_bot": 0.0,
        "albedo": 0.3,
        "T_earth": 255.0,
        # Bottom-side solar is off by default (rear tiles/ASICs are not Sun-facing nominally).
        "solar_on_bottom": False,
        "alpha_bot": 0.9,
        # CYCLED mode controls (used when env_mode == "CYCLED")
        "cycled_use_eclipse_lock": True,
        "cycled_duty": 0.5,
        "cycled_F_earth_high": 0.7,
        "cycled_F_earth_low": 0.0,
        "cycled_period_s": orbit_period_s,
        "view_factor_bot": 1.0,
    }

    # -------------------------------------------------------------------------
    # USER-EDITABLE: Heat source layout
    # -------------------------------------------------------------------------
    # Option A: use built-in layout templates ("4ASIC_ROW_X", "4ASIC_ROW_Y", etc.)
    use_layout = "4ASIC_ROW_X"
    row_axis = "x"               # 'x' or 'y'
    power_mode = "typical"       # 'typical' -> 12 W/ASIC (48 W total for 4 ASIC), 'max' -> 15 W/ASIC (60 W total)
    n_asics = 4                  # default for 4ASIC_ROW
    asic_power_W = None          # None -> derived from power_mode
    panel_center = (0.5 * width_x, 0.5 * height_y)
    heat_source_target_layer = "6-layer FR4+Cu PCB"
    source_z_mode = "top_cells"  # 'full_layer' or 'top_cells'
    n_top_cells = 1

    if power_mode.strip().lower() == "typical":
        asic_power_default = 12.0
    elif power_mode.strip().lower() == "max":
        asic_power_default = 15.0
    else:
        raise ValueError("power_mode must be 'typical' or 'max'.")
    if asic_power_W is None:
        asic_power_W = asic_power_default

    heat_sources = build_default_asic_sources(
        layout=use_layout,
        panel_center=panel_center,
        row_axis=row_axis,
        asic_power_W=asic_power_W,
        n_asics=n_asics,
        power_mode=power_mode,
        pitch_asic=0.025 if "4ASIC_ROW" in use_layout.strip().upper() else None,
        asic_size=None,
        asic_size_2=(0.03, 0.02),    # approx rectangular footprint for 2-ASIC case
        asic_size_4=(0.012, 0.012),  # per-ASIC footprint for 4-ASIC case
        pitch_2asic=0.05,
        pitch_4asic=0.025,           # requested 25 mm center-to-center
    )

    # Option B: provide custom source list directly (overrides template if not None)
    custom_sources = None
    if custom_sources is not None:
        heat_sources = custom_sources

    # -------------------------------------------------------------------------
    # USER-EDITABLE: Simple thermal throttling controller
    # -------------------------------------------------------------------------
    throttle_params = {
        "enabled": True,
        "T_high_C": 110.0,
        "T_low_C": 90.0,
        "T_shutdown_C": 125.0,
        "ramp_down_per_s": 0.10,
        "ramp_up_per_s": 0.05,
        "control_mode": "hottest",  # 'hottest' or 'average'
    }

    # -------------------------------------------------------------------------
    # USER-EDITABLE: Plot mode
    # -------------------------------------------------------------------------
    plot_mode = "interactive"  # "interactive" or "save"
    output_dir = "."

    results = run_simulation(
        layers=layers,
        width_x=width_x,
        height_y=height_y,
        Nx=Nx,
        Ny=Ny,
        dt=dt,
        t_total=t_total,
        T_init=T_init,
        dz_target=dz_target,
        Nz_per_layer=Nz_per_layer,
        orbit_period_s=orbit_period_s,
        eclipse_duration_s=eclipse_duration_s,
        bc_params=bc_params,
        heat_sources=heat_sources,
        heat_source_target_layer=heat_source_target_layer,
        source_z_mode=source_z_mode,
        n_top_cells=n_top_cells,
        throttle_params=throttle_params,
        energy_diag_every_n=1,
        env_mode=env_mode,
    )

    plot_results(
        results,
        pcb_layer_name_for_plot=heat_source_target_layer,
        plot_mode=plot_mode,
        output_dir=output_dir,
    )
