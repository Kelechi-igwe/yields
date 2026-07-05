import numpy as np
import copy
import yaml
from src.data_pull import config

def root_fraction(layer_depths, Zr, k):
    """Exponential root-depth distribution (Gerwitz & Page 1974). Sums to 1."""
    cum, rf = 0.0, []
    for d in layer_depths:
        z_eff = min(cum + d / 2.0, Zr)
        rf.append(0.0 if cum >= Zr else np.exp(-k * z_eff / Zr))
        cum += d
    rf = np.array(rf)
    return rf / rf.sum() if rf.sum() > 0 else rf


def init_cell_state(template, soil_layers):
    """Initialise state for one grid cell; SM starts at 70% of θ_fc."""
    s = copy.deepcopy(template)

    s["TT"] = float(s.get("TT", 0.0))
    s["LAI"] = float(s.get("LAI", 0.1))
    s["B"] = float(s.get("B", 0.0))
    s["B_leaf"] = float(s.get("B_leaf", 0.0))
    s["B_grain"] = float(s.get("B_grain", 0.0))
    s["Zr"] = float(s.get("Zr", 0.15))

    s["soil_layers"] = soil_layers
    s["SM_layers"]   = [l["theta_fc"] * l["depth"] * 1000 * 0.70 for l in soil_layers]
    return s


# ── Daily step function ───────────────────────────────────────────────────────

def step(weather, state, params, alloc, ET_override=None):

    T    = weather.get("T_mean")
    Tmax = weather.get("TMAX", weather.get("T_max"))
    Tmin = weather.get("TMIN", weather.get("T_min"))
    R    = weather.get("solar_rad", 0)
    rain = weather.get("rain", 0)
    if T is None:
        T = 0.5 * (Tmax + Tmin) if (Tmax and Tmin) else 20.0
    Tmax = Tmax if Tmax is not None else T + 2
    Tmin = Tmin if Tmin is not None else T - 2

    # Thermal time
    state["TT"] += max(0, T - params["T_base"])
    TT = state["TT"]

    # Phenological stage
    if   TT < params["TT_emergence"]: stage = "emergence"
    elif TT < params["TT_veg_end"]: stage = "vegetative"
    elif TT < params["TT_grainfill"]: stage = "reproductive"
    elif TT < params["TT_maturity"]: stage = "grain_fill"
    else:
        stage = "maturity"

    # Light interception (Beer-Lambert; McCree 1971 for PAR fraction)
    fIPAR = 1 - np.exp(-params["k"] * state["LAI"])
    APAR  = fIPAR * 0.48 * R

    # ET
    ET_pot = (params["ET_coeff"] * 0.0023
              * (T + 17.8) * np.sqrt(max(Tmax - Tmin, 0.1)) * max(R, 0))
    ET_act = float(ET_override) if ET_override is not None else ET_pot
    state.update(ET_pot=ET_pot, ET_act=ET_act, ET=ET_act, rainfall=rain)

    # Soil water balance
    state["Zr"] = min(state["Zr"] + 0.02, params["Zr_max"])
    soil  = state["soil_layers"]
    ldeps = [l["depth"] for l in soil]
    rf    = root_fraction(ldeps, state["Zr"], params["root_k"])

    state["SM_layers"][0] += rain
    for k in range(len(soil) - 1):                          # gravity drainage
        cap = soil[k]["theta_fc"] * soil[k]["depth"] * 1000
        if state["SM_layers"][k] > cap:
            state["SM_layers"][k+1] += state["SM_layers"][k] - cap
            state["SM_layers"][k]    = cap

    layer_stress = []
    for k, lyr in enumerate(soil):
        theta = state["SM_layers"][k] / (lyr["depth"] * 1000)
        rng   = lyr["theta_fc"] - lyr["theta_wp"]
        layer_stress.append(np.clip((theta - lyr["theta_wp"]) / rng, 0, 1) if rng > 0 else 0)

    f_water_soil = float(np.clip(np.dot(rf, layer_stress), 0, 1))
    state["f_water_soil"] = f_water_soil

    if ET_override is None:
        ET_act = ET_pot * f_water_soil
        state.update(ET_act=ET_act, ET=ET_act)

    f_water_et = np.clip(ET_act / ET_pot, 0, 1) if ET_pot > 0 else 1.0
    state["f_water_et"] = f_water_et
    f_water = f_water_et if ET_override is not None else f_water_soil
    state["f_water"] = f_water

    uptake = rf * ET_act
    for k, lyr in enumerate(soil):
        state["SM_layers"][k] = max(
            state["SM_layers"][k] - uptake[k],
            lyr["theta_wp"] * lyr["depth"] * 1000,
        )

    state["SM_total"] = sum(state["SM_layers"])
    pd_tot = sum(l["depth"] for l in soil)
    state["VWC"] = state["SM_total"] / (pd_tot * 1000) if pd_tot > 0 else np.nan

    # Biomass
    f_temp = np.clip((T - params["T_base"]) / (params["T_opt"] - params["T_base"]), 0.0, 1.0)
    dB = params["LUE"] * APAR * f_temp * f_water

    stage_alloc = alloc[stage] # config_file['biomass_allo_stages']
    dB_leaf = stage_alloc['f_leaf'] * dB

    if stage in ("reproductive", "grain_fill"):
        prog = (TT - params["TT_veg_end"]) / (params["TT_maturity"] - params["TT_veg_end"])
        dB_leaf -= np.clip(params["k_sen"] * prog, 0, params["k_sen_max"]) * state["B_leaf"]

    state["B"] += dB
    state["B_leaf"] = max(state["B_leaf"] + dB_leaf, 0)
    state["B_grain"] += stage_alloc['f_grain'] * dB
    state["LAI"] = min(params["SLA"] * state["B_leaf"], params["LAI_max"])
    state["stage"] = stage

    return state


