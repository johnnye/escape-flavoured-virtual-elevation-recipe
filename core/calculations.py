import numpy as np


def accel_calc(v, dt):
    """
    Calculate acceleration from velocity data.

    Args:
        v (array-like): Velocity in m/s
        dt (float): Time interval in seconds

    Returns:
        numpy.ndarray: Acceleration values
    """
    v = np.array(v)
    # Replace NaN or zero values with a small positive number
    v[np.isnan(v) | (v < 0.001)] = 0.001

    # Calculate acceleration
    a = np.zeros_like(v)
    for i in range(1, len(v)):
        # Safe division by ensuring denominator is never zero
        if v[i] < 0.001:  # Additional safety check
            a[i] = 0
        else:
            a[i] = (v[i] ** 2 - v[i - 1] ** 2) / (2 * v[i] * dt)

    # Clean up any invalid values
    a[np.isnan(a)] = 0
    a[np.isinf(a)] = 0

    return a


def cda_est(
    crr,
    df=None,
    v=None,
    watts=None,
    m=0,
    n=None,
    net_elev_change=0,
    vw=0,
    kg=None,
    rho=None,
    dt=1,
    eta=1,
):
    """
    Estimate coefficient of drag area (CdA).

    Args:
        crr (float): Coefficient of rolling resistance
        df (pandas.DataFrame, optional): DataFrame containing 'v' and 'watts'
        v (array-like, optional): Velocity in m/s if not using df
        watts (array-like, optional): Power in watts if not using df
        m (int): Start index
        n (int, optional): End index
        net_elev_change (float): Net elevation change in meters
        vw (float): Wind velocity in m/s (positive = headwind)
        kg (float): Rider mass in kg (required)
        rho (float): Air density in kg/m³ (required)
        dt (float): Time interval in seconds
        eta (float): Drivetrain efficiency

    Returns:
        float: Estimated CdA value
    """
    if kg is None or rho is None:
        raise ValueError(
            "Rider mass (kg) and air density (rho) are required parameters"
        )

    if df is not None:
        w = df["watts"].values * eta
        vg = df["v"].values
    else:
        w = np.array(watts) * eta
        vg = np.array(v)

    if n is None:
        n = len(w)

    # Ground velocity and air velocity
    va = vg + vw

    # Kinetic energy change
    JKE = kg * (vg[n - 1] ** 2 - vg[m] ** 2) / 2

    # Potential energy change
    JPE = kg * 9.807 * net_elev_change

    # Rolling resistance work
    Jrr = dt * crr * kg * 9.807 * np.sum(vg[m : n - 1])

    # Total work done
    Jtot = dt * np.sum(w[m : n - 1])

    # Aerodynamic denominator
    dAero = dt * rho * np.sum(vg[m : n - 1] * va[m : n - 1] ** 2) / 2

    # Calculate CdA
    return (Jtot - Jrr - JKE - JPE) / dAero


def crr_est(
    cda,
    df=None,
    v=None,
    watts=None,
    m=0,
    n=None,
    net_elev_change=0,
    vw=0,
    kg=None,
    rho=None,
    dt=1,
    eta=1,
):
    """
    Estimate coefficient of rolling resistance (Crr).

    Args:
        cda (float): Coefficient of drag area
        df (pandas.DataFrame, optional): DataFrame containing 'v' and 'watts'
        v (array-like, optional): Velocity in m/s if not using df
        watts (array-like, optional): Power in watts if not using df
        m (int): Start index
        n (int, optional): End index
        net_elev_change (float): Net elevation change in meters
        vw (float): Wind velocity in m/s (positive = headwind)
        kg (float): Rider mass in kg (required)
        rho (float): Air density in kg/m³ (required)
        dt (float): Time interval in seconds
        eta (float): Drivetrain efficiency

    Returns:
        float: Estimated Crr value
    """
    if kg is None or rho is None:
        raise ValueError(
            "Rider mass (kg) and air density (rho) are required parameters"
        )

    if df is not None:
        w = df["watts"].values * eta
        vg = df["v"].values
    else:
        w = np.array(watts) * eta
        vg = np.array(v)

    if n is None:
        n = len(w)

    # Ground velocity and air velocity
    va = vg + vw

    # Potential energy change
    JPE = kg * 9.807 * net_elev_change

    # Kinetic energy change
    JKE = kg * (vg[n - 1] ** 2 - vg[m] ** 2) / 2

    # Total work done
    Jtot = dt * np.sum(w[m : n - 1])

    # Rolling resistance denominator
    drr = dt * kg * 9.807 * np.sum(vg[m : n - 1])

    # Aerodynamic work
    Jaero = dt * cda * rho * np.sum(vg[m : n - 1] * va[m : n - 1] ** 2) / 2

    # Calculate Crr
    return (Jtot - JKE - JPE - Jaero) / drr


def calculate_distance(df, dt=1):
    """
    Calculate cumulative distance from velocity data.

    Args:
        df (pandas.DataFrame): DataFrame with velocity in 'v' column (m/s)
        dt (float): Time interval in seconds

    Returns:
        numpy.ndarray: Cumulative distance in meters
    """
    # For each time step, distance = velocity * time
    segment_distances = df["v"].values * dt

    # Cumulative sum to get total distance at each point
    cumulative_distance = np.cumsum(segment_distances)

    return cumulative_distance


def should_reset_elevation(current_lap, next_lap):
    """
    Check if elevation should be reset between laps.

    Args:
        current_lap (int): Current lap number
        next_lap (int): Next lap number

    Returns:
        bool: True if elevation should be reset
    """
    # Check if the next lap immediately follows the current one
    # If not, we should reset elevation
    return next_lap != current_lap + 1


def delta_ve(
    cda,
    crr=0.005,
    df=None,
    v=None,
    watts=None,
    a=None,
    vw=0,
    kg=None,
    rho=None,
    dt=1,
    eta=1,
):
    """
    Calculate virtual elevation change.

    Args:
        cda (float): Coefficient of drag area
        crr (float): Coefficient of rolling resistance
        df (pandas.DataFrame, optional): DataFrame containing 'v', 'watts', and 'a'
        v (array-like, optional): Velocity in m/s if not using df
        watts (array-like, optional): Power in watts if not using df
        a (array-like, optional): Acceleration in m/s² if not using df
        vw (float): Wind velocity in m/s (positive = headwind)
        kg (float): Rider mass in kg (required)
        rho (float): Air density in kg/m³ (required)
        dt (float): Time interval in seconds
        eta (float): Drivetrain efficiency

    Returns:
        numpy.ndarray: Virtual elevation changes
    """
    if kg is None or rho is None:
        raise ValueError(
            "Rider mass (kg) and air density (rho) are required parameters"
        )

    if df is not None:
        w = df["watts"].values * eta
        acc = df["a"].values
        vg = df["v"].values
    else:
        w = np.array(watts) * eta
        acc = np.array(a)
        vg = np.array(v)

    # Use our fixed virtual_slope function to calculate slope
    slope = virtual_slope(
        cda=cda, crr=crr, v=vg, watts=w, a=acc, vw=vw, kg=kg, rho=rho, dt=dt, eta=eta
    )

    # Calculate virtual elevation change
    return vg * dt * np.sin(np.arctan(slope))


def lap_work(W, vg, vw=0, kg=None, rho=None, dt=1, eta=1):
    """
    Calculate work components for segments with zero net elevation.

    Args:
        W (array-like): Power in watts
        vg (array-like): Velocity in m/s
        vw (float): Wind velocity in m/s (positive = headwind)
        kg (float): Rider mass in kg (required)
        rho (float): Air density in kg/m³ (required)
        dt (float): Time interval in seconds
        eta (float): Drivetrain efficiency

    Returns:
        tuple: (net work, rolling resistance work, aerodynamic work)
    """
    if kg is None or rho is None:
        raise ValueError(
            "Rider mass (kg) and air density (rho) are required parameters"
        )

    w = np.array(W) * eta
    vg = np.array(vg)
    va = vg + vw

    last_vg = len(vg) - 1

    # Net work
    netJ = dt * np.sum(w[1:]) - kg * (vg[last_vg] ** 2 - vg[0] ** 2) / 2

    # Rolling resistance work
    drr = dt * kg * 9.807 * np.sum(vg[1:])

    # Aerodynamic work
    dAero = dt * rho * np.sum(vg[1:] * va[1:] ** 2) / 2

    return (netJ, drr, dAero)


def rho_calc(hPa=1013.25, Tc=20, RH=50, h=0):
    """
    Calculate air density based on environmental conditions.

    Args:
        hPa (float): Barometric pressure in hectopascals
        Tc (float): Temperature in Celsius
        RH (float): Relative humidity as percentage (e.g., 70 for 70%)
        h (float): Altitude in meters

    Returns:
        float: Air density in kg/m³
    """
    # Constants
    g = 9.80665  # Gravity in m/s²
    Md = 0.0289644  # Molar mass of dry air in kg/mol
    Mv = 0.018016  # Molar mass of water vapor in kg/mol
    R = 8.31432  # Universal gas constant in J/(mol*K)
    Rd = R / Md  # Gas constant for dry air
    Rv = R / Mv  # Gas constant for water vapor
    L = 0.0065  # Temperature lapse rate in K/m
    Tk = Tc + 273.15  # Temperature in Kelvin

    # Saturation vapor pressure (Teten's formula)
    PVsat = 6.1078 * 10 ** (7.5 * Tc / (237.3 + Tc))

    # Humidity corrected vapor pressure
    PV = PVsat * RH / 100

    # Altitude corrected pressure
    hPa_corrected = hPa * ((1 - (L * h) / Tk) ** (g * Md / (R * L)))

    # Virtual temperature in Kelvin
    Tv = Tk / (1 - (0.378 * PV / hPa_corrected))

    # Calculate air density
    rho = round(100 * hPa_corrected / (Rd * Tv), 3)

    return rho


def v_est(w, cda=0.25, crr=0.005, rho=None, kg=None, vw=0, slope=0):
    """
    Estimate speed given power and environmental conditions.

    Args:
        w (float): Power in watts
        cda (float): Coefficient of drag area
        crr (float): Coefficient of rolling resistance
        rho (float): Air density in kg/m³ (required)
        kg (float): Rider mass in kg (required)
        vw (float): Wind velocity in m/s (positive = headwind)
        slope (float): Road gradient

    Returns:
        float: Estimated velocity in m/s
    """
    if kg is None or rho is None:
        raise ValueError(
            "Rider mass (kg) and air density (rho) are required parameters"
        )

    # Coefficients for cubic equation
    a1 = (slope + crr) * kg * 9.8 + 0.5 * rho * cda * vw**2
    a2 = rho * cda * vw
    a3 = 0.5 * rho * cda

    # Solve cubic equation: a3*v³ + a2*v² + a1*v - w = 0
    coeffs = [a3, a2, a1, -w]
    roots = np.roots(coeffs)

    # Find the positive real root
    valid_roots = [
        root.real for root in roots if root.real > 0 and abs(root.imag) < 1e-10
    ]

    if valid_roots:
        return valid_roots[0]
    else:
        return None  # No valid solution found


def virtual_slope(
    cda,
    crr=0.005,
    df=None,
    v=None,
    watts=None,
    a=None,
    vw=0,
    kg=None,
    rho=None,
    dt=1,
    eta=1,
):
    """
    Calculate virtual slope.

    Args:
        cda (float): Coefficient of drag area
        crr (float): Coefficient of rolling resistance
        df (pandas.DataFrame, optional): DataFrame containing 'v', 'watts', and 'a'
        v (array-like, optional): Velocity in m/s if not using df
        watts (array-like, optional): Power in watts if not using df
        a (array-like, optional): Acceleration in m/s² if not using df
        vw (float): Wind velocity in m/s (positive = headwind)
        kg (float): Rider mass in kg (required)
        rho (float): Air density in kg/m³ (required)
        dt (float): Time interval in seconds
        eta (float): Drivetrain efficiency

    Returns:
        numpy.ndarray: Virtual slope values
    """
    if kg is None or rho is None:
        raise ValueError(
            "Rider mass (kg) and air density (rho) are required parameters"
        )

    if df is not None:
        w = df["watts"].values * eta
        acc = df["a"].values
        vg = df["v"].values
    else:
        w = np.array(watts) * eta
        acc = np.array(a)
        vg = np.array(v)

    # Initialize result array with zeros (default slope)
    slope = np.zeros_like(vg, dtype=float)

    # Filter out zero velocities first to avoid division by zero
    valid_idx = np.where(vg > 0.001)[0]

    if len(valid_idx) > 0:
        # Only process entries with valid velocity
        valid_vg = vg[valid_idx]
        valid_w = w[valid_idx]
        valid_acc = acc[valid_idx]

        # Calculate air velocity
        valid_va = valid_vg + vw

        # Calculate slope for valid entries (no division by zero possible)
        valid_slopes = (
            (valid_w / (valid_vg * kg * 9.807))
            - (cda * rho * valid_va**2 / (2 * kg * 9.807))
            - crr
            - valid_acc / 9.807
        )

        # Assign results back to full array
        slope[valid_idx] = valid_slopes

    return slope


def virtual_wind(
    cda,
    crr=0.005,
    df=None,
    v=None,
    watts=None,
    a=None,
    vs=0,
    kg=None,
    rho=None,
    dt=1,
    eta=1,
):
    """
    Calculate virtual wind.

    Args:
        cda (float): Coefficient of drag area
        crr (float): Coefficient of rolling resistance
        df (pandas.DataFrame, optional): DataFrame containing 'v', 'watts', and 'a'
        v (array-like, optional): Velocity in m/s if not using df
        watts (array-like, optional): Power in watts if not using df
        a (array-like, optional): Acceleration in m/s² if not using df
        vs (float): Virtual slope
        kg (float): Rider mass in kg (required)
        rho (float): Air density in kg/m³ (required)
        dt (float): Time interval in seconds
        eta (float): Drivetrain efficiency

    Returns:
        numpy.ndarray: Virtual wind values
    """
    if kg is None or rho is None:
        raise ValueError(
            "Rider mass (kg) and air density (rho) are required parameters"
        )

    if df is not None:
        w = df["watts"].values * eta
        acc = df["a"].values
        vg = df["v"].values
    else:
        w = np.array(watts) * eta
        acc = np.array(a)
        vg = np.array(v)

    # Calculate net watts available for overcoming wind resistance
    net_watts = w - (crr + vs) * vg * kg * 9.807 - acc * kg * vg

    # Get signs (direction) of net watts
    signs = np.sign(net_watts)

    # Calculate virtual wind
    vw = np.sqrt(2 * np.abs(net_watts) / (rho * cda * vg)) * signs - vg

    return vw
