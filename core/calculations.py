import numpy as np

from core.config import VirtualElevationConfig


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
    eta=0.98,
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
    eta=0.98,
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
    config,
    cda,
    crr=0.005,
    df=None,
    v=None,
    watts=None,
    a=None,
):
    """
    Calculate virtual elevation change with wind direction support.

    Args:
        config (VirtualElevationConfig): Configuration parameters
        cda (float): Coefficient of drag area
        crr (float): Coefficient of rolling resistance
        df (pandas.DataFrame, optional): DataFrame containing 'v', 'watts', and 'a'
        v (array-like, optional): Velocity in m/s if not using df
        watts (array-like, optional): Power in watts if not using df
        a (array-like, optional): Acceleration in m/s² if not using df

    Returns:
        numpy.ndarray: Virtual elevation changes
    """
    import numpy as np

    if config.kg is None or config.rho is None:
        raise ValueError(
            "Rider mass (kg) and air density (rho) are required parameters"
        )

    if df is not None:
        w = df["watts"].values * config.eta
        acc = df["a"].values
        vg = df["v"].values

        # Calculate effective wind based on direction if available
        if (
            config.wind_direction is not None
            and "latitude" in df.columns
            and "longitude" in df.columns
        ):
            effective_wind = calculate_effective_wind(df, config)
        else:
            effective_wind = np.full_like(vg, config.vw)
    else:
        w = np.array(watts) * config.eta
        acc = np.array(a)
        vg = np.array(v)
        # When no DataFrame provided, use constant wind
        effective_wind = np.full_like(vg, config.vw)

    # Calculate virtual slope with effective wind
    slope = virtual_slope_with_direction(
        config=config,
        cda=cda,
        crr=crr,
        v=vg,
        watts=w,
        a=acc,
        effective_wind=effective_wind,
    )

    # Calculate virtual elevation change
    return vg * config.dt * np.sin(np.arctan(slope))


def virtual_slope_with_direction(
    config,
    cda,
    crr=0.005,
    v=None,
    watts=None,
    a=None,
    effective_wind=None,
):
    """
    Calculate virtual slope using effective wind based on direction.

    Args:
        config (VirtualElevationConfig): Configuration parameters
        cda (float): Coefficient of drag area
        crr (float): Coefficient of rolling resistance
        v (array-like): Velocity in m/s
        watts (array-like): Power in watts
        a (array-like): Acceleration in m/s²
        effective_wind (array-like, optional): Effective wind velocity accounting for direction

    Returns:
        numpy.ndarray: Virtual slope values
    """
    import numpy as np

    if config.kg is None or config.rho is None:
        raise ValueError(
            "Rider mass (kg) and air density (rho) are required parameters"
        )

    w = np.array(watts) * config.eta
    acc = np.array(a)
    vg = np.array(v)

    # If effective_wind not provided, use constant wind
    if effective_wind is None:
        effective_wind = np.full_like(vg, config.vw)

    # Initialize result array with zeros (default slope)
    slope = np.zeros_like(vg, dtype=float)

    # Filter out zero velocities first to avoid division by zero
    valid_idx = np.where(vg > 0.001)[0]

    if len(valid_idx) > 0:
        # Only process entries with valid velocity
        valid_vg = vg[valid_idx]
        valid_w = w[valid_idx]
        valid_acc = acc[valid_idx]
        valid_wind = effective_wind[valid_idx]

        # Calculate air velocity with direction-aware wind
        valid_va = valid_vg + valid_wind

        # Calculate slope for valid entries (no division by zero possible)
        valid_slopes = (
            (valid_w / (valid_vg * config.kg * 9.807))
            - (cda * config.rho * valid_va**2 / (2 * config.kg * 9.807))
            - crr
            - valid_acc / 9.807
        )

        # Assign results back to full array
        slope[valid_idx] = valid_slopes

    return slope


def lap_work(W, vg, vw=0, kg=None, rho=None, dt=1, eta=0.98):
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
    config: VirtualElevationConfig,
    cda,
    crr=0.005,
    df=None,
    v=None,
    watts=None,
    a=None,
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
    if config.kg is None or config.rho is None:
        raise ValueError(
            "Rider mass (kg) and air density (rho) are required parameters"
        )

    if df is not None:
        w = df["watts"].values * config.eta
        acc = df["a"].values
        vg = df["v"].values
    else:
        w = np.array(watts) * config.eta
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
        valid_va = valid_vg + config.vw

        # Calculate slope for valid entries (no division by zero possible)
        valid_slopes = (
            (valid_w / (valid_vg * config.kg * 9.807))
            - (cda * config.rho * valid_va**2 / (2 * config.kg * 9.807))
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
    eta=0.98,
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


def calculate_bearing(lat1, lon1, lat2, lon2):
    """
    Calculate the bearing (direction of travel) between two GPS points.

    Args:
        lat1, lon1: Coordinates of first point (in degrees)
        lat2, lon2: Coordinates of second point (in degrees)

    Returns:
        float: Bearing in degrees (0-360, where 0=North, 90=East)
    """
    import numpy as np

    # Convert to radians
    lat1, lon1 = np.radians(lat1), np.radians(lon1)
    lat2, lon2 = np.radians(lat2), np.radians(lon2)

    # Calculate bearing
    y = np.sin(lon2 - lon1) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(lon2 - lon1)
    bearing = np.arctan2(y, x)

    # Convert to degrees and normalize to 0-360
    bearing = np.degrees(bearing)
    bearing = (bearing + 360) % 360

    return bearing


def calculate_rider_directions(df):
    """
    Calculate the direction of travel for each point in the DataFrame.

    Args:
        df (pandas.DataFrame): DataFrame with latitude/longitude data

    Returns:
        numpy.ndarray: Direction of travel at each point (in degrees, 0-360)
    """
    import numpy as np

    if "latitude" not in df.columns or "longitude" not in df.columns:
        raise ValueError("DataFrame must contain latitude and longitude columns")

    # Get lat/lon values
    lat = df["latitude"].values
    lon = df["longitude"].values

    # Calculate bearing for each point
    n = len(df)
    directions = np.zeros(n)

    # First point needs special handling - use direction of next segment
    if n > 1:
        directions[0] = calculate_bearing(lat[0], lon[0], lat[1], lon[1])

    # Calculate directions between points
    for i in range(1, n - 1):
        # For smoother direction changes, use average of incoming and outgoing directions
        incoming = calculate_bearing(lat[i - 1], lon[i - 1], lat[i], lon[i])
        outgoing = calculate_bearing(lat[i], lon[i], lat[i + 1], lon[i + 1])

        # Average the directions (handling the 0/360 wrap-around)
        directions[i] = average_directions(incoming, outgoing)

    # Last point also needs special handling - use direction of previous segment
    if n > 1:
        directions[n - 1] = calculate_bearing(
            lat[n - 2], lon[n - 2], lat[n - 1], lon[n - 1]
        )

    return directions


def average_directions(dir1, dir2):
    """
    Average two directions, handling the 0/360 degree wrap-around properly.

    Args:
        dir1, dir2: Directions in degrees (0-360)

    Returns:
        float: Average direction in degrees (0-360)
    """
    import numpy as np

    # Convert to unit vectors
    x1, y1 = np.cos(np.radians(dir1)), np.sin(np.radians(dir1))
    x2, y2 = np.cos(np.radians(dir2)), np.sin(np.radians(dir2))

    # Average the vectors
    x_avg, y_avg = (x1 + x2) / 2, (y1 + y2) / 2

    # Convert back to angle
    avg_dir = np.degrees(np.arctan2(y_avg, x_avg))

    # Normalize to 0-360
    return (avg_dir + 360) % 360


# Modified calculate_effective_wind function for better smoothing and direction calculation


def calculate_effective_wind(df, config):
    """
    Calculate effective wind velocity based on wind direction and rider's direction of travel.

    Args:
        df (pandas.DataFrame): DataFrame with latitude/longitude data
        config (VirtualElevationConfig): Configuration with wind parameters

    Returns:
        numpy.ndarray: Effective wind velocity at each point (positive=headwind, negative=tailwind)
    """
    import numpy as np

    # If no wind direction specified, just return constant wind velocity
    if (
        config.wind_direction is None
        or "latitude" not in df.columns
        or "longitude" not in df.columns
    ):
        return np.full(len(df), config.vw)

    # Calculate rider directions (smoothed)
    rider_directions = calculate_rider_directions_smoothed(df)

    # Calculate the angle between wind and rider direction
    # Wind direction is where wind is coming FROM (meteorological convention)
    wind_direction = config.wind_direction

    # Calculate relative angle between wind direction and rider direction
    # We need to find the angle between the direction the rider is going
    # and the direction the wind is coming from
    angle_diff = np.abs(wind_direction - rider_directions)

    # Ensure angle is <= 180 degrees (shortest angle between directions)
    angle_diff = np.minimum(angle_diff, 360 - angle_diff)

    # Calculate effective wind component (projection of wind vector onto rider direction)
    # When angle_diff = 0, it's a direct headwind (cos(0°) = 1)
    # When angle_diff = 180, it's a direct tailwind (cos(180°) = -1)
    effective_wind = config.vw * np.cos(np.radians(angle_diff))

    # Correct the sign based on whether wind is coming from front or behind
    # If the angle between rider direction and wind source is > 90° and < 270°,
    # the wind is coming from behind (tailwind)
    is_from_behind = (angle_diff > 90) & (angle_diff < 270)
    effective_wind[is_from_behind] = -np.abs(effective_wind[is_from_behind])

    # Smooth the effective wind values to avoid abrupt changes
    window_size = min(
        11, len(effective_wind) // 10
    )  # Use at most 10% of points but at least 1
    if window_size % 2 == 0:
        window_size += 1  # Ensure odd window size
    window_size = max(1, window_size)  # Ensure at least 1

    if window_size > 1:
        from scipy.signal import savgol_filter

        try:
            effective_wind = savgol_filter(effective_wind, window_size, 2)
        except:
            # If savgol filter fails, use simple moving average
            kernel = np.ones(window_size) / window_size
            effective_wind = np.convolve(effective_wind, kernel, mode="same")

    return effective_wind


def calculate_rider_directions_smoothed(df, window_size=5):
    """
    Calculate smoothed direction of travel for each point in the DataFrame.

    Args:
        df (pandas.DataFrame): DataFrame with latitude/longitude data
        window_size (int): Size of window for smoothing directions

    Returns:
        numpy.ndarray: Smoothed direction of travel at each point (in degrees, 0-360)
    """
    import numpy as np
    from scipy.signal import savgol_filter

    if "latitude" not in df.columns or "longitude" not in df.columns:
        raise ValueError("DataFrame must contain latitude and longitude columns")

    # Get lat/lon values
    lat = df["latitude"].values
    lon = df["longitude"].values

    # Calculate bearing for each segment
    n = len(df)
    directions = np.zeros(n)

    # Calculate directions between points
    for i in range(1, n):
        directions[i - 1] = calculate_bearing(lat[i - 1], lon[i - 1], lat[i], lon[i])

    # Last point gets same direction as second-to-last point
    if n > 1:
        directions[n - 1] = directions[n - 2]

    # Convert directions to x,y components to handle the circular nature of angles
    x_comp = np.cos(np.radians(directions))
    y_comp = np.sin(np.radians(directions))

    # Smooth the x,y components
    window_size = min(window_size, len(x_comp) // 3)
    if window_size >= 3:
        try:
            x_smooth = savgol_filter(x_comp, window_size, 2)
            y_smooth = savgol_filter(y_comp, window_size, 2)
        except:
            # Fall back to simple moving average if savgol fails
            kernel = np.ones(window_size) / window_size
            x_smooth = np.convolve(x_comp, kernel, mode="same")
            y_smooth = np.convolve(y_comp, kernel, mode="same")
    else:
        x_smooth = x_comp
        y_smooth = y_comp

    # Convert back to angles
    smoothed_directions = (np.degrees(np.arctan2(y_smooth, x_smooth)) + 360) % 360

    return smoothed_directions
