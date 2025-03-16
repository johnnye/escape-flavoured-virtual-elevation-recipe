import numpy as np
import time
from scipy.optimize import (
    differential_evolution,
    minimize_scalar,
    basinhopping,
)
from scipy.stats import pearsonr

from core.calculations import delta_ve


def calculate_virtual_profile(ve_changes, actual_elevation, lap_column, df):
    """
    Helper function to build the virtual elevation profile from elevation changes.
    This avoids duplicating code in the optimization functions.

    Args:
        ve_changes (numpy.ndarray): Virtual elevation changes
        actual_elevation (numpy.ndarray): Actual elevation data
        lap_column (str): Column name containing lap numbers, or None for single lap
        df (pandas.DataFrame): DataFrame containing lap information

    Returns:
        numpy.ndarray: Virtual elevation profile
    """
    virtual_profile = np.zeros_like(actual_elevation)

    if lap_column is not None and lap_column in df.columns:
        # For multi-lap analysis: reset elevation only at non-consecutive lap boundaries
        lap_numbers = df[lap_column].values
        unique_laps = sorted(np.unique(lap_numbers))

        # Set initial elevation for the first lap
        first_lap = unique_laps[0]
        first_lap_indices = np.where(lap_numbers == first_lap)[0]
        start_idx = first_lap_indices[0]
        virtual_profile[start_idx] = actual_elevation[start_idx]

        # Process each lap
        prev_lap = None
        for lap in unique_laps:
            # Find indices for this lap
            lap_indices = np.where(lap_numbers == lap)[0]
            if len(lap_indices) == 0:
                continue

            # Set initial elevation for this lap if needed
            start_idx = lap_indices[0]

            # Check if we should reset elevation
            should_reset = prev_lap is None or should_reset_elevation(prev_lap, lap)
            # If this is not the first lap and the laps are not consecutive, reset elevation
            if should_reset:
                virtual_profile[start_idx] = actual_elevation[start_idx]

            start_range = 1 if should_reset else 0
            # Calculate cumulative elevation for this lap
            for i in range(start_range, len(lap_indices)):
                if i == 0:
                    idx = lap_indices[i]
                    prev_idx = idx - 1
                else:
                    idx = lap_indices[i]
                    prev_idx = lap_indices[i - 1]
                virtual_profile[idx] = virtual_profile[prev_idx] + ve_changes[prev_idx]

            prev_lap = lap
    else:
        # For single lap analysis: standard cumulative calculation
        virtual_profile[0] = actual_elevation[0]
        for i in range(1, len(virtual_profile)):
            virtual_profile[i] = virtual_profile[i - 1] + ve_changes[i - 1]

    return virtual_profile


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


def optimize_both_params_balanced(
    df,
    actual_elevation,
    kg,
    rho,
    n_grid=100,  # Reduced grid size to compensate for multiple starts
    r2_weight=0.5,  # Weight for R² in the composite objective (0-1)
    dt=1,
    eta=1,
    vw=0,
    lap_column=None,
    rmse_scale=None,  # Auto-calculated scaling factor for RMSE
    cda_bounds=(0.1, 0.5),  # CdA typically between 0.1 and 0.5 m²
    crr_bounds=(0.001, 0.01),  # Crr typically between 0.001 and 0.01
    n_random_starts=5,  # Number of random starting points
    basin_hopping_steps=30,  # Number of basin hopping steps
    basin_hopping_temp=1.0,  # Temperature parameter for basin hopping
    verbose=True,  # Whether to print detailed progress
):
    """
    Optimize both CdA and Crr using a balanced approach that combines multiple
    global optimization strategies to avoid local minima.

    Args:
        df (pandas.DataFrame): DataFrame with cycling data
        actual_elevation (array-like): Actual measured elevation data
        kg (float): Rider mass in kg
        rho (float): Air density in kg/m³
        n_grid (int): Number of grid points to use in parameter search
        r2_weight (float): Weight for R² in objective function (0-1)
        dt (float): Time interval in seconds
        eta (float): Drivetrain efficiency
        vw (float): Wind velocity in m/s (positive = headwind)
        lap_column (str): Column name containing lap numbers
        rmse_scale (float): Scaling factor for RMSE in objective function
        cda_bounds (tuple): (min, max) bounds for CdA optimization
        crr_bounds (tuple): (min, max) bounds for Crr optimization
        n_random_starts (int): Number of random starting points
        basin_hopping_steps (int): Number of basin hopping steps
        basin_hopping_temp (float): Temperature parameter for basin hopping
        verbose (bool): Whether to print detailed progress

    Returns:
        tuple: (optimized_cda, optimized_crr, rmse, r2, virtual_profile)
    """
    # Start timing
    start_time = time.time()

    # Convert actual elevation to numpy array if it's not already
    actual_elevation = np.array(actual_elevation)

    # Define parameter bounds
    bounds = [cda_bounds, crr_bounds]

    # Calculate baseline values and scaling factor for RMSE
    if rmse_scale is None:
        initial_cda = (cda_bounds[0] + cda_bounds[1]) / 2  # Midpoint of cda range
        initial_crr = (crr_bounds[0] + crr_bounds[1]) / 2  # Midpoint of crr range

        initial_ve_changes = delta_ve(
            cda=initial_cda,
            crr=initial_crr,
            df=df,
            vw=vw,
            kg=kg,
            rho=rho,
            dt=dt,
            eta=eta,
        )
        initial_virtual_profile = calculate_virtual_profile(
            initial_ve_changes, actual_elevation, lap_column, df
        )
        baseline_rmse = np.sqrt(
            np.mean((initial_virtual_profile - actual_elevation) ** 2)
        )
        # Use larger of baseline_rmse or 10% of elevation range as scale
        elev_range = np.max(actual_elevation) - np.min(actual_elevation)
        rmse_scale = max(baseline_rmse, 0.1 * elev_range)

        if verbose:
            print(f"Auto-calculated RMSE scaling factor: {rmse_scale:.2f}m")
            print(f"Using R² weight: {r2_weight:.2f}, RMSE weight: {1-r2_weight:.2f}")

    # Define the composite objective function
    def objective(params):
        """Objective function that balances R² and RMSE"""
        cda, crr = params

        # Calculate virtual elevation changes
        ve_changes = delta_ve(
            cda=cda, crr=crr, df=df, vw=vw, kg=kg, rho=rho, dt=dt, eta=eta
        )

        # Build virtual elevation profile
        virtual_profile = calculate_virtual_profile(
            ve_changes, actual_elevation, lap_column, df
        )

        # Calculate R² between virtual and actual elevation profiles
        r2 = pearsonr(virtual_profile, actual_elevation)[0] ** 2

        # Calculate normalized RMSE
        rmse = np.sqrt(np.mean((virtual_profile - actual_elevation) ** 2))
        normalized_rmse = rmse / rmse_scale

        # Weighted objective: lower is better
        # Use (1-R²) since we want to maximize R² but minimize the objective
        weighted_obj = r2_weight * (1 - r2) + (1 - r2_weight) * normalized_rmse

        return weighted_obj

    # For calculating metrics from a parameter set
    def calculate_metrics(cda, crr):
        ve_changes = delta_ve(
            cda=cda, crr=crr, df=df, vw=vw, kg=kg, rho=rho, dt=dt, eta=eta
        )
        virtual_profile = calculate_virtual_profile(
            ve_changes, actual_elevation, lap_column, df
        )
        r2 = pearsonr(virtual_profile, actual_elevation)[0] ** 2
        rmse = np.sqrt(np.mean((virtual_profile - actual_elevation) ** 2))
        return r2, rmse, virtual_profile

    if verbose:
        print("Step 1: Grid search to identify promising regions...")

    # Create a grid of points to evaluate
    cda_grid = np.linspace(cda_bounds[0], cda_bounds[1], int(np.sqrt(n_grid)))
    crr_grid = np.linspace(crr_bounds[0], crr_bounds[1], int(np.sqrt(n_grid)))

    # Evaluate objective function at each grid point
    grid_results = []
    for cda in cda_grid:
        for crr in crr_grid:
            # Get weighted objective
            weighted_obj = objective([cda, crr])

            # Calculate individual metrics for reporting
            r2, rmse, _ = calculate_metrics(cda, crr)

            grid_results.append((cda, crr, weighted_obj, r2, rmse))

    # Sort grid points by objective (ascending - lower is better)
    grid_results.sort(key=lambda x: x[2])

    if verbose:
        print(f"Grid search completed in {time.time() - start_time:.1f} seconds")
        print(f"Top 3 grid search results:")
        for i in range(min(3, len(grid_results))):
            print(
                f"  CdA={grid_results[i][0]:.4f} m², Crr={grid_results[i][1]:.5f}, "
                f"R²={grid_results[i][3]:.4f}, RMSE={grid_results[i][4]:.2f}m"
            )

    # Initialize storage for global best results across all optimization attempts
    global_best_results = []

    # Define the step-taking function for basin hopping to respect bounds
    def take_step(x):
        # Use a smaller step size as we get closer to convergence
        step_size = np.array(
            [
                (cda_bounds[1] - cda_bounds[0]) * 0.1,
                (crr_bounds[1] - crr_bounds[0]) * 0.1,
            ]
        )
        # Random step within bounds
        new_x = x + np.random.uniform(-1, 1, size=x.shape) * step_size
        # Clip to bounds
        new_x[0] = np.clip(new_x[0], cda_bounds[0], cda_bounds[1])
        new_x[1] = np.clip(new_x[1], crr_bounds[0], crr_bounds[1])
        return new_x

    # Define acceptance test function for basin hopping
    def accept_test(f_new, x_new, f_old, x_old):
        # Always accept if better
        if f_new < f_old:
            return True

        # Sometimes accept worse solutions based on temperature
        # More likely to accept small deteriorations
        delta_f = f_new - f_old
        prob = np.exp(-delta_f / basin_hopping_temp)
        return np.random.random() < prob

    # Set up for differential evolution first
    if verbose:
        print(
            "\nStep 2: Running multiple optimization attempts with different starting points..."
        )

    # Start with top grid results and add random starts
    starting_points = [
        (grid_results[i][0], grid_results[i][1])
        for i in range(min(3, len(grid_results)))
    ]

    # Add random starting points
    for _ in range(n_random_starts - len(starting_points)):
        random_cda = np.random.uniform(cda_bounds[0], cda_bounds[1])
        random_crr = np.random.uniform(crr_bounds[0], crr_bounds[1])
        starting_points.append((random_cda, random_crr))

    # Run optimization from each starting point
    for start_idx, (start_cda, start_crr) in enumerate(starting_points):
        if verbose:
            print(f"\nAttempt {start_idx + 1}/{len(starting_points)}:")
            print(f"Starting point: CdA={start_cda:.4f}m², Crr={start_crr:.5f}")

        start_attempt_time = time.time()

        # Use differential evolution first for global search
        de_result = differential_evolution(
            objective,
            bounds,
            popsize=15,
            mutation=(0.5, 1.0),
            recombination=0.7,
            strategy="best1bin",
            tol=0.01,
            maxiter=100,  # Limit iterations for speed
            init="sobol",  # Use Sobol sequence for better coverage
            updating="deferred",  # Update after a generation
            workers=1,
        )

        de_cda, de_crr = de_result.x
        de_obj = de_result.fun

        # Calculate metrics for reporting
        de_r2, de_rmse, _ = calculate_metrics(de_cda, de_crr)

        if verbose:
            print(
                f"DE result: CdA={de_cda:.4f}m², Crr={de_crr:.5f}, "
                f"R²={de_r2:.4f}, RMSE={de_rmse:.2f}m"
            )

        # Use basin hopping for exploring multiple basins
        bh_minimizer_kwargs = {
            "method": "L-BFGS-B",
            "bounds": bounds,
            "options": {"ftol": 1e-6, "gtol": 1e-5},
        }

        bh_result = basinhopping(
            objective,
            de_result.x,  # Start from DE result
            niter=basin_hopping_steps,
            T=basin_hopping_temp,
            stepsize=1.0,  # Initial step size, our custom function will scale
            take_step=take_step,
            accept_test=accept_test,
            minimizer_kwargs=bh_minimizer_kwargs,
        )

        bh_cda, bh_crr = bh_result.x
        bh_obj = bh_result.fun

        # Calculate metrics for this attempt
        bh_r2, bh_rmse, bh_profile = calculate_metrics(bh_cda, bh_crr)

        if verbose:
            print(
                f"Basin hopping result: CdA={bh_cda:.4f}m², Crr={bh_crr:.5f}, "
                f"R²={bh_r2:.4f}, RMSE={bh_rmse:.2f}m"
            )
            print(
                f"Attempt {start_idx + 1} completed in {time.time() - start_attempt_time:.1f} seconds"
            )

        # Store result from this attempt
        global_best_results.append((bh_cda, bh_crr, bh_obj, bh_r2, bh_rmse, bh_profile))

    # Find the global best result across all attempts
    global_best_results.sort(key=lambda x: x[2])  # Sort by objective value
    best_cda, best_crr, best_obj, best_r2, best_rmse, best_profile = (
        global_best_results[0]
    )

    if verbose:
        print("\nAll optimization attempts completed.")
        print(f"Total optimization time: {time.time() - start_time:.1f} seconds")
        print(
            f"Global best result: CdA={best_cda:.4f}m², Crr={best_crr:.5f}, "
            f"R²={best_r2:.4f}, RMSE={best_rmse:.2f}m"
        )

    return best_cda, best_crr, best_rmse, best_r2, best_profile


def optimize_cda_only_balanced(
    df,
    actual_elevation,
    fixed_crr,
    kg,
    rho,
    n_points=100,  # Reduced for efficiency with multiple starts
    r2_weight=0.5,  # Weight for R² in the composite objective (0-1)
    dt=1,
    eta=1,
    vw=0,
    lap_column=None,
    rmse_scale=None,  # Auto-calculated scaling factor for RMSE
    cda_bounds=(0.1, 0.5),  # CdA typically between 0.1 and 0.5 m²
    n_random_starts=5,  # Number of random starting points
    basin_hopping_steps=30,  # Number of basin hopping steps
    basin_hopping_temp=1.0,  # Temperature parameter for basin hopping
    verbose=True,  # Whether to print detailed progress
):
    """
    Optimize only CdA with a fixed Crr value using a balanced approach with
    multiple optimization strategies to avoid local minima.

    Args:
        df (pandas.DataFrame): DataFrame with cycling data
        actual_elevation (array-like): Actual measured elevation data
        fixed_crr (float): Fixed Crr value to use
        kg (float): Rider mass in kg
        rho (float): Air density in kg/m³
        n_points (int): Number of points to use in parameter search
        r2_weight (float): Weight for R² in objective function (0-1)
        dt (float): Time interval in seconds
        eta (float): Drivetrain efficiency
        vw (float): Wind velocity in m/s (positive = headwind)
        lap_column (str): Column name containing lap numbers
        rmse_scale (float): Scaling factor for RMSE in objective function
        cda_bounds (tuple): (min, max) bounds for CdA optimization
        n_random_starts (int): Number of random starting points
        basin_hopping_steps (int): Number of basin hopping steps
        basin_hopping_temp (float): Temperature parameter for basin hopping
        verbose (bool): Whether to print detailed progress

    Returns:
        tuple: (optimized_cda, fixed_crr, rmse, r2, virtual_profile)
    """
    # Start timing
    start_time = time.time()

    # Convert actual elevation to numpy array if it's not already
    actual_elevation = np.array(actual_elevation)

    # Calculate baseline values and scaling factor for RMSE
    if rmse_scale is None:
        initial_cda = (cda_bounds[0] + cda_bounds[1]) / 2  # Midpoint of cda range

        initial_ve_changes = delta_ve(
            cda=initial_cda, crr=fixed_crr, df=df, vw=vw, kg=kg, rho=rho, dt=dt, eta=eta
        )
        initial_virtual_profile = calculate_virtual_profile(
            initial_ve_changes, actual_elevation, lap_column, df
        )
        baseline_rmse = np.sqrt(
            np.mean((initial_virtual_profile - actual_elevation) ** 2)
        )
        # Use larger of baseline_rmse or 10% of elevation range as scale
        elev_range = np.max(actual_elevation) - np.min(actual_elevation)
        rmse_scale = max(baseline_rmse, 0.1 * elev_range)

        if verbose:
            print(f"Auto-calculated RMSE scaling factor: {rmse_scale:.2f}m")
            print(f"Using R² weight: {r2_weight:.2f}, RMSE weight: {1-r2_weight:.2f}")

    # Define the composite objective function
    def objective(cda):
        """Objective function that balances R² and RMSE"""
        # Handle single value input for basin_hopping compatibility
        # Handle array input (ensure we're using a scalar if there's just one element)
        if hasattr(cda, "__len__") and len(cda) == 1:
            cda = float(cda[0])

        # Calculate virtual elevation changes
        ve_changes = delta_ve(
            cda=cda, crr=fixed_crr, df=df, vw=vw, kg=kg, rho=rho, dt=dt, eta=eta
        )

        # Build virtual elevation profile
        virtual_profile = calculate_virtual_profile(
            ve_changes, actual_elevation, lap_column, df
        )

        # Calculate R² between virtual and actual elevation profiles
        r2 = pearsonr(virtual_profile, actual_elevation)[0] ** 2

        # Calculate normalized RMSE
        rmse = np.sqrt(np.mean((virtual_profile - actual_elevation) ** 2))
        normalized_rmse = rmse / rmse_scale

        # Weighted objective: lower is better
        weighted_obj = r2_weight * (1 - r2) + (1 - r2_weight) * normalized_rmse

        return weighted_obj

    # For calculating metrics from a parameter
    def calculate_metrics(cda):
        ve_changes = delta_ve(
            cda=cda, crr=fixed_crr, df=df, vw=vw, kg=kg, rho=rho, dt=dt, eta=eta
        )
        virtual_profile = calculate_virtual_profile(
            ve_changes, actual_elevation, lap_column, df
        )
        r2 = pearsonr(virtual_profile, actual_elevation)[0] ** 2
        rmse = np.sqrt(np.mean((virtual_profile - actual_elevation) ** 2))
        return r2, rmse, virtual_profile

    if verbose:
        print("Step 1: Initial search across CdA range...")

    # Create a linearly spaced array of CdA values to test
    cda_values = np.linspace(cda_bounds[0], cda_bounds[1], n_points)

    # Evaluate the objective function for each CdA value
    initial_results = []
    for cda in cda_values:
        # Get weighted objective
        weighted_obj = objective(cda)

        # Calculate individual metrics for reporting
        r2, rmse, _ = calculate_metrics(cda)

        initial_results.append((cda, weighted_obj, r2, rmse))

    # Sort by objective (lower is better)
    initial_results.sort(key=lambda x: x[1])

    if verbose:
        print(f"Initial search completed in {time.time() - start_time:.1f} seconds")
        print(f"Top 3 initial search results:")
        for i in range(min(3, len(initial_results))):
            print(
                f"  CdA={initial_results[i][0]:.4f}m², "
                f"R²={initial_results[i][2]:.4f}, RMSE={initial_results[i][3]:.2f}m"
            )

    # Initialize storage for global best results
    global_best_results = []

    # Define the step-taking function for basin hopping to respect bounds
    def take_step(x):
        # Scale based on the bounds range
        step_size = (cda_bounds[1] - cda_bounds[0]) * 0.1
        # Random step within bounds - ensure we keep the same dimension as x
        new_x = x + np.random.uniform(-1, 1, size=x.shape) * step_size
        # Clip to bounds and ensure dimension is preserved
        return np.clip(new_x, cda_bounds[0], cda_bounds[1])

    # Define acceptance test function for basin hopping
    def accept_test(f_new, x_new, f_old, x_old):
        # Always accept if better
        if f_new < f_old:
            return True

        # Sometimes accept worse solutions based on temperature
        delta_f = f_new - f_old
        prob = np.exp(-delta_f / basin_hopping_temp)
        return np.random.random() < prob

    if verbose:
        print(
            "\nStep 2: Running multiple optimization attempts with different starting points..."
        )

    # Start with top grid results and add random starts
    starting_points = [
        initial_results[i][0] for i in range(min(3, len(initial_results)))
    ]

    # Add random starting points
    for _ in range(n_random_starts - len(starting_points)):
        random_cda = np.random.uniform(cda_bounds[0], cda_bounds[1])
        starting_points.append(random_cda)

    # Run optimization from each starting point
    for start_idx, start_cda in enumerate(starting_points):
        if verbose:
            print(f"\nAttempt {start_idx + 1}/{len(starting_points)}:")
            print(f"Starting point: CdA={start_cda:.4f}m²")

        start_attempt_time = time.time()

        # Use basin hopping for exploring multiple basins
        minimizer_kwargs = {
            "method": "L-BFGS-B",
            "bounds": [(cda_bounds[0], cda_bounds[1])],  # Proper format for L-BFGS-B
        }

        # Use basin hopping with starting point as a scalar
        # Use scipy.optimize.minimize_scalar first for single parameter optimization
        scalar_result = minimize_scalar(
            objective, bounds=cda_bounds, method="bounded", options={"xatol": 1e-6}
        )

        # Then use basin hopping starting from this point
        bh_result = basinhopping(
            objective,
            np.array([scalar_result.x]),  # Start from scalar optimization result
            niter=basin_hopping_steps,
            T=basin_hopping_temp,
            stepsize=1.0,  # Initial step size, custom function will scale
            take_step=take_step,
            accept_test=accept_test,
            minimizer_kwargs=minimizer_kwargs,
        )

        # Extract result (basin hopping returns an array)
        bh_cda = bh_result.x[0]
        bh_obj = bh_result.fun

        # Calculate metrics for this attempt
        bh_r2, bh_rmse, bh_profile = calculate_metrics(bh_cda)

        if verbose:
            print(
                f"Basin hopping result: CdA={bh_cda:.4f}m², "
                f"R²={bh_r2:.4f}, RMSE={bh_rmse:.2f}m"
            )
            print(
                f"Attempt {start_idx + 1} completed in {time.time() - start_attempt_time:.1f} seconds"
            )

        # Store result from this attempt
        global_best_results.append((bh_cda, bh_obj, bh_r2, bh_rmse, bh_profile))

    # Find the global best result
    global_best_results.sort(key=lambda x: x[1])  # Sort by objective value
    best_cda, best_obj, best_r2, best_rmse, best_profile = global_best_results[0]

    if verbose:
        print("\nAll optimization attempts completed.")
        print(f"Total optimization time: {time.time() - start_time:.1f} seconds")
        print(
            f"Global best result: CdA={best_cda:.4f}m², "
            f"R²={best_r2:.4f}, RMSE={best_rmse:.2f}m"
        )

    return best_cda, fixed_crr, best_rmse, best_r2, best_profile


def optimize_crr_only_balanced(
    df,
    actual_elevation,
    fixed_cda,
    kg,
    rho,
    n_points=100,  # Reduced for efficiency with multiple starts
    r2_weight=0.5,  # Weight for R² in the composite objective (0-1)
    dt=1,
    eta=1,
    vw=0,
    lap_column=None,
    rmse_scale=None,  # Auto-calculated scaling factor for RMSE
    crr_bounds=(0.001, 0.01),  # Crr typically between 0.001 and 0.01
    n_random_starts=5,  # Number of random starting points
    basin_hopping_steps=30,  # Number of basin hopping steps
    basin_hopping_temp=1.0,  # Temperature parameter for basin hopping
    verbose=True,  # Whether to print detailed progress
):
    """
    Optimize only Crr with a fixed CdA value using a balanced approach with
    multiple optimization strategies to avoid local minima.

    Args:
        df (pandas.DataFrame): DataFrame with cycling data
        actual_elevation (array-like): Actual measured elevation data
        fixed_cda (float): Fixed CdA value to use
        kg (float): Rider mass in kg
        rho (float): Air density in kg/m³
        n_points (int): Number of points to use in parameter search
        r2_weight (float): Weight for R² in objective function (0-1)
        dt (float): Time interval in seconds
        eta (float): Drivetrain efficiency
        vw (float): Wind velocity in m/s (positive = headwind)
        lap_column (str): Column name containing lap numbers
        rmse_scale (float): Scaling factor for RMSE in objective function
        crr_bounds (tuple): (min, max) bounds for Crr optimization
        n_random_starts (int): Number of random starting points
        basin_hopping_steps (int): Number of basin hopping steps
        basin_hopping_temp (float): Temperature parameter for basin hopping
        verbose (bool): Whether to print detailed progress

    Returns:
        tuple: (fixed_cda, optimized_crr, rmse, r2, virtual_profile)
    """
    # Start timing
    start_time = time.time()

    # Convert actual elevation to numpy array if it's not already
    actual_elevation = np.array(actual_elevation)

    # Calculate baseline values and scaling factor for RMSE
    if rmse_scale is None:
        initial_crr = (crr_bounds[0] + crr_bounds[1]) / 2  # Midpoint of crr range

        initial_ve_changes = delta_ve(
            cda=fixed_cda, crr=initial_crr, df=df, vw=vw, kg=kg, rho=rho, dt=dt, eta=eta
        )
        initial_virtual_profile = calculate_virtual_profile(
            initial_ve_changes, actual_elevation, lap_column, df
        )
        baseline_rmse = np.sqrt(
            np.mean((initial_virtual_profile - actual_elevation) ** 2)
        )
        # Use larger of baseline_rmse or 10% of elevation range as scale
        elev_range = np.max(actual_elevation) - np.min(actual_elevation)
        rmse_scale = max(baseline_rmse, 0.1 * elev_range)

        if verbose:
            print(f"Auto-calculated RMSE scaling factor: {rmse_scale:.2f}m")
            print(f"Using R² weight: {r2_weight:.2f}, RMSE weight: {1-r2_weight:.2f}")

    # Define the composite objective function
    def objective(crr):
        """Objective function that balances R² and RMSE"""
        # Handle single value input for basin_hopping compatibility
        # Handle array input (ensure we're using a scalar if there's just one element)
        if hasattr(crr, "__len__") and len(crr) == 1:
            crr = float(crr[0])

        # Calculate virtual elevation changes
        ve_changes = delta_ve(
            cda=fixed_cda, crr=crr, df=df, vw=vw, kg=kg, rho=rho, dt=dt, eta=eta
        )

        # Build virtual elevation profile
        virtual_profile = calculate_virtual_profile(
            ve_changes, actual_elevation, lap_column, df
        )

        # Calculate R² between virtual and actual elevation profiles
        r2 = pearsonr(virtual_profile, actual_elevation)[0] ** 2

        # Calculate normalized RMSE
        rmse = np.sqrt(np.mean((virtual_profile - actual_elevation) ** 2))
        normalized_rmse = rmse / rmse_scale

        # Weighted objective: lower is better
        weighted_obj = r2_weight * (1 - r2) + (1 - r2_weight) * normalized_rmse

        return weighted_obj

    # For calculating metrics from a parameter
    def calculate_metrics(crr):
        ve_changes = delta_ve(
            cda=fixed_cda, crr=crr, df=df, vw=vw, kg=kg, rho=rho, dt=dt, eta=eta
        )
        virtual_profile = calculate_virtual_profile(
            ve_changes, actual_elevation, lap_column, df
        )
        r2 = pearsonr(virtual_profile, actual_elevation)[0] ** 2
        rmse = np.sqrt(np.mean((virtual_profile - actual_elevation) ** 2))
        return r2, rmse, virtual_profile

    if verbose:
        print("Step 1: Initial search across Crr range...")

    # Create a linearly spaced array of Crr values to test
    crr_values = np.linspace(crr_bounds[0], crr_bounds[1], n_points)

    # Evaluate the objective function for each Crr value
    initial_results = []
    for crr in crr_values:
        # Get weighted objective
        weighted_obj = objective(crr)

        # Calculate individual metrics for reporting
        r2, rmse, _ = calculate_metrics(crr)

        initial_results.append((crr, weighted_obj, r2, rmse))

    # Sort by objective (lower is better)
    initial_results.sort(key=lambda x: x[1])

    if verbose:
        print(f"Initial search completed in {time.time() - start_time:.1f} seconds")
        print(f"Top 3 initial search results:")
        for i in range(min(3, len(initial_results))):
            print(
                f"  Crr={initial_results[i][0]:.5f}, "
                f"R²={initial_results[i][2]:.4f}, RMSE={initial_results[i][3]:.2f}m"
            )

    # Initialize storage for global best results
    global_best_results = []

    # Define the step-taking function for basin hopping to respect bounds
    def take_step(x):
        # Scale based on the bounds range
        step_size = (crr_bounds[1] - crr_bounds[0]) * 0.1
        # Random step within bounds - ensure we keep the same dimension as x
        new_x = x + np.random.uniform(-1, 1, size=x.shape) * step_size
        # Clip to bounds and ensure dimension is preserved
        return np.clip(new_x, crr_bounds[0], crr_bounds[1])

    # Define acceptance test function for basin hopping
    def accept_test(f_new, x_new, f_old, x_old):
        # Always accept if better
        if f_new < f_old:
            return True

        # Sometimes accept worse solutions based on temperature
        delta_f = f_new - f_old
        prob = np.exp(-delta_f / basin_hopping_temp)
        return np.random.random() < prob

    if verbose:
        print(
            "\nStep 2: Running multiple optimization attempts with different starting points..."
        )

    # Start with top grid results and add random starts
    starting_points = [
        initial_results[i][0] for i in range(min(3, len(initial_results)))
    ]

    # Add random starting points
    for _ in range(n_random_starts - len(starting_points)):
        random_crr = np.random.uniform(crr_bounds[0], crr_bounds[1])
        starting_points.append(random_crr)

    # Run optimization from each starting point
    for start_idx, start_crr in enumerate(starting_points):
        if verbose:
            print(f"\nAttempt {start_idx + 1}/{len(starting_points)}:")
            print(f"Starting point: Crr={start_crr:.5f}")

        start_attempt_time = time.time()

        # Use basin hopping for exploring multiple basins
        minimizer_kwargs = {
            "method": "L-BFGS-B",
            "bounds": [(crr_bounds[0], crr_bounds[1])],  # Proper format for L-BFGS-B
        }

        # Use basin hopping with starting point as a scalar
        # Use scipy.optimize.minimize_scalar first for single parameter optimization
        scalar_result = minimize_scalar(
            objective, bounds=crr_bounds, method="bounded", options={"xatol": 1e-6}
        )

        # Then use basin hopping starting from this point
        bh_result = basinhopping(
            objective,
            np.array([scalar_result.x]),  # Start from scalar optimization result
            niter=basin_hopping_steps,
            T=basin_hopping_temp,
            stepsize=1.0,  # Initial step size, custom function will scale
            take_step=take_step,
            accept_test=accept_test,
            minimizer_kwargs=minimizer_kwargs,
        )

        # Extract result (basin hopping returns an array)
        bh_crr = bh_result.x[0]
        bh_obj = bh_result.fun

        # Calculate metrics for this attempt
        bh_r2, bh_rmse, bh_profile = calculate_metrics(bh_crr)

        if verbose:
            print(
                f"Basin hopping result: Crr={bh_crr:.5f}, "
                f"R²={bh_r2:.4f}, RMSE={bh_rmse:.2f}m"
            )
            print(
                f"Attempt {start_idx + 1} completed in {time.time() - start_attempt_time:.1f} seconds"
            )

        # Store result from this attempt
        global_best_results.append((bh_crr, bh_obj, bh_r2, bh_rmse, bh_profile))

    # Find the global best result
    global_best_results.sort(key=lambda x: x[1])  # Sort by objective value
    best_crr, best_obj, best_r2, best_rmse, best_profile = global_best_results[0]

    if verbose:
        print("\nAll optimization attempts completed.")
        print(f"Total optimization time: {time.time() - start_time:.1f} seconds")
        print(
            f"Global best result: Crr={best_crr:.5f}, "
            f"R²={best_r2:.4f}, RMSE={best_rmse:.2f}m"
        )

    return fixed_cda, best_crr, best_rmse, best_r2, best_profile


def optimize_both_params_for_target_elevation(
    df,
    actual_elevation,
    kg,
    rho,
    target_elevation_gain=0,
    n_grid=250,
    dt=1,
    eta=1,
    vw=0,
    lap_column=None,
    is_combined_laps=False,
    cda_bounds=(0.1, 0.5),
    crr_bounds=(0.001, 0.01),
    n_random_starts=5,
    basin_hopping_steps=30,
    basin_hopping_temp=1.0,
    verbose=True,
):
    """
    Optimize both CdA and Crr to achieve a specific target elevation gain.
    For individual laps, optimizes for the target gain per lap.
    For combined laps, optimizes for the target gain across all laps together.

    Args:
        df (pandas.DataFrame): DataFrame with cycling data
        actual_elevation (array-like): Actual measured elevation data
        kg (float): Rider mass in kg
        rho (float): Air density in kg/m³
        target_elevation_gain (float): Target elevation gain in meters (default: 0)
        n_grid (int): Number of grid points to use in parameter search
        dt (float): Time interval in seconds
        eta (float): Drivetrain efficiency
        vw (float): Wind velocity in m/s (positive = headwind)
        lap_column (str): Column name containing lap numbers
        is_combined_laps (bool): Whether this is for combined laps analysis
        cda_bounds (tuple): (min, max) bounds for CdA optimization
        crr_bounds (tuple): (min, max) bounds for Crr optimization
        n_random_starts (int): Number of random starting points
        basin_hopping_steps (int): Number of basin hopping steps
        basin_hopping_temp (float): Temperature parameter for basin hopping
        verbose (bool): Whether to print detailed progress

    Returns:
        tuple: (optimized_cda, optimized_crr, rmse, r2, virtual_profile)
    """
    # Start timing
    start_time = time.time()

    # Convert actual elevation to numpy array if it's not already
    actual_elevation = np.array(actual_elevation)

    # Define parameter bounds
    bounds = [cda_bounds, crr_bounds]

    # Define the objective function
    def objective(params):
        """Objective function that minimizes difference from target elevation gain"""
        cda, crr = params

        # Calculate virtual elevation changes
        ve_changes = delta_ve(
            cda=cda, crr=crr, df=df, vw=vw, kg=kg, rho=rho, dt=dt, eta=eta
        )

        # Build virtual elevation profile
        virtual_profile = calculate_virtual_profile(
            ve_changes, actual_elevation, lap_column, df
        )

        # Calculate deviation from target elevation gain
        if lap_column is not None and lap_column in df.columns:
            lap_numbers = df[lap_column].values
            unique_laps = sorted(np.unique(lap_numbers))

            if is_combined_laps:
                # For combined laps: Calculate total elevation gain across all laps
                total_elevation_gain = 0

                # Get the elevation gain from the first point to the last point
                first_idx = 0
                last_idx = len(virtual_profile) - 1
                total_elevation_gain = (
                    virtual_profile[last_idx] - virtual_profile[first_idx]
                )

                # Compare total gain against target
                return abs(total_elevation_gain - target_elevation_gain)
            else:
                # For individual laps: Calculate average deviation from target per lap
                total_deviation = 0
                lap_count = 0

                for lap in unique_laps:
                    lap_indices = np.where(lap_numbers == lap)[0]
                    if len(lap_indices) > 1:
                        lap_start_idx = lap_indices[0]
                        lap_end_idx = lap_indices[-1]
                        lap_elevation_gain = (
                            virtual_profile[lap_end_idx]
                            - virtual_profile[lap_start_idx]
                        )
                        lap_deviation = abs(lap_elevation_gain - target_elevation_gain)
                        total_deviation += lap_deviation
                        lap_count += 1

                # Use average deviation across laps
                return total_deviation / max(1, lap_count)
        else:
            # Single lap case
            actual_gain = virtual_profile[-1] - virtual_profile[0]
            return abs(actual_gain - target_elevation_gain)

    # For calculating metrics from a parameter set
    def calculate_metrics(cda, crr):
        ve_changes = delta_ve(
            cda=cda, crr=crr, df=df, vw=vw, kg=kg, rho=rho, dt=dt, eta=eta
        )
        virtual_profile = calculate_virtual_profile(
            ve_changes, actual_elevation, lap_column, df
        )
        r2 = pearsonr(virtual_profile, actual_elevation)[0] ** 2
        rmse = np.sqrt(np.mean((virtual_profile - actual_elevation) ** 2))

        # Calculate elevation gains
        if lap_column is not None and lap_column in df.columns:
            lap_numbers = df[lap_column].values
            unique_laps = sorted(np.unique(lap_numbers))

            if is_combined_laps:
                # For combined laps: report total elevation gain
                first_idx = 0
                last_idx = len(virtual_profile) - 1
                total_elevation_gain = (
                    virtual_profile[last_idx] - virtual_profile[first_idx]
                )
                elevation_gains = [total_elevation_gain]  # Just report total
            else:
                # For individual laps: report gain per lap
                elevation_gains = []
                for lap in unique_laps:
                    lap_indices = np.where(lap_numbers == lap)[0]
                    if len(lap_indices) > 1:
                        lap_start_idx = lap_indices[0]
                        lap_end_idx = lap_indices[-1]
                        lap_elevation_gain = (
                            virtual_profile[lap_end_idx]
                            - virtual_profile[lap_start_idx]
                        )
                        elevation_gains.append(lap_elevation_gain)
        else:
            # Single lap case
            elevation_gains = [virtual_profile[-1] - virtual_profile[0]]

        return r2, rmse, virtual_profile, elevation_gains

    if verbose:
        if is_combined_laps:
            if target_elevation_gain == 0:
                print(
                    "Step 1: Grid search to identify promising regions for zero total elevation gain..."
                )
            else:
                print(
                    f"Step 1: Grid search to identify promising regions for {target_elevation_gain:.1f}m total elevation gain..."
                )
        else:
            if target_elevation_gain == 0:
                print(
                    "Step 1: Grid search to identify promising regions for zero net elevation gain per lap..."
                )
            else:
                print(
                    f"Step 1: Grid search to identify promising regions for {target_elevation_gain:.1f}m elevation gain per lap..."
                )

    # Create a grid of points to evaluate
    cda_grid = np.linspace(cda_bounds[0], cda_bounds[1], int(np.sqrt(n_grid)))
    crr_grid = np.linspace(crr_bounds[0], crr_bounds[1], int(np.sqrt(n_grid)))

    # Evaluate objective function at each grid point
    grid_results = []
    for cda in cda_grid:
        for crr in crr_grid:
            # Get target elevation deviation objective
            elev_deviation_obj = objective([cda, crr])

            # Calculate individual metrics for reporting
            r2, rmse, _, elevation_gains = calculate_metrics(cda, crr)

            avg_gain = (
                sum(elevation_gains) / len(elevation_gains) if elevation_gains else 0
            )
            grid_results.append((cda, crr, elev_deviation_obj, r2, rmse, avg_gain))

    # Sort grid points by objective (ascending - lower is better)
    grid_results.sort(key=lambda x: x[2])

    if verbose:
        print(f"Grid search completed in {time.time() - start_time:.1f} seconds")
        print(f"Top 3 grid search results:")
        for i in range(min(3, len(grid_results))):
            print(
                f"  CdA={grid_results[i][0]:.4f} m², Crr={grid_results[i][1]:.5f}, "
                f"Elevation Gain={grid_results[i][2]:.2f}m, R²={grid_results[i][3]:.4f}, RMSE={grid_results[i][4]:.2f}m"
            )

    # Initialize storage for global best results across all optimization attempts
    global_best_results = []

    # Define the step-taking function for basin hopping to respect bounds
    def take_step(x):
        # Use a smaller step size as we get closer to convergence
        step_size = np.array(
            [
                (cda_bounds[1] - cda_bounds[0]) * 0.1,
                (crr_bounds[1] - crr_bounds[0]) * 0.1,
            ]
        )
        # Random step within bounds
        new_x = x + np.random.uniform(-1, 1, size=x.shape) * step_size
        # Clip to bounds
        new_x[0] = np.clip(new_x[0], cda_bounds[0], cda_bounds[1])
        new_x[1] = np.clip(new_x[1], crr_bounds[0], crr_bounds[1])
        return new_x

    # Define acceptance test function for basin hopping
    def accept_test(f_new, x_new, f_old, x_old):
        # Always accept if better
        if f_new < f_old:
            return True

        # Sometimes accept worse solutions based on temperature
        delta_f = f_new - f_old
        prob = np.exp(-delta_f / basin_hopping_temp)
        return np.random.random() < prob

    if verbose:
        print(
            "\nStep 2: Running multiple optimization attempts with different starting points..."
        )

    # Start with top grid results and add random starts
    starting_points = [
        (grid_results[i][0], grid_results[i][1])
        for i in range(min(3, len(grid_results)))
    ]

    # Add random starting points
    for _ in range(n_random_starts - len(starting_points)):
        random_cda = np.random.uniform(cda_bounds[0], cda_bounds[1])
        random_crr = np.random.uniform(crr_bounds[0], crr_bounds[1])
        starting_points.append((random_cda, random_crr))

    # Run optimization from each starting point
    for start_idx, (start_cda, start_crr) in enumerate(starting_points):
        if verbose:
            print(f"\nAttempt {start_idx + 1}/{len(starting_points)}:")
            print(f"Starting point: CdA={start_cda:.4f}m², Crr={start_crr:.5f}")

        start_attempt_time = time.time()

        # Use differential evolution first for global search
        de_result = differential_evolution(
            objective,
            bounds,
            popsize=15,
            mutation=(0.5, 1.0),
            recombination=0.7,
            strategy="best1bin",
            tol=0.01,
            maxiter=100,
            init="sobol",
            updating="deferred",
            workers=1,
        )

        de_cda, de_crr = de_result.x
        de_obj = de_result.fun

        # Calculate metrics for reporting
        de_r2, de_rmse, _, de_elevation_gains = calculate_metrics(de_cda, de_crr)
        de_avg_gain = (
            sum(de_elevation_gains) / len(de_elevation_gains)
            if de_elevation_gains
            else 0
        )

        if verbose:
            print(
                f"DE result: CdA={de_cda:.4f}m², Crr={de_crr:.5f}, "
                f"Avg Elevation Gain={de_avg_gain:.2f}m, R²={de_r2:.4f}, RMSE={de_rmse:.2f}m"
            )

        # Use basin hopping for exploring multiple basins
        bh_minimizer_kwargs = {
            "method": "L-BFGS-B",
            "bounds": bounds,
            "options": {"ftol": 1e-6, "gtol": 1e-5},
        }

        bh_result = basinhopping(
            objective,
            de_result.x,  # Start from DE result
            niter=basin_hopping_steps,
            T=basin_hopping_temp,
            stepsize=1.0,  # Initial step size, our custom function will scale
            take_step=take_step,
            accept_test=accept_test,
            minimizer_kwargs=bh_minimizer_kwargs,
        )

        bh_cda, bh_crr = bh_result.x
        bh_obj = bh_result.fun

        # Calculate metrics for this attempt
        bh_r2, bh_rmse, bh_profile, bh_elevation_gains = calculate_metrics(
            bh_cda, bh_crr
        )
        bh_avg_gain = (
            sum(bh_elevation_gains) / len(bh_elevation_gains)
            if bh_elevation_gains
            else 0
        )

        if verbose:
            print(
                f"Basin hopping result: CdA={bh_cda:.4f}m², Crr={bh_crr:.5f}, "
                f"Avg Elevation Gain={bh_avg_gain:.2f}m, R²={bh_r2:.4f}, RMSE={bh_rmse:.2f}m"
            )
            print(
                f"Attempt {start_idx + 1} completed in {time.time() - start_attempt_time:.1f} seconds"
            )

        # Store result from this attempt
        global_best_results.append(
            (bh_cda, bh_crr, bh_obj, bh_r2, bh_rmse, bh_profile, bh_elevation_gains)
        )

    # Find the global best result across all attempts
    global_best_results.sort(key=lambda x: x[2])  # Sort by objective value
    (
        best_cda,
        best_crr,
        best_obj,
        best_r2,
        best_rmse,
        best_profile,
        best_elevation_gains,
    ) = global_best_results[0]

    best_avg_gain = (
        sum(best_elevation_gains) / len(best_elevation_gains)
        if best_elevation_gains
        else 0
    )

    if verbose:
        print("\nAll optimization attempts completed.")
        print(f"Total optimization time: {time.time() - start_time:.1f} seconds")

        if is_combined_laps:
            if target_elevation_gain == 0:
                print(
                    f"Global best result: CdA={best_cda:.4f}m², Crr={best_crr:.5f}, "
                    f"Total Elevation Gain={best_avg_gain:.2f}m, R²={best_r2:.4f}, RMSE={best_rmse:.2f}m"
                )
            else:
                print(
                    f"Global best result: CdA={best_cda:.4f}m², Crr={best_crr:.5f}, "
                    f"Total Elevation Gain={best_avg_gain:.2f}m (target: {target_elevation_gain:.1f}m), "
                    f"R²={best_r2:.4f}, RMSE={best_rmse:.2f}m"
                )
        else:
            if target_elevation_gain == 0:
                print(
                    f"Global best result: CdA={best_cda:.4f}m², Crr={best_crr:.5f}, "
                    f"Avg Elevation Gain per Lap={best_avg_gain:.2f}m, R²={best_r2:.4f}, RMSE={best_rmse:.2f}m"
                )
            else:
                print(
                    f"Global best result: CdA={best_cda:.4f}m², Crr={best_crr:.5f}, "
                    f"Avg Elevation Gain per Lap={best_avg_gain:.2f}m (target: {target_elevation_gain:.1f}m), "
                    f"R²={best_r2:.4f}, RMSE={best_rmse:.2f}m"
                )

    return best_cda, best_crr, best_rmse, best_r2, best_profile


def optimize_cda_only_for_target_elevation(
    df,
    actual_elevation,
    fixed_crr,
    kg,
    rho,
    target_elevation_gain=0,
    n_points=100,
    dt=1,
    eta=1,
    vw=0,
    lap_column=None,
    is_combined_laps=False,
    cda_bounds=(0.1, 0.5),
    n_random_starts=5,
    basin_hopping_steps=30,
    basin_hopping_temp=1.0,
    verbose=True,
):
    """
    Optimize only CdA with a fixed Crr value, aiming for a specific target elevation gain.
    For individual laps, optimizes for the target gain per lap.
    For combined laps, optimizes for the target gain across all laps together.

    Args:
        df (pandas.DataFrame): DataFrame with cycling data
        actual_elevation (array-like): Actual measured elevation data
        fixed_crr (float): Fixed Crr value to use
        kg (float): Rider mass in kg
        rho (float): Air density in kg/m³
        target_elevation_gain (float): Target elevation gain in meters (default: 0)
        n_points (int): Number of points to use in parameter search
        dt (float): Time interval in seconds
        eta (float): Drivetrain efficiency
        vw (float): Wind velocity in m/s (positive = headwind)
        lap_column (str): Column name containing lap numbers
        is_combined_laps (bool): Whether this is for combined laps analysis
        cda_bounds (tuple): (min, max) bounds for CdA optimization
        n_random_starts (int): Number of random starting points
        basin_hopping_steps (int): Number of basin hopping steps
        basin_hopping_temp (float): Temperature parameter for basin hopping
        verbose (bool): Whether to print detailed progress

    Returns:
        tuple: (optimized_cda, fixed_crr, rmse, r2, virtual_profile)
    """
    # Start timing
    start_time = time.time()

    # Convert actual elevation to numpy array if it's not already
    actual_elevation = np.array(actual_elevation)

    # Define the objective function
    def objective(cda):
        """Objective function that minimizes difference from target elevation gain"""
        # Handle single value input for basin_hopping compatibility
        if hasattr(cda, "__len__") and len(cda) == 1:
            cda = float(cda[0])

        # Calculate virtual elevation changes
        ve_changes = delta_ve(
            cda=cda, crr=fixed_crr, df=df, vw=vw, kg=kg, rho=rho, dt=dt, eta=eta
        )

        # Build virtual elevation profile
        virtual_profile = calculate_virtual_profile(
            ve_changes, actual_elevation, lap_column, df
        )

        # Calculate deviation from target elevation gain
        if lap_column is not None and lap_column in df.columns:
            lap_numbers = df[lap_column].values
            unique_laps = sorted(np.unique(lap_numbers))

            if is_combined_laps:
                # For combined laps: Calculate total elevation gain across all laps
                first_idx = 0
                last_idx = len(virtual_profile) - 1
                total_elevation_gain = (
                    virtual_profile[last_idx] - virtual_profile[first_idx]
                )

                # Compare total gain against target
                return abs(total_elevation_gain - target_elevation_gain)
            else:
                # For individual laps: Calculate average deviation from target per lap
                total_deviation = 0
                lap_count = 0

                for lap in unique_laps:
                    lap_indices = np.where(lap_numbers == lap)[0]
                    if len(lap_indices) > 1:
                        lap_start_idx = lap_indices[0]
                        lap_end_idx = lap_indices[-1]
                        lap_elevation_gain = (
                            virtual_profile[lap_end_idx]
                            - virtual_profile[lap_start_idx]
                        )
                        lap_deviation = abs(lap_elevation_gain - target_elevation_gain)
                        total_deviation += lap_deviation
                        lap_count += 1

                # Use average deviation across laps
                return total_deviation / max(1, lap_count)
        else:
            # Single lap case
            actual_gain = virtual_profile[-1] - virtual_profile[0]
            return abs(actual_gain - target_elevation_gain)

    # For calculating metrics from a parameter
    def calculate_metrics(cda):
        ve_changes = delta_ve(
            cda=cda, crr=fixed_crr, df=df, vw=vw, kg=kg, rho=rho, dt=dt, eta=eta
        )
        virtual_profile = calculate_virtual_profile(
            ve_changes, actual_elevation, lap_column, df
        )
        r2 = pearsonr(virtual_profile, actual_elevation)[0] ** 2
        rmse = np.sqrt(np.mean((virtual_profile - actual_elevation) ** 2))

        # Calculate elevation gains
        if lap_column is not None and lap_column in df.columns:
            lap_numbers = df[lap_column].values
            unique_laps = sorted(np.unique(lap_numbers))

            if is_combined_laps:
                # For combined laps: report total elevation gain
                first_idx = 0
                last_idx = len(virtual_profile) - 1
                total_elevation_gain = (
                    virtual_profile[last_idx] - virtual_profile[first_idx]
                )
                elevation_gains = [total_elevation_gain]  # Just report total
            else:
                # For individual laps: report gain per lap
                elevation_gains = []
                for lap in unique_laps:
                    lap_indices = np.where(lap_numbers == lap)[0]
                    if len(lap_indices) > 1:
                        lap_start_idx = lap_indices[0]
                        lap_end_idx = lap_indices[-1]
                        lap_elevation_gain = (
                            virtual_profile[lap_end_idx]
                            - virtual_profile[lap_start_idx]
                        )
                        elevation_gains.append(lap_elevation_gain)
        else:
            # Single lap case
            elevation_gains = [virtual_profile[-1] - virtual_profile[0]]

        return r2, rmse, virtual_profile, elevation_gains

    if verbose:
        if is_combined_laps:
            if target_elevation_gain == 0:
                print(
                    "Step 1: Initial search across CdA range for zero total elevation gain..."
                )
            else:
                print(
                    f"Step 1: Initial search across CdA range for {target_elevation_gain:.1f}m total elevation gain..."
                )
        else:
            if target_elevation_gain == 0:
                print(
                    "Step 1: Initial search across CdA range for zero elevation gain per lap..."
                )
            else:
                print(
                    f"Step 1: Initial search across CdA range for {target_elevation_gain:.1f}m elevation gain per lap..."
                )

    # Create a linearly spaced array of CdA values to test
    cda_values = np.linspace(cda_bounds[0], cda_bounds[1], n_points)

    # Evaluate the objective function for each CdA value
    initial_results = []
    for cda in cda_values:
        # Get target elevation deviation objective
        elev_deviation_obj = objective(cda)

        # Calculate individual metrics for reporting
        r2, rmse, _, elevation_gains = calculate_metrics(cda)
        avg_gain = sum(elevation_gains) / len(elevation_gains) if elevation_gains else 0

        initial_results.append((cda, elev_deviation_obj, r2, rmse, avg_gain))

    # Sort by objective (lower is better)
    initial_results.sort(key=lambda x: x[1])

    if verbose:
        print(f"Initial search completed in {time.time() - start_time:.1f} seconds")
        print(f"Top 3 initial search results:")
        for i in range(min(3, len(initial_results))):
            print(
                f"  CdA={initial_results[i][0]:.4f}m², "
                f"Elevation Gain={initial_results[i][4]:.2f}m, R²={initial_results[i][2]:.4f}, RMSE={initial_results[i][3]:.2f}m"
            )

    # Initialize storage for global best results
    global_best_results = []

    # Define the step-taking function for basin hopping to respect bounds
    def take_step(x):
        # Scale based on the bounds range
        step_size = (cda_bounds[1] - cda_bounds[0]) * 0.1
        # Random step within bounds - ensure we keep the same dimension as x
        new_x = x + np.random.uniform(-1, 1, size=x.shape) * step_size
        # Clip to bounds and ensure dimension is preserved
        return np.clip(new_x, cda_bounds[0], cda_bounds[1])

    # Define acceptance test function for basin hopping
    def accept_test(f_new, x_new, f_old, x_old):
        # Always accept if better
        if f_new < f_old:
            return True

        # Sometimes accept worse solutions based on temperature
        delta_f = f_new - f_old
        prob = np.exp(-delta_f / basin_hopping_temp)
        return np.random.random() < prob

    if verbose:
        print(
            "\nStep 2: Running multiple optimization attempts with different starting points..."
        )

    # Start with top grid results and add random starts
    starting_points = [
        initial_results[i][0] for i in range(min(3, len(initial_results)))
    ]

    # Add random starting points
    for _ in range(n_random_starts - len(starting_points)):
        random_cda = np.random.uniform(cda_bounds[0], cda_bounds[1])
        starting_points.append(random_cda)

    # Run optimization from each starting point
    for start_idx, start_cda in enumerate(starting_points):
        if verbose:
            print(f"\nAttempt {start_idx + 1}/{len(starting_points)}:")
            print(f"Starting point: CdA={start_cda:.4f}m²")

        start_attempt_time = time.time()

        # Use basin hopping for exploring multiple basins
        minimizer_kwargs = {
            "method": "L-BFGS-B",
            "bounds": [(cda_bounds[0], cda_bounds[1])],  # Proper format for L-BFGS-B
        }

        # Use scipy.optimize.minimize_scalar first for single parameter optimization
        scalar_result = minimize_scalar(
            objective, bounds=cda_bounds, method="bounded", options={"xatol": 1e-6}
        )

        # Then use basin hopping starting from this point
        bh_result = basinhopping(
            objective,
            np.array([scalar_result.x]),  # Start from scalar optimization result
            niter=basin_hopping_steps,
            T=basin_hopping_temp,
            stepsize=1.0,  # Initial step size, custom function will scale
            take_step=take_step,
            accept_test=accept_test,
            minimizer_kwargs=minimizer_kwargs,
        )

        # Extract result (basin hopping returns an array)
        bh_cda = bh_result.x[0]
        bh_obj = bh_result.fun

        # Calculate metrics for this attempt
        bh_r2, bh_rmse, bh_profile, bh_elevation_gains = calculate_metrics(bh_cda)
        bh_avg_gain = (
            sum(bh_elevation_gains) / len(bh_elevation_gains)
            if bh_elevation_gains
            else 0
        )

        if verbose:
            if is_combined_laps:
                print(
                    f"Basin hopping result: CdA={bh_cda:.4f}m², "
                    f"Total Elevation Gain={bh_avg_gain:.2f}m, R²={bh_r2:.4f}, RMSE={bh_rmse:.2f}m"
                )
            else:
                print(
                    f"Basin hopping result: CdA={bh_cda:.4f}m², "
                    f"Avg Elevation Gain per Lap={bh_avg_gain:.2f}m, R²={bh_r2:.4f}, RMSE={bh_rmse:.2f}m"
                )
            print(
                f"Attempt {start_idx + 1} completed in {time.time() - start_attempt_time:.1f} seconds"
            )

        # Store result from this attempt
        global_best_results.append(
            (bh_cda, bh_obj, bh_r2, bh_rmse, bh_profile, bh_elevation_gains)
        )

    # Find the global best result
    global_best_results.sort(key=lambda x: x[1])  # Sort by objective value
    best_cda, best_obj, best_r2, best_rmse, best_profile, best_elevation_gains = (
        global_best_results[0]
    )
    best_avg_gain = (
        sum(best_elevation_gains) / len(best_elevation_gains)
        if best_elevation_gains
        else 0
    )

    if verbose:
        print("\nAll optimization attempts completed.")
        print(f"Total optimization time: {time.time() - start_time:.1f} seconds")

        if is_combined_laps:
            if target_elevation_gain == 0:
                print(
                    f"Global best result: CdA={best_cda:.4f}m², "
                    f"Total Elevation Gain={best_avg_gain:.2f}m, R²={best_r2:.4f}, RMSE={best_rmse:.2f}m"
                )
            else:
                print(
                    f"Global best result: CdA={best_cda:.4f}m², "
                    f"Total Elevation Gain={best_avg_gain:.2f}m (target: {target_elevation_gain:.1f}m), "
                    f"R²={best_r2:.4f}, RMSE={best_rmse:.2f}m"
                )
        else:
            if target_elevation_gain == 0:
                print(
                    f"Global best result: CdA={best_cda:.4f}m², "
                    f"Avg Elevation Gain per Lap={best_avg_gain:.2f}m, R²={best_r2:.4f}, RMSE={best_rmse:.2f}m"
                )
            else:
                print(
                    f"Global best result: CdA={best_cda:.4f}m², "
                    f"Avg Elevation Gain per Lap={best_avg_gain:.2f}m (target: {target_elevation_gain:.1f}m), "
                    f"R²={best_r2:.4f}, RMSE={best_rmse:.2f}m"
                )

    return best_cda, fixed_crr, best_rmse, best_r2, best_profile


def optimize_crr_only_for_target_elevation(
    df,
    actual_elevation,
    fixed_cda,
    kg,
    rho,
    target_elevation_gain=0,
    n_points=100,
    dt=1,
    eta=1,
    vw=0,
    lap_column=None,
    is_combined_laps=False,
    crr_bounds=(0.001, 0.01),
    n_random_starts=5,
    basin_hopping_steps=30,
    basin_hopping_temp=1.0,
    verbose=True,
):
    """
    Optimize only Crr with a fixed CdA value, aiming for a specific target elevation gain.
    For individual laps, optimizes for the target gain per lap.
    For combined laps, optimizes for the target gain across all laps together.

    Args:
        df (pandas.DataFrame): DataFrame with cycling data
        actual_elevation (array-like): Actual measured elevation data
        fixed_cda (float): Fixed CdA value to use
        kg (float): Rider mass in kg
        rho (float): Air density in kg/m³
        target_elevation_gain (float): Target elevation gain in meters (default: 0)
        n_points (int): Number of points to use in parameter search
        dt (float): Time interval in seconds
        eta (float): Drivetrain efficiency
        vw (float): Wind velocity in m/s (positive = headwind)
        lap_column (str): Column name containing lap numbers
        is_combined_laps (bool): Whether this is for combined laps analysis
        crr_bounds (tuple): (min, max) bounds for Crr optimization
        n_random_starts (int): Number of random starting points
        basin_hopping_steps (int): Number of basin hopping steps
        basin_hopping_temp (float): Temperature parameter for basin hopping
        verbose (bool): Whether to print detailed progress

    Returns:
        tuple: (fixed_cda, optimized_crr, rmse, r2, virtual_profile)
    """
    # Start timing
    start_time = time.time()

    # Convert actual elevation to numpy array if it's not already
    actual_elevation = np.array(actual_elevation)

    # Define the objective function
    def objective(crr):
        """Objective function that minimizes difference from target elevation gain"""
        # Handle single value input for basin_hopping compatibility
        if hasattr(crr, "__len__") and len(crr) == 1:
            crr = float(crr[0])

        # Calculate virtual elevation changes
        ve_changes = delta_ve(
            cda=fixed_cda, crr=crr, df=df, vw=vw, kg=kg, rho=rho, dt=dt, eta=eta
        )

        # Build virtual elevation profile
        virtual_profile = calculate_virtual_profile(
            ve_changes, actual_elevation, lap_column, df
        )

        # Calculate deviation from target elevation gain
        if lap_column is not None and lap_column in df.columns:
            lap_numbers = df[lap_column].values
            unique_laps = sorted(np.unique(lap_numbers))

            if is_combined_laps:
                # For combined laps: Calculate total elevation gain across all laps
                first_idx = 0
                last_idx = len(virtual_profile) - 1
                total_elevation_gain = (
                    virtual_profile[last_idx] - virtual_profile[first_idx]
                )

                # Compare total gain against target
                return abs(total_elevation_gain - target_elevation_gain)
            else:
                # For individual laps: Calculate average deviation from target per lap
                total_deviation = 0
                lap_count = 0

                for lap in unique_laps:
                    lap_indices = np.where(lap_numbers == lap)[0]
                    if len(lap_indices) > 1:
                        lap_start_idx = lap_indices[0]
                        lap_end_idx = lap_indices[-1]
                        lap_elevation_gain = (
                            virtual_profile[lap_end_idx]
                            - virtual_profile[lap_start_idx]
                        )
                        lap_deviation = abs(lap_elevation_gain - target_elevation_gain)
                        total_deviation += lap_deviation
                        lap_count += 1

                # Use average deviation across laps
                return total_deviation / max(1, lap_count)
        else:
            # Single lap case
            actual_gain = virtual_profile[-1] - virtual_profile[0]
            return abs(actual_gain - target_elevation_gain)

    # For calculating metrics from a parameter
    def calculate_metrics(crr):
        ve_changes = delta_ve(
            cda=fixed_cda, crr=crr, df=df, vw=vw, kg=kg, rho=rho, dt=dt, eta=eta
        )
        virtual_profile = calculate_virtual_profile(
            ve_changes, actual_elevation, lap_column, df
        )
        r2 = pearsonr(virtual_profile, actual_elevation)[0] ** 2
        rmse = np.sqrt(np.mean((virtual_profile - actual_elevation) ** 2))

        # Calculate elevation gains
        if lap_column is not None and lap_column in df.columns:
            lap_numbers = df[lap_column].values
            unique_laps = sorted(np.unique(lap_numbers))

            if is_combined_laps:
                # For combined laps: report total elevation gain
                first_idx = 0
                last_idx = len(virtual_profile) - 1
                total_elevation_gain = (
                    virtual_profile[last_idx] - virtual_profile[first_idx]
                )
                elevation_gains = [total_elevation_gain]  # Just report total
            else:
                # For individual laps: report gain per lap
                elevation_gains = []
                for lap in unique_laps:
                    lap_indices = np.where(lap_numbers == lap)[0]
                    if len(lap_indices) > 1:
                        lap_start_idx = lap_indices[0]
                        lap_end_idx = lap_indices[-1]
                        lap_elevation_gain = (
                            virtual_profile[lap_end_idx]
                            - virtual_profile[lap_start_idx]
                        )
                        elevation_gains.append(lap_elevation_gain)
        else:
            # Single lap case
            elevation_gains = [virtual_profile[-1] - virtual_profile[0]]

        return r2, rmse, virtual_profile, elevation_gains

    if verbose:
        if is_combined_laps:
            if target_elevation_gain == 0:
                print(
                    "Step 1: Initial search across Crr range for zero total elevation gain..."
                )
            else:
                print(
                    f"Step 1: Initial search across Crr range for {target_elevation_gain:.1f}m total elevation gain..."
                )
        else:
            if target_elevation_gain == 0:
                print(
                    "Step 1: Initial search across Crr range for zero elevation gain per lap..."
                )
            else:
                print(
                    f"Step 1: Initial search across Crr range for {target_elevation_gain:.1f}m elevation gain per lap..."
                )

    # Create a linearly spaced array of Crr values to test
    crr_values = np.linspace(crr_bounds[0], crr_bounds[1], n_points)

    # Evaluate the objective function for each Crr value
    initial_results = []
    for crr in crr_values:
        # Get target elevation deviation objective
        elev_deviation_obj = objective(crr)

        # Calculate individual metrics for reporting
        r2, rmse, _, elevation_gains = calculate_metrics(crr)
        avg_gain = sum(elevation_gains) / len(elevation_gains) if elevation_gains else 0

        initial_results.append((crr, elev_deviation_obj, r2, rmse, avg_gain))

    # Sort by objective (lower is better)
    initial_results.sort(key=lambda x: x[1])

    if verbose:
        print(f"Initial search completed in {time.time() - start_time:.1f} seconds")
        print(f"Top 3 initial search results:")
        for i in range(min(3, len(initial_results))):
            print(
                f"  Crr={initial_results[i][0]:.5f}, "
                f"Elevation Gain={initial_results[i][4]:.2f}m, R²={initial_results[i][2]:.4f}, RMSE={initial_results[i][3]:.2f}m"
            )

    # Initialize storage for global best results
    global_best_results = []

    # Define the step-taking function for basin hopping to respect bounds
    def take_step(x):
        # Scale based on the bounds range
        step_size = (crr_bounds[1] - crr_bounds[0]) * 0.1
        # Random step within bounds - ensure we keep the same dimension as x
        new_x = x + np.random.uniform(-1, 1, size=x.shape) * step_size
        # Clip to bounds and ensure dimension is preserved
        return np.clip(new_x, crr_bounds[0], crr_bounds[1])

    # Define acceptance test function for basin hopping
    def accept_test(f_new, x_new, f_old, x_old):
        # Always accept if better
        if f_new < f_old:
            return True

        # Sometimes accept worse solutions based on temperature
        delta_f = f_new - f_old
        prob = np.exp(-delta_f / basin_hopping_temp)
        return np.random.random() < prob

    if verbose:
        print(
            "\nStep 2: Running multiple optimization attempts with different starting points..."
        )

    # Start with top grid results and add random starts
    starting_points = [
        initial_results[i][0] for i in range(min(3, len(initial_results)))
    ]

    # Add random starting points
    for _ in range(n_random_starts - len(starting_points)):
        random_crr = np.random.uniform(crr_bounds[0], crr_bounds[1])
        starting_points.append(random_crr)

    # Run optimization from each starting point
    for start_idx, start_crr in enumerate(starting_points):
        if verbose:
            print(f"\nAttempt {start_idx + 1}/{len(starting_points)}:")
            print(f"Starting point: Crr={start_crr:.5f}")

        start_attempt_time = time.time()

        # Use basin hopping for exploring multiple basins
        minimizer_kwargs = {
            "method": "L-BFGS-B",
            "bounds": [(crr_bounds[0], crr_bounds[1])],  # Proper format for L-BFGS-B
        }

        # Use scipy.optimize.minimize_scalar first for single parameter optimization
        scalar_result = minimize_scalar(
            objective, bounds=crr_bounds, method="bounded", options={"xatol": 1e-6}
        )

        # Then use basin hopping starting from this point
        bh_result = basinhopping(
            objective,
            np.array([scalar_result.x]),  # Start from scalar optimization result
            niter=basin_hopping_steps,
            T=basin_hopping_temp,
            stepsize=1.0,  # Initial step size, custom function will scale
            take_step=take_step,
            accept_test=accept_test,
            minimizer_kwargs=minimizer_kwargs,
        )

        # Extract result (basin hopping returns an array)
        bh_crr = bh_result.x[0]
        bh_obj = bh_result.fun

        # Calculate metrics for this attempt
        bh_r2, bh_rmse, bh_profile, bh_elevation_gains = calculate_metrics(bh_crr)
        bh_avg_gain = (
            sum(bh_elevation_gains) / len(bh_elevation_gains)
            if bh_elevation_gains
            else 0
        )

        if verbose:
            if is_combined_laps:
                print(
                    f"Basin hopping result: Crr={bh_crr:.5f}, "
                    f"Total Elevation Gain={bh_avg_gain:.2f}m, R²={bh_r2:.4f}, RMSE={bh_rmse:.2f}m"
                )
            else:
                print(
                    f"Basin hopping result: Crr={bh_crr:.5f}, "
                    f"Avg Elevation Gain per Lap={bh_avg_gain:.2f}m, R²={bh_r2:.4f}, RMSE={bh_rmse:.2f}m"
                )
            print(
                f"Attempt {start_idx + 1} completed in {time.time() - start_attempt_time:.1f} seconds"
            )

        # Store result from this attempt
        global_best_results.append(
            (bh_crr, bh_obj, bh_r2, bh_rmse, bh_profile, bh_elevation_gains)
        )

    # Find the global best result
    global_best_results.sort(key=lambda x: x[1])  # Sort by objective value
    best_crr, best_obj, best_r2, best_rmse, best_profile, best_elevation_gains = (
        global_best_results[0]
    )
    best_avg_gain = (
        sum(best_elevation_gains) / len(best_elevation_gains)
        if best_elevation_gains
        else 0
    )

    if verbose:
        print("\nAll optimization attempts completed.")
        print(f"Total optimization time: {time.time() - start_time:.1f} seconds")

        if is_combined_laps:
            if target_elevation_gain == 0:
                print(
                    f"Global best result: Crr={best_crr:.5f}, "
                    f"Total Elevation Gain={best_avg_gain:.2f}m, R²={best_r2:.4f}, RMSE={best_rmse:.2f}m"
                )
            else:
                print(
                    f"Global best result: Crr={best_crr:.5f}, "
                    f"Total Elevation Gain={best_avg_gain:.2f}m (target: {target_elevation_gain:.1f}m), "
                    f"R²={best_r2:.4f}, RMSE={best_rmse:.2f}m"
                )
        else:
            if target_elevation_gain == 0:
                print(
                    f"Global best result: Crr={best_crr:.5f}, "
                    f"Avg Elevation Gain per Lap={best_avg_gain:.2f}m, R²={best_r2:.4f}, RMSE={best_rmse:.2f}m"
                )
            else:
                print(
                    f"Global best result: Crr={best_crr:.5f}, "
                    f"Avg Elevation Gain per Lap={best_avg_gain:.2f}m (target: {target_elevation_gain:.1f}m), "
                    f"R²={best_r2:.4f}, RMSE={best_rmse:.2f}m"
                )

    return fixed_cda, best_crr, best_rmse, best_r2, best_profile
