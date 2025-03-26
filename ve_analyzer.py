#!/usr/bin/env python3
# ve_analyzer.py - Virtual Elevation Analyzer

import argparse
import os
import traceback

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Import from our modules
from core.calculations import accel_calc, calculate_distance, delta_ve
from core.config import VirtualElevationConfig
from core.optimization import (
    calculate_virtual_profile,
    optimize_both_params_balanced,
    optimize_both_params_for_target_elevation,
    optimize_cda_only_balanced,
    optimize_cda_only_for_target_elevation,
    optimize_crr_only_balanced,
    optimize_crr_only_for_target_elevation,
)
from core.visualization import (
    create_interactive_map,
    plot_elevation_profiles,
    plot_static_map,
)
from data_io.data_processing import resample_data, trim_data_by_distance
from data_io.fit_parser import parse_fit_file


def analyze_and_plot_ve(
    df,
    config: VirtualElevationConfig,
    actual_elevation_col="elevation",
    distance_col=None,
    save_path=None,
    lap_column=None,
    r2_weight=0.5,
    n_grid=250,
    target_elevation_gain=None,
    is_combined_laps=False,
    interactive_plot=False,
    lap_num=None,  # Added parameter for lap number
):
    """
    Complete workflow function to analyze data, optimize parameters, and plot results.

    Args:
        df (pandas.DataFrame): DataFrame with cycling data
        actual_elevation_col (str): Column name for actual elevation data
        kg (float): Rider mass in kg
        rho (float): Air density in kg/m³
        dt (float): Time interval in seconds
        eta (float): Drivetrain efficiency
        vw (float): Wind velocity in m/s (positive = headwind)
        distance_col (str, optional): Column name for distance data
        save_path (str, optional): Path to save the plot image
        lap_column (str, optional): Column name containing lap numbers
        fixed_cda (float, optional): Fixed CdA value (if provided, only Crr will be optimized)
        fixed_crr (float, optional): Fixed Crr value (if provided, only CdA will be optimized)
        r2_weight (float): Weight for R² in the composite objective (0-1)
        n_grid (int): Number of grid points to use in parameter search
        cda_bounds (tuple): (min, max) bounds for CdA optimization
        crr_bounds (tuple): (min, max) bounds for Crr optimization
        target_elevation_gain (float, optional): Target elevation gain to optimize for (default: None uses R²/RMSE)
        is_combined_laps (bool): Whether this is for combined laps analysis (affects how target_elevation_gain is interpreted)
        interactive_plot (bool): Whether to show interactive plot for parameter fine-tuning
        lap_num (int, optional): Current lap number being processed

    Returns:
        tuple: (optimized_cda, optimized_crr, rmse, r2, fig)
    """
    if config.kg is None or config.rho is None:
        raise ValueError(
            "Rider mass (kg) and air density (rho) are required parameters"
        )

    from scipy.stats import pearsonr

    # Ensure we have acceleration data
    if "a" not in df.columns:
        df["a"] = accel_calc(df["v"].values, config.dt)

    # Get actual elevation data
    actual_elevation = df[actual_elevation_col].values

    # Get distance data if available
    distance = None
    if distance_col and distance_col in df.columns:
        distance = df[distance_col].values
    else:
        distance = calculate_distance(df, config.dt)

    # Determine which parameters to optimize
    if config.fixed_cda is not None and config.fixed_crr is not None:
        # Both parameters fixed - no optimization needed
        print(
            f"Using fixed parameters: CdA={config.fixed_cda:.4f} m², Crr={config.fixed_crr:.5f}"
        )
        # Calculate virtual elevation with fixed parameters
        ve_changes = delta_ve(
            config,
            cda=config.fixed_cda,
            crr=config.fixed_crr,
            df=df,
        )

        # Build virtual elevation profile
        virtual_elevation = calculate_virtual_profile(
            ve_changes, actual_elevation, lap_column, df
        )

        # Calculate RMSE
        rmse = np.sqrt(np.mean((virtual_elevation - actual_elevation) ** 2))
        optimized_cda = config.fixed_cda
        optimized_crr = config.fixed_crr

    elif config.fixed_cda is not None:
        # Only optimize Crr with fixed CdA
        if target_elevation_gain is not None:
            if is_combined_laps:
                if target_elevation_gain == 0:
                    print(
                        f"Using fixed CdA={config.fixed_cda:.4f} m² and optimizing Crr for zero total elevation gain"
                    )
                else:
                    print(
                        f"Using fixed CdA={config.fixed_cda:.4f} m² and optimizing Crr for {target_elevation_gain:.1f}m total elevation gain"
                    )
            else:
                if target_elevation_gain == 0:
                    print(
                        f"Using fixed CdA={config.fixed_cda:.4f} m² and optimizing Crr for zero elevation gain per lap"
                    )
                else:
                    print(
                        f"Using fixed CdA={config.fixed_cda:.4f} m² and optimizing Crr for {target_elevation_gain:.1f}m elevation gain per lap"
                    )

            optimized_cda, optimized_crr, rmse, r2, virtual_elevation = (
                optimize_crr_only_for_target_elevation(
                    df=df,
                    actual_elevation=actual_elevation,
                    config=config,
                    target_elevation_gain=target_elevation_gain,
                    lap_column=lap_column,
                    n_points=n_grid,
                    is_combined_laps=is_combined_laps,
                )
            )
        else:
            print(f"Using fixed CdA={config.fixed_cda:.4f} m² and optimizing Crr")
            optimized_cda, optimized_crr, rmse, r2, virtual_elevation = (
                optimize_crr_only_balanced(
                    df=df,
                    actual_elevation=actual_elevation,
                    config=config,
                    fixed_cda=config.fixed_cda,
                    lap_column=lap_column,
                    n_points=n_grid,
                    r2_weight=r2_weight,
                )
            )

    elif config.fixed_crr is not None:
        # Only optimize CdA with fixed Crr
        if target_elevation_gain is not None:
            if is_combined_laps:
                if target_elevation_gain == 0:
                    print(
                        f"Using fixed Crr={config.fixed_crr:.5f} and optimizing CdA for zero total elevation gain"
                    )
                else:
                    print(
                        f"Using fixed Crr={config.fixed_crr:.5f} and optimizing CdA for {target_elevation_gain:.1f}m total elevation gain"
                    )
            else:
                if target_elevation_gain == 0:
                    print(
                        f"Using fixed Crr={config.fixed_crr:.5f} and optimizing CdA for zero elevation gain per lap"
                    )
                else:
                    print(
                        f"Using fixed Crr={config.fixed_crr:.5f} and optimizing CdA for {target_elevation_gain:.1f}m elevation gain per lap"
                    )

            optimized_cda, optimized_crr, rmse, r2, virtual_elevation = (
                optimize_cda_only_for_target_elevation(
                    df=df,
                    actual_elevation=actual_elevation,
                    config=config,
                    target_elevation_gain=target_elevation_gain,
                    lap_column=lap_column,
                    n_points=n_grid,
                    is_combined_laps=is_combined_laps,
                )
            )
        else:
            print(f"Using fixed Crr={config.fixed_crr:.5f} and optimizing CdA")
            optimized_cda, optimized_crr, rmse, r2, virtual_elevation = (
                optimize_cda_only_balanced(
                    df=df,
                    actual_elevation=actual_elevation,
                    config=config,
                    lap_column=lap_column,
                    n_points=n_grid,
                    r2_weight=r2_weight,
                )
            )

    else:
        # Optimize both parameters
        if target_elevation_gain is not None:
            if is_combined_laps:
                if target_elevation_gain == 0:
                    print("Optimizing both CdA and Crr for zero total elevation gain")
                else:
                    print(
                        f"Optimizing both CdA and Crr for {target_elevation_gain:.1f}m total elevation gain"
                    )
            else:
                if target_elevation_gain == 0:
                    print("Optimizing both CdA and Crr for zero elevation gain per lap")
                else:
                    print(
                        f"Optimizing both CdA and Crr for {target_elevation_gain:.1f}m elevation gain per lap"
                    )

            optimized_cda, optimized_crr, rmse, r2, virtual_elevation = (
                optimize_both_params_for_target_elevation(
                    df=df,
                    actual_elevation=actual_elevation,
                    config=config,
                    target_elevation_gain=target_elevation_gain,
                    lap_column=lap_column,
                    n_grid=n_grid,
                    is_combined_laps=is_combined_laps,
                )
            )
        else:
            print("Optimizing both CdA and Crr")
            optimized_cda, optimized_crr, rmse, r2, virtual_elevation = (
                optimize_both_params_balanced(
                    df=df,
                    actual_elevation=actual_elevation,
                    config=config,
                    lap_column=lap_column,
                    n_grid=n_grid,
                    r2_weight=r2_weight,
                )
            )

    # Calculate R²
    r2 = pearsonr(actual_elevation, virtual_elevation)[0] ** 2

    # Calculate elevation gains for reporting
    elevation_gains = []
    if lap_column is not None and lap_column in df.columns:
        lap_numbers = df[lap_column].values
        unique_laps = sorted(np.unique(lap_numbers))

        if is_combined_laps:
            # For combined laps, calculate the total elevation gain
            first_idx = 0
            last_idx = len(virtual_elevation) - 1
            total_elevation_gain = (
                virtual_elevation[last_idx] - virtual_elevation[first_idx]
            )
            elevation_gains = [total_elevation_gain]  # Just report the total
        else:
            # For individual laps, calculate gain per lap
            for lap in unique_laps:
                lap_indices = np.where(lap_numbers == lap)[0]
                if len(lap_indices) > 1:
                    lap_start_idx = lap_indices[0]
                    lap_end_idx = lap_indices[-1]
                    lap_elevation_gain = (
                        virtual_elevation[lap_end_idx]
                        - virtual_elevation[lap_start_idx]
                    )
                    elevation_gains.append(lap_elevation_gain)
    else:
        # Single lap case
        elevation_gains = [virtual_elevation[-1] - virtual_elevation[0]]

    avg_elevation_gain = (
        sum(elevation_gains) / len(elevation_gains) if elevation_gains else 0
    )

    # Plot results if not in interactive mode
    fig = None
    if not interactive_plot:
        fig = plot_elevation_profiles(
            df=df,
            actual_elevation=actual_elevation,
            virtual_elevation=virtual_elevation,
            distance=distance,
            cda=optimized_cda,
            crr=optimized_crr,
            rmse=rmse,
            r2=r2,
            save_path=save_path,
            lap_column=lap_column,
        )

    # Print results
    print(f"Analysis Results:")
    print(f"  CdA: {optimized_cda:.4f} m²")
    print(f"  Crr: {optimized_crr:.5f}")
    print(f"  RMSE: {rmse:.2f} meters")
    print(f"  R²: {r2:.4f}")

    # Add elevation gain information
    if is_combined_laps:
        if len(elevation_gains) > 0:
            print(f"  Total elevation gain: {elevation_gains[0]:.2f} meters")

            if target_elevation_gain is not None:
                deviation = abs(elevation_gains[0] - target_elevation_gain)
                print(f"  Target elevation gain: {target_elevation_gain:.2f} meters")
                print(f"  Deviation from target: {deviation:.2f} meters")
    else:
        if lap_column is not None and len(elevation_gains) > 1:
            print(f"  Average elevation gain per lap: {avg_elevation_gain:.2f} meters")
            print(
                f"  Elevation gains by lap: {', '.join([f'{gain:.2f}m' for gain in elevation_gains])}"
            )

            if target_elevation_gain is not None:
                avg_deviation = sum(
                    [abs(gain - target_elevation_gain) for gain in elevation_gains]
                ) / len(elevation_gains)
                print(
                    f"  Target elevation gain per lap: {target_elevation_gain:.2f} meters"
                )
                print(f"  Average deviation from target: {avg_deviation:.2f} meters")
        else:
            print(f"  Net elevation gain: {avg_elevation_gain:.2f} meters")

            if target_elevation_gain is not None:
                deviation = abs(avg_elevation_gain - target_elevation_gain)
                print(f"  Target elevation gain: {target_elevation_gain:.2f} meters")
                print(f"  Deviation from target: {deviation:.2f} meters")

    # For interactive plotting, we handle this in the analyze_lap_data function now
    # to ensure the proper workflow with save functionality
    if interactive_plot and not lap_num:  # Only use this for non-lap based analysis
        print("\nLaunching interactive plot for parameter fine-tuning...")
        saved_params = show_interactive_plot(
            df=df,
            actual_elevation=actual_elevation,
            optimized_cda=optimized_cda,
            optimized_crr=optimized_crr,
            distance=distance,
            config=config,
            lap_column=lap_column,
            rmse=rmse,
            r2=r2,
            save_path=None,  # Don't save by default
        )

        # Update parameters if the user saved new ones
        if saved_params:
            optimized_cda, optimized_crr = saved_params

            # Recalculate virtual elevation with new parameters
            ve_changes = delta_ve(config, cda=optimized_cda, crr=optimized_crr, df=df)
            virtual_elevation = calculate_virtual_profile(
                ve_changes, actual_elevation, lap_column, df
            )

            # Recalculate metrics
            rmse = np.sqrt(np.mean((virtual_elevation - actual_elevation) ** 2))
            r2 = pearsonr(virtual_elevation, actual_elevation)[0] ** 2

            print(f"\nUpdated parameters from interactive session:")
            print(f"  CdA: {optimized_cda:.4f} m²")
            print(f"  Crr: {optimized_crr:.5f}")
            print(f"  RMSE: {rmse:.2f} meters")
            print(f"  R²: {r2:.4f}")

    return optimized_cda, optimized_crr, rmse, r2, fig


# Updates to analyze_lap_data function
def analyze_lap_data(
    df,
    lap_messages,
    config: VirtualElevationConfig,
    fit_file_path,  # Added parameter to get filename
    save_dir=None,
    min_lap_duration=30,
    debug=False,
    trim_distance=0,
    trim_start=None,
    trim_end=None,
    r2_weight=0.5,
    n_grid=250,
    target_elevation_gain=None,
    show_map=False,
    interactive_plot=False,
):
    """
    Process and analyze data by laps.
    """
    import json
    import os

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    from core.calculations import accel_calc, calculate_distance, delta_ve
    from core.optimization import (
        calculate_virtual_profile,
        optimize_both_params_balanced,
    )
    from core.visualization import create_combined_interactive_plot
    from data_io.data_processing import resample_data, trim_data_by_distance

    if config.kg is None or config.rho is None:
        raise ValueError("Rider mass and air density are required parameters")

    # Create result directory based on filename if not exists
    file_basename = os.path.basename(fit_file_path)
    file_name, _ = os.path.splitext(file_basename)

    if save_dir:
        result_dir = os.path.join(save_dir, file_name)
        os.makedirs(result_dir, exist_ok=True)
    else:
        result_dir = None

    # Path for saved parameters
    params_file = None
    if result_dir:
        params_file = os.path.join(result_dir, "saved_parameters.json")

    # Load saved parameters if available
    saved_params = {}
    if params_file and os.path.exists(params_file):
        try:
            with open(params_file, "r") as f:
                saved_params = json.load(f)
            print(f"Loaded saved parameters from {params_file}")
        except Exception as e:
            print(f"Error loading saved parameters: {e}")
            saved_params = {}

    # Handle lap data
    if not lap_messages:
        print("No lap markers found in .fit file. Treating entire ride as one lap.")
        # Create a single lap covering the entire activity
        lap_segments = [
            {"lap_number": 1, "start_time": df.index.min(), "end_time": df.index.max()}
        ]
    else:
        print(f"Found {len(lap_messages)} laps in the .fit file")

        # Process each lap to extract timing info
        lap_segments = []

        for i, lap in enumerate(lap_messages):
            if "start_time" in lap and "total_elapsed_time" in lap:
                # Convert start time to datetime
                if isinstance(lap["start_time"], (int, float)):
                    start_time = pd.to_datetime(lap["start_time"], unit="s")
                else:
                    start_time = pd.to_datetime(lap["start_time"])

                # Get the lap duration in seconds
                lap_duration = lap["total_elapsed_time"]

                # Calculate end time correctly using duration
                end_time = start_time + pd.Timedelta(seconds=lap_duration)

                if debug:
                    print(
                        f"  Lap {i+1}: Start={start_time}, Duration={lap_duration:.1f}s, End={end_time}"
                    )

                if lap_duration >= min_lap_duration:
                    lap_segments.append(
                        {
                            "lap_number": i + 1,
                            "start_time": start_time,
                            "end_time": end_time,
                            "duration": lap_duration,
                        }
                    )
                else:
                    print(
                        f"  Skipping lap {i+1}: Too short ({lap_duration:.1f}s < {min_lap_duration}s)"
                    )

        if not lap_segments:
            print(
                f"No valid laps found with duration >= {min_lap_duration}s. Treating entire ride as one lap."
            )
            lap_segments = [
                {
                    "lap_number": 1,
                    "start_time": df.index.min(),
                    "end_time": df.index.max(),
                    "duration": (df.index.max() - df.index.min()).total_seconds(),
                }
            ]

    # Split data by laps and analyze each lap
    results = {}

    for lap_info in lap_segments:
        lap_num = lap_info["lap_number"]
        start_time = lap_info["start_time"]
        end_time = lap_info["end_time"]
        duration = lap_info.get("duration", (end_time - start_time).total_seconds())

        print(f"\nAnalyzing lap {lap_num}: {duration:.1f} seconds")

        # Extract lap data
        lap_df = df[(df.index >= start_time) & (df.index <= end_time)].copy()[
            ["v", "watts", "elevation"]
        ]

        # Add coordinates if available
        if "latitude" in df.columns and "longitude" in df.columns:
            lap_df["latitude"] = df[(df.index >= start_time) & (df.index <= end_time)][
                "latitude"
            ]
            lap_df["longitude"] = df[(df.index >= start_time) & (df.index <= end_time)][
                "longitude"
            ]

        if len(lap_df) < 10:
            print(f"  Skipping lap {lap_num}: Not enough data points ({len(lap_df)})")
            continue

        # Resample to constant time interval
        try:
            resampled_df = resample_data(lap_df, config.resample_freq)
            if len(resampled_df) < 10:
                print(
                    f"  Skipping lap {lap_num}: Not enough data points after resampling ({len(resampled_df)})"
                )
                continue
        except Exception as e:
            print(f"  Error resampling lap {lap_num}: {str(e)}")
            continue

        # Reset index to get timestamp as column
        resampled_df = resampled_df.reset_index()

        # Calculate time differences to get dt
        dt_values = resampled_df["timestamp"].diff().dt.total_seconds()
        avg_dt = dt_values[1:].mean()  # skip first row which is NaN
        dt = avg_dt if not np.isnan(avg_dt) else 1.0
        config.dt = dt

        # Calculate acceleration
        resampled_df["a"] = accel_calc(resampled_df["v"].values, dt)

        # Calculate distance before any trimming
        distance = calculate_distance(resampled_df, dt)

        # Set up paths for saving files
        lap_save_dir = result_dir if result_dir else save_dir
        save_path = None
        if lap_save_dir:
            save_path = os.path.join(lap_save_dir, f"lap_{lap_num}_elevation.png")

        # Get saved parameters for this lap if available
        lap_key = f"lap_{lap_num}"
        initial_trim_start = trim_start if trim_start is not None else trim_distance
        initial_trim_end = trim_end if trim_end is not None else trim_distance
        initial_cda = config.fixed_cda if config.fixed_cda is not None else 0.3
        initial_crr = config.fixed_crr if config.fixed_crr is not None else 0.005

        if lap_key in saved_params:
            saved_lap_params = saved_params[lap_key]
            initial_trim_start = saved_lap_params.get("trim_start", initial_trim_start)
            initial_trim_end = saved_lap_params.get("trim_end", initial_trim_end)
            initial_cda = saved_lap_params.get("cda", initial_cda)
            initial_crr = saved_lap_params.get("crr", initial_crr)
            print(
                f"  Using saved parameters: trim_start={initial_trim_start:.1f}m, trim_end={initial_trim_end:.1f}m, CdA={initial_cda:.4f}, Crr={initial_crr:.5f}"
            )

        # Create optimization function for interactive plot
        def optimization_function(
            df,
            actual_elevation,
            config: VirtualElevationConfig,
            initial_cda=None,
            initial_crr=None,
            target_elevation_gain=None,
        ):
            """Enhanced optimization function that properly handles target_elevation_gain"""

            # When using target elevation gain
            if target_elevation_gain is not None:
                if config.fixed_cda is not None and config.fixed_crr is not None:
                    # Both parameters fixed - no optimization needed
                    print(
                        f"Using fixed parameters: CdA={config.fixed_cda:.4f} m², Crr={config.fixed_crr:.5f}"
                    )

                    # Calculate virtual elevation with fixed parameters
                    ve_changes = delta_ve(
                        config, cda=config.fixed_cda, crr=config.fixed_crr, df=df
                    )

                    # Build virtual elevation profile
                    virtual_elevation = calculate_virtual_profile(
                        ve_changes, actual_elevation, None, df
                    )

                    # Calculate RMSE and R²
                    rmse = np.sqrt(np.mean((virtual_elevation - actual_elevation) ** 2))
                    from scipy.stats import pearsonr

                    r2 = pearsonr(virtual_elevation, actual_elevation)[0] ** 2

                    return (
                        config.fixed_cda,
                        config.fixed_crr,
                        rmse,
                        r2,
                        virtual_elevation,
                    )

                elif config.fixed_cda is not None:
                    # Optimize Crr only with fixed CdA for target elevation
                    print(
                        f"Using fixed CdA={config.fixed_cda:.4f} m² and optimizing Crr for target elevation gain {target_elevation_gain:.1f}m"
                    )
                    from core.optimization import optimize_crr_only_for_target_elevation

                    return optimize_crr_only_for_target_elevation(
                        df=df,
                        actual_elevation=actual_elevation,
                        config=config,
                        target_elevation_gain=target_elevation_gain,
                        is_combined_laps=False if lap_num > 0 else True,
                    )

                elif config.fixed_crr is not None:
                    # Optimize CdA only with fixed Crr for target elevation
                    print(
                        f"Using fixed Crr={config.fixed_crr:.5f} and optimizing CdA for target elevation gain {target_elevation_gain:.1f}m"
                    )
                    from core.optimization import optimize_cda_only_for_target_elevation

                    return optimize_cda_only_for_target_elevation(
                        df=df,
                        actual_elevation=actual_elevation,
                        config=config,
                        target_elevation_gain=target_elevation_gain,
                        is_combined_laps=False if lap_num > 0 else True,
                    )

                else:
                    # Optimize both parameters for target elevation
                    print(
                        f"Optimizing both CdA and Crr for target elevation gain {target_elevation_gain:.1f}m"
                    )
                    from core.optimization import (
                        optimize_both_params_for_target_elevation,
                    )

                    return optimize_both_params_for_target_elevation(
                        df=df,
                        actual_elevation=actual_elevation,
                        config=config,
                        target_elevation_gain=target_elevation_gain,
                        is_combined_laps=False if lap_num > 0 else True,
                    )
            else:
                # Standard R²/RMSE optimization
                if config.fixed_cda is not None and config.fixed_crr is not None:
                    # Both parameters fixed - no optimization needed
                    print(
                        f"Using fixed parameters: CdA={config.fixed_cda:.4f} m², Crr={config.fixed_crr:.5f}"
                    )
                    # Calculate virtual elevation with fixed parameters
                    ve_changes = delta_ve(
                        config, cda=config.fixed_cda, crr=config.fixed_crr, df=df
                    )
                    # Build virtual elevation profile
                    virtual_elevation = calculate_virtual_profile(
                        ve_changes, actual_elevation, None, df
                    )
                    # Calculate RMSE and R²
                    rmse = np.sqrt(np.mean((virtual_elevation - actual_elevation) ** 2))
                    from scipy.stats import pearsonr

                    r2 = pearsonr(virtual_elevation, actual_elevation)[0] ** 2
                    return (
                        config.fixed_cda,
                        config.fixed_crr,
                        rmse,
                        r2,
                        virtual_elevation,
                    )

                elif config.fixed_cda is not None:
                    # Only optimize Crr with fixed CdA
                    print(
                        f"Using fixed CdA={config.fixed_cda:.4f} m² and optimizing Crr"
                    )
                    from core.optimization import optimize_crr_only_balanced

                    return optimize_crr_only_balanced(
                        df=df,
                        actual_elevation=actual_elevation,
                        config=config,
                        r2_weight=r2_weight,
                    )

                elif config.fixed_crr is not None:
                    # Only optimize CdA with fixed Crr
                    print(f"Using fixed Crr={config.fixed_crr:.5f} and optimizing CdA")
                    from core.optimization import optimize_cda_only_balanced

                    return optimize_cda_only_balanced(
                        df=df,
                        actual_elevation=actual_elevation,
                        config=config,
                        r2_weight=r2_weight,
                    )

                else:
                    # Optimize both parameters
                    print("Optimizing both CdA and Crr")
                    from core.optimization import optimize_both_params_balanced

                    return optimize_both_params_balanced(
                        df=df,
                        actual_elevation=actual_elevation,
                        config=config,
                        r2_weight=r2_weight,
                        n_grid=n_grid,
                    )

        # ---------------------------------------------------------------------------
        # INTERACTIVE MODE WITH COMBINED PLOT
        # ---------------------------------------------------------------------------
        if interactive_plot:
            map_save_path = None
            if lap_save_dir:
                map_save_path = os.path.join(
                    lap_save_dir, f"lap_{lap_num}_interactive.png"
                )

            # Use the combined interactive plot
            (
                action,
                user_trim_start,
                user_trim_end,
                final_cda,
                final_crr,
                final_rmse,
                final_r2,
            ) = create_combined_interactive_plot(
                df=resampled_df,
                actual_elevation=resampled_df["elevation"].values,
                lap_num=lap_num,
                config=config,
                initial_cda=initial_cda,
                initial_crr=initial_crr,
                initial_trim_start=initial_trim_start,
                initial_trim_end=initial_trim_end,
                save_path=map_save_path,
                # Pass target_elevation_gain in the lambda function
                optimization_function=lambda df, actual_elevation, config, initial_cda=None, initial_crr=None, target_elevation_gain=None: optimization_function(
                    df,
                    actual_elevation,
                    config,
                    initial_cda,
                    initial_crr,
                    target_elevation_gain,
                ),
                distance=distance,
                lap_column=None,
                target_elevation_gain=target_elevation_gain,  # Pass target_elevation_gain here
            )

            # Check if user chose to skip this lap
            if action == "skip":
                print(f"  User chose to skip lap {lap_num}")
                continue

            # Apply trimming with user-selected values
            print(
                f"  Using trim values: start={user_trim_start:.1f}m, end={user_trim_end:.1f}m"
            )

            # Calculate distance for trimming
            trimmed_df = trim_data_by_distance(
                resampled_df, distance, 0, user_trim_start, user_trim_end
            )

            if len(trimmed_df) < 10:
                print(
                    f"  Skipping lap {lap_num}: Not enough data points after trimming ({len(trimmed_df)})"
                )
                continue

            # Recalculate distance after trimming
            trimmed_distance = calculate_distance(trimmed_df, dt)

            # Save parameters if user saved results
            if action == "save" and lap_save_dir:
                # Store parameters for this lap
                lap_params = {
                    "trim_start": user_trim_start,
                    "trim_end": user_trim_end,
                    "cda": final_cda,
                    "crr": final_crr,
                    "rmse": final_rmse,
                    "r2": final_r2,
                }

                # Update saved parameters
                saved_params[lap_key] = lap_params

                # Save to file
                try:
                    with open(params_file, "w") as f:
                        json.dump(saved_params, f, indent=2)
                    print(f"  Saved parameters to {params_file}")
                except Exception as e:
                    print(f"  Error saving parameters: {e}")

            # Use the final values
            cda, crr, rmse, r2 = final_cda, final_crr, final_rmse, final_r2

            # Calculate virtual elevation profile for saving
            ve_changes = delta_ve(config, cda=cda, crr=crr, df=trimmed_df)
            virtual_profile = calculate_virtual_profile(
                ve_changes, trimmed_df["elevation"].values, None, trimmed_df
            )

            # Save a plot with the updated parameters
            if lap_save_dir and action == "save":
                from core.visualization import plot_elevation_profiles

                updated_save_path = os.path.join(
                    lap_save_dir, f"lap_{lap_num}_elevation_{action}.png"
                )
                updated_fig = plot_elevation_profiles(
                    df=trimmed_df,
                    actual_elevation=trimmed_df["elevation"].values,
                    virtual_elevation=virtual_profile,
                    distance=trimmed_distance,
                    cda=cda,
                    crr=crr,
                    rmse=rmse,
                    r2=r2,
                    save_path=updated_save_path,
                    lap_column=None,
                )
                plt.close(updated_fig)
                print(f"  Saved final plot to {updated_save_path}")

        # ---------------------------------------------------------------------------
        # REGULAR WORKFLOW (non-interactive)
        # ---------------------------------------------------------------------------
        else:
            # Traditional approach with separate trimming and optimization
            if trim_distance > 0 or trim_start is not None or trim_end is not None:
                # Apply trimming with prioritized individual trim values
                resampled_df = trim_data_by_distance(
                    resampled_df, distance, trim_distance, trim_start, trim_end
                )
                if len(resampled_df) < 10:
                    print(
                        f"  Skipping lap {lap_num}: Not enough data points after trimming ({len(resampled_df)})"
                    )
                    continue

            # Recalculate distance after trimming
            distance = calculate_distance(resampled_df, dt)

            try:
                # Run the analysis/optimization
                print("  Running parameter optimization...")
                cda, crr, rmse, r2, fig = analyze_and_plot_ve(
                    df=resampled_df,
                    config=config,
                    actual_elevation_col="elevation",
                    save_path=save_path,
                    r2_weight=r2_weight,
                    n_grid=n_grid,
                    target_elevation_gain=target_elevation_gain,
                    interactive_plot=False,
                )
            except Exception as e:
                print(f"  Error analyzing lap {lap_num}: {str(e)}")
                import traceback

                traceback_info = traceback.format_exc()
                print(f"  Traceback: {traceback_info}")
                results[f"lap_{lap_num}"] = {
                    "error": str(e),
                    "data": resampled_df,
                    "lap_number": lap_num,
                }
                continue

        # Calculate lap statistics
        lap_stats = {
            "duration_seconds": duration,
            "distance_meters": np.sum(resampled_df["v"].values * dt),
            "avg_power": resampled_df["watts"].mean(),
            "max_power": resampled_df["watts"].max(),
            "avg_speed": resampled_df["v"].mean(),
            "max_speed": resampled_df["v"].max(),
            "elevation_gain": np.sum(
                np.maximum(0, np.diff(resampled_df["elevation"].values))
            ),
            "start_time": start_time,
            "end_time": end_time,
            "lap_number": lap_num,
        }

        # Store results
        trimmed_or_resampled_df = trimmed_df if interactive_plot else resampled_df
        results[f"lap_{lap_num}"] = {
            "cda": cda,
            "crr": crr,
            "rmse": rmse,
            "r2": r2,
            "data": trimmed_or_resampled_df,
            "trim_start": user_trim_start if interactive_plot else trim_start,
            "trim_end": user_trim_end if interactive_plot else trim_end,
            **lap_stats,
        }

        print(
            f"  Results: CdA={cda:.4f}m², Crr={crr:.5f}, RMSE={rmse:.2f}m, R²={r2:.4f}"
        )
        print(
            f"  Avg Power: {lap_stats['avg_power']:.1f}W, Avg Speed: {lap_stats['avg_speed']*3.6:.1f}km/h"
        )

        # Generate maps if requested
        if (
            show_map and not interactive_plot
        ):  # Don't need separate maps in interactive mode
            from core.visualization import create_interactive_map, plot_static_map

            # Get the original lap data with GPS coordinates
            original_lap_df = df[
                (df.index >= start_time) & (df.index <= end_time)
            ].copy()

            if (
                "latitude" in original_lap_df.columns
                and "longitude" in original_lap_df.columns
            ):
                # Create static map
                map_dir = lap_save_dir if lap_save_dir else save_dir
                static_map_path = os.path.join(map_dir, f"lap_{lap_num}_map.png")
                try:
                    plot_static_map(original_lap_df, save_path=static_map_path)
                    print(f"  Generated static map: {static_map_path}")
                except Exception as e:
                    print(f"  Error creating static map: {str(e)}")

                # Create interactive map
                interactive_map_path = os.path.join(map_dir, f"lap_{lap_num}_map.html")
                try:
                    create_interactive_map(
                        original_lap_df, save_path=interactive_map_path
                    )
                    print(f"  Generated interactive map: {interactive_map_path}")
                except Exception as e:
                    print(f"  Error creating interactive map: {str(e)}")
            else:
                print("  Cannot generate maps: GPS data not available")

    return results


def analyze_combined_laps(
    df,
    lap_messages,
    selected_laps,
    fit_file_path,  # Added parameter to get filename
    config: VirtualElevationConfig,
    save_dir=None,
    min_lap_duration=30,
    debug=False,
    trim_distance=0,
    trim_start=None,
    trim_end=None,
    r2_weight=0.5,
    n_grid=250,
    target_elevation_gain=None,
    show_map=False,
    interactive_plot=False,
):
    """
    Process and analyze a specific set of laps combined as one segment.
    If target_elevation_gain is provided, it's interpreted as the target total elevation gain
    across all combined laps.
    """
    import json
    import os

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from scipy.stats import pearsonr

    from core.calculations import accel_calc, calculate_distance, delta_ve
    from core.optimization import calculate_virtual_profile
    from core.visualization import create_combined_interactive_plot
    from data_io.data_processing import resample_data, trim_data_by_distance

    if config.kg is None or config.rho is None:
        raise ValueError("Rider mass and air density are required parameters")

    # Create result directory based on filename if not exists
    file_basename = os.path.basename(fit_file_path)
    file_name, _ = os.path.splitext(file_basename)

    if save_dir:
        result_dir = os.path.join(save_dir, file_name)
        os.makedirs(result_dir, exist_ok=True)
    else:
        result_dir = None

    # Path for saved parameters
    params_file = None
    if result_dir:
        params_file = os.path.join(result_dir, "saved_parameters.json")

    # Load saved parameters if available
    saved_params = {}
    if params_file and os.path.exists(params_file):
        try:
            with open(params_file, "r") as f:
                saved_params = json.load(f)
            print(f"Loaded saved parameters from {params_file}")
        except Exception as e:
            print(f"Error loading saved parameters: {e}")
            saved_params = {}

    # Extract lap information
    lap_info = []
    for i, lap in enumerate(lap_messages):
        if "start_time" in lap and "total_elapsed_time" in lap:
            # Convert start time to datetime
            if isinstance(lap["start_time"], (int, float)):
                start_time = pd.to_datetime(lap["start_time"], unit="s")
            else:
                start_time = pd.to_datetime(lap["start_time"])

            # Get the lap duration in seconds
            lap_duration = lap["total_elapsed_time"]

            # Calculate end time
            end_time = start_time + pd.Timedelta(seconds=lap_duration)

            if lap_duration >= min_lap_duration:
                lap_info.append(
                    {
                        "lap_number": i + 1,
                        "start_time": start_time,
                        "end_time": end_time,
                        "duration": lap_duration,
                    }
                )

    # Filter to only the selected laps
    selected_lap_info = [lap for lap in lap_info if lap["lap_number"] in selected_laps]

    if not selected_lap_info:
        print(f"None of the requested laps {selected_laps} were found or valid")
        return None

    # Sort laps by start time
    selected_lap_info.sort(key=lambda x: x["start_time"])

    print(f"Found {len(selected_lap_info)} of the {len(selected_laps)} requested laps")
    for lap in selected_lap_info:
        print(f"  Lap {lap['lap_number']}: Duration={lap['duration']:.1f}s")

    # Combine data from all selected laps
    combined_df = pd.DataFrame()
    total_duration = 0

    for lap in selected_lap_info:
        # Extract lap data
        columns_to_extract = ["v", "watts", "elevation"]
        if "latitude" in df.columns and "longitude" in df.columns:
            columns_to_extract.extend(["latitude", "longitude"])

        lap_df = df[
            (df.index >= lap["start_time"]) & (df.index <= lap["end_time"])
        ].copy()[columns_to_extract]

        if len(lap_df) < 10:
            print(
                f"  Skipping lap {lap['lap_number']}: Not enough data points ({len(lap_df)})"
            )
            continue

        # Resample each lap individually
        try:
            resampled_lap_df = resample_data(lap_df, config.resample_freq)

            # Add a lap identifier column
            resampled_lap_df["lap_number"] = lap["lap_number"]

            # Reset index to get timestamp as column before concatenation
            resampled_lap_df = resampled_lap_df.reset_index()

            # Add to combined data
            combined_df = pd.concat([combined_df, resampled_lap_df], ignore_index=True)
            total_duration += lap["duration"]

        except Exception as e:
            print(f"  Error resampling lap {lap['lap_number']}: {str(e)}")
            continue

    if len(combined_df) < 10:
        print(f"Not enough data points in the combined laps ({len(combined_df)})")
        return None

    print(
        f"Combined {len(selected_lap_info)} laps with {len(combined_df)} data points and {total_duration:.1f}s duration"
    )

    # Calculate time differences to get dt (using the median to avoid large gaps between laps)
    dt_values = []

    for i in range(1, len(combined_df)):
        if combined_df.loc[i, "lap_number"] == combined_df.loc[i - 1, "lap_number"]:
            dt = (
                combined_df.loc[i, "timestamp"] - combined_df.loc[i - 1, "timestamp"]
            ).total_seconds()
            dt_values.append(dt)

    # Calculate average dt from valid intervals (within the same lap)
    if dt_values:
        avg_dt = np.median(dt_values)  # Use median to be robust to outliers
    else:
        avg_dt = 1.0  # Default if no valid intervals

    dt = avg_dt if not np.isnan(avg_dt) else 1.0
    print(f"Using time interval dt={dt:.2f}s for acceleration calculations")

    # set dt in config for use in optimization
    config.dt = dt

    # Calculate acceleration (only within each lap, not across lap boundaries)
    a_values = np.zeros(len(combined_df))

    for lap_num in combined_df["lap_number"].unique():
        lap_mask = combined_df["lap_number"] == lap_num
        lap_indices = np.where(lap_mask)[0]

        if len(lap_indices) > 1:
            lap_v = combined_df.loc[lap_mask, "v"].values
            lap_a = accel_calc(lap_v, config.dt)
            a_values[lap_indices] = lap_a

    combined_df["a"] = a_values

    # Calculate distance before any trimming
    distance = calculate_distance(combined_df, config.dt)

    # Get saved parameters for combined laps if available
    combined_key = f"combined_laps_{'-'.join(map(str, selected_laps))}"
    initial_trim_start = trim_start if trim_start is not None else trim_distance
    initial_trim_end = trim_end if trim_end is not None else trim_distance
    initial_cda = config.fixed_cda if config.fixed_cda is not None else 0.3
    initial_crr = config.fixed_crr if config.fixed_crr is not None else 0.005

    if combined_key in saved_params:
        saved_combined_params = saved_params[combined_key]
        initial_trim_start = saved_combined_params.get("trim_start", initial_trim_start)
        initial_trim_end = saved_combined_params.get("trim_end", initial_trim_end)
        initial_cda = saved_combined_params.get("cda", initial_cda)
        initial_crr = saved_combined_params.get("crr", initial_crr)
        print(
            f"Using saved parameters for combined laps: trim_start={initial_trim_start:.1f}m, trim_end={initial_trim_end:.1f}m, CdA={initial_cda:.4f}, Crr={initial_crr:.5f}"
        )

    # Set up paths for saving files
    combined_save_dir = result_dir if result_dir else save_dir
    save_path = None
    if combined_save_dir:
        lap_str = "-".join(map(str, selected_laps))
        save_path = os.path.join(combined_save_dir, f"laps_{lap_str}_combined.png")

    # Create optimization function for interactive plot
    def optimization_function(
        df,
        actual_elevation,
        config: VirtualElevationConfig,
        initial_cda=None,
        initial_crr=None,
        target_elevation_gain=None,
    ):
        """Enhanced optimization function that properly handles target_elevation_gain"""

        # When using target elevation gain
        if target_elevation_gain is not None:
            if config.fixed_cda is not None and config.fixed_crr is not None:
                # Both parameters fixed - no optimization needed
                print(
                    f"Using fixed parameters: CdA={config.fixed_cda:.4f} m², Crr={config.fixed_crr:.5f}"
                )

                # Calculate virtual elevation with fixed parameters
                ve_changes = delta_ve(
                    config,
                    df=df,
                )

                # Build virtual elevation profile
                virtual_elevation = calculate_virtual_profile(
                    ve_changes, actual_elevation, "lap_number", df
                )

                # Calculate RMSE and R²
                rmse = np.sqrt(np.mean((virtual_elevation - actual_elevation) ** 2))
                from scipy.stats import pearsonr

                r2 = pearsonr(virtual_elevation, actual_elevation)[0] ** 2

                return config.fixed_cda, config.fixed_crr, rmse, r2, virtual_elevation

            elif config.fixed_cda is not None:
                # Optimize Crr only with fixed CdA for target elevation
                print(
                    f"Using fixed CdA={config.fixed_cda:.4f} m² and optimizing Crr for target elevation gain {target_elevation_gain:.1f}m"
                )
                from core.optimization import optimize_crr_only_for_target_elevation

                return optimize_crr_only_for_target_elevation(
                    df=df,
                    actual_elevation=actual_elevation,
                    config=config,
                    target_elevation_gain=target_elevation_gain,
                    is_combined_laps=True,  # Always true for combined laps
                )

            elif config.fixed_crr is not None:
                # Optimize CdA only with fixed Crr for target elevation
                print(
                    f"Using fixed Crr={config.fixed_crr:.5f} and optimizing CdA for target elevation gain {target_elevation_gain:.1f}m"
                )
                from core.optimization import optimize_cda_only_for_target_elevation

                return optimize_cda_only_for_target_elevation(
                    df=df,
                    actual_elevation=actual_elevation,
                    config=config,
                    target_elevation_gain=target_elevation_gain,
                    is_combined_laps=True,  # Always true for combined laps
                )

            else:
                # Optimize both parameters for target elevation
                print(
                    f"Optimizing both CdA and Crr for target elevation gain {target_elevation_gain:.1f}m"
                )
                from core.optimization import optimize_both_params_for_target_elevation

                return optimize_both_params_for_target_elevation(
                    df=df,
                    actual_elevation=actual_elevation,
                    config=config,
                    target_elevation_gain=target_elevation_gain,
                    is_combined_laps=True,  # Always true for combined laps
                )
        else:
            # Standard R²/RMSE optimization
            if config.fixed_cda is not None and config.fixed_crr is not None:
                # Both parameters fixed - no optimization needed
                print(
                    f"Using fixed parameters: CdA={config.fixed_cda:.4f} m², Crr={config.fixed_crr:.5f}"
                )
                # Calculate virtual elevation with fixed parameters
                ve_changes = delta_ve(
                    config, cda=config.fixed_cda, crr=config.fixed_crr, df=df
                )
                # Build virtual elevation profile
                virtual_elevation = calculate_virtual_profile(
                    ve_changes, actual_elevation, "lap_number", df
                )
                # Calculate RMSE and R²
                rmse = np.sqrt(np.mean((virtual_elevation - actual_elevation) ** 2))
                from scipy.stats import pearsonr

                r2 = pearsonr(virtual_elevation, actual_elevation)[0] ** 2
                return config.fixed_cda, config.fixed_crr, rmse, r2, virtual_elevation

            elif config.fixed_cda is not None:
                # Only optimize Crr with fixed CdA
                print(f"Using fixed CdA={config.fixed_cda:.4f} m² and optimizing Crr")
                from core.optimization import optimize_crr_only_balanced

                return optimize_crr_only_balanced(
                    df=df,
                    actual_elevation=actual_elevation,
                    config=config,
                    r2_weight=r2_weight,
                )

            elif config.fixed_crr is not None:
                # Only optimize CdA with fixed Crr
                print(f"Using fixed Crr={config.fixed_crr:.5f} and optimizing CdA")
                from core.optimization import optimize_cda_only_balanced

                return optimize_cda_only_balanced(
                    df=df,
                    actual_elevation=actual_elevation,
                    config=config,
                    r2_weight=r2_weight,
                )

            else:
                # Optimize both parameters
                print("Optimizing both CdA and Crr")
                from core.optimization import optimize_both_params_balanced

                return optimize_both_params_balanced(
                    df=df,
                    actual_elevation=actual_elevation,
                    config=config,
                    r2_weight=r2_weight,
                    n_grid=n_grid,
                )

    # INTERACTIVE MODE WITH COMBINED PLOT
    if interactive_plot:
        map_save_path = None
        if combined_save_dir:
            lap_str = "-".join(map(str, selected_laps))
            map_save_path = os.path.join(
                combined_save_dir, f"laps_{lap_str}_combined_interactive.png"
            )

        # Use the combined interactive plot
        (
            action,
            user_trim_start,
            user_trim_end,
            final_cda,
            final_crr,
            final_rmse,
            final_r2,
        ) = create_combined_interactive_plot(
            df=combined_df,
            actual_elevation=combined_df["elevation"].values,
            lap_num=0,  # 0 indicates combined laps
            config=config,
            initial_cda=initial_cda,
            initial_crr=initial_crr,
            initial_trim_start=initial_trim_start,
            initial_trim_end=initial_trim_end,
            save_path=map_save_path,
            optimization_function=lambda df, actual_elevation, config, initial_cda=None, initial_crr=None, target_elevation_gain=None: optimization_function(
                df,
                actual_elevation,
                config,
                initial_cda,
                initial_crr,
                target_elevation_gain,
            ),
            distance=distance,
            lap_column="lap_number",
            target_elevation_gain=target_elevation_gain,  # Add this parameter
        )

        # Check if user chose to skip
        if action == "skip":
            print(f"User chose to skip combined lap analysis")
            return None

        # Apply trimming with user-selected values
        print(
            f"Using trim values: start={user_trim_start:.1f}m, end={user_trim_end:.1f}m"
        )

        # Apply trimming
        trimmed_df = trim_data_by_distance(
            combined_df, distance, 0, user_trim_start, user_trim_end
        )

        if len(trimmed_df) < 10:
            print(f"Not enough data points after trimming ({len(trimmed_df)})")
            return None

        # Recalculate distance after trimming
        trimmed_distance = calculate_distance(trimmed_df, dt)

        # Save parameters if user saved results
        if action == "save" and combined_save_dir:
            # Store parameters for combined laps
            lap_params = {
                "trim_start": user_trim_start,
                "trim_end": user_trim_end,
                "cda": final_cda,
                "crr": final_crr,
                "rmse": final_rmse,
                "r2": final_r2,
            }

            # Update saved parameters
            saved_params[combined_key] = lap_params

            # Save to file
            try:
                with open(params_file, "w") as f:
                    json.dump(saved_params, f, indent=2)
                print(f"Saved parameters to {params_file}")
            except Exception as e:
                print(f"Error saving parameters: {e}")

        # Use the final values
        cda, crr, rmse, r2 = final_cda, final_crr, final_rmse, final_r2

        # Calculate virtual elevation profile for saving
        ve_changes = delta_ve(config, cda=cda, crr=crr, df=trimmed_df)
        virtual_profile = calculate_virtual_profile(
            ve_changes, trimmed_df["elevation"].values, "lap_number", trimmed_df
        )

        # Save a plot with the updated parameters
        if combined_save_dir and action == "save":
            from core.visualization import plot_elevation_profiles

            lap_str = "-".join(map(str, selected_laps))
            updated_save_path = os.path.join(
                combined_save_dir, f"laps_{lap_str}_combined_{action}.png"
            )
            updated_fig = plot_elevation_profiles(
                df=trimmed_df,
                actual_elevation=trimmed_df["elevation"].values,
                virtual_elevation=virtual_profile,
                distance=trimmed_distance,
                cda=cda,
                crr=crr,
                rmse=rmse,
                r2=r2,
                save_path=updated_save_path,
                lap_column="lap_number",
            )
            plt.close(updated_fig)
            print(f"Saved final plot to {updated_save_path}")

        # Calculate statistics for the result
        combined_stats = {
            "laps": selected_laps,
            "duration_seconds": total_duration,
            "distance_meters": np.sum(trimmed_df["v"] * dt),
            "avg_power": trimmed_df["watts"].mean(),
            "max_power": trimmed_df["watts"].max(),
            "avg_speed": trimmed_df["v"].mean(),
            "max_speed": trimmed_df["v"].max(),
            "elevation_gain": np.sum(
                np.maximum(0, np.diff(trimmed_df["elevation"].values))
            ),
        }

        # Store results
        result = {
            "cda": cda,
            "crr": crr,
            "rmse": rmse,
            "r2": r2,
            "fig": None,  # No figure to return from interactive mode
            "data": trimmed_df,
            "trim_start": user_trim_start,
            "trim_end": user_trim_end,
            **combined_stats,
        }

        return result

    # REGULAR WORKFLOW (non-interactive)
    else:
        # Apply distance trimming if requested
        if trim_distance > 0 or trim_start is not None or trim_end is not None:
            # Calculate distance
            distance = calculate_distance(combined_df, dt)
            # Apply trimming with prioritized individual trim values
            combined_df = trim_data_by_distance(
                combined_df, distance, trim_distance, trim_start, trim_end
            )
            if len(combined_df) < 10:
                print(f"Not enough data points after trimming ({len(combined_df)})")
                return None

        # Run analysis using the existing analyze_and_plot_ve function
        try:
            from ve_analyzer import analyze_and_plot_ve

            # In combined laps, target_elevation_gain is interpreted as the total elevation gain across all selected laps
            if target_elevation_gain is not None:
                print(
                    f"Interpreting {target_elevation_gain:.1f}m as the target TOTAL elevation gain across all combined laps"
                )

            # Perform the analysis with is_combined_laps=True when target_elevation_gain is specified
            if target_elevation_gain is not None and config.fixed_cda is not None:
                # Optimize Crr only with fixed CdA
                cda, crr, rmse, r2, fig = analyze_and_plot_ve(
                    df=combined_df,
                    config=config,
                    actual_elevation_col="elevation",
                    save_path=save_path,
                    lap_column="lap_number",
                    r2_weight=r2_weight,
                    n_grid=n_grid,
                    target_elevation_gain=target_elevation_gain,
                    is_combined_laps=True,  # Indicate this is for combined laps
                    interactive_plot=False,
                )
            elif target_elevation_gain is not None and config.fixed_crr is not None:
                # Optimize CdA only with fixed Crr
                cda, crr, rmse, r2, fig = analyze_and_plot_ve(
                    df=combined_df,
                    config=config,
                    actual_elevation_col="elevation",
                    save_path=save_path,
                    lap_column="lap_number",
                    r2_weight=r2_weight,
                    n_grid=n_grid,
                    target_elevation_gain=target_elevation_gain,
                    is_combined_laps=True,  # Indicate this is for combined laps
                    interactive_plot=False,
                )
            elif target_elevation_gain is not None:
                # Optimize both parameters
                cda, crr, rmse, r2, fig = analyze_and_plot_ve(
                    df=combined_df,
                    config=config,
                    actual_elevation_col="elevation",
                    save_path=save_path,
                    lap_column="lap_number",
                    r2_weight=r2_weight,
                    n_grid=n_grid,
                    target_elevation_gain=target_elevation_gain,
                    is_combined_laps=True,  # Indicate this is for combined laps
                    interactive_plot=False,
                )
            else:
                # Standard r2/RMSE optimization
                cda, crr, rmse, r2, fig = analyze_and_plot_ve(
                    df=combined_df,
                    config=config,
                    actual_elevation_col="elevation",
                    save_path=save_path,
                    lap_column="lap_number",
                    r2_weight=r2_weight,
                    n_grid=n_grid,
                    target_elevation_gain=None,
                    interactive_plot=False,
                )

            # Save parameters for non-interactive mode too
            if combined_save_dir:
                # Store parameters for combined laps
                lap_params = {
                    "trim_start": (
                        trim_start if trim_start is not None else trim_distance
                    ),
                    "trim_end": trim_end if trim_end is not None else trim_distance,
                    "cda": cda,
                    "crr": crr,
                    "rmse": rmse,
                    "r2": r2,
                }

                # Update saved parameters
                saved_params[combined_key] = lap_params

                # Save to file
                try:
                    with open(params_file, "w") as f:
                        json.dump(saved_params, f, indent=2)
                    print(f"Saved parameters to {params_file}")
                except Exception as e:
                    print(f"Error saving parameters: {e}")

            # Calculate statistics
            combined_stats = {
                "laps": selected_laps,
                "duration_seconds": total_duration,
                "distance_meters": np.sum(combined_df["v"] * dt),
                "avg_power": combined_df["watts"].mean(),
                "max_power": combined_df["watts"].max(),
                "avg_speed": combined_df["v"].mean(),
                "max_speed": combined_df["v"].max(),
                "elevation_gain": np.sum(
                    np.maximum(0, np.diff(combined_df["elevation"].values))
                ),
            }

            # Store results
            result = {
                "cda": cda,
                "crr": crr,
                "rmse": rmse,
                "r2": r2,
                "fig": fig,
                "data": combined_df,
                "trim_start": trim_start,
                "trim_end": trim_end,
                **combined_stats,
            }

            # Generate maps if requested
            if show_map:
                from core.visualization import create_interactive_map, plot_static_map

                # Get the original lap data with GPS coordinates for each selected lap
                original_combined_df = pd.DataFrame()

                for lap in selected_lap_info:
                    lap_df = df[
                        (df.index >= lap["start_time"]) & (df.index <= lap["end_time"])
                    ].copy()
                    original_combined_df = pd.concat([original_combined_df, lap_df])

                if (
                    "latitude" in original_combined_df.columns
                    and "longitude" in original_combined_df.columns
                ):
                    # Create static map
                    lap_str = "-".join(map(str, selected_laps))
                    static_map_path = os.path.join(
                        combined_save_dir, f"laps_{lap_str}_combined_map.png"
                    )
                    try:
                        plot_static_map(original_combined_df, save_path=static_map_path)
                        print(f"Generated static map: {static_map_path}")
                    except Exception as e:
                        print(f"Error creating static map: {str(e)}")

                    # Create interactive map
                    interactive_map_path = os.path.join(
                        combined_save_dir, f"laps_{lap_str}_combined_map.html"
                    )
                    try:
                        create_interactive_map(
                            original_combined_df, save_path=interactive_map_path
                        )
                        print(f"Generated interactive map: {interactive_map_path}")
                    except Exception as e:
                        print(f"Error creating interactive map: {str(e)}")
                else:
                    print("Cannot generate maps: GPS data not available")

            return result

        except Exception as e:
            print(f"Error analyzing combined laps: {str(e)}")
            import traceback

            traceback_info = traceback.format_exc()
            print(f"Traceback: {traceback_info}")
            return None


def summarize_lap_results(results, save_path=None):
    """
    Create a summary of lap analysis results.
    """
    # Extract lap data into a DataFrame
    summary_data = []

    for lap_id, lap_result in results.items():
        if "error" in lap_result:
            continue

        lap_summary = {
            "lap_number": lap_result["lap_number"],
            "cda": lap_result["cda"],
            "crr": lap_result["crr"],
            "rmse": lap_result["rmse"],
            "r2": lap_result["r2"],
            "start_time": lap_result["start_time"],
            "end_time": lap_result["end_time"],
            "duration_seconds": lap_result["duration_seconds"],
            "distance_meters": lap_result["distance_meters"],
            "avg_power": lap_result["avg_power"],
            "avg_speed": lap_result["avg_speed"],
            "elevation_gain": lap_result["elevation_gain"],
        }

        summary_data.append(lap_summary)

    # Create summary DataFrame
    summary_df = pd.DataFrame(summary_data)

    if not summary_df.empty:
        # Sort by lap number
        summary_df = summary_df.sort_values("lap_number")

        # Create a comparison plot
        fig, axs = plt.subplots(2, 2, figsize=(12, 10))

        # Plot CdA by lap
        axs[0, 0].bar(summary_df["lap_number"], summary_df["cda"])
        axs[0, 0].set_title("CdA by Lap")
        axs[0, 0].set_xlabel("Lap Number")
        axs[0, 0].set_ylabel("CdA (m²)")
        axs[0, 0].grid(True, linestyle="--", alpha=0.7)

        # Plot Crr by lap
        axs[0, 1].bar(summary_df["lap_number"], summary_df["crr"])
        axs[0, 1].set_title("Crr by Lap")
        axs[0, 1].set_xlabel("Lap Number")
        axs[0, 1].set_ylabel("Crr")
        axs[0, 1].grid(True, linestyle="--", alpha=0.7)

        # Plot RMSE by lap
        axs[1, 0].bar(summary_df["lap_number"], summary_df["rmse"])
        axs[1, 0].set_title("Elevation RMSE by Lap")
        axs[1, 0].set_xlabel("Lap Number")
        axs[1, 0].set_ylabel("RMSE (m)")
        axs[1, 0].grid(True, linestyle="--", alpha=0.7)

        # Plot R² by lap
        axs[1, 1].bar(summary_df["lap_number"], summary_df["r2"])
        axs[1, 1].set_title("Elevation R² by Lap")
        axs[1, 1].set_xlabel("Lap Number")
        axs[1, 1].set_ylabel("R²")
        axs[1, 1].grid(True, linestyle="--", alpha=0.7)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return summary_df


def show_interactive_plot(
    df,
    actual_elevation,
    optimized_cda,
    optimized_crr,
    distance=None,
    config: VirtualElevationConfig = None,
    lap_column=None,
    rmse=None,
    r2=None,
    save_path=None,
    lap_num=None,  # Added parameter for lap number
):
    """
    Show an interactive plot with sliders for CdA and Crr parameters.

    Args:
        df (pandas.DataFrame): DataFrame with cycling data
        actual_elevation (array-like): Actual measured elevation data
        optimized_cda (float): Optimized CdA value
        optimized_crr (float): Optimized Crr value
        distance (array-like, optional): Distance data in meters
        kg (float): Rider mass in kg
        rho (float): Air density in kg/m³
        dt (float): Time interval in seconds
        eta (float): Drivetrain efficiency
        vw (float): Wind velocity in m/s (positive = headwind)
        lap_column (str, optional): Column name containing lap numbers
        rmse (float, optional): RMSE value from optimization
        r2 (float, optional): R² value from optimization
        save_path (str, optional): Path to save a screenshot of the plot
        cda_range (tuple, optional): (min, max) range for CdA slider
        crr_range (tuple, optional): (min, max) range for Crr slider
        lap_num (int, optional): Current lap number being processed

    Returns:
        tuple: (final_cda, final_crr) or None if no parameters were saved
    """
    from core.visualization import create_interactive_elevation_plot

    if config.kg is None or config.rho is None:
        raise ValueError(
            "Rider mass (kg) and air density (rho) are required parameters"
        )

    # Set default ranges if not provided
    if cda_range is None:
        # Set range to ±50% of optimized value, but within reasonable bounds
        cda_min = max(0.05, optimized_cda * 0.5)
        cda_max = min(0.8, optimized_cda * 1.5)
        cda_range = (cda_min, cda_max)

    if crr_range is None:
        # Set range to ±50% of optimized value, but within reasonable bounds
        crr_min = max(0.0005, optimized_crr * 0.5)
        crr_max = min(0.02, optimized_crr * 1.5)
        crr_range = (crr_min, crr_max)

    print(f"\nShowing interactive plot with initial parameters:")
    print(
        f"  CdA: {optimized_cda:.4f} m² (range: {cda_range[0]:.4f} - {cda_range[1]:.4f})"
    )
    print(
        f"  Crr: {optimized_crr:.5f} (range: {crr_range[0]:.5f} - {crr_range[1]:.5f})"
    )
    print("Use sliders to adjust parameters and observe changes in real-time")
    print("Click 'Save Results' when you're satisfied with the parameters")

    # Create and display the interactive plot with save button
    fig, cda_slider, crr_slider, reset_button, save_button, saved_params = (
        create_interactive_elevation_plot(
            df=df,
            actual_elevation=actual_elevation,
            initial_cda=optimized_cda,
            initial_crr=optimized_crr,
            distance=distance,
            config=config,
            lap_column=lap_column,
            cda_range=cda_range,
            crr_range=crr_range,
            initial_rmse=rmse,
            initial_r2=r2,
            save_path=save_path,
            lap_num=lap_num,
        )
    )

    # Show the plot
    plt.show()

    # Return the saved parameters if any
    if saved_params[0] is not None:
        return saved_params[0]
    else:
        return None


def main():
    """
    Command-line tool for analyzing .fit files by laps.
    """
    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(
        description="Virtual Elevation Analyzer: Estimate CdA and Crr from cycling data."
    )

    # Required arguments
    parser.add_argument("fit_file", help="Path to the .fit file to analyze")
    parser.add_argument(
        "--mass", type=float, required=True, help="Rider + bike mass in kg (required)"
    )
    parser.add_argument(
        "--rho", type=float, required=True, help="Air density in kg/m³ (required)"
    )

    # Optional arguments
    parser.add_argument(
        "--resample", default="1s", help="Resampling frequency (default: 1 second)"
    )
    parser.add_argument(
        "--output", default="fit_analysis_results", help="Output directory for results"
    )
    parser.add_argument(
        "--min-lap",
        type=int,
        default=30,
        help="Minimum lap duration in seconds (default: 30)",
    )
    parser.add_argument(
        "--selected-laps",
        type=str,
        help='Comma-separated list of lap numbers to analyze together (e.g., "2,4,6,8,10")',
    )
    parser.add_argument(
        "--cda",
        type=float,
        help="Fixed CdA value to use (if provided, only Crr will be optimized)",
    )
    parser.add_argument(
        "--crr",
        type=float,
        help="Fixed Crr value to use (if provided, only CdA will be optimized)",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug output")
    parser.add_argument(
        "--show-map", action="store_true", help="Show lap routes on a map"
    )

    # Trimming parameters
    parser.add_argument(
        "--trim-distance",
        type=float,
        default=0,
        help="Distance in meters to trim from start and end of recording (default: 0)",
    )
    parser.add_argument(
        "--trim-start",
        type=float,
        default=None,
        help="Distance in meters to trim from start of recording (overrides --trim-distance for start)",
    )
    parser.add_argument(
        "--trim-end",
        type=float,
        default=None,
        help="Distance in meters to trim from end of recording (overrides --trim-distance for end)",
    )

    # Other parameters
    parser.add_argument(
        "--r2-weight",
        type=float,
        default=0.5,
        help="Weight for R² in the composite objective (0-1, default: 0.5)",
    )
    parser.add_argument(
        "--grid-points",
        type=int,
        default=250,
        help="Number of grid points to use in parameter search (default: 250)",
    )
    parser.add_argument(
        "--cda-bounds",
        type=str,
        default="0.1,0.5",
        help="Bounds for CdA optimization as min,max (default: '0.1,0.5')",
    )
    parser.add_argument(
        "--crr-bounds",
        type=str,
        default="0.001,0.01",
        help="Bounds for Crr optimization as min,max (default: '0.001,0.01')",
    )
    parser.add_argument(
        "--optimize-elevation-gain",
        type=float,
        default=None,
        help="Optimize for a specific elevation gain (in meters). For individual lap analysis, this is the target gain per lap. For combined lap analysis (--selected-laps), this is the target total gain across all selected laps.",
    )
    parser.add_argument(
        "--eta",
        type=float,
        default=0.98,
        help="Drivetrain efficiency (default: 0.98)",
    )

    # Interactive mode parameter
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Enable interactive mode for data trimming and parameter fine-tuning",
    )

    args = parser.parse_args()

    # Create configuration
    config = VirtualElevationConfig(
        rider_mass=args.mass,
        air_density=args.rho,
        drivetrain_efficiency=args.eta,
        fixed_cda=args.cda,
        fixed_crr=args.crr,
        resample_freq=args.resample,
    )

    # After parsing args, add some helpful messaging
    if args.optimize_elevation_gain is not None:
        if args.selected_laps:
            if args.optimize_elevation_gain == 0:
                print(
                    f"Will optimize for zero total elevation gain across all selected laps: {args.selected_laps}"
                )
            else:
                print(
                    f"Will optimize for {args.optimize_elevation_gain:.1f}m total elevation gain across all selected laps: {args.selected_laps}"
                )
        else:
            if args.optimize_elevation_gain == 0:
                print(f"Will optimize for zero elevation gain per lap")
            else:
                print(
                    f"Will optimize for {args.optimize_elevation_gain:.1f}m elevation gain per lap"
                )

    # Parse optimization bounds
    try:
        cda_min, cda_max = map(float, args.cda_bounds.split(","))
        cda_bounds = (cda_min, cda_max)
        config.cda_bounds = cda_bounds
    except ValueError:
        print(
            f"Error: Invalid CdA bounds format: {args.cda_bounds}. Use format 'min,max'"
        )
        return None

    try:
        crr_min, crr_max = map(float, args.crr_bounds.split(","))
        crr_bounds = (crr_min, crr_max)
        config.crr_bounds = crr_bounds
    except ValueError:
        print(
            f"Error: Invalid Crr bounds format: {args.crr_bounds}. Use format 'min,max'"
        )
        return None

    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    # Print analysis parameters
    print("\nAnalysis Parameters:")
    print(f"Rider mass: {args.mass} kg")
    print(f"Air density: {args.rho} kg/m³")
    print(f"Drivetrain efficiency: {args.eta}")
    print(f"Resampling: {args.resample}")
    print(f"Output directory: {args.output}")
    print(f"Minimum lap duration: {args.min_lap} seconds")

    # Print trimming parameters
    if args.trim_start is not None:
        print(f"Trim start: {args.trim_start} meters")
    if args.trim_end is not None:
        print(f"Trim end: {args.trim_end} meters")
    if args.trim_distance > 0 and args.trim_start is None and args.trim_end is None:
        print(f"Trim distance: {args.trim_distance} meters from start and end")

    print(f"R² weight: {args.r2_weight}")
    print(f"Grid points: {args.grid_points}")
    print(f"CdA bounds: {cda_bounds}")
    print(f"Crr bounds: {crr_bounds}")

    if args.cda is not None:
        print(f"Fixed CdA: {args.cda:.4f} m² (only optimizing Crr)")
    if args.crr is not None:
        print(f"Fixed Crr: {args.crr:.5f} (only optimizing CdA)")
    if args.cda is None and args.crr is None:
        print("Optimizing both CdA and Crr")

    if args.show_map:
        print("Will generate route maps for analyzed laps")

    if args.interactive:
        print("Interactive mode enabled - will show combined map and optimization plot")

    # Parse the FIT file
    df, lap_messages = parse_fit_file(args.fit_file, args.debug)

    if df is None or lap_messages is None:
        return None

    # Check if we should analyze specific laps together
    if args.selected_laps:
        try:
            selected_laps = [int(lap.strip()) for lap in args.selected_laps.split(",")]
            print(f"Analyzing selected laps together: {selected_laps}")

            # Analyze the combined laps
            combined_result = analyze_combined_laps(
                df=df,
                lap_messages=lap_messages,
                selected_laps=selected_laps,
                fit_file_path=args.fit_file,  # Added parameter
                config=config,
                save_dir=args.output,
                min_lap_duration=args.min_lap,
                debug=args.debug,
                trim_distance=args.trim_distance,
                trim_start=args.trim_start,
                trim_end=args.trim_end,
                r2_weight=args.r2_weight,
                n_grid=args.grid_points,
                target_elevation_gain=args.optimize_elevation_gain,
                show_map=args.show_map,
                interactive_plot=args.interactive,
            )

            if combined_result:
                # Get the result directory based on filename
                file_basename = os.path.basename(args.fit_file)
                file_name, _ = os.path.splitext(file_basename)
                result_dir = os.path.join(args.output, file_name)
                os.makedirs(result_dir, exist_ok=True)

                # Save combined lap data to CSV
                combined_csv_path = os.path.join(result_dir, "combined_laps_data.csv")
                combined_result["data"].to_csv(combined_csv_path, index=False)

                print("\nCombined analysis complete!")
                print(f"Combined data saved to: {combined_csv_path}")
                print(f"CdA: {combined_result['cda']:.4f} m²")
                print(f"Crr: {combined_result['crr']:.5f}")

                # Display the plot if it exists
                if combined_result["fig"] is not None:
                    plt.figure(combined_result["fig"].number)
                    plt.show()

                return combined_result
            else:
                print("\nCombined lap analysis failed.")
        except ValueError as e:
            print(f"Error parsing selected laps: {e}")
            print(
                "Please specify laps as comma-separated integers (e.g., '2,4,6,8,10')."
            )

    # Process by individual laps
    results = analyze_lap_data(
        df=df,
        lap_messages=lap_messages,
        config=config,
        fit_file_path=args.fit_file,  # Added parameter
        save_dir=args.output,
        min_lap_duration=args.min_lap,
        debug=args.debug,
        trim_distance=args.trim_distance,
        trim_start=args.trim_start,
        trim_end=args.trim_end,
        r2_weight=args.r2_weight,
        n_grid=args.grid_points,
        target_elevation_gain=args.optimize_elevation_gain,
        show_map=args.show_map,
        interactive_plot=args.interactive,
    )

    if results:
        # Get the result directory based on filename
        file_basename = os.path.basename(args.fit_file)
        file_name, _ = os.path.splitext(file_basename)
        result_dir = os.path.join(args.output, file_name)
        os.makedirs(result_dir, exist_ok=True)

        # Create summary
        print("\nCreating lap summary...")
        summary_df = summarize_lap_results(
            results=results, save_path=os.path.join(result_dir, "lap_comparison.png")
        )

        # Save summary to CSV
        summary_csv_path = os.path.join(result_dir, "lap_summary.csv")
        if not summary_df.empty:
            summary_df.to_csv(summary_csv_path, index=False)

            print(f"\nAnalysis complete!")
            print(f"Individual lap results saved to: {result_dir}")
            print(f"Summary saved to: {summary_csv_path}")

            # Display summary
            print("\nLap Summary:")
            display_cols = [
                "lap_number",
                "cda",
                "crr",
                "rmse",
                "r2",
                "avg_power",
                "avg_speed",
                "distance_meters",
            ]
            print(summary_df[display_cols].to_string(index=False))
        else:
            print("\nNo valid lap results to summarize.")

        # Save additional metrics for each lap
        for lap_id, lap_results in results.items():
            if "error" not in lap_results:
                lap_num = lap_results["lap_number"]
                lap_data = lap_results["data"]

                # Save lap data to CSV
                lap_csv_path = os.path.join(result_dir, f"lap_{lap_num}_data.csv")
                lap_data.to_csv(lap_csv_path, index=False)

        print("\nPlots are saved in the output directory.")
        print(
            "To view the plots, you can run: 'plt.show()' if running in interactive mode."
        )

        # Show plots
        plt.show()

        return results, summary_df
    else:
        print("\nAnalysis failed. Check the input .fit file.")
        return None, None


if __name__ == "__main__":
    results = main()
