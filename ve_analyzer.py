#!/usr/bin/env python3
# ve_analyzer.py - Virtual Elevation Analyzer

import os
import sys
import argparse
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import traceback

# Import from our modules
from core.calculations import accel_calc, delta_ve, calculate_distance
from core.optimization import (
    optimize_both_params_balanced,
    optimize_cda_only_balanced,
    optimize_crr_only_balanced,
    calculate_virtual_profile,
)
from core.visualization import plot_elevation_profiles
from data_io.fit_parser import parse_fit_file
from data_io.data_processing import trim_data_by_distance, resample_data


def analyze_and_plot_ve(
    df,
    actual_elevation_col="elevation",
    kg=None,
    rho=None,
    dt=1,
    eta=1,
    vw=0,
    distance_col=None,
    save_path=None,
    lap_column=None,
    fixed_cda=None,
    fixed_crr=None,
    r2_weight=0.5,
    n_grid=250,
    cda_bounds=(0.1, 0.5),
    crr_bounds=(0.001, 0.01),
):
    """
    Complete workflow function to analyze data, optimize parameters, and plot results.
    """
    if kg is None or rho is None:
        raise ValueError(
            "Rider mass (kg) and air density (rho) are required parameters"
        )

    from scipy.stats import pearsonr

    # Ensure we have acceleration data
    if "a" not in df.columns:
        df["a"] = accel_calc(df["v"].values, dt)

    # Get actual elevation data
    actual_elevation = df[actual_elevation_col].values

    # Get distance data if available
    distance = None
    if distance_col and distance_col in df.columns:
        distance = df[distance_col].values
    else:
        distance = calculate_distance(df, dt)

    # Determine which parameters to optimize
    if fixed_cda is not None and fixed_crr is not None:
        # Both parameters fixed - no optimization needed
        print(f"Using fixed parameters: CdA={fixed_cda:.4f} m², Crr={fixed_crr:.5f}")
        # Calculate virtual elevation with fixed parameters
        ve_changes = delta_ve(
            cda=fixed_cda, crr=fixed_crr, df=df, vw=vw, kg=kg, rho=rho, dt=dt, eta=eta
        )

        # Build virtual elevation profile
        virtual_elevation = calculate_virtual_profile(
            ve_changes, actual_elevation, lap_column, df
        )

        # Calculate RMSE
        rmse = np.sqrt(np.mean((virtual_elevation - actual_elevation) ** 2))
        optimized_cda = fixed_cda
        optimized_crr = fixed_crr

    elif fixed_cda is not None:
        # Only optimize Crr with fixed CdA
        print(f"Using fixed CdA={fixed_cda:.4f} m² and optimizing Crr")
        optimized_cda, optimized_crr, rmse, r2, virtual_elevation = (
            optimize_crr_only_balanced(
                df=df,
                actual_elevation=actual_elevation,
                fixed_cda=fixed_cda,
                kg=kg,
                rho=rho,
                dt=dt,
                eta=eta,
                vw=vw,
                lap_column=lap_column,
                n_points=n_grid,
                r2_weight=r2_weight,
                crr_bounds=crr_bounds,
            )
        )

    elif fixed_crr is not None:
        # Only optimize CdA with fixed Crr
        print(f"Using fixed Crr={fixed_crr:.5f} and optimizing CdA")
        optimized_cda, optimized_crr, rmse, r2, virtual_elevation = (
            optimize_cda_only_balanced(
                df=df,
                actual_elevation=actual_elevation,
                fixed_crr=fixed_crr,
                kg=kg,
                rho=rho,
                dt=dt,
                eta=eta,
                vw=vw,
                lap_column=lap_column,
                n_points=n_grid,
                r2_weight=r2_weight,
                cda_bounds=cda_bounds,
            )
        )

    else:
        # Optimize both parameters
        print("Optimizing both CdA and Crr")
        optimized_cda, optimized_crr, rmse, r2, virtual_elevation = (
            optimize_both_params_balanced(
                df=df,
                actual_elevation=actual_elevation,
                kg=kg,
                rho=rho,
                dt=dt,
                eta=eta,
                vw=vw,
                lap_column=lap_column,
                n_grid=n_grid,
                r2_weight=r2_weight,
                cda_bounds=cda_bounds,
                crr_bounds=crr_bounds,
            )
        )

    # Calculate R²
    r2 = pearsonr(actual_elevation, virtual_elevation)[0] ** 2

    # Plot results
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

    return optimized_cda, optimized_crr, rmse, r2, fig


def analyze_lap_data(
    df,
    lap_messages,
    rider_mass,
    air_density,
    resample_freq="1s",
    save_dir=None,
    min_lap_duration=30,
    debug=False,
    fixed_cda=None,
    fixed_crr=None,
    trim_distance=0,
    r2_weight=0.5,
    n_grid=250,
    cda_bounds=(0.1, 0.5),
    crr_bounds=(0.001, 0.01),
):
    """
    Process and analyze data by laps.
    """
    if rider_mass is None or air_density is None:
        raise ValueError("Rider mass and air density are required parameters")

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

        if len(lap_df) < 10:
            print(f"  Skipping lap {lap_num}: Not enough data points ({len(lap_df)})")
            continue

        # Resample to constant time interval
        try:
            resampled_df = resample_data(lap_df, resample_freq)
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

        # Calculate acceleration
        resampled_df["a"] = accel_calc(resampled_df["v"].values, dt)

        # Calculate distance for trimming
        if trim_distance > 0:
            # Calculate distance
            distance = calculate_distance(resampled_df, dt)
            # Apply trimming
            resampled_df = trim_data_by_distance(resampled_df, distance, trim_distance)
            if len(resampled_df) < 10:
                print(
                    f"  Skipping lap {lap_num}: Not enough data points after trimming ({len(resampled_df)})"
                )
                continue
            print(f"  Trimmed {trim_distance}m from start and end of lap")

        # Run analysis
        save_path = None
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f"lap_{lap_num}_elevation.png")

        try:
            # Perform the analysis
            cda, crr, rmse, r2, fig = analyze_and_plot_ve(
                df=resampled_df,
                actual_elevation_col="elevation",
                kg=rider_mass,
                rho=air_density,
                dt=dt,
                save_path=save_path,
                fixed_cda=fixed_cda,
                fixed_crr=fixed_crr,
                r2_weight=r2_weight,
                n_grid=n_grid,
                cda_bounds=cda_bounds,
                crr_bounds=crr_bounds,
            )

            # Calculate lap statistics
            lap_stats = {
                "duration_seconds": duration,
                "distance_meters": resampled_df["v"].sum() * dt,  # Approximate distance
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
            results[f"lap_{lap_num}"] = {
                "cda": cda,
                "crr": crr,
                "rmse": rmse,
                "r2": r2,
                "fig": fig,
                "data": resampled_df,
                **lap_stats,
            }

            print(
                f"  Results: CdA={cda:.4f}m², Crr={crr:.5f}, RMSE={rmse:.2f}m, R²={r2:.4f}"
            )
            print(
                f"  Avg Power: {lap_stats['avg_power']:.1f}W, Avg Speed: {lap_stats['avg_speed']*3.6:.1f}km/h"
            )

        except Exception as e:
            print(f"  Error analyzing lap {lap_num}: {str(e)}")
            traceback_info = traceback.format_exc()
            print(f"  Traceback: {traceback_info}")
            results[f"lap_{lap_num}"] = {
                "error": str(e),
                "data": resampled_df,
                "lap_number": lap_num,
            }

    return results


def analyze_combined_laps(
    df,
    lap_messages,
    selected_laps,
    rider_mass,
    air_density,
    resample_freq="1s",
    save_dir=None,
    min_lap_duration=30,
    debug=False,
    fixed_cda=None,
    fixed_crr=None,
    trim_distance=0,
    r2_weight=0.5,
    n_grid=250,
    cda_bounds=(0.1, 0.5),
    crr_bounds=(0.001, 0.01),
):
    """
    Process and analyze a specific set of laps combined as one segment.
    """
    if rider_mass is None or air_density is None:
        raise ValueError("Rider mass and air density are required parameters")

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
        lap_df = df[
            (df.index >= lap["start_time"]) & (df.index <= lap["end_time"])
        ].copy()[["v", "watts", "elevation"]]

        if len(lap_df) < 10:
            print(
                f"  Skipping lap {lap['lap_number']}: Not enough data points ({len(lap_df)})"
            )
            continue

        # Resample each lap individually
        try:
            resampled_lap_df = resample_data(lap_df, resample_freq)

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

    # Calculate acceleration (only within each lap, not across lap boundaries)
    a_values = np.zeros(len(combined_df))

    for lap_num in combined_df["lap_number"].unique():
        lap_mask = combined_df["lap_number"] == lap_num
        lap_indices = np.where(lap_mask)[0]

        if len(lap_indices) > 1:
            lap_v = combined_df.loc[lap_mask, "v"].values
            lap_a = accel_calc(lap_v, dt)
            a_values[lap_indices] = lap_a

    combined_df["a"] = a_values

    # Apply distance trimming if requested
    if trim_distance > 0:
        # Calculate distance
        distance = calculate_distance(combined_df, dt)
        # Apply trimming
        combined_df = trim_data_by_distance(combined_df, distance, trim_distance)
        if len(combined_df) < 10:
            print(f"Not enough data points after trimming ({len(combined_df)})")
            return None
        print(f"Trimmed {trim_distance}m from start and end of recording")

    # Run analysis
    save_path = None
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        lap_str = "-".join(map(str, selected_laps))
        save_path = os.path.join(save_dir, f"laps_{lap_str}_combined.png")

    try:
        # Perform the analysis
        cda, crr, rmse, r2, fig = analyze_and_plot_ve(
            df=combined_df,
            actual_elevation_col="elevation",
            kg=rider_mass,
            rho=air_density,
            dt=dt,
            save_path=save_path,
            lap_column="lap_number",  # Use the lap_number column for proper lap handling
            fixed_cda=fixed_cda,
            fixed_crr=fixed_crr,
            r2_weight=r2_weight,
            n_grid=n_grid,
            cda_bounds=cda_bounds,
            crr_bounds=crr_bounds,
        )

        # Calculate statistics
        combined_stats = {
            "laps": selected_laps,
            "duration_seconds": total_duration,
            "distance_meters": np.sum(
                combined_df["v"] * dt
            ),  # More accurate distance calculation
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
            **combined_stats,
        }

        print(f"\nCombined analysis results:")
        print(f"  CdA: {cda:.4f}m², Crr: {crr:.5f}")
        print(f"  RMSE: {rmse:.2f}m, R²: {r2:.4f}")
        print(f"  Avg Power: {combined_stats['avg_power']:.1f}W")
        print(f"  Avg Speed: {combined_stats['avg_speed']*3.6:.1f}km/h")
        print(f"  Distance: {combined_stats['distance_meters']/1000:.2f}km")

        return result

    except Exception as e:
        print(f"Error analyzing combined laps: {str(e)}")
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

    # New parameters
    parser.add_argument(
        "--trim-distance",
        type=float,
        default=0,
        help="Distance in meters to trim from start and end of recording (default: 0)",
    )
    parser.add_argument(
        "--velodrome",
        action="store_true",
        help="Set to true if recording was done in a velodrome (forces elevation to zero)",
    )
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

    args = parser.parse_args()

    # Parse optimization bounds
    try:
        cda_min, cda_max = map(float, args.cda_bounds.split(","))
        cda_bounds = (cda_min, cda_max)
    except ValueError:
        print(
            f"Error: Invalid CdA bounds format: {args.cda_bounds}. Use format 'min,max'"
        )
        return None

    try:
        crr_min, crr_max = map(float, args.crr_bounds.split(","))
        crr_bounds = (crr_min, crr_max)
    except ValueError:
        print(
            f"Error: Invalid Crr bounds format: {args.crr_bounds}. Use format 'min,max'"
        )
        return None

    # Override R2 weight for velodrome mode if not explicitly provided
    if args.velodrome and "--r2-weight" not in sys.argv:
        args.r2_weight = 0.0  # Use only RMSE for velodrome
        print(
            "Velodrome mode: Setting R² weight to 0 (using only RMSE for optimization)"
        )

    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    # Print analysis parameters
    print("\nAnalysis Parameters:")
    print(f"Rider mass: {args.mass} kg")
    print(f"Air density: {args.rho} kg/m³")
    print(f"Resampling: {args.resample}")
    print(f"Output directory: {args.output}")
    print(f"Minimum lap duration: {args.min_lap} seconds")
    print(f"Trim distance: {args.trim_distance} meters from start and end")

    if args.velodrome:
        print("Velodrome mode: Forcing elevation to zero")

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

    # Parse the FIT file
    df, lap_messages = parse_fit_file(args.fit_file, args.debug)

    if df is None or lap_messages is None:
        return None

    # Apply velodrome mode if requested
    if args.velodrome:
        df["elevation"] = 1.0
        print("Applied velodrome mode: Set all elevation values to zero")

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
                rider_mass=args.mass,
                air_density=args.rho,
                resample_freq=args.resample,
                save_dir=args.output,
                min_lap_duration=args.min_lap,
                debug=args.debug,
                fixed_cda=args.cda,
                fixed_crr=args.crr,
                trim_distance=args.trim_distance,
                r2_weight=args.r2_weight,
                n_grid=args.grid_points,
                cda_bounds=cda_bounds,
                crr_bounds=crr_bounds,
            )

            if combined_result:
                # Save combined lap data to CSV
                combined_csv_path = os.path.join(args.output, "combined_laps_data.csv")
                combined_result["data"].to_csv(combined_csv_path, index=False)

                print("\nCombined analysis complete!")
                print(f"Combined data saved to: {combined_csv_path}")
                print(f"CdA: {combined_result['cda']:.4f} m²")
                print(f"Crr: {combined_result['crr']:.5f}")

                # Display the plot
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
        rider_mass=args.mass,
        air_density=args.rho,
        resample_freq=args.resample,
        save_dir=args.output,
        min_lap_duration=args.min_lap,
        debug=args.debug,
        fixed_cda=args.cda,
        fixed_crr=args.crr,
        trim_distance=args.trim_distance,
        r2_weight=args.r2_weight,
        n_grid=args.grid_points,
        cda_bounds=cda_bounds,
        crr_bounds=crr_bounds,
    )

    if results:
        # Create summary
        print("\nCreating lap summary...")
        summary_df = summarize_lap_results(
            results=results, save_path=os.path.join(args.output, "lap_comparison.png")
        )

        # Save summary to CSV
        summary_csv_path = os.path.join(args.output, "lap_summary.csv")
        if not summary_df.empty:
            summary_df.to_csv(summary_csv_path, index=False)

            print(f"\nAnalysis complete!")
            print(f"Individual lap results saved to: {args.output}")
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
                lap_csv_path = os.path.join(args.output, f"lap_{lap_num}_data.csv")
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
