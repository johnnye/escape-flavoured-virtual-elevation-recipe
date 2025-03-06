import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import pearsonr

from core.calculations import calculate_distance

plt.style.use("fivethirtyeight")


def plot_elevation_profiles(
    df,
    actual_elevation,
    virtual_elevation,
    distance=None,
    cda=None,
    crr=None,
    rmse=None,
    r2=None,
    save_path=None,
    lap_column=None,
):
    """
    Plot actual and virtual elevation profiles against distance.

    Args:
        df (pandas.DataFrame): DataFrame with cycling data
        actual_elevation (array-like): Actual measured elevation data
        virtual_elevation (array-like): Calculated virtual elevation data
        distance (array-like, optional): Distance data in meters
        cda (float, optional): CdA value to display in plot title
        crr (float, optional): Crr value to display in plot title
        rmse (float, optional): RMSE value to display in plot title
        r2 (float, optional): R² value to display in plot title
        save_path (str, optional): Path to save the plot image
        lap_column (str, optional): Column name containing lap numbers for multi-lap visualization

    Returns:
        matplotlib.figure.Figure: The created figure
    """
    # If distance is not provided, calculate it from velocity
    if distance is None and "v" in df.columns:
        distance = calculate_distance(df)
    elif distance is None:
        distance = np.arange(len(actual_elevation))

    # Convert distance to kilometers for better readability
    distance_km = distance / 1000

    # Calculate R² if not provided
    if r2 is None:
        r2 = pearsonr(actual_elevation, virtual_elevation)[0] ** 2

    # Create a figure with two subplots (profiles and residuals)
    fig = plt.figure(figsize=(12, 8))
    gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])

    # Plot elevation profiles
    ax1 = plt.subplot(gs[0])
    ax1.plot(distance_km, actual_elevation, "b-", linewidth=2, label="Actual Elevation")
    ax1.plot(
        distance_km, virtual_elevation, "r-", linewidth=2, label="Virtual Elevation"
    )

    # Add vertical lines for lap boundaries if lap column is provided
    if lap_column is not None and lap_column in df.columns:
        lap_numbers = df[lap_column].values
        unique_laps = sorted(np.unique(lap_numbers))

        # Add vertical lines at lap transitions
        for i in range(1, len(unique_laps)):
            # Find the first index of the new lap
            transition_idx = np.where(lap_numbers == unique_laps[i])[0][0]

            # Check if the laps are consecutive
            if unique_laps[i] != unique_laps[i - 1] + 1:
                # Non-consecutive laps: add a green dashed line
                ax1.axvline(
                    x=distance_km[transition_idx],
                    color="green",
                    linestyle="--",
                    alpha=0.7,
                )
                line_label = f"Lap {unique_laps[i]} (Reset)"
            else:
                # Consecutive laps: add a blue dotted line
                ax1.axvline(
                    x=distance_km[transition_idx],
                    color="blue",
                    linestyle=":",
                    alpha=0.5,
                )
                line_label = f"Lap {unique_laps[i]}"

            # Add lap number labels
            if transition_idx > 0:
                ax1.text(
                    distance_km[transition_idx] + 0.1,
                    np.min(actual_elevation)
                    + 0.1 * (np.max(actual_elevation) - np.min(actual_elevation)),
                    line_label,
                    rotation=90,
                    va="bottom",
                )

    ax1.set_ylabel("Elevation (m)", fontsize=12)
    ax1.set_title("Elevation Profiles Comparison", fontsize=14)
    ax1.grid(True, linestyle="--", alpha=0.7)
    ax1.legend(loc="best")

    # Add parameter values in text box if provided
    if cda is not None or crr is not None or rmse is not None:
        textstr = ""
        if cda is not None:
            textstr += f"CdA: {cda:.4f} m²\n"
        if crr is not None:
            textstr += f"Crr: {crr:.5f}\n"
        if rmse is not None:
            textstr += f"RMSE: {rmse:.2f} m\n"
        if r2 is not None:
            textstr += f"R²: {r2:.4f}"

        # Position text box in upper right
        props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)
        ax1.text(
            0.95,
            0.95,
            textstr,
            transform=ax1.transAxes,
            fontsize=10,
            verticalalignment="top",
            horizontalalignment="right",
            bbox=props,
        )

    # Plot residuals (differences between actual and virtual)
    ax2 = plt.subplot(gs[1], sharex=ax1)
    residuals = actual_elevation - virtual_elevation
    ax2.plot(distance_km, residuals, "g-", linewidth=1.5)
    ax2.axhline(y=0, color="k", linestyle="-", alpha=0.3)

    # Add vertical lines for lap boundaries on residual plot too
    if lap_column is not None and lap_column in df.columns:
        lap_numbers = df[lap_column].values
        unique_laps = sorted(np.unique(lap_numbers))

        for i in range(1, len(unique_laps)):
            transition_idx = np.where(lap_numbers == unique_laps[i])[0][0]

            # Check if the laps are consecutive
            if unique_laps[i] != unique_laps[i - 1] + 1:
                # Non-consecutive laps: add a green dashed line
                ax2.axvline(
                    x=distance_km[transition_idx],
                    color="green",
                    linestyle="--",
                    alpha=0.7,
                )
            else:
                # Consecutive laps: add a blue dotted line
                ax2.axvline(
                    x=distance_km[transition_idx],
                    color="blue",
                    linestyle=":",
                    alpha=0.5,
                )

    ax2.set_xlabel("Distance (km)", fontsize=12)
    ax2.set_ylabel("Residual (m)", fontsize=12)
    ax2.set_title("Elevation Difference (Actual - Virtual)", fontsize=12)
    ax2.grid(True, linestyle="--", alpha=0.7)

    plt.tight_layout()

    # Save the figure if a path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def plot_summary(summary_df, save_path=None):
    """
    Create a summary plot of analysis results across laps.

    Args:
        summary_df (pandas.DataFrame): DataFrame containing summary data
        save_path (str, optional): Path to save the plot

    Returns:
        matplotlib.figure.Figure: The created figure
    """
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

    return fig
