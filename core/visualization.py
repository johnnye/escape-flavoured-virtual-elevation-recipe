import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
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


def plot_static_map(df, save_path=None):
    """
    Generate a static map with OpenStreetMap background showing the route.

    Args:
        df (pandas.DataFrame): DataFrame with cycling data including latitude/longitude
        save_path (str, optional): Path to save the map image

    Returns:
        matplotlib.figure.Figure: The created figure
    """
    try:
        import contextily as ctx
        import geopandas as gpd
        import matplotlib.pyplot as plt
        from shapely.geometry import LineString, Point
    except ImportError:
        print("To use mapping functionality, install the required packages:")
        print("pip install matplotlib contextily geopandas shapely")
        return None

    # Ensure we have latitude and longitude data
    if "latitude" not in df.columns or "longitude" not in df.columns:
        print("GPS data not available in the file")
        return None

    # Create a figure
    fig, ax = plt.subplots(figsize=(10, 8))

    # Create a list of Point geometries
    points = [Point(lon, lat) for lon, lat in zip(df["longitude"], df["latitude"])]

    # Create a line from the points (for the route)
    route_line = LineString(points)

    # Create GeoDataFrame for points and line
    # Use correct CRS for GPS data (WGS84)
    geometry = [route_line]
    gdf_line = gpd.GeoDataFrame(geometry=geometry, crs="EPSG:4326")

    # Transform to Web Mercator projection (what OSM uses)
    gdf_line = gdf_line.to_crs(epsg=3857)

    # Create GeoDataFrame for start and end points
    start_point = Point(df["longitude"].iloc[0], df["latitude"].iloc[0])
    end_point = Point(df["longitude"].iloc[-1], df["latitude"].iloc[-1])

    gdf_points = gpd.GeoDataFrame(
        geometry=[start_point, end_point],
        data={"type": ["start", "end"]},
        crs="EPSG:4326",
    )
    gdf_points = gdf_points.to_crs(epsg=3857)

    # Plot on the map
    gdf_line.plot(ax=ax, color="blue", linewidth=2)

    # Plot start and end points
    start_mask = gdf_points["type"] == "start"
    end_mask = gdf_points["type"] == "end"

    gdf_points[start_mask].plot(ax=ax, color="green", markersize=50, zorder=10)
    gdf_points[end_mask].plot(ax=ax, color="red", markersize=50, zorder=10)

    # Add OSM background with dynamic zoom level
    try:
        # Calculate the bounds of the data to determine appropriate zoom
        bounds = gdf_line.total_bounds  # [minx, miny, maxx, maxy]

        # Calculate the width and height in meters (in web mercator)
        width_meters = bounds[2] - bounds[0]
        height_meters = bounds[3] - bounds[1]

        # Maximum dimension in meters
        max_dimension = max(width_meters, height_meters)

        # Calculate appropriate zoom level
        # The formula is based on the relationship between zoom level and map resolution
        # Higher zoom = more detailed but covering less area
        # Zoom level 19 (max) shows approximately 0.15m per pixel
        # Zoom level 1 (near min) shows approximately 40000m per pixel

        # Determine zoom based on the track's extent
        if max_dimension < 500:  # Very short track (< 500m)
            zoom = 19  # Highest detail
        elif max_dimension < 1000:  # Short track (< 1km)
            zoom = 18
        elif max_dimension < 2000:  # Medium-short track (< 2km)
            zoom = 17
        elif max_dimension < 5000:  # Medium track (< 5km)
            zoom = 16
        elif max_dimension < 10000:  # Medium-long track (< 10km)
            zoom = 15
        elif max_dimension < 20000:  # Long track (< 20km)
            zoom = 14
        elif max_dimension < 50000:  # Very long track (< 50km)
            zoom = 13
        else:  # Extremely long track
            zoom = 12

        print(
            f"Track length: approximately {max_dimension/1000:.1f}km, using zoom level {zoom}"
        )

        ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik, zoom=zoom)
    except Exception as e:
        print(f"Error adding map background: {str(e)}")

    # Remove axis labels which aren't meaningful in this projection
    ax.set_axis_off()

    # Add a title
    ax.set_title("Route Map", fontsize=16, pad=20)

    # Save the figure if a path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def create_interactive_map(df, save_path=None):
    """
    Create an interactive, zoomable map showing the route.

    Args:
        df (pandas.DataFrame): DataFrame with cycling data including latitude/longitude
        save_path (str, optional): Path to save the map HTML file

    Returns:
        folium.Map: The created map object
    """
    try:
        import folium
    except ImportError:
        print("To use interactive mapping functionality, install folium:")
        print("pip install folium")
        return None

    # Ensure we have latitude and longitude data
    if "latitude" not in df.columns or "longitude" not in df.columns:
        print("GPS data not available in the file")
        return None

    # Calculate the map center
    center_lat = (df["latitude"].min() + df["latitude"].max()) / 2
    center_lon = (df["longitude"].min() + df["longitude"].max()) / 2

    # Create map
    map_obj = folium.Map(location=[center_lat, center_lon], zoom_start=13)

    # Add route polyline
    points = list(zip(df["latitude"], df["longitude"]))
    folium.PolyLine(points, color="blue", weight=3, opacity=0.7).add_to(map_obj)

    # Add start marker (green)
    start_lat = df["latitude"].iloc[0]
    start_lon = df["longitude"].iloc[0]
    folium.Marker(
        location=[start_lat, start_lon],
        icon=folium.Icon(color="green", icon="play", prefix="fa"),
        popup="Start",
    ).add_to(map_obj)

    # Add end marker (red/checkered flag)
    end_lat = df["latitude"].iloc[-1]
    end_lon = df["longitude"].iloc[-1]
    folium.Marker(
        location=[end_lat, end_lon],
        icon=folium.Icon(color="red", icon="flag-checkered", prefix="fa"),
        popup="Finish",
    ).add_to(map_obj)

    # Save to HTML file if path provided
    if save_path:
        map_obj.save(save_path)

    return map_obj
