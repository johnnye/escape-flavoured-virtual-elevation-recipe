import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button, Slider
from scipy.stats import pearsonr

from core.calculations import calculate_distance, delta_ve
from core.config import VirtualElevationConfig
from core.optimization import calculate_virtual_profile

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


def create_interactive_elevation_plot(
    df,
    actual_elevation,
    initial_cda,
    initial_crr,
    distance=None,
    config: VirtualElevationConfig = None,
    lap_column=None,
    cda_range=None,
    crr_range=None,
    initial_rmse=None,
    initial_r2=None,
    save_path=None,
    lap_num=None,  # Added parameter for lap number
):
    """
    Create an interactive plot with sliders to adjust CdA and Crr parameters.

    Args:
        df (pandas.DataFrame): DataFrame with cycling data
        actual_elevation (array-like): Actual measured elevation data
        initial_cda (float): Initial CdA value (typically from optimization)
        initial_crr (float): Initial Crr value (typically from optimization)
        distance (array-like, optional): Distance data in meters
        kg (float): Rider mass in kg
        rho (float): Air density in kg/m³
        dt (float): Time interval in seconds
        eta (float): Drivetrain efficiency
        vw (float): Wind velocity in m/s (positive = headwind)
        lap_column (str, optional): Column name containing lap numbers
        cda_range (tuple): (min, max) range for CdA slider
        crr_range (tuple): (min, max) range for Crr slider
        initial_rmse (float, optional): Initial RMSE value
        initial_r2 (float, optional): Initial R² value
        save_path (str, optional): Path to save a screenshot of the plot
        lap_num (int, optional): Current lap number being processed

    Returns:
        tuple: (fig, cda_slider, crr_slider, reset_button, save_button, saved_params)
               where saved_params is (cda, crr) if save button was clicked, else None
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.widgets import Button, Slider
    from scipy.stats import pearsonr

    plt.ioff()

    if config.kg is None or config.rho is None:
        raise ValueError(
            "Rider mass (kg) and air density (rho) are required parameters"
        )

    # If distance is not provided, calculate it from velocity
    if distance is None and "v" in df.columns:
        distance = calculate_distance(df)
    elif distance is None:
        distance = np.arange(len(actual_elevation))

    # Convert distance to kilometers for better readability
    distance_km = distance / 1000

    # Calculate initial virtual elevation using provided parameters
    initial_ve_changes = delta_ve(config, cda=initial_cda, crr=initial_crr, df=df)
    initial_virtual_elevation = calculate_virtual_profile(
        initial_ve_changes, actual_elevation, lap_column, df
    )

    # Calculate initial metrics if not provided
    if initial_rmse is None:
        initial_rmse = np.sqrt(
            np.mean((initial_virtual_elevation - actual_elevation) ** 2)
        )
    if initial_r2 is None:
        initial_r2 = pearsonr(initial_virtual_elevation, actual_elevation)[0] ** 2

    # Set up the figure and axes
    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(3, 1, height_ratios=[3, 1, 0.5])

    # Add some extra space at the bottom for the sliders
    plt.subplots_adjust(bottom=0.25)

    # Create axes for the plots
    ax_elevation = plt.subplot(gs[0])
    ax_residual = plt.subplot(gs[1], sharex=ax_elevation)

    # Create a rectangular area for the sliders and buttons
    ax_sliders = plt.subplot(gs[2])
    ax_sliders.set_visible(False)  # Hide this axis

    # Create separate axes for each slider
    ax_cda_slider = plt.axes([0.15, 0.15, 0.7, 0.03])
    ax_crr_slider = plt.axes([0.15, 0.1, 0.7, 0.03])

    # Create axes for buttons with adjusted positions
    ax_reset_button = plt.axes([0.15, 0.03, 0.15, 0.05])
    ax_save_button = plt.axes([0.70, 0.03, 0.15, 0.05])  # New save button

    # Set up the residual axis
    ax_residual.set_xlabel("Distance (km)", fontsize=12)
    ax_residual.set_ylabel("Residual (m)", fontsize=12)
    ax_residual.set_title("Elevation Difference (Actual - Virtual)", fontsize=12)
    ax_residual.grid(True, linestyle="--", alpha=0.7)

    # Set up the elevation axis
    ax_elevation.set_ylabel("Elevation (m)", fontsize=12)
    title = "Elevation Profiles Comparison"
    if lap_num is not None:
        title = f"Lap {lap_num} - {title}"
    ax_elevation.set_title(title, fontsize=14)
    ax_elevation.grid(True, linestyle="--", alpha=0.7)

    # Plot initial data
    (actual_line,) = ax_elevation.plot(
        distance_km, actual_elevation, "b-", linewidth=2, label="Actual Elevation"
    )
    (virtual_line,) = ax_elevation.plot(
        distance_km,
        initial_virtual_elevation,
        "r-",
        linewidth=2,
        label="Virtual Elevation",
    )

    # Add legend
    ax_elevation.legend(loc="best")

    # Plot initial residuals
    initial_residuals = actual_elevation - initial_virtual_elevation
    (residual_line,) = ax_residual.plot(
        distance_km, initial_residuals, "g-", linewidth=1.5
    )
    ax_residual.axhline(y=0, color="k", linestyle="-", alpha=0.3)

    # Add vertical lines for lap boundaries if lap column is provided
    lap_lines_elevation = []
    lap_lines_residual = []
    lap_texts = []

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
                line_style = {"color": "green", "linestyle": "--", "alpha": 0.7}
                line_label = f"Lap {unique_laps[i]} (Reset)"
            else:
                # Consecutive laps: add a blue dotted line
                line_style = {"color": "blue", "linestyle": ":", "alpha": 0.5}
                line_label = f"Lap {unique_laps[i]}"

            # Add lines to both plots
            line_elev = ax_elevation.axvline(
                x=distance_km[transition_idx], **line_style
            )
            line_resid = ax_residual.axvline(
                x=distance_km[transition_idx], **line_style
            )

            lap_lines_elevation.append(line_elev)
            lap_lines_residual.append(line_resid)

            # Add lap number labels
            if transition_idx > 0:
                text = ax_elevation.text(
                    distance_km[transition_idx] + 0.1,
                    np.min(actual_elevation)
                    + 0.1 * (np.max(actual_elevation) - np.min(actual_elevation)),
                    line_label,
                    rotation=90,
                    va="bottom",
                )
                lap_texts.append(text)

    # Add metrics text box
    props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)
    metrics_text = (
        f"CdA: {initial_cda:.4f} m²\n"
        f"Crr: {initial_crr:.5f}\n"
        f"RMSE: {initial_rmse:.2f} m\n"
        f"R²: {initial_r2:.4f}"
    )
    metrics_box = ax_elevation.text(
        0.95,
        0.95,
        metrics_text,
        transform=ax_elevation.transAxes,
        fontsize=10,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=props,
    )

    # Set default ranges if not provided
    if cda_range is None:
        # Set range to ±50% of optimized value, but within reasonable bounds
        cda_min = max(0.05, initial_cda * 0.5)
        cda_max = min(0.8, initial_cda * 1.5)
        cda_range = (cda_min, cda_max)

    if crr_range is None:
        # Set range to ±50% of optimized value, but within reasonable bounds
        crr_min = max(0.0005, initial_crr * 0.5)
        crr_max = min(0.02, initial_crr * 1.5)
        crr_range = (crr_min, crr_max)

    # Create sliders
    cda_slider = Slider(
        ax=ax_cda_slider,
        label="CdA (m²)",
        valmin=cda_range[0],
        valmax=cda_range[1],
        valinit=initial_cda,
        valfmt="%.4f",
    )

    crr_slider = Slider(
        ax=ax_crr_slider,
        label="Crr",
        valmin=crr_range[0],
        valmax=crr_range[1],
        valinit=initial_crr,
        valfmt="%.5f",
    )

    # Create reset button
    reset_button = Button(
        ax_reset_button, "Reset", color="lightgoldenrodyellow", hovercolor="0.975"
    )

    # Create save button with a different color to make it stand out
    save_button = Button(
        ax_save_button, "Save Results", color="lightgreen", hovercolor="palegreen"
    )

    # Variable to store saved parameters
    saved_params = [None]  # Use list to allow modification in nested functions

    # Function to update the plot when sliders change
    def update(val):
        # Get current values from sliders
        cda = cda_slider.val
        crr = crr_slider.val

        # Calculate new virtual elevation
        ve_changes = delta_ve(config, cda=cda, crr=crr, df=df)
        virtual_elevation = calculate_virtual_profile(
            ve_changes, actual_elevation, lap_column, df
        )

        # Update the virtual elevation line
        virtual_line.set_ydata(virtual_elevation)

        # Calculate and update residuals
        residuals = actual_elevation - virtual_elevation
        residual_line.set_ydata(residuals)

        # Calculate metrics
        rmse = np.sqrt(np.mean((virtual_elevation - actual_elevation) ** 2))
        r2 = pearsonr(virtual_elevation, actual_elevation)[0] ** 2

        # Update metrics text
        metrics_text = (
            f"CdA: {cda:.4f} m²\n"
            f"Crr: {crr:.5f}\n"
            f"RMSE: {rmse:.2f} m\n"
            f"R²: {r2:.4f}"
        )
        metrics_box.set_text(metrics_text)

        # Update plot limits if needed
        ax_residual.relim()
        ax_residual.autoscale_view()

        # Redraw the figure
        fig.canvas.draw_idle()

    # Function to reset sliders to initial values
    def reset(event):
        cda_slider.set_val(initial_cda)
        crr_slider.set_val(initial_crr)

    def save_results(event):
        # Store the current parameter values
        saved_params[0] = (cda_slider.val, crr_slider.val)

        # Visual feedback that save was successful
        save_button.label.set_text("Saved!")
        save_button.color = "palegreen"
        fig.canvas.draw_idle()

        # Remove threading/timer approach which is causing crashes
        # Mark the figure as closed in a safer way
        plt.close()  # Close all figures

    # Function to reset the save button text
    def reset_save_button(button, fig):
        button.label.set_text("Save Results")
        button.color = "lightgreen"
        fig.canvas.draw_idle()

    # Connect the update function to the sliders
    cda_slider.on_changed(update)
    crr_slider.on_changed(update)

    # Connect the reset function to the reset button
    reset_button.on_clicked(reset)

    # Connect the save function to the save button
    save_button.on_clicked(save_results)

    # Save a screenshot if requested
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig, cda_slider, crr_slider, reset_button, save_button, saved_params


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


def create_interactive_trim_map(
    df,
    lap_num,
    initial_trim_start=0,
    initial_trim_end=0,
    save_path=None,
):
    """
    Create an interactive map with sliders to trim the start and end of a lap route.

    Args:
        df (pandas.DataFrame): DataFrame with cycling data including latitude/longitude
        lap_num (int): Current lap number being processed
        initial_trim_start (float): Initial trim start distance in meters
        initial_trim_end (float): Initial trim end distance in meters
        save_path (str, optional): Path to save a screenshot of the map

    Returns:
        tuple: (action, trim_start, trim_end) where action is "optimize" or "skip"
    """
    try:
        import contextily as ctx
        import geopandas as gpd
        import matplotlib.pyplot as plt
        import numpy as np
        from matplotlib.widgets import Button, Slider
        from shapely.geometry import LineString, Point
    except ImportError:
        print("To use interactive mapping, install the required packages:")
        print("pip install matplotlib contextily geopandas shapely")
        return "optimize", initial_trim_start, initial_trim_end

    # Ensure we have latitude and longitude data
    if "latitude" not in df.columns or "longitude" not in df.columns:
        print("GPS data not available for interactive trimming, using default values")
        return "optimize", initial_trim_start, initial_trim_end

    # Calculate distance along route
    distances = [0]
    total_distance = 0

    # Calculate cumulative distance along route
    for i in range(1, len(df)):
        lat1, lon1 = df["latitude"].iloc[i - 1], df["longitude"].iloc[i - 1]
        lat2, lon2 = df["latitude"].iloc[i], df["longitude"].iloc[i]

        # Use haversine formula to calculate distance in meters
        import math

        R = 6371000  # Earth radius in meters
        phi1 = math.radians(lat1)
        phi2 = math.radians(lat2)
        delta_phi = math.radians(lat2 - lat1)
        delta_lambda = math.radians(lon2 - lon1)

        a = math.sin(delta_phi / 2) * math.sin(delta_phi / 2) + math.cos(
            phi1
        ) * math.cos(phi2) * math.sin(delta_lambda / 2) * math.sin(delta_lambda / 2)
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

        segment_distance = R * c
        total_distance += segment_distance
        distances.append(total_distance)

    fig = plt.figure(figsize=(12, 12))  # Increase height
    plt.subplots_adjust(bottom=0.3)  # Increase bottom margin for sliders/buttons

    # Create the map axis
    ax = plt.subplot(111)

    # Create points for the full route
    points = [Point(lon, lat) for lon, lat in zip(df["longitude"], df["latitude"])]
    route_line = LineString(points)

    # Create GeoDataFrame for the route line
    gdf_line = gpd.GeoDataFrame(geometry=[route_line], crs="EPSG:4326")
    gdf_line = gdf_line.to_crs(epsg=3857)  # Convert to Web Mercator

    # Plot original route in blue
    gdf_line.plot(ax=ax, color="blue", linewidth=2, alpha=0.5)

    # Initial start and end points (0% and 100% of route)
    start_point = Point(df["longitude"].iloc[0], df["latitude"].iloc[0])
    end_point = Point(df["longitude"].iloc[-1], df["latitude"].iloc[-1])

    # Convert to GeoDataFrame
    gdf_points = gpd.GeoDataFrame(
        geometry=[start_point, end_point],
        data={"type": ["start", "end"]},
        crs="EPSG:4326",
    )
    gdf_points = gdf_points.to_crs(epsg=3857)

    # Plot start and end points
    start_point_plot = ax.scatter(
        gdf_points[gdf_points["type"] == "start"].geometry.x,
        gdf_points[gdf_points["type"] == "start"].geometry.y,
        color="green",
        s=100,
        zorder=10,
    )
    end_point_plot = ax.scatter(
        gdf_points[gdf_points["type"] == "end"].geometry.x,
        gdf_points[gdf_points["type"] == "end"].geometry.y,
        color="red",
        s=100,
        zorder=10,
    )

    # Add OSM background
    bounds = gdf_line.total_bounds
    width_meters = bounds[2] - bounds[0]
    height_meters = bounds[3] - bounds[1]
    max_dimension = max(width_meters, height_meters)

    # Determine zoom level based on track extent
    if max_dimension < 500:
        zoom = 19
    elif max_dimension < 1000:
        zoom = 18
    elif max_dimension < 2000:
        zoom = 17
    elif max_dimension < 5000:
        zoom = 16
    elif max_dimension < 10000:
        zoom = 15
    elif max_dimension < 20000:
        zoom = 14
    elif max_dimension < 50000:
        zoom = 13
    else:
        zoom = 12

    try:
        ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik, zoom=zoom)
    except Exception as e:
        print(f"Error adding map background: {str(e)}")

    # Remove axis labels
    ax.set_axis_off()

    # Add title
    ax.set_title(f"Lap {lap_num} - Adjust Start and End Trim Points", fontsize=16)

    # Create slider axes
    ax_trim_start = plt.axes([0.2, 0.2, 0.65, 0.03])  # Move up
    ax_trim_end = plt.axes([0.2, 0.15, 0.65, 0.03])  # Move up

    # Create button axes
    ax_skip = plt.axes([0.2, 0.07, 0.3, 0.05])  # Left button (now Skip)
    ax_optimize = plt.axes([0.55, 0.07, 0.3, 0.05])  # Right button (now Optimize)

    # Create sliders
    max_trim = total_distance * 0.45  # Maximum 45% trim from each end

    # If initial values are larger than max, cap them
    initial_trim_start = min(initial_trim_start, max_trim)
    initial_trim_end = min(initial_trim_end, max_trim)

    trim_start_slider = Slider(
        ax=ax_trim_start,
        label="Trim Start (m)",
        valmin=0,
        valmax=max_trim,
        valinit=initial_trim_start,
        valfmt="%.0f",
    )

    trim_end_slider = Slider(
        ax=ax_trim_end,
        label="Trim End (m)",
        valmin=0,
        valmax=max_trim,
        valinit=initial_trim_end,
        valfmt="%.0f",
    )

    # Create buttons
    optimize_button = Button(
        ax_optimize, "Start Optimization", color="lightgreen", hovercolor="palegreen"
    )
    skip_button = Button(ax_skip, "Skip Lap", color="lightsalmon", hovercolor="salmon")

    info_text = plt.figtext(
        0.5,
        0.02,  # Position slightly higher
        f"Trim Start: {initial_trim_start:.0f}m, Trim End: {initial_trim_end:.0f}m\n"
        f"Total distance: {total_distance:.0f}m",
        ha="center",
        bbox={"facecolor": "wheat", "alpha": 0.5, "pad": 5},
    )

    # Function to update the map when sliders change
    def update(val):
        # Get current values from sliders
        trim_start = trim_start_slider.val
        trim_end = trim_end_slider.val

        # Find the indices corresponding to the trim distances
        start_idx = 0
        end_idx = len(distances) - 1

        for i, dist in enumerate(distances):
            if dist >= trim_start:
                start_idx = i
                break

        for i in range(len(distances) - 1, -1, -1):
            if distances[i] <= (total_distance - trim_end):
                end_idx = i
                break

        # Check if trim is valid (at least 10% of route remaining)
        if start_idx >= end_idx or (end_idx - start_idx) < len(distances) * 0.1:
            # Invalid trim, don't update
            return

        # Update start and end points
        start_lon, start_lat = (
            df["longitude"].iloc[start_idx],
            df["latitude"].iloc[start_idx],
        )
        end_lon, end_lat = df["longitude"].iloc[end_idx], df["latitude"].iloc[end_idx]

        start_point = Point(start_lon, start_lat)
        end_point = Point(end_lon, end_lat)

        # Update GeoDataFrame
        new_points = gpd.GeoDataFrame(
            geometry=[start_point, end_point],
            data={"type": ["start", "end"]},
            crs="EPSG:4326",
        )
        new_points = new_points.to_crs(epsg=3857)

        # Update the plotted points
        start_point_plot.set_offsets(
            [
                [
                    new_points[new_points["type"] == "start"].geometry.x.iloc[0],
                    new_points[new_points["type"] == "start"].geometry.y.iloc[0],
                ]
            ]
        )

        end_point_plot.set_offsets(
            [
                [
                    new_points[new_points["type"] == "end"].geometry.x.iloc[0],
                    new_points[new_points["type"] == "end"].geometry.y.iloc[0],
                ]
            ]
        )

        # Create trimmed route line
        trimmed_points = [
            Point(lon, lat)
            for lon, lat in zip(
                df["longitude"].iloc[start_idx : end_idx + 1],
                df["latitude"].iloc[start_idx : end_idx + 1],
            )
        ]

        if len(trimmed_points) >= 2:
            trimmed_line = LineString(trimmed_points)

            # Clear old route
            ax.collections = [
                c for c in ax.collections if c not in [start_point_plot, end_point_plot]
            ]

            # Plot full route (blue, transparent)
            gdf_line.plot(ax=ax, color="blue", linewidth=2, alpha=0.3)

            # Plot trimmed route
            trimmed_gdf = gpd.GeoDataFrame(geometry=[trimmed_line], crs="EPSG:4326")
            trimmed_gdf = trimmed_gdf.to_crs(epsg=3857)
            trimmed_gdf.plot(ax=ax, color="red", linewidth=3, alpha=0.7)

            # Re-add the points on top
            ax.add_collection(start_point_plot)
            ax.add_collection(end_point_plot)

        # Update info text
        trimmed_distance = distances[end_idx] - distances[start_idx]
        info_text.set_text(
            f"Trim Start: {trim_start:.0f}m, Trim End: {trim_end:.0f}m\n"
            f"Trimmed distance: {trimmed_distance:.0f}m of {total_distance:.0f}m"
        )

        # Redraw the figure
        fig.canvas.draw_idle()

    # Connect the update function to the sliders
    trim_start_slider.on_changed(update)
    trim_end_slider.on_changed(update)

    # Variable to store the result
    result = ["optimize", initial_trim_start, initial_trim_end]

    # Button click events
    def on_optimize(event):
        result[0] = "optimize"
        result[1] = trim_start_slider.val
        result[2] = trim_end_slider.val
        plt.close(fig)

    def on_skip(event):
        result[0] = "skip"
        result[1] = trim_start_slider.val
        result[2] = trim_end_slider.val
        plt.close(fig)

    # Connect button click events
    optimize_button.on_clicked(on_optimize)
    skip_button.on_clicked(on_skip)

    # Save a screenshot if requested
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    # Show the plot (this blocks until the plot is closed)
    plt.show()

    # Return the result: action, trim_start, trim_end
    return result[0], result[1], result[2]


def create_combined_interactive_plot(
    df,
    actual_elevation,
    lap_num,
    config: VirtualElevationConfig,
    initial_cda=0.3,
    initial_crr=0.005,
    initial_trim_start=0,
    initial_trim_end=0,
    cda_range=None,
    crr_range=None,
    save_path=None,
    optimization_function=None,
    distance=None,
    lap_column=None,
    target_elevation_gain=None,  # Added this parameter
):
    """
    Create a combined interactive plot with map, elevation profiles, and parameter sliders.

    Args:
        df (pandas.DataFrame): DataFrame with cycling data including latitude/longitude
        actual_elevation (array-like): Actual measured elevation data
        lap_num (int): Current lap number being processed (0 for combined laps)
        rider_mass (float): Rider mass in kg
        air_density (float): Air density in kg/m³
        initial_cda (float): Initial CdA value
        initial_crr (float): Initial Crr value
        initial_trim_start (float): Initial trim start distance in meters
        initial_trim_end (float): Initial trim end distance in meters
        dt (float): Time interval in seconds
        eta (float): Drivetrain efficiency
        vw (float): Wind velocity in m/s (positive = headwind)
        cda_range (tuple): (min, max) range for CdA slider
        crr_range (tuple): (min, max) range for Crr slider
        save_path (str, optional): Path to save a screenshot of the plot
        optimization_function (callable): Function to perform optimization
        distance (array-like, optional): Distance data in meters
        lap_column (str, optional): Column name containing lap numbers
        target_elevation_gain (float, optional): Target elevation gain in meters

    Returns:
        tuple: (action, trim_start, trim_end, cda, crr, rmse, r2)
    """
    import time

    import matplotlib.gridspec as gridspec
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.widgets import Button, Slider
    from scipy.stats import pearsonr

    try:
        import contextily as ctx
        import geopandas as gpd
        from shapely.geometry import LineString, Point

        map_libs_available = True
    except ImportError:
        print(
            "Map libraries not available. Install them with: pip install contextily geopandas shapely"
        )
        map_libs_available = False

    plt.ioff()  # Turn off interactive mode to prevent unexpected plot displays

    from core.calculations import calculate_distance, delta_ve
    from core.optimization import calculate_virtual_profile

    # Set default ranges if not provided
    if cda_range is None:
        cda_min = max(0.05, initial_cda * 0.5)
        cda_max = min(0.8, initial_cda * 1.5)
        cda_range = (cda_min, cda_max)

    if crr_range is None:
        crr_min = max(0.0005, initial_crr * 0.5)
        crr_max = min(0.02, initial_crr * 1.5)
        crr_range = (crr_min, crr_max)

    # Calculate distance if not provided
    if distance is None:
        distance = calculate_distance(df, config.dt)

    # Calculate total distance
    total_distance = distance[-1]

    # Calculate minimum distance needed for 30 seconds of data
    avg_speed = df["v"].mean()
    min_distance_for_30s = avg_speed * 30  # 30 seconds * average speed

    # Set maximum trim to nearly the full distance (95%)
    max_trim = total_distance * 0.95

    # Cap initial trim values
    initial_trim_start = min(initial_trim_start, max_trim)
    initial_trim_end = min(initial_trim_end, max_trim)

    # Create a figure with a grid layout - MODIFIED to remove route plot
    fig = plt.figure(figsize=(15, 10))

    # Create grid layout with different areas for map, elevation plot, residuals
    # Removed the route plot and adjusted the grid
    gs = gridspec.GridSpec(3, 1, height_ratios=[3, 2, 1])

    # Map view
    ax_map = fig.add_subplot(gs[0])

    # Elevation plot for both actual and virtual profiles
    ax_elevation = fig.add_subplot(gs[1])

    # Residuals plot aligned with elevation plot (share x-axis for alignment)
    ax_residual = fig.add_subplot(gs[2], sharex=ax_elevation)

    # Status area at bottom (hidden)
    ax_status = fig.add_subplot(gs[2, :])
    ax_status.set_visible(False)  # Hide but keep for layout

    # Add space for sliders and buttons
    plt.subplots_adjust(bottom=0.30)

    # Create slider axes
    ax_trim_start = plt.axes([0.15, 0.20, 0.70, 0.02])
    ax_trim_end = plt.axes([0.15, 0.16, 0.70, 0.02])
    ax_cda_slider = plt.axes([0.15, 0.12, 0.70, 0.02])
    ax_crr_slider = plt.axes([0.15, 0.08, 0.70, 0.02])

    # Create button axes
    ax_skip = plt.axes([0.15, 0.03, 0.20, 0.04])
    ax_optimize = plt.axes([0.40, 0.03, 0.20, 0.04])
    ax_save = plt.axes([0.65, 0.03, 0.20, 0.04])

    # Make sure the sliders are created with this new max_trim value
    trim_start_slider = Slider(
        ax=ax_trim_start,
        label="Trim Start (m)",
        valmin=0,
        valmax=max_trim,
        valinit=initial_trim_start,
        valfmt="%.0f",
    )

    trim_end_slider = Slider(
        ax=ax_trim_end,
        label="Trim End (m)",
        valmin=0,
        valmax=max_trim,
        valinit=initial_trim_end,
        valfmt="%.0f",
    )

    cda_slider = Slider(
        ax=ax_cda_slider,
        label="CdA (m²)",
        valmin=cda_range[0],
        valmax=cda_range[1],
        valinit=initial_cda,
        valfmt="%.4f",
    )

    crr_slider = Slider(
        ax=ax_crr_slider,
        label="Crr",
        valmin=crr_range[0],
        valmax=crr_range[1],
        valinit=initial_crr,
        valfmt="%.5f",
    )

    # Display optimization type
    optimization_type = (
        "Target Elevation Gain" if target_elevation_gain is not None else "R²/RMSE"
    )
    optimization_info = f"Optimization: {optimization_type}"
    if target_elevation_gain is not None:
        optimization_info += f" ({target_elevation_gain}m)"

    optimization_text = plt.figtext(
        0.5, 0.24, optimization_info, ha="center", fontsize=10
    )

    # Create buttons
    skip_button = Button(ax_skip, "Skip Lap", color="lightsalmon", hovercolor="salmon")
    optimize_button = Button(
        ax_optimize, "Optimize", color="lightblue", hovercolor="skyblue"
    )
    save_button = Button(
        ax_save, "Save Results", color="lightgreen", hovercolor="palegreen"
    )

    # Status variables to store results and state
    results = {
        "action": "optimize",  # default action
        "trim_start": initial_trim_start,
        "trim_end": initial_trim_end,
        "cda": initial_cda,
        "crr": initial_crr,
        "rmse": None,
        "r2": None,
        "optimizing": False,
        "saved": False,
        "current_virtual_profile": None,  # Store the current virtual profile
        "trimmed_df": None,  # Store the currently trimmed DataFrame
        "trimmed_distance": None,  # Store the currently trimmed distance
        "trimmed_elevation": None,  # Store the currently trimmed elevation
    }

    # Set up the map if libraries are available
    map_features = {
        "start_point_plot": None,
        "end_point_plot": None,
        "trim_start_line": None,
        "trim_end_line": None,
    }

    elevation_features = {
        "actual_line": None,
        "virtual_line": None,
        "residual_line": None,
        "trim_start_line": None,
        "trim_end_line": None,
        "metrics_box": None,
        "grayed_start": None,  # For gray area before trim start
        "grayed_end": None,  # For gray area after trim end
    }

    # Calculate distances along the route for the map view
    def calculate_route_distances():
        if (
            map_libs_available
            and "latitude" in df.columns
            and "longitude" in df.columns
        ):
            distances = [0]
            total_dist = 0
            # Calculate cumulative distance along route
            for i in range(1, len(df)):
                lat1, lon1 = df["latitude"].iloc[i - 1], df["longitude"].iloc[i - 1]
                lat2, lon2 = df["latitude"].iloc[i], df["longitude"].iloc[i]

                # Use haversine formula to calculate distance in meters
                import math

                R = 6371000  # Earth radius in meters
                phi1 = math.radians(lat1)
                phi2 = math.radians(lat2)
                delta_phi = math.radians(lat2 - lat1)
                delta_lambda = math.radians(lon2 - lon1)

                a = math.sin(delta_phi / 2) * math.sin(delta_phi / 2) + math.cos(
                    phi1
                ) * math.cos(phi2) * math.sin(delta_lambda / 2) * math.sin(
                    delta_lambda / 2
                )
                c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

                segment_distance = R * c
                total_dist += segment_distance
                distances.append(total_dist)
            return distances
        return None

    route_distances = calculate_route_distances()

    # Prepare map with route
    def initialize_map():
        if (
            not map_libs_available
            or "latitude" not in df.columns
            or "longitude" not in df.columns
        ):
            ax_map.text(
                0.5,
                0.5,
                "Map view not available\n(missing GPS data or libraries)",
                ha="center",
                va="center",
                transform=ax_map.transAxes,
            )
            ax_map.set_visible(False)
            return False

        try:
            # Create points for the full route
            points = [
                Point(lon, lat) for lon, lat in zip(df["longitude"], df["latitude"])
            ]
            route_line = LineString(points)

            # Create GeoDataFrame for the route line
            gdf_line = gpd.GeoDataFrame(geometry=[route_line], crs="EPSG:4326")
            gdf_line = gdf_line.to_crs(epsg=3857)  # Convert to Web Mercator

            # Plot original route in blue
            gdf_line.plot(ax=ax_map, color="blue", linewidth=2, alpha=0.5)

            # Initial start and end points (0% and 100% of route)
            start_point = Point(df["longitude"].iloc[0], df["latitude"].iloc[0])
            end_point = Point(df["longitude"].iloc[-1], df["latitude"].iloc[-1])

            # Convert to GeoDataFrame
            gdf_points = gpd.GeoDataFrame(
                geometry=[start_point, end_point],
                data={"type": ["start", "end"]},
                crs="EPSG:4326",
            )
            gdf_points = gdf_points.to_crs(epsg=3857)

            # Plot start and end points
            map_features["start_point_plot"] = ax_map.scatter(
                gdf_points[gdf_points["type"] == "start"].geometry.x,
                gdf_points[gdf_points["type"] == "start"].geometry.y,
                color="green",
                s=100,
                zorder=10,
            )
            map_features["end_point_plot"] = ax_map.scatter(
                gdf_points[gdf_points["type"] == "end"].geometry.x,
                gdf_points[gdf_points["type"] == "end"].geometry.y,
                color="red",
                s=100,
                zorder=10,
            )

            # Add OSM background
            bounds = gdf_line.total_bounds
            width_meters = bounds[2] - bounds[0]
            height_meters = bounds[3] - bounds[1]
            max_dimension = max(width_meters, height_meters)

            # Determine zoom level based on track extent
            if max_dimension < 500:
                zoom = 19
            elif max_dimension < 1000:
                zoom = 18
            elif max_dimension < 2000:
                zoom = 17
            elif max_dimension < 5000:
                zoom = 16
            elif max_dimension < 10000:
                zoom = 15
            elif max_dimension < 20000:
                zoom = 14
            elif max_dimension < 50000:
                zoom = 13
            else:
                zoom = 12

            try:
                ctx.add_basemap(
                    ax_map, source=ctx.providers.OpenStreetMap.Mapnik, zoom=zoom
                )
            except Exception as e:
                print(f"Error adding map background: {str(e)}")

            # Remove axis labels for map
            ax_map.set_axis_off()

            # Add title to map
            lap_title = f"Lap {lap_num}" if lap_num > 0 else "Combined Laps"
            ax_map.set_title(
                f"{lap_title} - Adjust Start and End Trim Points", fontsize=14
            )

            return True

        except Exception as e:
            print(f"Error initializing map: {str(e)}")
            import traceback

            traceback.print_exc()
            ax_map.text(
                0.5,
                0.5,
                f"Error initializing map: {str(e)}",
                ha="center",
                va="center",
                transform=ax_map.transAxes,
            )
            ax_map.set_visible(False)
            return False

    # Initialize elevation plot with empty data
    def initialize_elevation_plot():
        # Set up plots
        ax_elevation.set_ylabel("Elevation (m)", fontsize=12)
        lap_title = f"Lap {lap_num}" if lap_num > 0 else "Combined Laps"
        ax_elevation.set_title(f"{lap_title} - Elevation Profiles", fontsize=14)
        ax_elevation.grid(True, linestyle="--", alpha=0.7)

        ax_residual.set_xlabel("Distance (km)", fontsize=12)
        ax_residual.set_ylabel("Residual (m)", fontsize=12)
        ax_residual.set_title("Elevation Difference (Actual - Virtual)", fontsize=12)
        ax_residual.grid(True, linestyle="--", alpha=0.7)

        # Convert distance to km
        distance_km = distance / 1000

        # Plot actual elevation
        (elevation_features["actual_line"],) = ax_elevation.plot(
            distance_km, actual_elevation, "b-", linewidth=2, label="Actual Elevation"
        )

        # Initialize virtual elevation lines - both active and inactive regions
        virtual_data = np.zeros_like(actual_elevation)
        virtual_data[:] = np.nan

        # Active region (full opacity)
        (elevation_features["virtual_line"],) = ax_elevation.plot(
            distance_km, virtual_data, "r-", linewidth=2, label="Virtual Elevation"
        )

        # Inactive region (reduced opacity)
        (elevation_features["virtual_line_inactive"],) = ax_elevation.plot(
            [], [], "r-", linewidth=2, alpha=0.3, label="_nolegend_"
        )

        # Initialize residual lines - both active and inactive regions
        residual_data = np.zeros_like(distance_km)
        residual_data[:] = np.nan

        # Active region (full opacity)
        (elevation_features["residual_line"],) = ax_residual.plot(
            distance_km, residual_data, "g-", linewidth=1.5, label="Residual"
        )

        # Inactive region (reduced opacity)
        (elevation_features["residual_line_inactive"],) = ax_residual.plot(
            [], [], "g-", linewidth=1.5, alpha=0.3, label="_nolegend_"
        )

        # Zero line in residual plot
        ax_residual.axhline(y=0, color="k", linestyle="-", alpha=0.3)

        # Add legend
        ax_elevation.legend(loc="best")

        # Add trim lines
        start_km = initial_trim_start / 1000
        end_km = (total_distance - initial_trim_end) / 1000

        # Elevation plot trim lines
        elevation_features["trim_start_line"] = ax_elevation.axvline(
            x=start_km, color="green", linewidth=1.5, linestyle="--", visible=False
        )
        elevation_features["trim_end_line"] = ax_elevation.axvline(
            x=end_km, color="red", linewidth=1.5, linestyle="--", visible=False
        )

        # Residual plot trim lines
        elevation_features["residual_trim_start"] = ax_residual.axvline(
            x=start_km, color="green", linewidth=1.5, linestyle="--", visible=False
        )
        elevation_features["residual_trim_end"] = ax_residual.axvline(
            x=end_km, color="red", linewidth=1.5, linestyle="--", visible=False
        )

        # Initialize storage for grayed areas
        elevation_features["ylim"] = (
            np.min(actual_elevation) - 10,
            np.max(actual_elevation) + 10,
        )
        elevation_features["res_ylim"] = (-10, 10)
        elevation_features["grayed_start"] = None
        elevation_features["grayed_end"] = None
        elevation_features["res_grayed_start"] = None
        elevation_features["res_grayed_end"] = None

        # Add metrics text box
        props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)
        metrics_text = "Press 'Optimize' to start analysis"
        elevation_features["metrics_box"] = ax_elevation.text(
            0.95,
            0.95,
            metrics_text,
            transform=ax_elevation.transAxes,
            fontsize=10,
            verticalalignment="top",
            horizontalalignment="right",
            bbox=props,
        )

    # Function to update the map trim points
    def update_map_trim_points(trim_start, trim_end):
        if not map_libs_available:
            return

        try:
            # Find indices corresponding to the trim distances
            start_idx = 0
            end_idx = len(distance) - 1

            for i, dist in enumerate(distance):
                if dist >= trim_start:
                    start_idx = i
                    break

            for i in range(len(distance) - 1, -1, -1):
                if distance[i] <= (total_distance - trim_end):
                    end_idx = i
                    break

            # Update map points if we have them
            if (
                map_features["start_point_plot"] is not None
                and "latitude" in df.columns
            ):
                # Update start point
                start_lon, start_lat = (
                    df["longitude"].iloc[start_idx],
                    df["latitude"].iloc[start_idx],
                )
                start_point = Point(start_lon, start_lat)
                gdf_start = gpd.GeoDataFrame(
                    geometry=[start_point], crs="EPSG:4326"
                ).to_crs(epsg=3857)

                map_features["start_point_plot"].set_offsets(
                    [[gdf_start.geometry.x.iloc[0], gdf_start.geometry.y.iloc[0]]]
                )

                # Update end point
                end_lon, end_lat = (
                    df["longitude"].iloc[end_idx],
                    df["latitude"].iloc[end_idx],
                )
                end_point = Point(end_lon, end_lat)
                gdf_end = gpd.GeoDataFrame(
                    geometry=[end_point], crs="EPSG:4326"
                ).to_crs(epsg=3857)

                map_features["end_point_plot"].set_offsets(
                    [[gdf_end.geometry.x.iloc[0], gdf_end.geometry.y.iloc[0]]]
                )

        except Exception as e:
            print(f"Error updating map trim points: {str(e)}")

    def update_elevation_trim_lines(trim_start, trim_end):
        try:
            # Convert to km for the plot
            start_km = trim_start / 1000
            end_km = (total_distance - trim_end) / 1000

            # Update the trim lines with proper sequences
            elevation_features["trim_start_line"].set_xdata([start_km, start_km])
            elevation_features["trim_start_line"].set_visible(True)

            elevation_features["trim_end_line"].set_xdata([end_km, end_km])
            elevation_features["trim_end_line"].set_visible(True)

            elevation_features["residual_trim_start"].set_xdata([start_km, start_km])
            elevation_features["residual_trim_start"].set_visible(True)

            elevation_features["residual_trim_end"].set_xdata([end_km, end_km])
            elevation_features["residual_trim_end"].set_visible(True)

            # Remove old grayed areas
            for area in [
                "grayed_start",
                "grayed_end",
                "res_grayed_start",
                "res_grayed_end",
            ]:
                if area in elevation_features and elevation_features[area] is not None:
                    try:
                        elevation_features[area].remove()
                    except:
                        pass
                    elevation_features[area] = None

            # Get current y-limits
            y_min, y_max = elevation_features["ylim"]
            res_y_min, res_y_max = elevation_features["res_ylim"]

            # Gray out areas with reduced opacity (0.15)
            if start_km > 0:
                # For elevation plot
                start_x = np.linspace(0, start_km, 100)
                start_y_min = np.ones_like(start_x) * y_min
                start_y_max = np.ones_like(start_x) * y_max

                elevation_features["grayed_start"] = ax_elevation.fill_between(
                    start_x, start_y_min, start_y_max, color="gray", alpha=0.15
                )

                # For residual plot
                res_start_y_min = np.ones_like(start_x) * res_y_min
                res_start_y_max = np.ones_like(start_x) * res_y_max

                elevation_features["res_grayed_start"] = ax_residual.fill_between(
                    start_x, res_start_y_min, res_start_y_max, color="gray", alpha=0.15
                )

            if end_km < total_distance / 1000:
                # For elevation plot
                end_x = np.linspace(end_km, total_distance / 1000, 100)
                end_y_min = np.ones_like(end_x) * y_min
                end_y_max = np.ones_like(end_x) * y_max

                elevation_features["grayed_end"] = ax_elevation.fill_between(
                    end_x, end_y_min, end_y_max, color="gray", alpha=0.15
                )

                # For residual plot
                res_end_y_min = np.ones_like(end_x) * res_y_min
                res_end_y_max = np.ones_like(end_x) * res_y_max

                elevation_features["res_grayed_end"] = ax_residual.fill_between(
                    end_x, res_end_y_min, res_end_y_max, color="gray", alpha=0.15
                )

        except Exception as e:
            print(f"Error updating elevation trim lines: {str(e)}")
            import traceback

            traceback.print_exc()

    # Function to run optimization with current parameters
    def run_optimization():
        if results["optimizing"]:
            return

        results["optimizing"] = True
        optimize_button.label.set_text("Optimizing...")
        optimize_button.color = "yellow"
        fig.canvas.draw_idle()

        try:
            # Get trim values from sliders
            trim_start = trim_start_slider.val
            trim_end = trim_end_slider.val

            # Find indices corresponding to the trim distances
            start_idx = 0
            end_idx = len(distance) - 1

            for i, dist in enumerate(distance):
                if dist >= trim_start:
                    start_idx = i
                    break

            for i in range(len(distance) - 1, -1, -1):
                if distance[i] <= (total_distance - trim_end):
                    end_idx = i
                    break

            # Extract just the trimmed section for optimization
            trimmed_df = df.iloc[start_idx : end_idx + 1].copy()
            trimmed_distance = distance[start_idx : end_idx + 1] - distance[start_idx]
            trimmed_elevation = actual_elevation[start_idx : end_idx + 1]

            # Store the trimmed data for later use
            results["trimmed_df"] = trimmed_df
            results["trimmed_distance"] = trimmed_distance
            results["trimmed_elevation"] = trimmed_elevation

            # Check if we have enough data points after trimming
            if len(trimmed_df) < 10:
                elevation_features["metrics_box"].set_text(
                    "Error: Not enough data points after trimming\n"
                    "Adjust trim sliders and try again"
                )
                fig.canvas.draw_idle()
                results["optimizing"] = False
                optimize_button.label.set_text("Optimize")
                optimize_button.color = "lightblue"
                return

            # Recalculate dt if time data is available
            if "timestamp" in trimmed_df.columns:
                dt_values = trimmed_df["timestamp"].diff().dt.total_seconds()
                avg_dt = dt_values[1:].mean()  # skip first row which is NaN
                adjusted_dt = avg_dt if not np.isnan(avg_dt) else config.dt
                config.dt = adjusted_dt

            # Calculate acceleration if needed
            if "a" not in trimmed_df.columns:
                from core.calculations import accel_calc

                trimmed_df["a"] = accel_calc(trimmed_df["v"].values, adjusted_dt)

            # Use optimization function if provided - ONLY OPTIMIZE THE ACTIVE REGION
            if optimization_function is not None:
                # Call the optimization function with trimmed data, passing target_elevation_gain
                optimized_cda, optimized_crr, rmse, r2, virtual_profile = (
                    optimization_function(
                        df=trimmed_df,
                        actual_elevation=trimmed_elevation,
                        config=config,
                        initial_cda=cda_slider.val,
                        initial_crr=crr_slider.val,
                        target_elevation_gain=target_elevation_gain,
                    )
                )
            else:
                # No optimization function, calculate with current parameters
                optimized_cda = cda_slider.val
                optimized_crr = crr_slider.val

                # Calculate virtual elevation with current parameters for trimmed region
                ve_changes = delta_ve(
                    config, cda=optimized_cda, crr=optimized_crr, df=trimmed_df
                )

                # Build virtual elevation profile for trimmed region
                virtual_profile = calculate_virtual_profile(
                    ve_changes, trimmed_elevation, lap_column, trimmed_df
                )

                # Calculate stats
                rmse = np.sqrt(np.mean((virtual_profile - trimmed_elevation) ** 2))
                r2 = pearsonr(virtual_profile, trimmed_elevation)[0] ** 2

            # Store results
            results["cda"] = optimized_cda
            results["crr"] = optimized_crr
            results["rmse"] = rmse
            results["r2"] = r2
            results["current_virtual_profile"] = virtual_profile

            # Update sliders to match optimized values
            cda_slider.set_val(optimized_cda)
            crr_slider.set_val(optimized_crr)

            # Update plots with the new virtual profile - will calculate full profile
            update_elevation_plot(
                trimmed_distance,
                trimmed_elevation,
                virtual_profile,
                optimized_cda,
                optimized_crr,
                rmse,
                r2,
            )

        except Exception as e:
            print(f"Error during optimization: {str(e)}")
            import traceback

            traceback.print_exc()
            elevation_features["metrics_box"].set_text(f"Optimization error: {str(e)}")

        finally:
            results["optimizing"] = False
            optimize_button.label.set_text("Optimize")
            optimize_button.color = "lightblue"
            fig.canvas.draw_idle()

    # Function to update elevation plot with new data
    def update_elevation_plot(
        trimmed_distance, trimmed_elevation, virtual_profile, cda, crr, rmse, r2
    ):
        try:
            # Use full distance range for display
            distance_km = distance / 1000

            # Find active region indices
            start_idx = 0
            end_idx = len(distance) - 1

            for i, dist in enumerate(distance):
                if dist >= trim_start_slider.val:
                    start_idx = i
                    break

            for i in range(len(distance) - 1, -1, -1):
                if distance[i] <= (total_distance - trim_end_slider.val):
                    end_idx = i
                    break

            # Show complete actual elevation profile
            elevation_features["actual_line"].set_xdata(distance_km)
            elevation_features["actual_line"].set_ydata(actual_elevation)

            # Calculate virtual elevation for the FULL profile
            ve_changes = delta_ve(config, cda=cda, crr=crr, df=df)

            # Build full virtual elevation profile
            full_virtual = calculate_virtual_profile(
                ve_changes, actual_elevation, lap_column, df
            )

            # Split virtual elevation into active and inactive regions
            active_x = distance_km[start_idx : end_idx + 1]
            active_y = full_virtual[start_idx : end_idx + 1]

            # Split inactive regions into pre and post sections
            pre_x = distance_km[:start_idx] if start_idx > 0 else []
            pre_y = full_virtual[:start_idx] if start_idx > 0 else []

            post_x = (
                distance_km[end_idx + 1 :] if end_idx < len(distance_km) - 1 else []
            )
            post_y = (
                full_virtual[end_idx + 1 :] if end_idx < len(full_virtual) - 1 else []
            )

            # Update active line (full opacity)
            elevation_features["virtual_line"].set_xdata(active_x)
            elevation_features["virtual_line"].set_ydata(active_y)
            elevation_features["virtual_line"].set_visible(True)

            # FIXED: Handle pre and post regions as separate lines
            # Create pre-trim line if it doesn't exist
            if "virtual_line_pre" not in elevation_features:
                (elevation_features["virtual_line_pre"],) = ax_elevation.plot(
                    [], [], "r-", linewidth=2, alpha=0.3, label="_nolegend_"
                )

            # Create post-trim line if it doesn't exist
            if "virtual_line_post" not in elevation_features:
                (elevation_features["virtual_line_post"],) = ax_elevation.plot(
                    [], [], "r-", linewidth=2, alpha=0.3, label="_nolegend_"
                )

            # Update pre-trim line
            elevation_features["virtual_line_pre"].set_xdata(pre_x)
            elevation_features["virtual_line_pre"].set_ydata(pre_y)
            elevation_features["virtual_line_pre"].set_visible(len(pre_x) > 0)

            # Update post-trim line
            elevation_features["virtual_line_post"].set_xdata(post_x)
            elevation_features["virtual_line_post"].set_ydata(post_y)
            elevation_features["virtual_line_post"].set_visible(len(post_x) > 0)

            # Hide the old inactive line if it exists
            if "virtual_line_inactive" in elevation_features:
                elevation_features["virtual_line_inactive"].set_visible(False)

            # Calculate residuals (full profile)
            full_residuals = actual_elevation - full_virtual

            # Get the residual value at trim start point to offset residuals
            residual_offset = (
                full_residuals[start_idx] if start_idx < len(full_residuals) else 0
            )

            # Zero the residuals at trim start point
            adjusted_residuals = full_residuals - residual_offset

            # Split residuals into active and inactive regions
            active_res_x = distance_km[start_idx : end_idx + 1]
            active_res_y = adjusted_residuals[start_idx : end_idx + 1]

            # Split inactive regions for residuals
            pre_res_x = distance_km[:start_idx] if start_idx > 0 else []
            pre_res_y = adjusted_residuals[:start_idx] if start_idx > 0 else []

            post_res_x = (
                distance_km[end_idx + 1 :] if end_idx < len(distance_km) - 1 else []
            )
            post_res_y = (
                adjusted_residuals[end_idx + 1 :]
                if end_idx < len(adjusted_residuals) - 1
                else []
            )

            # Update residual lines
            elevation_features["residual_line"].set_xdata(active_res_x)
            elevation_features["residual_line"].set_ydata(active_res_y)
            elevation_features["residual_line"].set_visible(True)

            # FIXED: Handle pre and post residual regions as separate lines
            # Create pre-trim residual line if it doesn't exist
            if "residual_line_pre" not in elevation_features:
                (elevation_features["residual_line_pre"],) = ax_residual.plot(
                    [], [], "g-", linewidth=1.5, alpha=0.3, label="_nolegend_"
                )

            # Create post-trim residual line if it doesn't exist
            if "residual_line_post" not in elevation_features:
                (elevation_features["residual_line_post"],) = ax_residual.plot(
                    [], [], "g-", linewidth=1.5, alpha=0.3, label="_nolegend_"
                )

            # Update pre-trim residual line
            elevation_features["residual_line_pre"].set_xdata(pre_res_x)
            elevation_features["residual_line_pre"].set_ydata(pre_res_y)
            elevation_features["residual_line_pre"].set_visible(len(pre_res_x) > 0)

            # Update post-trim residual line
            elevation_features["residual_line_post"].set_xdata(post_res_x)
            elevation_features["residual_line_post"].set_ydata(post_res_y)
            elevation_features["residual_line_post"].set_visible(len(post_res_y) > 0)

            # Hide the old inactive residual line if it exists
            if "residual_line_inactive" in elevation_features:
                elevation_features["residual_line_inactive"].set_visible(False)

            # Update metrics text
            optimization_type_text = ""
            if target_elevation_gain is not None:
                trimmed_gain = virtual_profile[-1] - virtual_profile[0]
                optimization_type_text = f"Target: {target_elevation_gain:.1f}m, Actual: {trimmed_gain:.1f}m\n"

            metrics_text = (
                f"CdA: {cda:.4f} m²\n"
                f"Crr: {crr:.5f}\n"
                f"RMSE: {rmse:.2f} m\n"
                f"R²: {r2:.4f}\n"
                + optimization_type_text
                + f"Trim: {trim_start_slider.val:.0f}m start, {trim_end_slider.val:.0f}m end\n"
                + f"Active region metrics only"
            )
            elevation_features["metrics_box"].set_text(metrics_text)

            # Update plot limits
            ax_elevation.relim()
            ax_elevation.autoscale_view()
            ax_residual.relim()
            ax_residual.autoscale_view()

            # Update y-limits for grayed areas
            elevation_features["ylim"] = ax_elevation.get_ylim()
            elevation_features["res_ylim"] = ax_residual.get_ylim()

        except Exception as e:
            print(f"Error updating elevation plot: {str(e)}")
            import traceback

            traceback.print_exc()

    # Function to handle trimming slider updates
    def update_trimming(val):
        # Get current values
        trim_start = trim_start_slider.val
        trim_end = trim_end_slider.val

        # Calculate minimum distance needed for 30 seconds of data
        avg_speed = df["v"].mean()
        min_distance_for_30s = max(30 * avg_speed, 10)  # At least 10m or 30s of data

        # Check if current trim values would leave enough data
        remaining_distance = total_distance - trim_start - trim_end
        if remaining_distance < min_distance_for_30s:
            # If the user is adjusting the start slider
            if val == trim_start:
                # Adjust the end trim to ensure we have enough data
                new_trim_end = total_distance - trim_start - min_distance_for_30s
                # Only if new_trim_end is valid
                if new_trim_end >= 0:
                    trim_end = new_trim_end
                    # Update slider without triggering another update
                    trim_end_slider.eventson = False
                    trim_end_slider.set_val(trim_end)
                    trim_end_slider.eventson = True
                else:
                    # Adjust start instead
                    trim_start = total_distance - trim_end - min_distance_for_30s
                    # Update slider without triggering another update
                    trim_start_slider.eventson = False
                    trim_start_slider.set_val(trim_start)
                    trim_start_slider.eventson = True
            else:
                # User is adjusting end slider, adjust start instead
                new_trim_start = total_distance - trim_end - min_distance_for_30s
                # Only if new_trim_start is valid
                if new_trim_start >= 0:
                    trim_start = new_trim_start
                    # Update slider without triggering another update
                    trim_start_slider.eventson = False
                    trim_start_slider.set_val(trim_start)
                    trim_start_slider.eventson = True
                else:
                    # Adjust end instead
                    trim_end = total_distance - trim_start - min_distance_for_30s
                    # Update slider without triggering another update
                    trim_end_slider.eventson = False
                    trim_end_slider.set_val(trim_end)
                    trim_end_slider.eventson = True

        # Store in results
        results["trim_start"] = trim_start
        results["trim_end"] = trim_end

        # Update map and elevation plot trim indicators
        update_map_trim_points(trim_start, trim_end)
        update_elevation_trim_lines(trim_start, trim_end)

        # Update metrics text with current trim values
        if elevation_features["metrics_box"] is not None:
            current_text = elevation_features["metrics_box"].get_text()
            if "Trim:" in current_text:
                # Replace the trim line
                lines = current_text.split("\n")
                for i, line in enumerate(lines):
                    if line.startswith("Trim:"):
                        lines[i] = f"Trim: {trim_start:.0f}m start, {trim_end:.0f}m end"
                elevation_features["metrics_box"].set_text("\n".join(lines))

        # Re-trim if we have already optimized
        if results["current_virtual_profile"] is not None:
            # Find indices corresponding to the trim distances
            start_idx = 0
            end_idx = len(distance) - 1

            for i, dist in enumerate(distance):
                if dist >= trim_start:
                    start_idx = i
                    break

            for i in range(len(distance) - 1, -1, -1):
                if distance[i] <= (total_distance - trim_end):
                    end_idx = i
                    break

            # Trim the data
            trimmed_df = df.iloc[start_idx : end_idx + 1].copy()
            trimmed_distance = distance[start_idx : end_idx + 1] - distance[start_idx]
            trimmed_elevation = actual_elevation[start_idx : end_idx + 1]

            # Store the trimmed data for later use
            results["trimmed_df"] = trimmed_df
            results["trimmed_distance"] = trimmed_distance
            results["trimmed_elevation"] = trimmed_elevation

            # Check if we have enough data points after trimming
            if len(trimmed_df) < 10:
                elevation_features["metrics_box"].set_text(
                    "Error: Not enough data points after trimming\n"
                    "Adjust trim sliders and try again"
                )
                fig.canvas.draw_idle()
                return

            # Update plots with current optimized parameters but new trimming
            current_cda = cda_slider.val
            current_crr = crr_slider.val

            # Calculate virtual elevation with current parameters
            ve_changes = delta_ve(
                config, cda=current_cda, crr=current_crr, df=trimmed_df
            )

            # Build virtual elevation profile
            virtual_profile = calculate_virtual_profile(
                ve_changes, trimmed_elevation, lap_column, trimmed_df
            )

            # Calculate stats
            rmse = np.sqrt(np.mean((virtual_profile - trimmed_elevation) ** 2))
            r2 = pearsonr(virtual_profile, trimmed_elevation)[0] ** 2

            # Store results
            results["rmse"] = rmse
            results["r2"] = r2
            results["current_virtual_profile"] = virtual_profile

            # Update plots with the new virtual profile
            update_elevation_plot(
                trimmed_distance,
                trimmed_elevation,
                virtual_profile,
                current_cda,
                current_crr,
                rmse,
                r2,
            )

        # Redraw
        fig.canvas.draw_idle()

    # Function to handle parameter slider updates
    def update_parameters(val):
        # Only update if we're not in the middle of optimization
        if results["optimizing"]:
            return

        # Get current values
        cda = cda_slider.val
        crr = crr_slider.val

        # Store in results
        results["cda"] = cda
        results["crr"] = crr

        # Skip if we don't have trimmed data yet
        if results["trimmed_df"] is None:
            return

        trimmed_df = results["trimmed_df"]
        trimmed_distance = results["trimmed_distance"]
        trimmed_elevation = results["trimmed_elevation"]

        # Check if we have enough data points
        if len(trimmed_df) < 10:
            return

        # Recalculate dt if time data is available
        if "timestamp" in trimmed_df.columns:
            dt_values = trimmed_df["timestamp"].diff().dt.total_seconds()
            avg_dt = dt_values[1:].mean()  # skip first row which is NaN
            adjusted_dt = avg_dt if not np.isnan(avg_dt) else config.dt
            config.dt = adjusted_dt

        # Calculate acceleration if needed
        if "a" not in trimmed_df.columns:
            from core.calculations import accel_calc

            trimmed_df["a"] = accel_calc(trimmed_df["v"].values, adjusted_dt)

        # Calculate virtual elevation with current parameters
        ve_changes = delta_ve(config, cda=cda, crr=crr, df=trimmed_df)

        # Build virtual elevation profile
        virtual_profile = calculate_virtual_profile(
            ve_changes, trimmed_elevation, lap_column, trimmed_df
        )

        # Calculate stats
        rmse = np.sqrt(np.mean((virtual_profile - trimmed_elevation) ** 2))
        r2 = pearsonr(virtual_profile, trimmed_elevation)[0] ** 2

        # Store results
        results["rmse"] = rmse
        results["r2"] = r2
        results["current_virtual_profile"] = virtual_profile

        # Update plots
        update_elevation_plot(
            trimmed_distance, trimmed_elevation, virtual_profile, cda, crr, rmse, r2
        )

    # Button callbacks
    def on_skip(event):
        results["action"] = "skip"
        plt.close(fig)

    def on_optimize(event):
        if not results["optimizing"]:
            run_optimization()

    def on_save(event):
        results["action"] = "save"
        results["saved"] = True
        save_button.label.set_text("Saved!")
        save_button.color = "palegreen"
        fig.canvas.draw_idle()
        plt.close(fig)

    # Connect callbacks
    trim_start_slider.on_changed(update_trimming)
    trim_end_slider.on_changed(update_trimming)
    cda_slider.on_changed(update_parameters)
    crr_slider.on_changed(update_parameters)

    skip_button.on_clicked(on_skip)
    optimize_button.on_clicked(on_optimize)
    save_button.on_clicked(on_save)

    # Initialize the plot components
    map_initialized = initialize_map()
    initialize_elevation_plot()

    # Set initial trim lines
    update_trimming(None)

    # Save a screenshot if requested
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    # Run initial optimization
    run_optimization()

    # Show the plot (blocks until closed)
    plt.show()

    # Return results
    return (
        results["action"],
        results["trim_start"],
        results["trim_end"],
        results["cda"],
        results["crr"],
        results["rmse"],
        results["r2"],
    )
