import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider, Button
from scipy.stats import pearsonr

from core.calculations import calculate_distance, delta_ve
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
    kg=None,
    rho=None,
    dt=1,
    eta=0.98,
    vw=0,
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
    from matplotlib.widgets import Slider, Button
    import numpy as np
    from scipy.stats import pearsonr

    plt.ioff()

    if kg is None or rho is None:
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
    initial_ve_changes = delta_ve(
        cda=initial_cda, crr=initial_crr, df=df, vw=vw, kg=kg, rho=rho, dt=dt, eta=eta
    )
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
        ve_changes = delta_ve(
            cda=cda, crr=crr, df=df, vw=vw, kg=kg, rho=rho, dt=dt, eta=eta
        )
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
        import matplotlib.pyplot as plt
        from matplotlib.widgets import Slider, Button
        import numpy as np
        import contextily as ctx
        import geopandas as gpd
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
