"""
Restructured visualization module with component-based architecture
"""

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button, Slider
from scipy.stats import pearsonr

from core.calculations import calculate_distance, delta_ve
from core.config import VirtualElevationConfig
from core.optimization import calculate_virtual_profile

plt.style.use("fivethirtyeight")


class EventEmitter:
    """Simple event system to allow components to communicate"""

    def __init__(self):
        self._subscribers = {}

    def subscribe(self, event_name, callback):
        """Subscribe to an event with a callback function"""
        if event_name not in self._subscribers:
            self._subscribers[event_name] = []
        self._subscribers[event_name].append(callback)

    def emit(self, event_name, *args, **kwargs):
        """Emit an event with arguments"""
        if event_name in self._subscribers:
            for callback in self._subscribers[event_name]:
                callback(*args, **kwargs)


class BaseComponent:
    """Base component class with common functionality"""

    def __init__(self, parent):
        self.parent = parent
        self.ax = None
        self.visible = True

    def set_axis(self, ax):
        """Set the matplotlib axis for this component"""
        self.ax = ax

    def update(self):
        """Update the component (to be implemented by subclasses)"""
        pass

    def clear(self):
        """Clear all artists from the axis"""
        if self.ax:
            self.ax.cla()

    def show(self):
        """Show the component"""
        if self.ax:
            self.ax.set_visible(True)
        self.visible = True

    def hide(self):
        """Hide the component"""
        if self.ax:
            self.ax.set_visible(False)
        self.visible = False


class MapComponent(BaseComponent):
    """Component for displaying the route map"""

    def __init__(self, parent):
        super().__init__(parent)
        self.start_point = None
        self.end_point = None
        self.route_line = None
        self.map_initialized = False

    # Update the initialize method in the MapComponent class to color the route by wind effect

    def initialize(self):
        """Initialize the map with the route colored by wind direction effect"""
        try:
            # Only import these if needed
            import contextily as ctx
            import geopandas as gpd
            import matplotlib.pyplot as plt
            import numpy as np
            from matplotlib.colors import Normalize
            from matplotlib.cm import ScalarMappable
            from matplotlib.patches import Patch
            from shapely.geometry import LineString, Point

            df = self.parent.df
            lap_column = self.parent.lap_column
            config = self.parent.config

            # Check if we have GPS data
            if "latitude" not in df.columns or "longitude" not in df.columns:
                self.ax.text(
                    0.5,
                    0.5,
                    "Map view not available\n(missing GPS data)",
                    ha="center",
                    va="center",
                    transform=self.ax.transAxes,
                )
                return False

            # Check if we have wind direction data
            has_wind_direction = (
                hasattr(config, "wind_direction")
                and config.wind_direction is not None
                and config.vw > 0
            )

            # Calculate effective wind if we have wind direction
            effective_wind = None
            if has_wind_direction:
                from core.calculations import calculate_effective_wind

                effective_wind = calculate_effective_wind(df, config)

            # Create line geometries for each lap or segment
            if lap_column is not None and lap_column in df.columns:
                lap_numbers = df[lap_column].values
                unique_laps = sorted(np.unique(lap_numbers))

                # For wind effect coloring, we'll need to plot each segment individually
                # rather than whole laps, if wind direction is available
                if has_wind_direction:
                    for lap in unique_laps:
                        lap_mask = lap_numbers == lap
                        lap_indices = np.where(lap_mask)[0]

                        if len(lap_indices) < 2:
                            continue

                        # Plot each segment within the lap with appropriate color
                        for i in range(len(lap_indices) - 1):
                            idx1 = lap_indices[i]
                            idx2 = lap_indices[i + 1]

                            if effective_wind is not None:
                                # Use cool-warm colormap (blue for tailwind, red for headwind)
                                # Normalize wind to [-wind_speed, wind_speed] range
                                norm = Normalize(vmin=-config.vw, vmax=config.vw)
                                cmap = plt.cm.coolwarm
                                color = cmap(norm(effective_wind[idx1]))
                            else:
                                color = "blue"

                            # Create a line for this segment
                            x1, y1 = (
                                df["longitude"].iloc[idx1],
                                df["latitude"].iloc[idx1],
                            )
                            x2, y2 = (
                                df["longitude"].iloc[idx2],
                                df["latitude"].iloc[idx2],
                            )
                            line = LineString([(x1, y1), (x2, y2)])

                            # Create GeoDataFrame and plot
                            gdf_segment = gpd.GeoDataFrame(
                                geometry=[line], crs="EPSG:4326"
                            ).to_crs(epsg=3857)
                            gdf_segment.plot(
                                ax=self.ax, color=color, linewidth=3, alpha=0.8
                            )
                else:
                    # Original code for lap lines without wind color
                    geometries = []
                    for lap in unique_laps:
                        lap_mask = lap_numbers == lap
                        lap_points = [
                            Point(lon, lat)
                            for lon, lat in zip(
                                df.loc[lap_mask, "longitude"],
                                df.loc[lap_mask, "latitude"],
                            )
                        ]
                        if len(lap_points) > 1:
                            geometries.append(LineString(lap_points))

                    # Create GeoDataFrame for all lap lines
                    gdf_line = gpd.GeoDataFrame(
                        geometry=geometries, crs="EPSG:4326"
                    ).to_crs(epsg=3857)
                    # Plot original route
                    gdf_line.plot(ax=self.ax, color="blue", linewidth=2, alpha=0.5)
            else:
                # For a single route (no laps)
                if has_wind_direction:
                    # Plot each segment with appropriate color
                    for i in range(len(df) - 1):
                        if effective_wind is not None:
                            # Use cool-warm colormap (blue for tailwind, red for headwind)
                            norm = Normalize(vmin=-config.vw, vmax=config.vw)
                            cmap = plt.cm.coolwarm
                            color = cmap(norm(effective_wind[i]))
                        else:
                            color = "blue"

                        # Create a line for this segment
                        x1, y1 = df["longitude"].iloc[i], df["latitude"].iloc[i]
                        x2, y2 = df["longitude"].iloc[i + 1], df["latitude"].iloc[i + 1]
                        line = LineString([(x1, y1), (x2, y2)])

                        # Create GeoDataFrame and plot
                        gdf_segment = gpd.GeoDataFrame(
                            geometry=[line], crs="EPSG:4326"
                        ).to_crs(epsg=3857)
                        gdf_segment.plot(
                            ax=self.ax, color=color, linewidth=3, alpha=0.8
                        )

                    # Add a colorbar if we have wind direction
                    if effective_wind is not None:
                        sm = ScalarMappable(cmap=cmap, norm=norm)
                        sm.set_array([])
                        cbar = plt.colorbar(
                            sm,
                            ax=self.ax,
                            orientation="horizontal",
                            pad=0.05,
                            shrink=0.5,
                        )
                        cbar.set_label("Wind Effect (m/s)", fontsize=10)
                        cbar.ax.text(
                            0.25,
                            0.5,
                            "Tailwind",
                            transform=cbar.ax.transAxes,
                            ha="center",
                            va="center",
                            fontsize=8,
                            color="blue",
                        )
                        cbar.ax.text(
                            0.75,
                            0.5,
                            "Headwind",
                            transform=cbar.ax.transAxes,
                            ha="center",
                            va="center",
                            fontsize=8,
                            color="red",
                        )
                else:
                    # Original code for single continuous route without wind color
                    points = [
                        Point(lon, lat)
                        for lon, lat in zip(df["longitude"], df["latitude"])
                    ]
                    route_line = LineString(points)
                    gdf_line = gpd.GeoDataFrame(
                        geometry=[route_line], crs="EPSG:4326"
                    ).to_crs(epsg=3857)
                    # Plot original route
                    gdf_line.plot(ax=self.ax, color="blue", linewidth=2, alpha=0.5)

            # Initial start and end points (0% and 100% of route)
            start_point = Point(df["longitude"].iloc[0], df["latitude"].iloc[0])
            end_point = Point(df["longitude"].iloc[-1], df["latitude"].iloc[-1])

            # Convert to GeoDataFrame and plot start/end points
            gdf_points = gpd.GeoDataFrame(
                geometry=[start_point, end_point],
                data={"type": ["start", "end"]},
                crs="EPSG:4326",
            ).to_crs(epsg=3857)

            # Plot start and end points
            self.start_point = self.ax.scatter(
                gdf_points[gdf_points["type"] == "start"].geometry.x,
                gdf_points[gdf_points["type"] == "start"].geometry.y,
                color="green",
                s=100,
                zorder=10,
            )

            self.end_point = self.ax.scatter(
                gdf_points[gdf_points["type"] == "end"].geometry.x,
                gdf_points[gdf_points["type"] == "end"].geometry.y,
                color="red",
                s=100,
                zorder=10,
            )

            # Add OpenStreetMap background
            bounds = self.ax.get_xlim() + self.ax.get_ylim()
            max_dimension = max(bounds[1] - bounds[0], bounds[3] - bounds[2])

            # Determine appropriate zoom level
            zoom = 12  # Default
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

            try:
                ctx.add_basemap(
                    self.ax, source=ctx.providers.OpenStreetMap.Mapnik, zoom=zoom
                )
            except Exception as e:
                print(f"Error adding map background: {str(e)}")

            # Remove axis labels
            self.ax.set_axis_off()

            # Add title
            lap_title = (
                f"Lap {self.parent.lap_num}"
                if self.parent.lap_num > 0
                else "Combined Laps"
            )

            title = f"{lap_title} - Adjust Start and End Trim Points"

            # Add wind info to title if available
            if has_wind_direction:
                title += f"\nWind: {config.vw:.1f} m/s from {config.wind_direction:.0f}° (Red=Headwind, Blue=Tailwind)"

            self.ax.set_title(title, fontsize=12)

            # Add wind arrow
            self.add_wind_arrow()

            self.map_initialized = True
            return True

        except ImportError:
            self.ax.text(
                0.5,
                0.5,
                "Map libraries not available.\nInstall with:\npip install contextily geopandas shapely",
                ha="center",
                va="center",
                transform=self.ax.transAxes,
            )
            return False
        except Exception as e:
            import traceback

            traceback.print_exc()
            self.ax.text(
                0.5,
                0.5,
                f"Error initializing map: {str(e)}",
                ha="center",
                va="center",
                transform=self.ax.transAxes,
            )
            return False

    def add_wind_arrow(self):
        """Add a wind direction arrow to the map."""
        # Check if we have wind direction in the config
        if (
            not hasattr(self.parent.config, "wind_direction")
            or self.parent.config.wind_direction is None
        ):
            return

        # Get wind parameters
        wind_speed = self.parent.config.vw
        wind_direction = self.parent.config.wind_direction

        # Skip if no wind
        if wind_speed == 0:
            return

        try:
            import numpy as np
            import matplotlib.patheffects as path_effects

            # Convert from meteorological to mathematical angles
            # Meteorological: 0° = North, 90° = East, 180° = South, 270° = West
            # Mathematical: 0° = East, 90° = North, 180° = West, 270° = South
            math_angle = (90 - wind_direction) % 360
            math_angle_rad = np.radians(math_angle)

            # Position in axes coordinates (top right corner)
            arrow_x = 0.95
            arrow_y = 0.95

            # Set arrow length based on wind speed (normalized to some extent)
            base_length = 0.5  # Base arrow length in axes coordinates
            arrow_length = min(base_length, base_length * (wind_speed / 10))

            # Calculate arrow starting point (wind source)
            dx = arrow_length * np.cos(math_angle_rad)
            dy = arrow_length * np.sin(math_angle_rad)
            start_x = arrow_x + dx
            start_y = arrow_y + dy

            # Create arrow pointing FROM the wind source TO the destination
            arrow = self.ax.annotate(
                "",
                xy=(arrow_x, arrow_y),  # Arrow end (destination)
                xytext=(start_x, start_y),  # Arrow start (wind source)
                xycoords="axes fraction",
                textcoords="axes fraction",
                arrowprops=dict(
                    arrowstyle="->",
                    lw=2,
                    color="blue",
                    shrinkA=5,
                    shrinkB=5,
                    path_effects=[
                        path_effects.withStroke(linewidth=4, foreground="white")
                    ],
                ),
            )

            # Add wind speed label
            text = self.ax.text(
                arrow_x,
                arrow_y + 0.05,
                f"Wind: {wind_speed:.1f} m/s from {wind_direction:.0f}°",
                transform=self.ax.transAxes,
                ha="right",
                va="bottom",
                fontsize=10,
                color="blue",
                fontweight="bold",
                path_effects=[path_effects.withStroke(linewidth=3, foreground="white")],
            )

        except Exception as e:
            print(f"Error adding wind arrow: {e}")

    def update_trim_points(self, start_idx, end_idx):
        """Update the map trim points based on dataframe indices"""
        if not self.map_initialized:
            return

        try:
            # Only import these if needed
            import geopandas as gpd
            from shapely.geometry import Point

            df = self.parent.df

            # Update start point
            start_lon, start_lat = (
                df["longitude"].iloc[start_idx],
                df["latitude"].iloc[start_idx],
            )
            start_point = Point(start_lon, start_lat)
            gdf_start = gpd.GeoDataFrame(
                geometry=[start_point], crs="EPSG:4326"
            ).to_crs(epsg=3857)

            # Update end point
            end_lon, end_lat = (
                df["longitude"].iloc[end_idx],
                df["latitude"].iloc[end_idx],
            )
            end_point = Point(end_lon, end_lat)
            gdf_end = gpd.GeoDataFrame(geometry=[end_point], crs="EPSG:4326").to_crs(
                epsg=3857
            )

            # Update the scatter points
            self.start_point.set_offsets(
                [[gdf_start.geometry.x.iloc[0], gdf_start.geometry.y.iloc[0]]]
            )

            self.end_point.set_offsets(
                [[gdf_end.geometry.x.iloc[0], gdf_end.geometry.y.iloc[0]]]
            )

        except Exception as e:
            print(f"Error updating map trim points: {str(e)}")


class ElevationProfileComponent(BaseComponent):
    """Component for displaying the elevation profile"""

    def __init__(self, parent):
        super().__init__(parent)
        self.actual_line = None
        self.virtual_line = None
        self.virtual_line_pre = None
        self.virtual_line_post = None
        self.trim_start_line = None
        self.trim_end_line = None
        self.grayed_start = None
        self.grayed_end = None
        self.ylim = (0, 1)  # Will be updated later

    def initialize(self):
        """Initialize the elevation profile plot"""
        df = self.parent.df
        distance = self.parent.distance
        actual_elevation = self.parent.actual_elevation
        lap_column = self.parent.lap_column

        # Convert distance to km for plotting
        distance_km = distance / 1000

        # Setup plot properties
        lap_title = (
            f"Lap {self.parent.lap_num}" if self.parent.lap_num > 0 else "Combined Laps"
        )
        self.ax.set_ylabel("Elevation (m)", fontsize=12)
        self.ax.set_title(f"{lap_title} - Elevation Profiles", fontsize=14)
        self.ax.grid(True, linestyle="--", alpha=0.7)

        # Plot actual elevation
        (self.actual_line,) = self.ax.plot(
            distance_km, actual_elevation, "b-", linewidth=2, label="Actual Elevation"
        )

        # Initialize virtual elevation line (with NaN to make it invisible initially)
        virtual_data = np.full_like(actual_elevation, np.nan)
        (self.virtual_line,) = self.ax.plot(
            distance_km, virtual_data, "r-", linewidth=2, label="Virtual Elevation"
        )

        # Create inactive region lines (pre and post trim)
        (self.virtual_line_pre,) = self.ax.plot(
            [], [], "r-", linewidth=2, alpha=0.3, label="_nolegend_"
        )

        (self.virtual_line_post,) = self.ax.plot(
            [], [], "r-", linewidth=2, alpha=0.3, label="_nolegend_"
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
                    self.ax.axvline(
                        x=distance_km[transition_idx],
                        color="green",
                        linestyle="--",
                        alpha=0.7,
                    )
                    line_label = f"Lap {unique_laps[i]} (Reset)"
                else:
                    # Consecutive laps: add a blue dotted line
                    self.ax.axvline(
                        x=distance_km[transition_idx],
                        color="blue",
                        linestyle=":",
                        alpha=0.5,
                    )
                    line_label = f"Lap {unique_laps[i]}"

                # Add lap number labels
                if transition_idx > 0:
                    self.ax.text(
                        distance_km[transition_idx] + 0.1,
                        np.min(actual_elevation)
                        + 0.1 * (np.max(actual_elevation) - np.min(actual_elevation)),
                        line_label,
                        rotation=90,
                        va="bottom",
                    )

        # Initialize trim lines
        self.trim_start_line = self.ax.axvline(
            x=0, color="green", linewidth=1.5, linestyle="--", visible=False
        )

        self.trim_end_line = self.ax.axvline(
            x=distance_km[-1], color="red", linewidth=1.5, linestyle="--", visible=False
        )

        # Set initial y limits for later use with grayed areas
        self.ylim = (np.min(actual_elevation) - 10, np.max(actual_elevation) + 10)

        # Create legend but return handles and labels
        # self.ax.legend(loc="upper right")
        return self.ax.get_legend_handles_labels()

    def update_trim_lines(self, trim_start, trim_end):
        """Update the trim line positions and grayed areas"""
        distance = self.parent.distance
        total_distance = distance[-1]

        # Convert to km for the plot
        start_km = trim_start / 1000
        end_km = (total_distance - trim_end) / 1000

        # Update trim lines
        self.trim_start_line.set_xdata([start_km, start_km])
        self.trim_start_line.set_visible(True)

        self.trim_end_line.set_xdata([end_km, end_km])
        self.trim_end_line.set_visible(True)

        # Remove old grayed areas
        if self.grayed_start is not None:
            self.grayed_start.remove()
            self.grayed_start = None

        if self.grayed_end is not None:
            self.grayed_end.remove()
            self.grayed_end = None

        # Add new grayed areas
        y_min, y_max = self.ylim

        if start_km > 0:
            start_x = np.linspace(0, start_km, 100)
            start_y_min = np.ones_like(start_x) * y_min
            start_y_max = np.ones_like(start_x) * y_max

            self.grayed_start = self.ax.fill_between(
                start_x, start_y_min, start_y_max, color="gray", alpha=0.15
            )

        if end_km < total_distance / 1000:
            end_x = np.linspace(end_km, total_distance / 1000, 100)
            end_y_min = np.ones_like(end_x) * y_min
            end_y_max = np.ones_like(end_x) * y_max

            self.grayed_end = self.ax.fill_between(
                end_x, end_y_min, end_y_max, color="gray", alpha=0.15
            )

    def update_elevation_data(self, full_virtual, start_idx, end_idx):
        """Update the elevation profile with new data"""
        df = self.parent.df
        distance = self.parent.distance
        total_distance = distance[-1]

        # Convert to km for plotting
        distance_km = distance / 1000

        # Split virtual elevation into active and inactive regions
        active_x = distance_km[start_idx : end_idx + 1]
        active_y = full_virtual[start_idx : end_idx + 1]

        # Split inactive regions
        pre_x = distance_km[:start_idx] if start_idx > 0 else []
        pre_y = full_virtual[:start_idx] if start_idx > 0 else []

        post_x = distance_km[end_idx + 1 :] if end_idx < len(distance_km) - 1 else []
        post_y = full_virtual[end_idx + 1 :] if end_idx < len(full_virtual) - 1 else []

        # Update active region line
        self.virtual_line.set_xdata(active_x)
        self.virtual_line.set_ydata(active_y)
        self.virtual_line.set_visible(True)

        # Update pre-trim line
        self.virtual_line_pre.set_xdata(pre_x)
        self.virtual_line_pre.set_ydata(pre_y)
        self.virtual_line_pre.set_visible(len(pre_x) > 0)

        # Update post-trim line
        self.virtual_line_post.set_xdata(post_x)
        self.virtual_line_post.set_ydata(post_y)
        self.virtual_line_post.set_visible(len(post_x) > 0)

        # Update y limits for grayed area calculations
        self.ax.relim()
        self.ax.autoscale_view()
        self.ylim = self.ax.get_ylim()


class ResidualPlotComponent(BaseComponent):
    """Component for displaying the residual plot"""

    def __init__(self, parent):
        super().__init__(parent)
        self.residual_line = None
        self.residual_line_pre = None
        self.residual_line_post = None
        self.trim_start_line = None
        self.trim_end_line = None
        self.grayed_start = None
        self.grayed_end = None
        self.ylim = (-10, 10)  # Will be updated later

    def initialize(self):
        """Initialize the residual plot"""
        df = self.parent.df
        distance = self.parent.distance
        lap_column = self.parent.lap_column

        # Convert distance to km for plotting
        distance_km = distance / 1000

        # Setup plot properties
        self.ax.set_xlabel("Distance (km)", fontsize=12)
        self.ax.set_ylabel("Residual (m)", fontsize=12)
        self.ax.set_title("Elevation Difference (Actual - Virtual)", fontsize=12)
        self.ax.grid(True, linestyle="--", alpha=0.7)

        # Add zero line
        self.ax.axhline(y=0, color="k", linestyle="-", alpha=0.3)

        # Initialize residual lines with NaN
        residual_data = np.full(len(distance_km), np.nan)

        # Active region line
        (self.residual_line,) = self.ax.plot(
            distance_km, residual_data, "g-", linewidth=1.5, label="Residual"
        )

        # Inactive region lines
        (self.residual_line_pre,) = self.ax.plot(
            [], [], "g-", linewidth=1.5, alpha=0.3, label="_nolegend_"
        )

        (self.residual_line_post,) = self.ax.plot(
            [], [], "g-", linewidth=1.5, alpha=0.3, label="_nolegend_"
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
                    self.ax.axvline(
                        x=distance_km[transition_idx],
                        color="green",
                        linestyle="--",
                        alpha=0.7,
                    )
                else:
                    # Consecutive laps: add a blue dotted line
                    self.ax.axvline(
                        x=distance_km[transition_idx],
                        color="blue",
                        linestyle=":",
                        alpha=0.5,
                    )

        # Initialize trim lines
        self.trim_start_line = self.ax.axvline(
            x=0, color="green", linewidth=1.5, linestyle="--", visible=False
        )

        self.trim_end_line = self.ax.axvline(
            x=distance_km[-1], color="red", linewidth=1.5, linestyle="--", visible=False
        )

    def update_trim_lines(self, trim_start, trim_end):
        """Update the trim line positions and grayed areas"""
        distance = self.parent.distance
        total_distance = distance[-1]

        # Convert to km for the plot
        start_km = trim_start / 1000
        end_km = (total_distance - trim_end) / 1000

        # Update trim lines
        self.trim_start_line.set_xdata([start_km, start_km])
        self.trim_start_line.set_visible(True)

        self.trim_end_line.set_xdata([end_km, end_km])
        self.trim_end_line.set_visible(True)

        # Remove old grayed areas
        if self.grayed_start is not None:
            self.grayed_start.remove()
            self.grayed_start = None

        if self.grayed_end is not None:
            self.grayed_end.remove()
            self.grayed_end = None

        # Add new grayed areas
        y_min, y_max = self.ylim

        if start_km > 0:
            start_x = np.linspace(0, start_km, 100)
            start_y_min = np.ones_like(start_x) * y_min
            start_y_max = np.ones_like(start_x) * y_max

            self.grayed_start = self.ax.fill_between(
                start_x, start_y_min, start_y_max, color="gray", alpha=0.15
            )

        if end_km < total_distance / 1000:
            end_x = np.linspace(end_km, total_distance / 1000, 100)
            end_y_min = np.ones_like(end_x) * y_min
            end_y_max = np.ones_like(end_x) * y_max

            self.grayed_end = self.ax.fill_between(
                end_x, end_y_min, end_y_max, color="gray", alpha=0.15
            )

    def update_residual_data(self, actual_elevation, full_virtual, start_idx, end_idx):
        """Update the residual plot with new data"""
        distance = self.parent.distance

        # Convert to km for plotting
        distance_km = distance / 1000

        # Calculate full residuals
        full_residuals = actual_elevation - full_virtual

        # Get the residual value at trim start point to offset residuals
        residual_offset = (
            full_residuals[start_idx] if start_idx < len(full_residuals) else 0
        )

        # Zero the residuals at trim start point
        adjusted_residuals = full_residuals - residual_offset

        # Split residuals into active and inactive regions
        active_x = distance_km[start_idx : end_idx + 1]
        active_y = adjusted_residuals[start_idx : end_idx + 1]

        # Split inactive regions
        pre_x = distance_km[:start_idx] if start_idx > 0 else []
        pre_y = adjusted_residuals[:start_idx] if start_idx > 0 else []

        post_x = distance_km[end_idx + 1 :] if end_idx < len(distance_km) - 1 else []
        post_y = (
            adjusted_residuals[end_idx + 1 :]
            if end_idx < len(adjusted_residuals) - 1
            else []
        )

        # Update active region line
        self.residual_line.set_xdata(active_x)
        self.residual_line.set_ydata(active_y)
        self.residual_line.set_visible(True)

        # Update pre-trim line
        self.residual_line_pre.set_xdata(pre_x)
        self.residual_line_pre.set_ydata(pre_y)
        self.residual_line_pre.set_visible(len(pre_x) > 0)

        # Update post-trim line
        self.residual_line_post.set_xdata(post_x)
        self.residual_line_post.set_ydata(post_y)
        self.residual_line_post.set_visible(len(post_x) > 0)

        # Update y limits for grayed area calculations
        self.ax.relim()
        self.ax.autoscale_view()
        self.ylim = self.ax.get_ylim()


class MetricsComponent(BaseComponent):
    """Component for displaying metrics and statistics"""

    def __init__(self, parent):
        super().__init__(parent)
        self.metrics_text = None

    def initialize(self):
        """Initialize the metrics display"""
        # Hide axis
        self.ax.axis("off")

        # Add text box for metrics
        props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)
        self.metrics_text = self.ax.text(
            0.95,
            0.50,
            "Press 'Optimize' to start analysis",
            transform=self.ax.transAxes,
            fontsize=10,
            verticalalignment="top",
            horizontalalignment="right",
            bbox=props,
        )

    def update_metrics(
        self, cda, crr, rmse, r2, trim_start, trim_end, virtual_profile=None
    ):
        """Update the metrics display"""
        target_elevation_gain = self.parent.target_elevation_gain

        # Format elevation gain info if needed
        optimization_type_text = ""
        if target_elevation_gain is not None and virtual_profile is not None:
            trimmed_gain = virtual_profile[-1] - virtual_profile[0]
            optimization_type_text = (
                f"Target: {target_elevation_gain:.1f}m, Actual: {trimmed_gain:.1f}m\n"
            )

        # Format metrics text
        metrics_text = (
            f"CdA: {cda:.4f} m²\n"
            f"Crr: {crr:.5f}\n"
            f"RMSE: {rmse:.2f} m\n"
            f"R²: {r2:.4f}\n"
            + optimization_type_text
            + f"Trim: {trim_start:.0f}m start, {trim_end:.0f}m end\n"
            + f"Active region metrics only"
        )

        self.metrics_text.set_text(metrics_text)


class ControlComponent(BaseComponent):
    """Component for sliders and buttons"""

    def __init__(self, parent):
        super().__init__(parent)
        self.trim_start_slider = None
        self.trim_end_slider = None
        self.cda_slider = None
        self.crr_slider = None
        self.skip_button = None
        self.optimize_button = None
        self.save_button = None

    def initialize(self, fig, trim_start, trim_end, cda, crr):
        """Initialize the control elements"""
        distance = self.parent.distance
        total_distance = distance[-1]

        # Calculate minimum distance needed for 30 seconds of data
        avg_speed = self.parent.df["v"].mean()
        min_distance_for_30s = max(30 * avg_speed, 10)  # At least 10m or 30s of data

        # Set maximum trim to 95% of total distance
        max_trim = total_distance * 0.45  # Max 45% from each end

        # Cap initial values
        trim_start = min(trim_start, max_trim)
        trim_end = min(trim_end, max_trim)

        # Get CDA and CRR ranges
        cda_range = self.parent.cda_range
        crr_range = self.parent.crr_range

        # Create slider axes
        ax_trim_start = plt.axes([0.15, 0.20, 0.70, 0.02])
        ax_trim_end = plt.axes([0.15, 0.16, 0.70, 0.02])
        ax_cda_slider = plt.axes([0.15, 0.12, 0.70, 0.02])
        ax_crr_slider = plt.axes([0.15, 0.08, 0.70, 0.02])

        # Create button axes
        ax_skip = plt.axes([0.15, 0.03, 0.20, 0.04])
        ax_optimize = plt.axes([0.40, 0.03, 0.20, 0.04])
        ax_save = plt.axes([0.65, 0.03, 0.20, 0.04])

        # Create sliders
        self.trim_start_slider = Slider(
            ax=ax_trim_start,
            label="Trim Start (m)",
            valmin=0,
            valmax=max_trim,
            valinit=trim_start,
            valfmt="%.0f",
        )

        self.trim_end_slider = Slider(
            ax=ax_trim_end,
            label="Trim End (m)",
            valmin=0,
            valmax=max_trim,
            valinit=trim_end,
            valfmt="%.0f",
        )

        self.cda_slider = Slider(
            ax=ax_cda_slider,
            label="CdA (m²)",
            valmin=cda_range[0],
            valmax=cda_range[1],
            valinit=cda,
            valfmt="%.4f",
        )

        self.crr_slider = Slider(
            ax=ax_crr_slider,
            label="Crr",
            valmin=crr_range[0],
            valmax=crr_range[1],
            valinit=crr,
            valfmt="%.5f",
        )

        # Create buttons
        self.skip_button = Button(
            ax_skip, "Skip Lap", color="lightsalmon", hovercolor="salmon"
        )

        self.optimize_button = Button(
            ax_optimize, "Optimize", color="lightblue", hovercolor="skyblue"
        )

        self.save_button = Button(
            ax_save, "Save Results", color="lightgreen", hovercolor="palegreen"
        )

        # Display optimization type
        target_text = (
            "Target Elevation Gain"
            if self.parent.target_elevation_gain is not None
            else "R²/RMSE"
        )
        if self.parent.target_elevation_gain is not None:
            target_text += f" ({self.parent.target_elevation_gain}m)"

        plt.figtext(0.5, 0.24, f"Optimization: {target_text}", ha="center", fontsize=10)

        # Connect callbacks
        self.trim_start_slider.on_changed(self.on_trim_changed)
        self.trim_end_slider.on_changed(self.on_trim_changed)
        self.cda_slider.on_changed(self.on_parameter_changed)
        self.crr_slider.on_changed(self.on_parameter_changed)

        self.skip_button.on_clicked(self.on_skip)
        self.optimize_button.on_clicked(self.on_optimize)
        self.save_button.on_clicked(self.on_save)

        return {
            "trim_start": self.trim_start_slider,
            "trim_end": self.trim_end_slider,
            "cda": self.cda_slider,
            "crr": self.crr_slider,
        }

    def on_trim_changed(self, val):
        """Handle trim slider changes"""
        self.parent.on_trim_changed(val)

    def on_parameter_changed(self, val):
        """Handle parameter slider changes"""
        self.parent.on_parameter_changed(val)

    def on_skip(self, event):
        """Handle skip button click"""
        self.parent.on_skip()

    def on_optimize(self, event):
        """Handle optimize button click"""
        if not self.parent.optimizing:
            self.optimize_button.label.set_text("Optimizing...")
            self.optimize_button.color = "yellow"
            plt.gcf().canvas.draw_idle()
            self.parent.on_optimize()

    def on_save(self, event):
        """Handle save button click"""
        self.save_button.label.set_text("Saved!")
        self.save_button.color = "palegreen"
        plt.gcf().canvas.draw_idle()
        self.parent.on_save()

    def get_values(self):
        """Get current slider values"""
        return {
            "trim_start": self.trim_start_slider.val,
            "trim_end": self.trim_end_slider.val,
            "cda": self.cda_slider.val,
            "crr": self.crr_slider.val,
        }


class VirtualElevationPlot:
    """Main visualization class that coordinates all components"""

    def __init__(
        self,
        df,
        actual_elevation,
        lap_num,
        config,
        initial_cda=0.3,
        initial_crr=0.005,
        initial_trim_start=0,
        initial_trim_end=0,
        cda_range=None,
        crr_range=None,
        distance=None,
        lap_column=None,
        target_elevation_gain=None,
        optimization_function=None,
    ):

        # Store input parameters
        self.df = df
        self.actual_elevation = actual_elevation
        self.lap_num = lap_num
        self.config = config
        self.initial_cda = initial_cda
        self.initial_crr = initial_crr
        self.initial_trim_start = initial_trim_start
        self.initial_trim_end = initial_trim_end
        self.lap_column = lap_column
        self.target_elevation_gain = target_elevation_gain
        self.optimization_function = optimization_function

        # Calculate distance if not provided
        if distance is None:
            self.distance = calculate_distance(df, config.dt)
        else:
            self.distance = distance

        # Set CDA and CRR ranges if not provided
        if cda_range is None:
            cda_min = max(0.05, initial_cda * 0.5)
            cda_max = min(0.8, initial_cda * 1.5)
            self.cda_range = (cda_min, cda_max)
        else:
            self.cda_range = cda_range

        if crr_range is None:
            crr_min = max(0.0005, initial_crr * 0.5)
            crr_max = min(0.02, initial_crr * 1.5)
            self.crr_range = (crr_min, crr_max)
        else:
            self.crr_range = crr_range

        # Initialize state variables
        self.trimmed_df = None
        self.trimmed_distance = None
        self.trimmed_elevation = None
        self.current_virtual_profile = None
        self.start_idx = 0
        self.end_idx = len(df) - 1
        self.rmse = None
        self.r2 = None
        self.optimizing = False
        self.saved = False
        self.action = "optimize"  # Default action

        # Create event system
        self.events = EventEmitter()

        # Create components
        self.map_component = MapComponent(self)
        self.elevation_component = ElevationProfileComponent(self)
        self.residual_component = ResidualPlotComponent(self)
        self.metrics_component = MetricsComponent(self)
        self.control_component = ControlComponent(self)

    def create_plot(self, save_path=None):
        """Create the interactive plot"""
        # Create figure and grid layout
        fig = plt.figure(figsize=(15, 10))

        # Create grid layout
        gs = gridspec.GridSpec(3, 2, height_ratios=[3, 2, 1], width_ratios=[4, 1])

        # Add space for controls at bottom
        plt.subplots_adjust(bottom=0.30)

        # Create and assign axes to components
        ax_map = fig.add_subplot(gs[0, 0])
        self.map_component.set_axis(ax_map)

        ax_legend = fig.add_subplot(gs[0, 1])
        self.metrics_component.set_axis(ax_legend)

        ax_elevation = fig.add_subplot(gs[1, :])
        self.elevation_component.set_axis(ax_elevation)

        ax_residual = fig.add_subplot(gs[2, :], sharex=ax_elevation)
        self.residual_component.set_axis(ax_residual)

        # Initialize all components
        self.map_component.initialize()
        handles, labels = self.elevation_component.initialize()
        self.residual_component.initialize()
        self.metrics_component.initialize()

        # Move the legend to the legend area
        if handles is not None and labels is not None:
            ax_legend.legend(handles, labels, loc="upper right")

        # Initialize the controls
        self.control_component.initialize(
            fig,
            self.initial_trim_start,
            self.initial_trim_end,
            self.initial_cda,
            self.initial_crr,
        )

        # Update trim lines based on initial values
        self.update_trim_indices(self.initial_trim_start, self.initial_trim_end)

        # Save screenshot if requested
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")

        # Run initial optimization
        self.on_optimize()

        return fig

    def update_trim_indices(self, trim_start, trim_end):
        """Calculate dataframe indices based on trim distances"""
        # Find indices corresponding to the trim distances
        start_idx = 0
        end_idx = len(self.distance) - 1

        for i, dist in enumerate(self.distance):
            if dist >= trim_start:
                start_idx = i
                break

        for i in range(len(self.distance) - 1, -1, -1):
            if self.distance[i] <= (self.distance[-1] - trim_end):
                end_idx = i
                break

        # Store the indices
        self.start_idx = start_idx
        self.end_idx = end_idx

        # Update components with the new trim values - only update visuals
        # This is a fast operation that just shows the trim lines
        self.elevation_component.update_trim_lines(trim_start, trim_end)
        self.residual_component.update_trim_lines(trim_start, trim_end)
        self.map_component.update_trim_points(start_idx, end_idx)

        # Fast redraw of just the trim lines
        plt.gcf().canvas.draw_idle()

    def on_trim_changed(self, val):
        """Handle changes to trim sliders"""
        if self.optimizing:
            return

        # Get current values
        values = self.control_component.get_values()
        trim_start = values["trim_start"]
        trim_end = values["trim_end"]

        # Ensure minimum trim length for valid data (30 seconds worth)
        avg_speed = self.df["v"].mean()
        min_distance_for_30s = max(30 * avg_speed, 10)  # At least 10m or 30s
        total_distance = self.distance[-1]

        # Check if there's enough data left after trimming
        remaining_distance = total_distance - trim_start - trim_end
        if remaining_distance < min_distance_for_30s:
            # Adjust trim values to ensure minimum length
            # Code to handle this case is simplified here
            pass

        # Update trim indices
        self.update_trim_indices(trim_start, trim_end)

        # If we already have a virtual profile, update it with new trim
        if self.current_virtual_profile is not None:
            # Extract trimmed data
            self.trimmed_df = self.df.iloc[self.start_idx : self.end_idx + 1].copy()
            self.trimmed_distance = (
                self.distance[self.start_idx : self.end_idx + 1]
                - self.distance[self.start_idx]
            )
            self.trimmed_elevation = self.actual_elevation[
                self.start_idx : self.end_idx + 1
            ]

            # Only recalculate if user has stopped sliding for a moment
            # This is a performance optimization
            self._schedule_update()

    def _schedule_update(self):
        """Schedule a delayed update to avoid excessive redrawing"""
        # Use a timer to delay the update
        if hasattr(self, "_update_timer") and self._update_timer is not None:
            self._update_timer.cancel()

        import threading

        self._update_timer = threading.Timer(0.2, self._do_delayed_update)
        self._update_timer.start()

    def _do_delayed_update(self):
        """Perform the delayed update"""
        self.recalculate_virtual_elevation()
        plt.gcf().canvas.draw_idle()

    def on_parameter_changed(self, val):
        """Handle changes to CdA and Crr sliders"""
        if self.optimizing or self.trimmed_df is None:
            return

        # Schedule update with delay for better performance
        self._schedule_update()

    def recalculate_virtual_elevation(self):
        """Recalculate virtual elevation with current parameters"""
        # Get current values
        values = self.control_component.get_values()
        cda = values["cda"]
        crr = values["crr"]

        # Calculate virtual elevation with current parameters
        ve_changes = delta_ve(self.config, cda=cda, crr=crr, df=self.trimmed_df)

        # Build virtual elevation profile
        virtual_profile = calculate_virtual_profile(
            ve_changes, self.trimmed_elevation, self.lap_column, self.trimmed_df
        )

        # Calculate stats
        rmse = np.sqrt(np.mean((virtual_profile - self.trimmed_elevation) ** 2))
        r2 = pearsonr(virtual_profile, self.trimmed_elevation)[0] ** 2

        # Store the results
        self.rmse = rmse
        self.r2 = r2
        self.current_virtual_profile = virtual_profile

        # Calculate full profile for the entire dataset - this is potentially expensive
        # We'll calculate this less frequently
        full_ve_changes = delta_ve(self.config, cda=cda, crr=crr, df=self.df)

        full_virtual = calculate_virtual_profile(
            full_ve_changes, self.actual_elevation, self.lap_column, self.df
        )

        # Update the plots
        self.elevation_component.update_elevation_data(
            full_virtual, self.start_idx, self.end_idx
        )

        self.residual_component.update_residual_data(
            self.actual_elevation, full_virtual, self.start_idx, self.end_idx
        )

        # Update metrics
        self.metrics_component.update_metrics(
            cda,
            crr,
            rmse,
            r2,
            values["trim_start"],
            values["trim_end"],
            self.current_virtual_profile,
        )

    def on_optimize(self):
        """Run the optimization"""
        self.optimizing = True

        try:
            # Get trim values
            values = self.control_component.get_values()
            trim_start = values["trim_start"]
            trim_end = values["trim_end"]

            # Update trim indices
            self.update_trim_indices(trim_start, trim_end)

            # Extract trimmed data
            self.trimmed_df = self.df.iloc[self.start_idx : self.end_idx + 1].copy()
            self.trimmed_distance = (
                self.distance[self.start_idx : self.end_idx + 1]
                - self.distance[self.start_idx]
            )
            self.trimmed_elevation = self.actual_elevation[
                self.start_idx : self.end_idx + 1
            ]

            # Check if we have enough data points
            if len(self.trimmed_df) < 10:
                self.metrics_component.metrics_text.set_text(
                    "Error: Not enough data points after trimming\n"
                    "Adjust trim sliders and try again"
                )
                plt.gcf().canvas.draw_idle()
                return

            # Calculate acceleration if needed
            if "a" not in self.trimmed_df.columns:
                from core.calculations import accel_calc

                self.trimmed_df["a"] = accel_calc(
                    self.trimmed_df["v"].values, self.config.dt
                )

            # Use optimization function if provided
            if self.optimization_function is not None:
                # Call with target_elevation_gain parameter
                (optimized_cda, optimized_crr, rmse, r2, virtual_profile) = (
                    self.optimization_function(
                        df=self.trimmed_df,
                        actual_elevation=self.trimmed_elevation,
                        config=self.config,
                        initial_cda=values["cda"],
                        initial_crr=values["crr"],
                        target_elevation_gain=self.target_elevation_gain,
                    )
                )
            else:
                # No optimization, just calculate with current parameters
                optimized_cda = values["cda"]
                optimized_crr = values["crr"]

                # Calculate virtual elevation
                ve_changes = delta_ve(
                    self.config,
                    cda=optimized_cda,
                    crr=optimized_crr,
                    df=self.trimmed_df,
                )

                # Build profile
                virtual_profile = calculate_virtual_profile(
                    ve_changes, self.trimmed_elevation, self.lap_column, self.trimmed_df
                )

                # Calculate stats
                rmse = np.sqrt(np.mean((virtual_profile - self.trimmed_elevation) ** 2))
                r2 = pearsonr(virtual_profile, self.trimmed_elevation)[0] ** 2

            # Store results
            self.rmse = rmse
            self.r2 = r2
            self.current_virtual_profile = virtual_profile

            # Update sliders without triggering events
            self.control_component.cda_slider.eventson = False
            self.control_component.crr_slider.eventson = False
            self.control_component.cda_slider.set_val(optimized_cda)
            self.control_component.crr_slider.set_val(optimized_crr)
            self.control_component.cda_slider.eventson = True
            self.control_component.crr_slider.eventson = True

            # Calculate full virtual elevation for entire dataset
            full_ve_changes = delta_ve(
                self.config, cda=optimized_cda, crr=optimized_crr, df=self.df
            )

            full_virtual = calculate_virtual_profile(
                full_ve_changes, self.actual_elevation, self.lap_column, self.df
            )

            # Update the plots
            self.elevation_component.update_elevation_data(
                full_virtual, self.start_idx, self.end_idx
            )

            self.residual_component.update_residual_data(
                self.actual_elevation, full_virtual, self.start_idx, self.end_idx
            )

            # Update metrics
            self.metrics_component.update_metrics(
                optimized_cda,
                optimized_crr,
                rmse,
                r2,
                trim_start,
                trim_end,
                self.current_virtual_profile,
            )

        except Exception as e:
            import traceback

            traceback.print_exc()
            self.metrics_component.metrics_text.set_text(
                f"Optimization error: {str(e)}"
            )

        finally:
            self.optimizing = False
            self.control_component.optimize_button.label.set_text("Optimize")
            self.control_component.optimize_button.color = "lightblue"
            plt.gcf().canvas.draw_idle()

    def on_skip(self):
        """Handle skip button click"""
        self.action = "skip"
        plt.close()

    def on_save(self):
        """Handle save button click"""
        self.action = "save"
        self.saved = True
        plt.close()

    def run(self, save_path=None):
        """Create and display the interactive plot"""
        fig = self.create_plot(save_path)
        plt.show()

        # Return results
        values = self.control_component.get_values()
        return (
            self.action,
            values["trim_start"],
            values["trim_end"],
            values["cda"],
            values["crr"],
            self.rmse,
            self.r2,
        )


def create_combined_interactive_plot(
    df,
    actual_elevation,
    lap_num,
    config,
    initial_cda=0.3,
    initial_crr=0.005,
    initial_trim_start=0,
    initial_trim_end=0,
    save_path=None,
    optimization_function=None,
    distance=None,
    lap_column=None,
    target_elevation_gain=None,
):
    """
    Create a combined interactive plot with map, elevation profiles, and parameter sliders.

    This is a wrapper around the VirtualElevationPlot class for backwards compatibility.
    """
    plot = VirtualElevationPlot(
        df=df,
        actual_elevation=actual_elevation,
        lap_num=lap_num,
        config=config,
        initial_cda=initial_cda,
        initial_crr=initial_crr,
        initial_trim_start=initial_trim_start,
        initial_trim_end=initial_trim_end,
        distance=distance,
        lap_column=lap_column,
        target_elevation_gain=target_elevation_gain,
        optimization_function=optimization_function,
    )

    return plot.run(save_path)


# Static plot functions from the original visualization module


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


def visualize_wind_effect(df, wind_speed, wind_direction):
    """
    Visualize the effect of wind on a rider throughout the course.

    Args:
        df (pandas.DataFrame): DataFrame with latitude/longitude data
        wind_speed (float): Wind speed in m/s
        wind_direction (float): Wind direction in degrees (meteorological convention)

    Returns:
        matplotlib.figure.Figure: The created figure
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable
    import matplotlib.patheffects as path_effects

    # Create VirtualElevationConfig with wind parameters
    from core.config import VirtualElevationConfig

    config = VirtualElevationConfig(
        rider_mass=70,  # Dummy value, not used for this visualization
        air_density=1.225,  # Dummy value, not used for this visualization
        wind_velocity=wind_speed,
        wind_direction=wind_direction,
    )

    # Calculate rider directions and effective wind
    from core.calculations import calculate_rider_directions, calculate_effective_wind

    rider_directions = calculate_rider_directions(df)
    effective_wind = calculate_effective_wind(df, config)

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))

    # Plot the route
    lat = df["latitude"].values
    lon = df["longitude"].values

    # Color the route based on effective wind
    # Positive = headwind (red), Negative = tailwind (blue), Zero = crosswind (white)
    norm = Normalize(vmin=-wind_speed, vmax=wind_speed)
    cmap = plt.cm.coolwarm

    # Plot the route with color gradient based on effective wind
    for i in range(len(df) - 1):
        ax.plot(
            [lon[i], lon[i + 1]],
            [lat[i], lat[i + 1]],
            color=cmap(norm(effective_wind[i])),
            linewidth=3,
            alpha=0.8,
            solid_capstyle="round",
        )

    # Add wind direction arrow in the corner
    # Convert from meteorological (where wind comes FROM) to math convention (where wind goes TO)
    wind_arrow_dir = (wind_direction + 180) % 360
    wind_arrow_rad = np.radians(wind_arrow_dir)
    arrow_length = 0.02  # Adjust based on your map scale

    # Calculate arrow coordinates
    arrow_x = 0.9  # Position in axes coordinates (0-1)
    arrow_y = 0.1
    dx = arrow_length * np.cos(wind_arrow_rad)
    dy = arrow_length * np.sin(wind_arrow_rad)

    # Convert to data coordinates for the arrow
    arrow_box = ax.transAxes.transform(
        [[arrow_x, arrow_y], [arrow_x + dx, arrow_y + dy]]
    )
    arrow_data_coords = ax.transData.inverted().transform(arrow_box)

    # Draw the arrow
    ax.annotate(
        f"Wind\n{wind_speed} m/s",
        xy=tuple(arrow_data_coords[0]),
        xytext=tuple(arrow_data_coords[1]),
        arrowprops=dict(
            arrowstyle="->", linewidth=2, color="black", shrinkA=0, shrinkB=0
        ),
        ha="center",
        va="center",
        fontsize=12,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.8),
    )

    # Add colorbar
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, label="Effective Wind (m/s)", pad=0.02)
    cbar.set_label("Effective Wind (m/s)", rotation=270, labelpad=20, fontsize=12)
    cbar.ax.text(
        0.5,
        0.25,
        "Tailwind",
        transform=cbar.ax.transAxes,
        ha="center",
        va="center",
        fontsize=10,
        color="blue",
        path_effects=[path_effects.withStroke(linewidth=3, foreground="white")],
    )
    cbar.ax.text(
        0.5,
        0.75,
        "Headwind",
        transform=cbar.ax.transAxes,
        ha="center",
        va="center",
        fontsize=10,
        color="red",
        path_effects=[path_effects.withStroke(linewidth=3, foreground="white")],
    )

    # Add start/end markers
    ax.scatter(lon[0], lat[0], color="green", s=100, label="Start", zorder=10)
    ax.scatter(lon[-1], lat[-1], color="red", s=100, label="End", zorder=10)

    # Add direction markers periodically
    num_markers = 10  # Adjust based on route complexity
    indices = np.linspace(0, len(df) - 1, num_markers, dtype=int)
    for i in indices:
        if i == 0 or i == len(df) - 1:
            continue  # Skip start/end points

        dir_rad = np.radians(rider_directions[i])
        marker_length = 0.0005  # Adjust based on your map scale
        ax.arrow(
            lon[i],
            lat[i],
            marker_length * np.cos(dir_rad),
            marker_length * np.sin(dir_rad),
            head_width=marker_length / 2,
            head_length=marker_length / 2,
            fc="black",
            ec="black",
            alpha=0.7,
            zorder=5,
        )

    # Set labels and title
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(f"Wind Effect Analysis: {wind_speed} m/s from {wind_direction}°")
    ax.legend()

    # Add general info
    info_text = (
        f"- Wind from: {wind_direction}° at {wind_speed} m/s\n"
        f"- Route colored by effective wind\n"
        f"- Red = Headwind, Blue = Tailwind\n"
        f"- Black arrows show rider direction"
    )
    ax.text(
        0.02,
        0.02,
        info_text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="bottom",
        horizontalalignment="left",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    # Equal aspect ratio
    ax.set_aspect("equal")

    plt.tight_layout()
    return fig
