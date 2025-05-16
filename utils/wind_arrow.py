import folium


class WindArrow(folium.FeatureGroup):
    """Custom folium plugin to display wind direction and speed"""

    def __init__(self, location, wind_speed, wind_dir, name=None):
        super().__init__(name=name)
        self._name = name
        self.location = location
        self.wind_speed = wind_speed
        self.wind_dir = wind_dir

    def _repr_html_(self):
        html = f"""
        <div id='wind-arrow' style="position: absolute; top: 10px; right: 10px; 
                 background-color: rgba(255, 255, 255, 0.9); padding: 10px; 
                 border-radius: 5px; border: 1px solid #4363d8; z-index: 1000;
                 box-shadow: 0 2px 5px rgba(0,0,0,0.2);">
            <div style="font-weight: bold; text-align: center; margin-bottom: 5px; color: #4363d8;">Wind</div>
            <div id="arrow" style="text-align: center;">
                <div style="font-size: 28px; transform: rotate({self.wind_dir + 180}deg); color: #4363d8;">↑</div>
                <div style="font-size: 12px; margin-top: 5px;">{self.wind_speed} m/s</div>
                <div style="font-size: 12px;">{self.wind_dir}°</div>
            </div>
        </div>
        """
        return html
