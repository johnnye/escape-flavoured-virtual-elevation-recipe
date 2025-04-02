class VirtualElevationConfig:
    def __init__(
        self,
        rider_mass,
        air_density,
        time_interval=1.0,
        resample_freq="1s",
        drivetrain_efficiency=0.98,
        wind_velocity=0.0,
        wind_direction=None,  # Added parameter: wind direction in degrees (0=North, 90=East)
        cda_bounds=(0.15, 0.5),  # Changed from (0.1, 0.5)
        crr_bounds=(0.001, 0.03),  # Changed from (0.001, 0.01)
        fixed_cda=None,
        fixed_crr=None,
    ):
        # Required parameters
        if rider_mass <= 0:
            raise ValueError("Rider mass must be positive")
        if air_density <= 0:
            raise ValueError("Air density must be positive")

        self.kg = rider_mass
        self.rho = air_density
        self.eta = drivetrain_efficiency
        self.vw = wind_velocity
        self.wind_direction = wind_direction  # Store wind direction
        self.cda_bounds = cda_bounds
        self.crr_bounds = crr_bounds
        self.fixed_cda = fixed_cda
        self.fixed_crr = fixed_crr

        # Time interval parameters
        self.dt = time_interval
        self.resample_freq = resample_freq

        # Ensure dt and resample_freq are consistent
        self._validate_time_parameters()

    def clone_with(self, **kwargs):
        """Create a new instance with updated parameters."""
        params = self.__dict__.copy()
        params.update(kwargs)
        return VirtualElevationConfig(**params)

    def _validate_time_parameters(self):
        """Ensure dt and resample_freq are consistent."""
        # Extract numeric value from resample_freq if it's a simple case like "1s"
        if self.resample_freq.endswith("s") and self.resample_freq[:-1].isdigit():
            resample_seconds = float(self.resample_freq[:-1])
            if abs(resample_seconds - self.dt) > 0.001:  # Allow small float difference
                print(
                    f"Warning: resample_freq '{self.resample_freq}' doesn't match dt={self.dt}"
                )
