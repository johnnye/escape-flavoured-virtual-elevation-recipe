# Escape Flavoured Virtual Elevation Analyzer Recipe

## Introduction

This tool implements the Virtual Elevation method ("Chung Method") developed by Robert Chung, which allows for the estimation of a cyclist's aerodynamic parameters (CdA) and rolling resistance (Crr) using power, speed and elevation data from cycling activities.

The Virtual Elevation method works by calculating what the elevation profile would look like based on the measured power output, speed, and assumed aerodynamic and rolling resistance parameters. By optimizing these parameters to make the virtual elevation match the actual measured elevation as closely as possible, we can hopefully accurately estimate a rider's CdA and Crr values.

## Requirements

- Python 3.7 or higher
- Dependencies: numpy, pandas, matplotlib, scipy, fitparse

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/dhanek/escape-flavoured-virtual-elevation-recipe.git
   cd escape-flavoured-virtual-elevation-recipe
   ```

2. Install required dependencies:
   ```
   pip install numpy pandas matplotlib scipy fitparse
   ```

## Usage

```
python ve_analyzer.py <fit_file> --mass <rider_mass_kg> --rho <air_density_kg_m3> [additional options]
```

### Required Parameters:
- `fit_file`: Path to the FIT file to analyze
- `--mass`: Rider mass in kg (mandatory)
- `--rho`: Air density in kg/m³ (mandatory)

### Optional Parameters:
- `--resample`: Resampling frequency (default: "1s")
- `--output`: Output directory for results (default: "fit_analysis_results")
- `--min-lap`: Minimum lap duration in seconds (default: 30)
- `--selected-laps`: Comma-separated list of lap numbers to analyze together (e.g., "2,4,6,8,10")
- `--cda`: Fixed CdA value to use (if provided, only Crr will be optimized)
- `--crr`: Fixed Crr value to use (if provided, only CdA will be optimized)
- `--debug`: Enable debug output
- `--trim-distance`: Distance in meters to trim from start and end of recording (default: 0)
- `--r2-weight`: Weight for R² in the composite objective (0-1, default: 0.5)
- `--grid-points`: Number of grid points to use in parameter search (default: 250)
- `--cda-bounds`: Bounds for CdA optimization as min,max (default: "0.1,0.5")
- `--crr-bounds`: Bounds for Crr optimization as min,max (default: "0.001,0.01")

### Examples

Basic usage:
```
python ve_analyzer.py my_ride.fit --mass 75 --rho 1.225
```

Analyze specific laps together with custom parameters:
```
python ve_analyzer.py my_ride.fit --mass 75 --rho 1.225 --selected-laps "2,4,6" --cda 0.3 --trim-distance 500
```

## Output

The tool will create an output directory containing:
- CSV files with processed data for each lap
- Elevation comparison plots
- Summary statistics and optimization results

## How It Works

1. The tool reads the FIT file and extracts power, speed, and elevation data
2. It separates the data into individual laps
3. For each lap (or combined laps), it:
   - Resamples the data to a consistent time interval
   - Calculates acceleration from velocity changes
   - Optimizes CdA and Crr parameters to match actual elevation
   - Plots virtual vs. actual elevation profiles
   - Calculates statistics like RMSE and R²
4. Generates summary reports and visualizations

## License

GNU GENERAL PUBLIC LICENSE

## Acknowledgements

- Robert Chung for developing the Virtual Elevation method"
- https://escapecollective.com/the-chung-method-explained-how-to-aero-test-in-the-real-world-2/