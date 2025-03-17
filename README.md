# Escape Flavoured Virtual Elevation Analyzer Recipe

## Introduction

This tool implements the Virtual Elevation method ("Chung Method") developed by Robert Chung, which allows for the estimation of a cyclist's aerodynamic parameters (CdA) and rolling resistance (Crr) using power, speed and elevation data from cycling activities.

The Virtual Elevation method works by calculating what the elevation profile would look like based on the measured power output, speed, and assumed aerodynamic and rolling resistance parameters. By optimizing these parameters to make the virtual elevation match the actual measured elevation as closely as possible, we can hopefully accurately estimate a rider's CdA and Crr values.

## Requirements

- Python 3.7 or higher
- Required dependencies: numpy, pandas, matplotlib, scipy, fitparse
- Optional mapping dependencies: contextily, folium, geopandas, shapely

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/dhanek/escape-flavoured-virtual-elevation-recipe.git
   cd escape-flavoured-virtual-elevation-recipe
   ```

2. Install required dependencies:
   ```
   pip install -r requirements.txt
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
- `--show-map`: Display route maps for analyzed laps (static and interactive versions)
- `--trim-distance`: Distance in meters to trim from start and end of recording (default: 0)
- `--trim-start`: Distance in meters to trim from start of recording (overrides --trim-distance for start)
- `--trim-end`: Distance in meters to trim from end of recording (overrides --trim-distance for end)
- `--r2-weight`: Weight for R² in the composite objective (0-1, default: 0.5)
- `--grid-points`: Number of grid points to use in parameter search (default: 250)
- `--cda-bounds`: Bounds for CdA optimization as min,max (default: "0.1,0.5")
- `--crr-bounds`: Bounds for Crr optimization as min,max (default: "0.001,0.01")
- `--optimize-elevation-gain`: Optimize for a specific elevation gain (in meters). For individual lap analysis, this is the target gain per lap. For combined lap analysis (--selected-laps), this is the target total gain across all selected laps.

### Examples

Basic usage:
```
python ve_analyzer.py my_ride.fit --mass 75 --rho 1.225
```

Analyze specific laps together with custom parameters:
```
python ve_analyzer.py my_ride.fit --mass 75 --rho 1.225 --selected-laps "2,4,6" --cda 0.3 --trim-distance 500
```

Optimize for zero elevation gain across combined laps:
```
python ve_analyzer.py my_ride.fit --mass 75 --rho 1.225 --selected-laps "2,4,6" --optimize-elevation-gain 0
```

Trim specific distances from start and end:
```
python ve_analyzer.py my_ride.fit --mass 75 --rho 1.225 --trim-start 300 --trim-end 200
```

Generate route maps for each lap:
```
python ve_analyzer.py my_ride.fit --mass 75 --rho 1.225 --show-map
```

## Output

The tool will create an output directory containing:
- CSV files with processed data for each lap
- Elevation comparison plots
- Summary statistics and optimization results
- Route maps (if `--show-map` is used):
  - Static maps (.png) with OpenStreetMap background
  - Interactive, zoomable maps (.html) for viewing in a web browser

## Map Visualization

When using the `--show-map` option, the tool generates two types of maps:

### Static Maps
- Shows the complete route on an OpenStreetMap background
- Green dot marks the start point
- Red dot marks the end point
- The zoom level is automatically adjusted based on the track length

### Interactive Maps
- Zoomable, pannable OpenStreetMap
- Green marker with play icon at the start
- Red marker with checkered flag at the finish
- Click on markers to see "Start" and "Finish" labels

Maps are generated for individual laps or for combined laps, depending on your analysis mode.

## How It Works

1. The tool reads the FIT file and extracts power, speed, elevation, and GPS data
2. It separates the data into individual laps
3. For each lap (or combined laps), it:
   - Resamples the data to a consistent time interval
   - Calculates acceleration from velocity changes
   - Optimizes CdA and Crr parameters to match actual elevation
   - Plots virtual vs. actual elevation profiles
   - Generates route maps (if requested)
   - Calculates statistics like RMSE and R²
4. Generates summary reports and visualizations

## License

GNU GENERAL PUBLIC LICENSE

## Acknowledgements

- Robert Chung for developing the "Virtual Elevation method"
- https://escapecollective.com/the-chung-method-explained-how-to-aero-test-in-the-real-world-2/