import pandas as pd


def parse_fit_file(fit_file_path, debug=False):
    """
    Parse a .fit file and extract data records and lap messages.

    Args:
        fit_file_path (str): Path to the .fit file
        debug (bool): Whether to print debug information

    Returns:
        tuple: (DataFrame with record data, list of lap messages)
    """
    try:
        # Import fitparse
        from fitparse import FitFile
    except ImportError:
        print("The fitparse library is required but not installed.")
        print("Install it using: pip install fitparse")
        return None, None

    print(f"Parsing .fit file: {fit_file_path}")

    # Parse the fit file
    try:
        fit_file = FitFile(fit_file_path)
        fit_file.parse()
    except Exception as e:
        print(f"Error parsing .fit file: {str(e)}")
        return None, None

    # Extract data messages
    data_messages = []
    lap_messages = []

    for message in fit_file.get_messages():
        if message.name == "record":
            data = {}
            for field in message:
                data[field.name] = field.value
            data_messages.append(data)
        elif message.name == "lap":
            lap_data = {}
            for field in message:
                lap_data[field.name] = field.value
            lap_messages.append(lap_data)

    if debug:
        print(
            f"Found {len(data_messages)} record messages and {len(lap_messages)} lap messages"
        )

    # Create DataFrame from records
    df = pd.DataFrame(data_messages)

    # Check if we have any data
    if df.empty:
        print("No record data found in the .fit file")
        return None, None

    # Convert units and rename columns
    if "speed" in df.columns:
        # Garmin typically records speed in m/s
        df["v"] = df["speed"]
    elif "enhanced_speed" in df.columns:
        df["v"] = df["enhanced_speed"]

    if "power" in df.columns:
        df["watts"] = df["power"]

    if "altitude" in df.columns:
        df["elevation"] = df["altitude"]
    elif "enhanced_altitude" in df.columns:
        df["elevation"] = df["enhanced_altitude"]

    # Extract timestamps
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df.set_index("timestamp", inplace=True)
    else:
        print("No timestamp data found in the .fit file")
        return None, None

    # Check if we have all required columns
    required_columns = ["v", "watts", "elevation"]
    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        print(f"Required columns missing: {', '.join(missing_columns)}")
        print("Available columns:", ", ".join(df.columns))
        return None, None

    # Debug lap messages to understand their structure
    if debug and lap_messages:
        print("\nLap Message Fields:")
        sample_lap = lap_messages[0]
        for key, value in sample_lap.items():
            print(f"  {key}: {value} ({type(value)})")

    return df, lap_messages


def extract_lap_info(fit_file_path, min_lap_duration=30, debug=False):
    """Extract lap information from a .fit file without processing the data.

    Args:
        fit_file_path (str): Path to the .fit file
        min_lap_duration (int): Minimum lap duration in seconds
        debug (bool): Whether to print debug information

    Returns:
        list: List of dictionaries containing lap information
    """
    try:
        from fitparse import FitFile
    except ImportError:
        print("The fitparse library is required but not installed.")
        return None

    # Parse the fit file
    try:
        fit_file = FitFile(fit_file_path)
        fit_file.parse()
    except Exception as e:
        print(f"Error parsing .fit file: {str(e)}")
        return None

    # Extract lap messages
    lap_messages = []
    for message in fit_file.get_messages():
        if message.name == "lap":
            lap_data = {}
            for field in message:
                lap_data[field.name] = field.value
            lap_messages.append(lap_data)

    if not lap_messages:
        print("No lap markers found in .fit file.")
        return None

    # Process each lap
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

            if debug:
                print(
                    f"  Lap {i+1}: Start={start_time}, Duration={lap_duration:.1f}s, End={end_time}"
                )

            if lap_duration >= min_lap_duration:
                lap_info.append(
                    {
                        "lap_number": i + 1,
                        "start_time": start_time,
                        "end_time": end_time,
                        "duration": lap_duration,
                    }
                )

    return lap_info
