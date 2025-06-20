import pandas as pd
import argparse
import os
import re # Import the regular expression library

# --- Configuration ---
# Updated to match the specific layout of the 'initial_aerobic_growth.xlsx' plate.
WELL_TO_SAMPLE_MAPPING = {
    'F4': 'Sample A',
    'G4': 'Sample B',
    'H4': 'Sample C',
    'F5': 'Positive control (pAX)',
    'G5': 'Negative control (pLS34)'
    # This script will only process data for wells listed here.
}

def parse_time_to_hours(time_str):
    """
    Robustly parses a time string (e.g., "1 h 15 min", "30min", "45 s")
    and converts it into a single float value of decimal hours.
    
    Args:
        time_str (str): The time string to parse.
        
    Returns:
        float: The total time in decimal hours.
    """
    hours = 0
    # This regex finds all number-unit pairs (e.g., '1 h', '15m')
    matches = re.findall(r'(\d+\.?\d*)\s*([hms])', str(time_str))
    
    for value, unit in matches:
        value = float(value)
        if unit == 'h':
            hours += value
        elif unit == 'm':
            hours += value / 60
        elif unit == 's':
            hours += value / 3600
    return hours

def parse_plate_reader_data(input_path):
    """
    Parses a raw plate reader Excel file where data for each time point
    is in separate blocks marked by "Cycle". It extracts, cleans, and
    normalizes the data for specified wells.
    
    Args:
        input_path (str): The full path to the raw input Excel file.

    Returns:
        pandas.DataFrame: A clean, tidy DataFrame ready for analysis, or None if an error occurs.
    """
    if not os.path.exists(input_path):
        print(f"--> DEBUG: ERROR. File not found at the specified path: {os.path.abspath(input_path)}")
        return None

    try:
        df_raw = pd.read_excel(input_path, header=None)
        print("--> DEBUG: Successfully read the Excel file.")
    except Exception as e:
        print(f"--> DEBUG: Failed to read the Excel file. Error: {e}")
        return None

    cycle_rows = df_raw[df_raw[0].astype(str).str.contains("Cycle", case=False, na=False)].index.tolist()

    if not cycle_rows:
        print("--> DEBUG: Error. No rows containing the word 'Cycle' were found. Cannot parse the file.")
        return None
    
    print(f"--> DEBUG: Found {len(cycle_rows)} 'Cycle' rows. Starting data extraction.")

    all_timepoints_df_list = []
    for i, row_start in enumerate(cycle_rows):
        row_end = cycle_rows[i + 1] if i + 1 < len(cycle_rows) else df_raw.shape[0]
        
        cycle_info = df_raw.iloc[row_start, 0]
        try:
            time_info_str = cycle_info.split('(')[-1].split(')')[0].strip()
        except IndexError:
            print(f"--> DEBUG: Warning: Could not parse time from header: '{cycle_info}'. Skipping row {row_start}.")
            continue

        plate_header_row = row_start + 2
        plate_data_start = row_start + 3
        
        plate_block = df_raw.iloc[plate_data_start:row_end, :13].copy()
        
        try:
            col_headers = [str(int(float(col))) for col in df_raw.iloc[plate_header_row, 1:13] if str(col).strip()]
            plate_block.columns = ['Row'] + col_headers
        except (ValueError, TypeError):
            print(f"--> DEBUG: Warning: Could not parse column headers correctly at row {plate_header_row}. Using raw strings.")
            plate_block.columns = ['Row'] + [str(col) for col in df_raw.iloc[plate_header_row, 1:13]]
            
        plate_long = plate_block.melt(id_vars=['Row'], var_name='Col', value_name='OD600')
        plate_long['Time_str'] = time_info_str
        
        all_timepoints_df_list.append(plate_long)

    if not all_timepoints_df_list:
        print("--> DEBUG: Error: Failed to extract any valid data blocks after looping through cycles.")
        return None
        
    df_long = pd.concat(all_timepoints_df_list, ignore_index=True)
    
    df_long.dropna(subset=['Row', 'Col', 'OD600'], inplace=True)
    df_long['Col'] = df_long['Col'].astype(str)
    df_long['Well'] = df_long['Row'] + df_long['Col']
    df_long['OD600'] = pd.to_numeric(df_long['OD600'], errors='coerce')
    
    df_long['Group'] = df_long['Well'].map(WELL_TO_SAMPLE_MAPPING)
    
    # --- Use the new, robust time parsing function ---
    df_long['Time_Hours'] = df_long['Time_str'].apply(parse_time_to_hours)
    
    df_long.dropna(subset=['Group', 'OD600', 'Time_Hours'], inplace=True)
    
    neg_control_group_name = 'Negative control (pLS34)'
    if neg_control_group_name not in df_long['Group'].unique():
        print(f"--> DEBUG: Error: Negative control group '{neg_control_group_name}' not found. Cannot normalize.")
        return None

    neg_control_means = df_long[df_long['Group'] == neg_control_group_name].groupby('Time_Hours')['OD600'].mean()
    df_long['OD600_Normalized'] = df_long['OD600'] - df_long['Time_Hours'].map(neg_control_means)

    final_cols = ['Time_Hours', 'Well', 'Group', 'OD600', 'OD600_Normalized']
    df_final = df_long[final_cols].sort_values(by=['Well', 'Time_Hours']).reset_index(drop=True)
    
    return df_final

def main():
    """
    Main function to execute the data wrangling pipeline for the initial growth assay.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.join(script_dir, '..')
    
    raw_data_path = os.path.join(project_root, 'data', 'raw', 'initial_aerobic_growth.xlsx')
    processed_data_dir = os.path.join(project_root, 'data', 'processed')
    output_path = os.path.join(processed_data_dir, 'clean_initial_growth.csv')

    os.makedirs(processed_data_dir, exist_ok=True)
    
    print(f"--- Starting: 01_wrangle_initial_growth ---")
    print(f"Attempting to read raw data from: {raw_data_path}")

    clean_data = parse_plate_reader_data(raw_data_path)

    if clean_data is not None and not clean_data.empty:
        clean_data.to_csv(output_path, index=False)
        print(f"\nData wrangling complete.")
        print(f"Clean data saved to: {output_path}")
        print("\nHead of the processed data:")
        print(clean_data.head())
    else:
        print("\nData wrangling failed. No output file was created.")

if __name__ == '__main__':
    main()
