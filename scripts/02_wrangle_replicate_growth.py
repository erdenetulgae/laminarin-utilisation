import pandas as pd
import argparse
import os

# --- Configuration ---
# This mapping is for the 'C_aerobic_growth.xlsx' experiment plate.
WELL_TO_SAMPLE_MAPPING = {
    'D1': 'Negative control', 'D2': 'Negative control', 'D3': 'Negative control',
    'D4': 'Positive control (pAX)', 'D5': 'Positive control (pAX)', 'D6': 'Positive control (pAX)',
    'D7': 'Sample C',         'D8': 'Sample C',         'D9': 'Sample C',
    'D10': 'Sample C + exo', 'D11': 'Sample C + exo', 'D12': 'Sample C + exo'
}

# The name of the sheet in the Excel file to read data from.
SHEET_NAME = 'Sheet2' 

def parse_custom_wide_format(input_path, sheet_name):
    """
    Parses a sheet from an Excel file with a specific custom wide format where
    the data is preceded by a multi-line header containing timestamps.
    
    Args:
        input_path (str): The full path to the raw input Excel file.
        sheet_name (str): The name of the sheet to read.

    Returns:
        pandas.DataFrame: A clean, tidy DataFrame ready for analysis, or None if an error occurs.
    """
    if not os.path.exists(input_path):
        print(f"--> DEBUG: ERROR. File not found at path: {os.path.abspath(input_path)}")
        return None

    try:
        # Read the entire sheet without headers to parse it manually.
        df_raw = pd.read_excel(input_path, sheet_name=sheet_name, header=None)
        print(f"--> DEBUG: Successfully read sheet '{sheet_name}' from the Excel file.")
    except Exception as e:
        print(f"--> DEBUG: Failed to read the Excel file or sheet. Error: {e}")
        return None

    # --- New, Corrected Parsing Logic for this Specific XLSX Structure ---
    # Find the row that contains "Cycle Nr." which marks the start of our header block.
    try:
        header_start_row = df_raw[df_raw[0].astype(str).str.contains("Cycle Nr.", na=False)].index[0]
    except IndexError:
        print("--> DEBUG: ERROR. Could not find the 'Cycle Nr.' header row. Cannot parse file.")
        return None
        
    print(f"--> DEBUG: Found header block starting at row {header_start_row}.")

    # Timestamps are in the row directly below the "Cycle Nr." header.
    timestamp_row_index = header_start_row + 1
    timestamps_seconds = pd.to_numeric(df_raw.iloc[timestamp_row_index, 1:], errors='coerce').dropna().values
    
    # The actual data block starts 3 rows below the "Cycle Nr." header.
    data_start_row = header_start_row + 3
    df_data_block = df_raw.iloc[data_start_row:].copy()
    
    # Rename the first column to 'Well' and set it as the index.
    df_data_block = df_data_block.rename(columns={0: 'Well'}).set_index('Well')
    
    # The OD data is in the subsequent columns. Select only as many as we have timestamps.
    df_data_block = df_data_block.iloc[:, :len(timestamps_seconds)]
    
    # Assign the extracted timestamps as the column headers.
    df_data_block.columns = timestamps_seconds
    
    # --- Reshaping and Cleaning ---
    df_long = df_data_block.reset_index().melt(
        id_vars=['Well'], 
        var_name='Time_Seconds', 
        value_name='OD600'
    )
    
    df_long['Well'] = df_long['Well'].str.strip()
    df_long['Group'] = df_long['Well'].map(WELL_TO_SAMPLE_MAPPING)
    
    df_long['Time_Hours'] = pd.to_numeric(df_long['Time_Seconds'], errors='coerce') / 3600.0
    df_long['OD600'] = pd.to_numeric(df_long['OD600'], errors='coerce')
    
    df_long.dropna(subset=['Group', 'OD600', 'Time_Hours'], inplace=True)
    
    if df_long.empty:
        print("--> DEBUG: Data is empty after cleaning and mapping. Check well names in mapping dictionary.")
        return None

    # --- Normalization Step ---
    neg_control_group_name = 'Negative control'
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
    Main function to execute the data wrangling pipeline for the second growth assay.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.join(script_dir, '..')
    
    raw_data_path = os.path.join(project_root, 'data', 'raw', 'C_aerobic_growth.xlsx')
    processed_data_dir = os.path.join(project_root, 'data', 'processed')
    output_path = os.path.join(processed_data_dir, 'clean_replicate_growth.csv')

    os.makedirs(processed_data_dir, exist_ok=True)
    
    print(f"--- Starting: 02_wrangle_replicate_growth ---")
    print(f"Attempting to read raw data from: {raw_data_path}")

    clean_data = parse_custom_wide_format(raw_data_path, sheet_name=SHEET_NAME)

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
