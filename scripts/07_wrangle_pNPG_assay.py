import pandas as pd
import os

# --- Configuration ---
# This mapping is for the pNPG assay plate layout.
WELL_TO_SAMPLE_MAPPING = {
    'A6': 'Blank',
    'B6': 'Positive control (pAX)',
    'C6': 'Negative control (pLS34)',
    'D6': 'Sample A',
    'E6': 'Sample B',
    'F6': 'Sample C'
}

# The number of metadata rows to skip at the top of the CSV file.
# Based on file metadata, this is ~13. Adjust if needed.
ROWS_TO_SKIP = 13
# The name of the sheet to read from the Excel file.
SHEET_NAME = 'End point'

def parse_endpoint_data(input_path, sheet_name, rows_to_skip, well_mapping):
    """
    Parses a simple endpoint assay from a specific sheet in an Excel file,
    reshapes it, and performs blank-subtraction normalization.
    
    Args:
        input_path (str): The full path to the raw input Excel file.
        sheet_name (str): The name of the sheet to read.
        rows_to_skip (int): The number of rows to skip at the top.
        well_mapping (dict): A dictionary mapping well IDs to sample names.

    Returns:
        pandas.DataFrame: A clean, tidy DataFrame ready for analysis, or None if an error occurs.
    """
    if not os.path.exists(input_path):
        print(f"--> DEBUG: ERROR. File not found at path: {os.path.abspath(input_path)}")
        return None

    try:
        df_raw = pd.read_excel(input_path, sheet_name=sheet_name, skiprows=rows_to_skip, header=0)
        print(f"--> DEBUG: Successfully read sheet '{sheet_name}' from the Excel file.")
    except Exception as e:
        print(f"--> DEBUG: Failed to read the Excel file or sheet. Error: {e}")
        return None

    # --- Data Cleaning and Structuring ---
    # The data is in a wide format (rows=A-H, cols=1-12). Melt into a long format.
    
    # --- FIX: Robustly rename the first column to 'Row' ---
    # This handles cases where the first column might not be named '<>'.
    if df_raw.columns[0] != 'Row':
        df_raw = df_raw.rename(columns={df_raw.columns[0]: 'Row'})

    df_long = df_raw.melt(id_vars=['Row'], var_name='Col', value_name='OD405')
    
    df_long['Well'] = df_long['Row'] + df_long['Col'].astype(str)
    df_long['OD405'] = pd.to_numeric(df_long['OD405'], errors='coerce')
    df_long['Group'] = df_long['Well'].map(well_mapping)
    
    # Filter for only the wells we have mapped
    df_final = df_long.dropna(subset=['Group', 'OD405']).copy()

    if df_final.empty:
        print("--> DEBUG: Data is empty after cleaning and mapping. Check well names in mapping dictionary.")
        return None

    # --- Normalization Step ---
    try:
        blank_value = df_final[df_final['Group'] == 'Blank']['OD405'].mean()
        print(f"--> DEBUG: Calculated blank value as: {blank_value:.4f}")
        df_final['OD405_Normalized'] = df_final['OD405'] - blank_value
    except (KeyError, IndexError):
        print("--> DEBUG: Error: 'Blank' well not found in data. Cannot perform blank subtraction.")
        df_final['OD405_Normalized'] = df_final['OD405']

    # Select and reorder final columns
    output_cols = ['Well', 'Group', 'OD405', 'OD405_Normalized']
    df_final = df_final[output_cols].reset_index(drop=True)
    
    return df_final

def main():
    """
    Main function to execute the data wrangling pipeline for the pNPG assay.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.join(script_dir, '..')
    
    raw_data_path = os.path.join(project_root, 'data', 'raw', 'pNPG_assay.xlsx')
    processed_data_dir = os.path.join(project_root, 'data', 'processed')
    output_path = os.path.join(processed_data_dir, 'clean_pNPG_assay.csv')

    os.makedirs(processed_data_dir, exist_ok=True)
    
    print(f"--- Starting: 07_wrangle_pNPG_assay ---")
    print(f"Attempting to read raw data from: {raw_data_path}")

    clean_data = parse_endpoint_data(raw_data_path, sheet_name=SHEET_NAME, rows_to_skip=ROWS_TO_SKIP, well_mapping=WELL_TO_SAMPLE_MAPPING)

    if clean_data is not None and not clean_data.empty:
        clean_data.to_csv(output_path, index=False)
        print(f"\nData wrangling complete.")
        print(f"Clean data saved to: {output_path}")
        print("\nProcessed Data:")
        print(clean_data)
    else:
        print("\nData wrangling failed. No output file was created.")

if __name__ == '__main__':
    main()
