import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# --- Configuration ---
# Use the same professional color palette for consistency across all figures.
COLOR_PALETTE = {
    'Positive control (pAX)': '#2ca02c',     # Green for Positive
    'Sample C': '#1f77b4',                 # Blue for Sample C
    'Negative control (pLS34)': '#333333',     # Dark Grey/Black for Negative
    'Sample A': '#d6b4e8',                 # Light Purple
    'Sample B': '#ffbf7f',                 # Light Orange
}

def plot_endpoint_barchart(df, title, output_filename):
    """
    Generates and saves a publication-quality bar chart from clean endpoint data.

    Args:
        df (pd.DataFrame): The clean DataFrame to plot.
        title (str): The title for the plot.
        output_filename (str): The full path to save the output PNG file.
    """
    print(f"Generating bar chart: '{title}'...")

    # Exclude the 'Blank' sample from the plot itself
    df_plot = df[df['Group'] != 'Blank'].copy()

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(4.5, 3.5))

    sns.barplot(
        data=df_plot,
        x='Group',
        y='OD405_Normalized',
        palette=COLOR_PALETTE,
        ax=ax,
        capsize=0.1,
        errorbar='sd' # Show standard deviation as error bars
    )
    
    # --- Formatting for Publication Quality ---
    ax.set_xlabel('')
    ax.set_ylabel('Blank-Subtracted Absorbance (OD405)', fontsize=8)
    ax.tick_params(axis='y', labelsize=7)
    plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right', fontsize=7)

    # Add a horizontal line at y=0 for reference
    ax.axhline(0, ls='--', color='black', alpha=0.7, linewidth=0.75)

    plt.tight_layout()
    try:
        plt.savefig(output_filename, dpi=300)
        print(f"Successfully saved bar chart to: {output_filename}")
    except Exception as e:
        print(f"Error saving plot: {e}")
    
    plt.close(fig)

def main():
    """
    Main function to load the processed pNPG data and generate a figure.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.join(script_dir, '..')
    
    processed_data_path = os.path.join(project_root, 'data', 'processed', 'clean_pNPG_assay.csv')
    figures_dir = os.path.join(project_root, 'results', 'figures')

    os.makedirs(figures_dir, exist_ok=True)
    
    output_filename = os.path.join(figures_dir, 'pNPG_assay_barchart.png')

    print("--- Starting: 08_plot_pNPG_assay ---")

    if not os.path.exists(processed_data_path):
        print(f"\nError: Processed data file not found: {processed_data_path}.")
        print("Please run '07_wrangle_pNPG_assay.py' first.")
        return
        
    print(f"\nLoading processed data from: {processed_data_path}")
    df = pd.read_csv(processed_data_path)
    
    plot_endpoint_barchart(df, 'Enzymatic Activity (pNPG Assay)', output_filename)

    print("\n--- pNPG plotting script finished. ---")

if __name__ == '__main__':
    main()
