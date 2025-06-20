import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# --- Configuration for Plotting ---
# Define figure size for A4 half-page width (in inches)
FIG_WIDTH = 3.25
FIG_HEIGHT = 2.4

# This list defines the full set of groups to be plotted on the graphs.
GROUPS_TO_PLOT_ORDER = [
    'Positive control (pAX)',
    'Negative control (pLS34)', # From the initial growth assay
    'Negative control',          # From the replicate growth assay
    'Sample A',
    'Sample B',
    'Sample C'
]

# Define the color palette. Using a dictionary ensures each group always gets the same color.
COLOR_PALETTE = {
    # Primary, high-contrast colors
    'Positive control (pAX)': '#2ca02c',     # Green for Positive
    'Sample C': '#1f77b4',                 # Blue for Sample C
    'Negative control (pLS34)': '#333333',     # Dark Grey/Black for Negative
    'Negative control': '#333333',         # Ensure both negative controls are the same color

    # Secondary, lighter colors
    'Sample A': '#d6b4e8',                 # Light Purple
    'Sample B': '#ffbf7f',                 # Light Orange
    
    # Ensure consistency for different experiment names if they appear
    'Positive control': '#2ca02c'
}

def plot_raw_growth_curve(df, title, output_filename):
    """
    Generates and saves a high-quality plot of the RAW, unnormalized growth data.
    """
    print(f"Generating RAW data plot: '{title}'...")

    df_filtered = df[df['Group'].isin(GROUPS_TO_PLOT_ORDER)].copy()
    print(f"--> Plotting groups: {df_filtered['Group'].unique().tolist()}")

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))

    sns.lineplot(
        data=df_filtered,
        x='Time_Hours',
        y='OD600', 
        hue='Group',
        palette=COLOR_PALETTE,
        hue_order=GROUPS_TO_PLOT_ORDER, 
        errorbar=('ci', 95),
        ax=ax,
        linewidth=1.5
    )
    
    ax.set_xlabel('Time (Hours)', fontsize=8)
    ax.set_ylabel('Raw Optical Density (OD600)', fontsize=8)
    ax.tick_params(axis='both', which='major', labelsize=7)
    
    if ax.get_legend() is not None:
        ax.get_legend().remove()
    
    plt.tight_layout()
    try:
        plt.savefig(output_filename, dpi=300)
        print(f"Successfully saved RAW plot to: {output_filename}")
    except Exception as e:
        print(f"Error saving plot: {e}")
    
    plt.close(fig)

def create_legend_file(handles, labels, output_filename):
    """
    Creates and saves a separate image file containing only the legend.
    """
    print(f"\nGenerating legend file: {os.path.basename(output_filename)}...")
    fig_legend = plt.figure(figsize=(2, 1.5))
    ax_legend = fig_legend.add_subplot(111)
    
    ax_legend.legend(handles, labels, loc='center', frameon=False, fontsize=8)
    ax_legend.axis('off')
    
    try:
        fig_legend.savefig(output_filename, dpi=300, bbox_inches='tight')
        print(f"Successfully saved legend to: {output_filename}")
    except Exception as e:
        print(f"Error saving legend: {e}")
        
    plt.close(fig_legend)


def main():
    """
    Main function to load all processed datasets and generate a raw data figure for each.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.join(script_dir, '..')
    
    processed_data_dir = os.path.join(project_root, 'data', 'processed')
    figures_dir = os.path.join(project_root, 'results', 'figures')

    os.makedirs(figures_dir, exist_ok=True)

    files_to_plot = {
        'clean_initial_growth.csv': 'Initial Aerobic Growth (Raw Data)',
        'clean_replicate_growth.csv': 'Replicate Aerobic Growth (Raw Data)'
    }

    print("--- Starting: 04_plot_raw_growth ---")

    # Generate the plots for each dataset
    for filename, title in files_to_plot.items():
        input_path = os.path.join(processed_data_dir, filename)
        output_filename = os.path.join(figures_dir, filename.replace('.csv', '_raw.png'))
        
        if not os.path.exists(input_path):
            print(f"\nWarning: Processed data file not found: {input_path}. Skipping plot.")
            continue
            
        print(f"\nLoading processed data from: {input_path}")
        df = pd.read_csv(input_path)
        
        plot_raw_growth_curve(df, title, output_filename)

    # --- Create a single, combined legend file ---
    # We create a dummy plot just to get the legend handles from our color palette
    fig, ax = plt.subplots()
    for group in GROUPS_TO_PLOT_ORDER:
        if group in COLOR_PALETTE:
            ax.plot([], [], color=COLOR_PALETTE[group], label=group)
    
    handles, labels = ax.get_legend_handles_labels()
    plt.close(fig)

    if handles:
        # Create the main legend with all items
        full_legend_filename = os.path.join(figures_dir, 'legend_all_raw.png')
        create_legend_file(handles, labels, full_legend_filename)

        # --- Create an alternative legend with only primary samples ---
        primary_legend_order = [
            'Positive control (pAX)',
            'Negative control (pLS34)',
            'Sample C'
        ]
        
        # Filter the handles and labels for the primary-only legend
        primary_handles = [h for h, l in zip(handles, labels) if l in primary_legend_order]
        primary_labels = [l for l in labels if l in primary_legend_order]
        
        # Re-order them to match the desired list
        label_handle_map = dict(zip(primary_labels, primary_handles))
        sorted_handles = [label_handle_map[label] for label in primary_legend_order if label in label_handle_map]
        sorted_labels = [label for label in primary_legend_order if label in label_handle_map]

        if sorted_handles:
            alt_legend_filename = os.path.join(figures_dir, 'legend_primary_raw.png')
            create_legend_file(sorted_handles, sorted_labels, alt_legend_filename)

    print("\n--- Raw plotting script finished. ---")

if __name__ == '__main__':
    main()
