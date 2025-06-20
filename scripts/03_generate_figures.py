import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# --- Configuration for Plotting ---
# Define figure size for A4 half-page width (in inches)
FIG_WIDTH = 3.25
FIG_HEIGHT = 2.4

# Define the exact groups to plot and their specific order for the legend
GROUPS_TO_PLOT_ORDER = [
    'Positive control (pAX)',
    'Negative control (pLS34)',
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

def plot_growth_curve(df, title, output_filename):
    """
    Generates and saves a high-quality growth curve plot WITHOUT a legend,
    only showing the specified groups. Returns the handles and labels needed
    to create a separate legend file.

    Args:
        df (pd.DataFrame): The clean, long-format DataFrame to plot.
        title (str): The title for the plot.
        output_filename (str): The full path to save the output PNG file.
        
    Returns:
        tuple: A tuple containing the legend handles and labels.
    """
    print(f"Generating plot: '{title}'...")

    # --- Filter to show only the specified groups ---
    df_filtered = df[df['Group'].isin(GROUPS_TO_PLOT_ORDER)].copy()
    print(f"--> Plotting groups: {df_filtered['Group'].unique().tolist()}")

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))

    # Create the line plot with the specified group order
    sns.lineplot(
        data=df_filtered,
        x='Time_Hours',
        y='OD600_Normalized',
        hue='Group',
        palette=COLOR_PALETTE,
        hue_order=GROUPS_TO_PLOT_ORDER, # Enforce the exact order
        errorbar=('ci', 95),
        ax=ax,
        linewidth=1.5
    )
    
    # --- Formatting ---

    ax.set_xlabel('Time (Hours)', fontsize=8)
    ax.set_ylabel('Background Subtracted OD600', fontsize=8)
    ax.tick_params(axis='both', which='major', labelsize=7)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.axhline(0, ls='--', color='black', alpha=0.7, linewidth=0.75)
    
    # Get handles and labels for creating a separate legend
    handles, labels = ax.get_legend_handles_labels()
    
    # Remove the legend from the plot itself
    if ax.get_legend() is not None:
        ax.get_legend().remove()
    
    plt.tight_layout()
    try:
        plt.savefig(output_filename, dpi=300)
        print(f"Successfully saved plot to: {output_filename}")
    except Exception as e:
        print(f"Error saving plot: {e}")
    
    plt.close(fig)
    
    return handles, labels

def create_legend_file(handles, labels, output_filename):
    """
    Creates and saves a separate image file containing only the legend.
    
    Args:
        handles: Legend handles from a matplotlib plot.
        labels: Legend labels from a matplotlib plot.
        output_filename (str): The full path to save the output PNG file.
    """
    print("\nGenerating separate legend file...")
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
    Main function to load all processed datasets and generate a figure for each.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.join(script_dir, '..')
    
    processed_data_dir = os.path.join(project_root, 'data', 'processed')
    figures_dir = os.path.join(project_root, 'results', 'figures')

    os.makedirs(figures_dir, exist_ok=True)

    files_to_plot = {
        'clean_initial_growth.csv': 'Initial Aerobic Growth Assay',
        'clean_replicate_growth.csv': 'Replicate Aerobic Growth Assay'
    }

    print("--- Starting: 03_generate_figures ---")
    
    master_legend_dict = {}

    for filename, title in files_to_plot.items():
        input_path = os.path.join(processed_data_dir, filename)
        output_filename = os.path.join(figures_dir, filename.replace('.csv', '.png'))
        
        if not os.path.exists(input_path):
            print(f"\nWarning: Processed data file not found: {input_path}. Skipping plot.")
            continue
            
        print(f"\nLoading processed data from: {input_path}")
        df = pd.read_csv(input_path)
        
        handles, labels = plot_growth_curve(df, title, output_filename)
        
        for handle, label in zip(handles, labels):
            if label not in master_legend_dict:
                master_legend_dict[label] = handle
    
    if master_legend_dict:
        # Sort the legend items according to our specified list for consistency
        sorted_labels = [label for label in GROUPS_TO_PLOT_ORDER if label in master_legend_dict]
        sorted_handles = [master_legend_dict[label] for label in sorted_labels]
        
        legend_output_filename = os.path.join(figures_dir, 'legend.png')
        create_legend_file(sorted_handles, sorted_labels, legend_output_filename)

    print("\n--- Plotting script finished. ---")

if __name__ == '__main__':
    main()
