import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.stats import f_oneway, ttest_ind
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from sklearn.metrics import auc
from itertools import combinations

# --- Configuration ---
# Define colors for the bar plot to maintain consistency
COLOR_PALETTE = {
    'Positive control (pAX)': '#2ca02c',     # Green for Positive
    'Sample C': '#1f77b4',                 # Blue for Sample C
    'Negative control (pLS34)': '#333333',     # Dark Grey/Black for Negative
    'Sample A': '#d6b4e8',                 # Light Purple
    'Sample B': '#ffbf7f',                 # Light Orange
    'Positive control': '#2ca02c',
    'Negative control': '#333333',
    'Sample C + exo': '#FF6347' # Tomato
}

def add_stat_annotation(ax, df, x_col, y_col):
    """
    Adds pairwise statistical annotation (p-values) to a seaborn barplot
    for specific, pre-defined comparisons, drawing wider bars first.
    
    Args:
        ax: The matplotlib axes object for the plot.
        df (pd.DataFrame): DataFrame containing the data.
        x_col (str): The name of the column for x-axis categories.
        y_col (str): The name of the column for y-axis values.
    """
    groups = df[x_col].unique()
    
    neg_control_name_initial = 'Negative control (pLS34)'
    neg_control_name_replicate = 'Negative control'

    neg_control_to_use = None
    if neg_control_name_initial in groups:
        neg_control_to_use = neg_control_name_initial
    elif neg_control_name_replicate in groups:
        neg_control_to_use = neg_control_name_replicate

    if not neg_control_to_use:
        print("--> Warning: No negative control found for statistical annotation.")
        return
        
    pairs_to_plot = [
        ('Positive control (pAX)', neg_control_to_use),
        ('Sample C', neg_control_to_use)
    ]
    pairs_to_plot = [p for p in pairs_to_plot if p[0] in groups and p[1] in groups]

    if not pairs_to_plot:
        return

    x_labels = [label.get_text() for label in ax.get_xticklabels()]
    x_coords = {group: i for i, group in enumerate(x_labels)}

    pairs_to_plot.sort(key=lambda pair: abs(x_coords.get(pair[0], -1) - x_coords.get(pair[1], -1)), reverse=True)
    
    y_max = df.groupby(x_col)[y_col].mean().max()
    y_err_max = df.groupby(x_col)[y_col].std().max()
    bar_height = y_max + y_err_max
    step = bar_height * 0.20

    for i, (g1, g2) in enumerate(pairs_to_plot):
        if g1 not in x_coords or g2 not in x_coords:
            continue

        data1 = df[df[x_col] == g1][y_col]
        data2 = df[df[x_col] == g2][y_col]
        
        _, p_val = ttest_ind(data1, data2, equal_var=False)
        
        if p_val < 0.001:
            p_text = 'p < 0.001'
        elif p_val < 0.05:
            p_text = f'p = {p_val:.3f}'
        else:
            p_text = 'ns'
            print(p_val, "this is the ns p value")  

        x1, x2 = x_coords[g1], x_coords[g2]
        
        y = bar_height + i * step
        ax.plot([x1, x1, x2, x2], [y, y + step*0.2, y + step*0.2, y], lw=1.5, c='black')
        ax.grid(True, which='both', axis='y', linestyle='--', linewidth=0.5)
        ax.set_xlabel('')
        ax.text((x1 + x2) / 2, y + step*0.2, p_text, ha='center', va='bottom', color='black', fontsize=8)

def analyze_growth_with_auc(df, title, output_dir):
    """
    Calculates the Area Under the Curve (AUC) for each replicate in the DataFrame,
    runs statistical analysis, and generates a bar plot of the results.

    Args:
        df (pd.DataFrame): The clean, long-format DataFrame to analyze.
        title (str): The base title for outputs.
        output_dir (str): The path to the root 'results' directory.
    """
    print(f"\n--- Starting AUC Analysis for: {title} ---")
    
    df = df[df['Group'] != 'Sample C + exo'].copy()
    
    df_auc_data = df[['Time_Hours', 'Well', 'Group', 'OD600_Normalized']].copy()

    auc_results = []
    for well_id in df_auc_data['Well'].unique():
        well_data = df_auc_data[df_auc_data['Well'] == well_id].sort_values(by='Time_Hours')
        
        time_points = well_data['Time_Hours'].values
        od_values = well_data['OD600_Normalized'].values
        
        od_values[od_values < 0] = 0
        well_auc = auc(time_points, od_values)
        
        group_name = well_data['Group'].iloc[0]
        auc_results.append({'Well': well_id, 'Group': group_name, 'AUC': well_auc})

    df_auc = pd.DataFrame(auc_results)
    
    auc_table_path = os.path.join(output_dir, 'tables', f'auc_results_{title.replace(" ", "_").lower()}.csv')
    df_auc.to_csv(auc_table_path, index=False)
    print(f"Saved AUC results table to: {auc_table_path}")

    print("Generating bar plot of AUC values...")
    # plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(3.0, 3.0))

    # --- FIX: Define the plot order explicitly ---
    groups_present = df_auc['Group'].unique()
    neg_control_name = 'Negative control (pLS34)' if 'Negative control (pLS34)' in groups_present else 'Negative control'
    plot_order = ['Positive control (pAX)', neg_control_name, 'Sample C']
    # Filter for only those groups that are actually in the data for this specific plot
    plot_order = [group for group in plot_order if group in groups_present]
    
    sns.barplot(
        data=df_auc,
        x='Group',
        y='AUC',
        order=plot_order, # Use the specified order
        palette=COLOR_PALETTE,
        ax=ax,
        capsize=0.1,
        errorbar='sd'
    )
    
    y_lim_top = ax.get_ylim()[1]
    ax.set_ylim(top=y_lim_top * 1.5)

    add_stat_annotation(ax, df=df_auc, x_col='Group', y_col='AUC')
    
    ax.set_ylabel('Area Under Curve (OD * Hours)', fontsize=8)
    
    # --- FIX: Set horizontal labels ---
    ax.tick_params(axis='y', labelsize=7)
    ax.tick_params(axis='x', rotation=30, labelsize=7) # Set rotation to 0
    
    plt.tight_layout()
    barplot_path = os.path.join(output_dir, 'figures', f'auc_barplot_{title.replace(" ", "_").lower()}.png')
    plt.savefig(barplot_path, dpi=300)
    print(f"Saved AUC bar plot to: {barplot_path}")
    plt.close(fig)

    # --- Statistical Analysis on AUC Values (Console Output) ---
    print("\n--- Statistical Analysis on AUC Values (Console Output) ---")
    
    groups_to_test = [g for g in df_auc['Group'].unique() if 'Negative control' not in g]
    groups_for_anova = [df_auc[df_auc['Group'] == g]['AUC'].dropna().values for g in groups_to_test]
    groups_for_anova = [g for g in groups_for_anova if len(g) > 1]
    
    if len(groups_for_anova) < 2:
         print("Could not perform stats: fewer than two experimental groups with sufficient data.")
         return

    f_stat, p_value = f_oneway(*groups_for_anova)
    print(f"ANOVA Test Results: F-statistic={f_stat:.4f}, P-value={p_value:.4f}")

    if p_value < 0.05:
        print("Tukey's HSD post-hoc test:\n")
        tukey_data = df_auc[df_auc['Group'].isin(groups_to_test)]
        tukey_result = pairwise_tukeyhsd(
            endog=tukey_data['AUC'], 
            groups=tukey_data['Group'], 
            alpha=0.05
        )
        print(tukey_result)
    else:
        print("The result is not statistically significant.")
        

def main():
    """
    Main function to load all processed datasets and run AUC analysis on each.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.join(script_dir, '..')
    
    processed_data_dir = os.path.join(project_root, 'data', 'processed')
    results_dir = os.path.join(project_root, 'results')

    os.makedirs(os.path.join(results_dir, 'figures'), exist_ok=True)
    os.makedirs(os.path.join(results_dir, 'tables'), exist_ok=True)

    files_to_analyze = {
        'clean_initial_growth.csv': 'Initial Growth Assay',
        'clean_replicate_growth.csv': 'Replicate Growth Assay'
    }

    print("--- Starting: 06_auc_analysis ---")

    for filename, title in files_to_analyze.items():
        input_path = os.path.join(processed_data_dir, filename)
        
        if not os.path.exists(input_path):
            print(f"\nWarning: Processed data file not found: {input_path}. Skipping analysis.")
            continue
            
        print(f"\nLoading processed data from: {input_path}")
        df = pd.read_csv(input_path)
        
        analyze_growth_with_auc(df, title, results_dir)

    print("\n--- AUC analysis script finished. ---")

if __name__ == '__main__':
    main()
