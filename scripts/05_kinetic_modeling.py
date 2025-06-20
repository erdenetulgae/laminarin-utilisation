import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.stats import f_oneway
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# Import the Gaussian Process library from scikit-learn
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel
# Import a class for creating custom legend handles
import matplotlib.patches as mpatches


# --- Configuration ---
# This mapping should match the group names in the clean_replicate_growth.csv file
GROUP_MAPPING = {
    'D1': 'Negative control', 'D2': 'Negative control', 'D3': 'Negative control',
    'D4': 'Positive control (pAX)', 'D5': 'Positive control (pAX)', 'D6': 'Positive control (pAX)',
    'D7': 'Sample C',         'D8': 'Sample C',         'D9': 'Sample C',
    'D10': 'Sample C + exo', 'D11': 'Sample C + exo', 'D12': 'Sample C + exo'
}

# Define the color palette for consistent plotting
COLOR_PALETTE = {
    'Positive control (pAX)': '#2ca02c',     # Green for Positive
    'Sample C': '#1f77b4',                 # Blue for Sample C
}

def analyze_growth_kinetics(df, output_dir):
    """
    Fits a Gaussian Process model to each replicate, extracts kinetic parameters
    from the smoothed curves, and performs statistical analysis on the parameters.
    Saves validation plots and parameter tables.

    Args:
        df (pd.DataFrame): The clean, long-format DataFrame to analyze.
        output_dir (str): The path to the root 'results' directory.
    """
    print("Preparing data for GP modeling...")
    # Use the background-subtracted data for modeling
    df_model = df[['Time_Hours', 'Well', 'Group', 'OD600_Normalized']].copy()
    df_model.rename(columns={'OD600_Normalized': 'OD'}, inplace=True)
    
    results = []
    # --- Define the specific groups we want to model ---
    groups_to_model = ['Positive control (pAX)', 'Sample C']
    
    # Define the kernel for the GP
    kernel = C(1.0) * RBF(length_scale=1.0) + WhiteKernel(noise_level=1.0)
    
    # --- Plot 1: Individual Replicate Fits (Portrait Style) ---
    fig_ind, axes_ind = plt.subplots(3, 2, figsize=(4.0, 6.0), sharex=True, sharey=True)

    print(f"Fitting Gaussian Process models for groups: {groups_to_model}...")
    
    for col_idx, group_name in enumerate(groups_to_model):
        wells_in_group = [well for well, g_name in GROUP_MAPPING.items() if g_name == group_name]
        axes_ind[0, col_idx].set_title(group_name, fontsize=10)

        for row_idx, well_id in enumerate(wells_in_group):
            ax = axes_ind[row_idx, col_idx]
            well_data = df_model[df_model['Well'] == well_id].copy()
            if well_data.empty:
                ax.text(0.5, 0.5, 'No Data', ha='center', va='center')
                continue
                
            t_data = well_data['Time_Hours'].values.reshape(-1, 1)
            od_data = well_data['OD'].values

            gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, random_state=42)
            gp.fit(t_data, od_data)

            t_pred = np.linspace(t_data.min(), t_data.max(), 200).reshape(-1, 1)
            od_pred, sigma = gp.predict(t_pred, return_std=True)

            growth_rate = np.gradient(od_pred, t_pred.flatten())
            max_rate_value = np.max(growth_rate)
            try:
                lag_idx = np.where(growth_rate >= 0.05 * max_rate_value)[0][0]
                lam = t_pred.flatten()[lag_idx]
            except IndexError:
                lam = t_pred.flatten()[-1] / 2
            
            post_lag_indices = np.where(t_pred.flatten() >= lam)
            K = np.max(od_pred[post_lag_indices])
            mu = np.max(growth_rate[post_lag_indices])
            
            results.append({'Well': well_id, 'Group': group_name, 'K': K, 'mu': mu, 'lambda': lam})

            plot_color = COLOR_PALETTE.get(group_name, 'black')
            ax.plot(t_data, od_data, 'o', alpha=0.5, color=plot_color, markersize=3)
            ax.plot(t_pred, od_pred, '-', color=plot_color, lw=2)
            ax.fill_between(t_pred.flatten(), od_pred - 1.96 * sigma, od_pred + 1.96 * sigma, alpha=0.2, color=plot_color)
            ax.grid(True, which='both', linestyle='--', linewidth=0.5)
            ax.tick_params(axis='both', which='major', labelsize=7)

    for i in range(3):
        axes_ind[i, 0].set_ylabel(f'Replicate {i+1}\n(OD600)', fontsize=8)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig_ind.supxlabel('Time (Hours)', fontsize=8, y = 0.01)

    validation_plot_path = os.path.join(output_dir, 'figures', 'kinetic_model_fits_individual.png')
    plt.savefig(validation_plot_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved individual validation plot to: {validation_plot_path}")
    plt.close(fig_ind)

    # --- Plot 2: Combined Replicate Fits ---
    fig_comb, axes_comb = plt.subplots(1, 2, figsize=(6.5, 3.0), sharey=True)
    
    # Store the master GP curves for the final overlay plot
    master_curves = {}

    for i, group_name in enumerate(groups_to_model):
        ax = axes_comb[i]
        group_data = df_model[df_model['Group'] == group_name]
        
        t_data_all = group_data['Time_Hours'].values.reshape(-1, 1)
        od_data_all = group_data['OD'].values

        gp_all = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, random_state=42)
        gp_all.fit(t_data_all, od_data_all)

        t_pred = np.linspace(t_data.min(), t_data.max(), 200).reshape(-1, 1)
        od_pred_all, sigma_all = gp_all.predict(t_pred, return_std=True)
        
        master_curves[group_name] = {'t_pred': t_pred, 'od_pred': od_pred_all, 'sigma': sigma_all}

        plot_color = COLOR_PALETTE.get(group_name, 'black')
        
        # Plot each replicate's noisy data separately for visibility
        for well in group_data['Well'].unique():
            well_data = group_data[group_data['Well'] == well]
            ax.plot(well_data['Time_Hours'], well_data['OD'], 'o', alpha=0.7, markersize=3, color=plot_color, label=None)
        
        ax.plot(t_pred, od_pred_all, '-', label='GP Mean Curve', color=plot_color, lw=2)
        ax.fill_between(t_pred.flatten(), od_pred_all - 1.96 * sigma_all, od_pred_all + 1.96 * sigma_all, alpha=0.3, color=plot_color, label='95% Confidence')
        
        ax.set_title(group_name, fontsize=10)
        ax.set_xlabel('Time (Hours)', fontsize=8)
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        ax.tick_params(axis='both', which='major', labelsize=7)

    # for i, group_name in enumerate(groups_to_model):
    #     ax = axes_comb[i]
    #     group_data = df_model[df_model['Group'] == group_name]
        
    #     t_data_all = group_data['Time_Hours'].values.reshape(-1, 1)
    #     od_data_all = group_data['OD'].values

    #     gp_all = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, random_state=42)
    #     gp_all.fit(t_data_all, od_data_all)

    #     t_pred = np.linspace(t_data.min(), t_data.max(), 200).reshape(-1, 1)
    #     od_pred_all, sigma_all = gp_all.predict(t_pred, return_std=True)
        
    #     master_curves[group_name] = {'t_pred': t_pred, 'od_pred': od_pred_all, 'sigma': sigma_all}

    #     plot_color = COLOR_PALETTE.get(group_name, 'black')
    #     ax.plot(t_data_all, od_data_all, 'o', label='All Replicates', alpha=1.0, color=plot_color, markersize=1)
    #     ax.plot(t_pred, od_pred_all, '-', label='GP Mean Curve', color='black', lw=2)
    #     ax.fill_between(t_pred.flatten(), od_pred_all - 1.96 * sigma_all, od_pred_all + 1.96 * sigma_all, alpha=0.3, color=plot_color, label='95% Confidence')
        
    #     ax.set_title(group_name, fontsize=10)
    #     ax.set_xlabel('Time (Hours)', fontsize=8)
    #     ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    #     ax.tick_params(axis='both', which='major', labelsize=7)

    axes_comb[0].set_ylabel('Background Subtracted OD600', fontsize=8)
    handles_comb, labels_comb = axes_comb[0].get_legend_handles_labels()
    fig_comb.legend(handles_comb, labels_comb, loc='lower center', bbox_to_anchor=(0.5, -0.05), ncol=3, frameon=False, fontsize=8)
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    
    combined_plot_path = os.path.join(output_dir, 'figures', 'kinetic_model_fits_by_group.png')
    plt.savefig(combined_plot_path, dpi=300, bbox_inches='tight')
    print(f"Saved combined validation plot to: {combined_plot_path}")
    plt.close(fig_comb)
    
    # # --- Plot 3: Overlay of Pooled Replicate Curves ---
    # fig_overlay, ax_overlay = plt.subplots(1, 1, figsize=(1.5, 3.0))
    
    # for group_name, curve_data in master_curves.items():
    #     t_pred = curve_data['t_pred']
    #     od_pred = curve_data['od_pred']
    #     sigma = curve_data['sigma']
    #     plot_color = COLOR_PALETTE.get(group_name, 'black')
        
    #     ax_overlay.plot(t_pred, od_pred, '-', label=group_name, color=plot_color, lw=2)
    #     ax_overlay.fill_between(t_pred.flatten(), od_pred - 1.96 * sigma, od_pred + 1.96 * sigma, alpha=0.2, color=plot_color)

    # ax_overlay.set_xlabel('Time (Hours)', fontsize=8)
    # ax_overlay.set_ylabel('Background Subtracted OD600', fontsize=8)
    # ax_overlay.tick_params(axis='both', which='major', labelsize=7)
    # ax_overlay.grid(True, which='both', linestyle='--', linewidth=0.5)
    # ax_overlay.legend(fontsize=8, loc='upper left')
    
    # plt.tight_layout()
    # overlay_plot_path = os.path.join(output_dir, 'figures', 'kinetic_model_fits_comparison.png')
    # plt.savefig(overlay_plot_path, dpi=300, bbox_inches='tight')
    # print(f"Saved overlay comparison plot to: {overlay_plot_path}")
    # plt.close(fig_overlay)


    # if not results:
    #     print("Error: Model fitting failed for all wells. Cannot proceed.")
    #     return

    # --- Plot 3: Overlay of Pooled Replicate Curves ---
    fig_overlay, ax_overlay = plt.subplots(1, 1, figsize=(3.7, 2.5))

    # Plot noisy data for each group
    for group_name in groups_to_model:
        group_data = df_model[df_model['Group'] == group_name]
        plot_color = COLOR_PALETTE.get(group_name, 'black')
        ax_overlay.plot(
            group_data['Time_Hours'], group_data['OD'],
            'o', alpha=0.3, markersize=2, color=plot_color, label=f'{group_name} raw' if group_name not in ax_overlay.get_legend_handles_labels()[1] else None
        )

    for group_name, curve_data in master_curves.items():
        t_pred = curve_data['t_pred']
        od_pred = curve_data['od_pred']
        sigma = curve_data['sigma']
        plot_color = COLOR_PALETTE.get(group_name, 'black')
        
        ax_overlay.plot(t_pred, od_pred, '-', label=group_name, color=plot_color, lw=2)
        ax_overlay.fill_between(t_pred.flatten(), od_pred - 1.96 * sigma, od_pred + 1.96 * sigma, alpha=0.2, color=plot_color)

    ax_overlay.set_xlabel('Time (Hours)', fontsize=8)
    ax_overlay.set_ylabel('Background Subtracted OD600', fontsize=8)
    ax_overlay.tick_params(axis='both', which='major', labelsize=7)
    ax_overlay.grid(True, which='both', linestyle='--', linewidth=0.5)
    # ax_overlay.legend(fontsize=8, loc='upper left')
    
    plt.tight_layout()
    overlay_plot_path = os.path.join(output_dir, 'figures', 'kinetic_model_fits_comparison.png')
    plt.savefig(overlay_plot_path, dpi=300, bbox_inches='tight')
    print(f"Saved overlay comparison plot to: {overlay_plot_path}")
    plt.close(fig_overlay)

    # --- Analyze the Extracted Parameters ---
    df_params = pd.DataFrame(results)
    
    params_table_path = os.path.join(output_dir, 'tables', 'kinetic_parameters.csv')
    os.makedirs(os.path.dirname(params_table_path), exist_ok=True)
    df_params.to_csv(params_table_path, index=False)
    print(f"Saved kinetic parameters table to: {params_table_path}")
    
    print("\n--- Extracted Kinetic Parameters from GP Models ---")
    print(df_params)

    for param in ['K', 'mu', 'lambda']:
        print(f"\n--- Statistical Analysis for Parameter: {param.upper()} ---")
        
        groups_for_anova = [df_params[df_params['Group'] == g][param].dropna().values for g in groups_to_model]
        groups_for_anova = [g for g in groups_for_anova if len(g) > 1]

        if len(groups_for_anova) < 2:
            print(f"Could not perform ANOVA for {param}: less than two groups with sufficient data.")
            continue
        
        f_stat, p_value = f_oneway(*groups_for_anova)
        print(f"ANOVA Test Results: F-statistic={f_stat:.4f}, P-value={p_value:.4f}")

        if p_value < 0.05 and len(df_params['Group'].unique()) > 1:
            print("Result is statistically significant. Performing Tukey's HSD test...\n")
            tukey_result = pairwise_tukeyhsd(endog=df_params[df_params['Group'].isin(groups_to_model)][param], 
                                             groups=df_params[df_params['Group'].isin(groups_to_model)]['Group'], 
                                             alpha=0.05)
            print(tukey_result)
        else:
            print("The result is not statistically significant.")

def main():
    """
    Main function to load the processed replicate growth data and run kinetic analysis.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.join(script_dir, '..')
    
    processed_data_path = os.path.join(project_root, 'data', 'processed', 'clean_replicate_growth.csv')
    results_dir = os.path.join(project_root, 'results')

    os.makedirs(os.path.join(results_dir, 'figures'), exist_ok=True)
    os.makedirs(os.path.join(results_dir, 'tables'), exist_ok=True)

    print("--- Starting: 05_kinetic_modeling ---")
    
    if not os.path.exists(processed_data_path):
        print(f"\nError: Processed data file not found: {processed_data_path}.")
        print("Please run '02_wrangle_replicate_growth.py' first.")
        return
        
    print(f"\nLoading processed data from: {processed_data_path}")
    df_processed = pd.read_csv(processed_data_path)
    
    analyze_growth_kinetics(df_processed, results_dir)
    
    print("\n--- Kinetic modeling script finished. ---")

if __name__ == '__main__':
    main()
