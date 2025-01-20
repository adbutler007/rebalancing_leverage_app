import argparse
import numpy as np
from multiprocessing import Pool, cpu_count
import time
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import pandas as pd
from plotting import MatPlotting
import os

def annualize_vol(daily_vol):
    return daily_vol * np.sqrt(252)

def calculate_sharpe(daily_returns):
    """Calculate proper annualized Sharpe ratio using arithmetic returns"""
    if len(daily_returns) < 2:
        return np.nan
    
    ann_return = np.mean(daily_returns) * 252
    ann_vol = np.std(daily_returns) * np.sqrt(252)
    
    if ann_vol == 0:
        return np.nan
    return ann_return / ann_vol

def compute_cagr_safe(final_value, years):
    """Safe CAGR calculation with bankruptcy handling"""
    if final_value <= 0:
        return np.nan
    return final_value ** (1 / years) - 1

def run_simulation(args):
    """Executes one full simulation with proper drift and bankruptcy handling"""
    sim_idx, params = args
    np.random.seed(params['seed'] + sim_idx if params['seed'] is not None else None)
    
    # Generate correlated returns using proper drift
    returns = np.random.multivariate_normal(
        params['daily_drift'],
        params['cov_matrix'],
        size=params['total_days']
    )
    
    # Initialize portfolios
    target_weights = np.array(params['target_weights'])
    rebalance_days = params['rebalance_days']
    bankruptcy_threshold = 0.001  # 99% loss threshold
    
    # Rebalanced portfolio
    current_weights = target_weights.copy()
    reb_daily_rets = []
    reb_value = 1.0
    
    # Non-rebalanced portfolio
    nonreb_daily_rets = []
    nonreb_value = 1.0
    nonreb_weights = target_weights.copy()
    
    for t in range(params['total_days']):
        asset_ret = returns[t]
        
        # ===== Rebalanced Portfolio =====
        port_ret = np.dot(current_weights, asset_ret)
        reb_value *= (1 + port_ret)
        
        # Bankruptcy check
        if reb_value < bankruptcy_threshold:
            reb_value = bankruptcy_threshold
            port_ret = -1.0  # Full loss for remaining period
            reb_daily_rets.extend([-1.0]*(params['total_days']-t))
            break
            
        reb_daily_rets.append(port_ret)
        new_weights = (current_weights * (1 + asset_ret)) / (1 + port_ret)
        if t in rebalance_days:
            new_weights = target_weights.copy()
        current_weights = new_weights
        
        # ===== Non-Rebalanced Portfolio =====
        port_ret_non = np.dot(nonreb_weights, asset_ret)
        nonreb_value *= (1 + port_ret_non)
        
        # Bankruptcy check
        if nonreb_value < bankruptcy_threshold:
            nonreb_value = bankruptcy_threshold
            port_ret_non = -1.0  # Full loss for remaining period
            nonreb_daily_rets.extend([-1.0]*(params['total_days']-t))
            break
            
        nonreb_daily_rets.append(port_ret_non)
        new_weights_non = (nonreb_weights * (1 + asset_ret)) / (1 + port_ret_non)
        nonreb_weights = new_weights_non
    
    # Calculate metrics
    cagr_reb = compute_cagr_safe(reb_value, params['horizon_years'])
    cagr_non = compute_cagr_safe(nonreb_value, params['horizon_years'])
    
    sharpe_reb = calculate_sharpe(np.array(reb_daily_rets))
    sharpe_non = calculate_sharpe(np.array(nonreb_daily_rets))
    
    vol_reb = annualize_vol(np.std(reb_daily_rets)) if len(reb_daily_rets) > 1 else np.nan
    vol_non = annualize_vol(np.std(nonreb_daily_rets)) if len(nonreb_daily_rets) > 1 else np.nan
    
    # Add NaN checks
    if np.isnan(sharpe_reb) or np.isnan(sharpe_non):
        print(f"Warning: NaN Sharpe ratio detected in simulation {sim_idx}")
        print(f"Vol_reb: {vol_reb}, Vol_non: {vol_non}")
        print(f"CAGR_reb: {cagr_reb}, CAGR_non: {cagr_non}")
    
    return (
        cagr_reb - cagr_non,
        vol_reb - vol_non,
        sharpe_reb - sharpe_non,
        sharpe_reb,
        sharpe_non
    )

def plot_simulation_results(results, params):
    """Create comprehensive visualization of simulation results with trimmed outliers"""
    # Create plots directory if it doesn't exist
    plots_dir = 'plots'
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)
        print(f"Created directory: {plots_dir}")
    
    # Extract and trim results
    cagr_diff = np.array([r[0] for r in results])
    vol_diff = np.array([r[1] for r in results])
    sharpe_diff = np.array([r[2] for r in results])
    reb_sharpes = np.array([r[3] for r in results])
    non_reb_sharpes = np.array([r[4] for r in results])
    
    # Debug: Print initial data stats
    print("\nInitial Data Statistics:")
    print(f"Total simulations: {len(results)}")
    print(f"NaN counts:")
    print(f"CAGR diff: {np.isnan(cagr_diff).sum()}")
    print(f"Vol diff: {np.isnan(vol_diff).sum()}")
    print(f"Sharpe diff: {np.isnan(sharpe_diff).sum()}")
    print(f"Reb Sharpes: {np.isnan(reb_sharpes).sum()}")
    print(f"Non-reb Sharpes: {np.isnan(non_reb_sharpes).sum()}")
    
    # Remove NaN values before analysis
    valid_mask = ~np.isnan(cagr_diff)
    cagr_diff = cagr_diff[valid_mask]
    vol_diff = vol_diff[valid_mask]
    sharpe_diff = sharpe_diff[valid_mask]
    reb_sharpes = reb_sharpes[valid_mask]
    non_reb_sharpes = non_reb_sharpes[valid_mask]
    
    print(f"\nAfter NaN removal:")
    print(f"Valid simulations: {len(cagr_diff)}")
    
    # Print pre-trimming stats
    print("\nBefore trimming:")
    print(f"CAGR diff range: [{np.min(cagr_diff):.2%}, {np.max(cagr_diff):.2%}]")
    print(f"Mean CAGR diff: {np.mean(cagr_diff):.2%}")
    print(f"Median CAGR diff: {np.median(cagr_diff):.2%}")
    
    # Debug: Print value ranges
    print("\nValue ranges:")
    print(f"CAGR diff: [{np.min(cagr_diff):.4f}, {np.max(cagr_diff):.4f}]")
    print(f"Vol diff: [{np.min(vol_diff):.4f}, {np.max(vol_diff):.4f}]")
    print(f"Sharpe diff: [{np.min(sharpe_diff):.4f}, {np.max(sharpe_diff):.4f}]")
    
    # Trim the extreme 1% from both tails based on CAGR diff
    lower_bound = np.percentile(cagr_diff, 1)
    upper_bound = np.percentile(cagr_diff, 99)
    mask = (cagr_diff >= lower_bound) & (cagr_diff <= upper_bound)
    
    # Apply mask to all arrays
    cagr_diff = cagr_diff[mask]
    vol_diff = vol_diff[mask]
    sharpe_diff = sharpe_diff[mask]
    reb_sharpes = reb_sharpes[mask]
    non_reb_sharpes = non_reb_sharpes[mask]
    
    print(f"\nAfter trimming:")
    print(f"Remaining simulations: {len(cagr_diff)}")
    print(f"CAGR diff range: [{np.min(cagr_diff):.2%}, {np.max(cagr_diff):.2%}]")
    
    # Generate KDE data with explicit bw_method
    kde = sns.kdeplot(data=cagr_diff, bw_method=0.2)
    x = kde.lines[0].get_xdata()
    y = kde.lines[0].get_ydata()
    plt.close()  # Close the temporary plot
    
    print(f"\nKDE data points: {len(x)}")
    print(f"x range: [{min(x):.4f}, {max(x):.4f}]")
    print(f"y range: [{min(y):.4f}, {max(y):.4f}]")
    
    # Create single density plot without splitting
    plot_df = pd.DataFrame({
        'Density': y
    }, index=x)
    
    print("\nFinal plot DataFrame:")
    print(f"Shape: {plot_df.shape}")
    print(f"Columns: {plot_df.columns}")
    print(f"Index range: [{plot_df.index.min():.4f}, {plot_df.index.max():.4f}]")
    print(f"Sample of data:\n{plot_df.head()}")
    
    density_plotter = MatPlotting(plot_df)
    density_plotter.plot(os.path.join(plots_dir, 'cagr_distribution'), plot_type='density')
    
    # 2. Performance Probability
    prob_data = {
        'Value': [np.mean(cagr_diff < 0), np.mean(cagr_diff > 0)]
    }
    prob_df = pd.DataFrame(prob_data, index=['Underperform', 'Outperform'])
    prob_plotter = MatPlotting(prob_df)
    prob_plotter.plot(os.path.join(plots_dir, 'performance_probability'), plot_type='bar')
    
    # 3. Magnitude Distribution
    magnitude_data = {
        'Mean': [np.mean(cagr_diff[cagr_diff < 0]), np.mean(cagr_diff[cagr_diff > 0])],
        'Median': [np.median(cagr_diff[cagr_diff < 0]), np.median(cagr_diff[cagr_diff > 0])]
    }
    magnitude_df = pd.DataFrame(magnitude_data, index=['Underperformance', 'Outperformance'])
    magnitude_plotter = MatPlotting(magnitude_df)
    magnitude_plotter.plot(os.path.join(plots_dir, 'magnitude_distribution'), plot_type='bar')
    
    # Print summary statistics
    print("\nAfter trimming:")
    print(f"Trimmed {len(results) - len(cagr_diff)} outliers ({(1 - len(cagr_diff)/len(results)):.1%} of samples)")
    print(f"CAGR diff range: [{np.min(cagr_diff):.2%}, {np.max(cagr_diff):.2%}]")
    print(f"Mean CAGR diff: {np.mean(cagr_diff):.2%}")
    print(f"Median CAGR diff: {np.median(cagr_diff):.2%}")
    print(f"Probability of underperformance: {np.mean(cagr_diff < 0):.1%}")
    print(f"Mean underperformance: {np.mean(cagr_diff[cagr_diff < 0]):.2%}")
    print(f"Mean outperformance: {np.mean(cagr_diff[cagr_diff > 0]):.2%}")
    
    return {
        'plots/cagr_distribution.png': 'CAGR Difference Distribution (Density)',
        'plots/performance_probability.png': 'Probability of Under/Outperformance',
        'plots/magnitude_distribution.png': 'Magnitude of Under/Outperformance'
    }

def main():
    start_time = time.time()
    
    parser = argparse.ArgumentParser(description='Portfolio Simulation')
    parser.add_argument('--num_simulations', type=int, required=True, 
                       help='Number of Monte Carlo simulations')
    parser.add_argument('--horizon_years', type=float, required=True,
                       help='Investment horizon in years')
    parser.add_argument('--rebalance_freq', type=str, required=True,
                       choices=['daily','weekly','monthly','quarterly','annual'],
                       help='Rebalancing frequency')
    parser.add_argument('--target_weights', type=float, nargs=2, required=True,
                       help='Target weights for assets [w1 w2]')
    parser.add_argument('--sharpe1', type=float, required=True,
                       help='Annualized Sharpe ratio for asset 1')
    parser.add_argument('--sharpe2', type=float, required=True,
                       help='Annualized Sharpe ratio for asset 2')
    parser.add_argument('--sigma1', type=float, required=True,
                       help='Annualized volatility for asset 1')
    parser.add_argument('--sigma2', type=float, required=True,
                       help='Annualized volatility for asset 2')
    parser.add_argument('--rho', type=float, required=True,
                       help='Correlation between assets')
    parser.add_argument('--seed', type=int, default=None,
                       help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not -1 <= args.rho <= 1:
        raise ValueError("Correlation must be between -1 and 1")
    
    # Convert annualized parameters to daily with Itô correction
    # daily_drift = [
    #     (args.sharpe1 * args.sigma1 - 0.5 * args.sigma1**2) / 252,
    #     (args.sharpe2 * args.sigma2 - 0.5 * args.sigma2**2) / 252
    # ]

    daily_drift = [
        (args.sharpe1 * args.sigma1) / 252,
        (args.sharpe2 * args.sigma2) / 252
    ]
    
    sigma1_daily = args.sigma1 / np.sqrt(252)
    sigma2_daily = args.sigma2 / np.sqrt(252)
    
    cov_matrix = np.array([
        [sigma1_daily**2, args.rho * sigma1_daily * sigma2_daily],
        [args.rho * sigma1_daily * sigma2_daily, sigma2_daily**2]
    ])
    
    # Calculate rebalance days
    freq_map = {
        'daily': 1,
        'weekly': 5,
        'monthly': 21,
        'quarterly': 63,
        'annual': 252
    }
    total_days = int(args.horizon_years * 252)
    freq = freq_map[args.rebalance_freq]
    rebalance_days = set(range(freq-1, total_days, freq))
    
    # Prepare simulation parameters
    params = {
        'daily_drift': daily_drift,
        'cov_matrix': cov_matrix,
        'target_weights': args.target_weights,
        'rebalance_days': rebalance_days,
        'total_days': total_days,
        'horizon_years': args.horizon_years,
        'seed': args.seed
    }
    
    # Run simulations in parallel
    with Pool(cpu_count()) as pool:
        tasks = [(i, params) for i in range(args.num_simulations)]
        results = pool.map(run_simulation, tasks)
    
    # Process results
    cagr_diff = [r[0] for r in results]
    vol_diff = [r[1] for r in results]
    sharpe_diff = [r[2] for r in results]
    reb_sharpes = [r[3] for r in results]
    non_reb_sharpes = [r[4] for r in results]
    
    # Filter valid results
    valid_mask = ~np.isnan(cagr_diff)
    cagr_diff = np.array(cagr_diff)[valid_mask]
    vol_diff = np.array(vol_diff)[valid_mask]
    sharpe_diff = np.array(sharpe_diff)[valid_mask]
    reb_sharpes = np.array(reb_sharpes)[valid_mask]
    non_reb_sharpes = np.array(non_reb_sharpes)[valid_mask]
    
    # Calculate statistics
    def calc_stats(arr):
        return {
            'mean': np.nanmean(arr),
            'median': np.nanmedian(arr),
            'std': np.nanstd(arr),
            '5th': np.nanpercentile(arr, 5),
            '95th': np.nanpercentile(arr, 95)
        }
    
    stats = {
        'cagr': calc_stats(cagr_diff),
        'vol': calc_stats(vol_diff),
        'sharpe': calc_stats(sharpe_diff)
    }
    
    # Print results
    print("\nPerformance Difference Statistics (Rebalanced - Non-Rebalanced):")
    for metric in ['cagr', 'vol', 'sharpe']:
        print(f"\n{metric.upper()}:")
        print(f"  Mean:   {stats[metric]['mean']:.6f}")
        print(f"  Median: {stats[metric]['median']:.6f}")
        print(f"  Std:    {stats[metric]['std']:.6f}")
        print(f"  5th-95th Percentiles: [{stats[metric]['5th']:.6f}, {stats[metric]['95th']:.6f}]")

    print("\nAggregate Sharpe Ratio Analysis:")
    print(f"Median Rebalanced Sharpe: {np.nanmedian(reb_sharpes):.6f}")
    print(f"Median Non-rebalanced Sharpe: {np.nanmedian(non_reb_sharpes):.6f}")
    
    # Add aggregate statistics
    all_reb_sharpes = [r[0]/r[1] if r[1] != 0 else np.nan for r in zip(cagr_diff, vol_diff)]
    all_non_reb_sharpes = [r[0]/r[1] if r[1] != 0 else np.nan for r in zip(cagr_diff, vol_diff)]
    
    print("\nAggregate Sharpe Ratio Analysis:")
    print(f"Mean Rebalanced Sharpe: {np.nanmean(all_reb_sharpes):.6f}")
    print(f"Mean Non-rebalanced Sharpe: {np.nanmean(all_non_reb_sharpes):.6f}")
    
    # Prepare parameters for plotting
    plot_params = {
        'num_simulations': args.num_simulations,
        'horizon_years': args.horizon_years,
        'rebalance_freq': args.rebalance_freq
    }
    
    # Create plots
    plot_results = plot_simulation_results(results, plot_params)
    
    elapsed_time = time.time() - start_time
    print(f"\nTotal execution time: {elapsed_time:.2f} seconds")

    return plot_results

if __name__ == '__main__':
    main()

## RUNNING INSTRUCTIONS
# python main.py   --num_simulations 1000   --horizon_years 10   --rebalance_freq monthly   --target_weights 1 1   --sharpe1 0.3   --sharpe2 0.3   --sigma1 0.15   --sigma2 0.15   --rho 0   --seed 42
