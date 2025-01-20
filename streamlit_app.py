import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from main import calculate_sharpe, run_simulation, compute_cagr_safe, annualize_vol
from plotting import MatPlotting
from multiprocessing import Pool, cpu_count
from scipy import stats
import seaborn as sns
import concurrent.futures
import time

# Set page config
st.set_page_config(
    page_title="Rebalancing Premium Simulator",
    page_icon="📈",
    layout="wide"
)

# Custom CSS styling
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,wght@0,400;0,500;0,700;1,400&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.stButton>button {
    background-color: #323A46;
    color: white;
    border-radius: 8px;
    padding: 0.5rem 1rem;
    border: none;
}

.stButton>button:hover {
    background-color: #3A6A9C;
    color: white;
}

.stSlider [data-baseweb="slider"] {
    color: #323A46;
}

[data-baseweb="select"] {
    border-radius: 8px;
}
</style>
""", unsafe_allow_html=True)

def run_simulation_wrapper(params):
    """Wrapper function to run simulations with ProcessPoolExecutor"""
    print(f"Debug: Starting simulation with params: {params}")
    start_time = time.time()
    
    results = []
    # Use ProcessPoolExecutor with max_workers based on CPU count
    max_workers = min(4, cpu_count())  # Limit to 4 processes for cloud deployment
    print(f"Debug: Using {max_workers} workers")
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Create tasks
        tasks = [(i, params) for i in range(params['num_simulations'])]
        # Submit all tasks at once since ProcessPoolExecutor handles memory better
        futures = [executor.submit(run_simulation, task) for task in tasks]
        
        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                print(f"Debug: Error in simulation: {e}")
                continue
    
    elapsed_time = time.time() - start_time
    print(f"Debug: Simulation completed in {elapsed_time:.2f} seconds")
    return results

def main():
    st.title("Portfolio Rebalancing Premium Simulator")
    st.markdown("Explore the impact of portfolio rebalancing under different market conditions")

    with st.sidebar:
        st.header("Simulation Parameters")
        
        col1, col2 = st.columns(2)
        with col1:
            num_simulations = st.slider("Number of Simulations", 100, 10000, 1000)
            horizon_years = st.slider("Investment Horizon (Years)", 1, 30, 10)
            rebalance_freq = st.selectbox(
                "Rebalance Frequency",
                ['daily', 'weekly', 'monthly', 'quarterly', 'annual'],
                index=2
            )
            
        with col2:
            sharpe1 = st.number_input("Asset 1 Sharpe Ratio", 0.0, 2.0, 0.3)
            sharpe2 = st.number_input("Asset 2 Sharpe Ratio", 0.0, 2.0, 0.3)
            sigma1 = st.number_input("Asset 1 Volatility", 0.05, 0.50, 0.15)
            sigma2 = st.number_input("Asset 2 Volatility", 0.05, 0.50, 0.15)
            
        rho = st.slider("Correlation (ρ)", -1.0, 1.0, 0.0)
        target_weights = st.columns(2)
        with target_weights[0]:
            w1 = st.number_input("Asset 1 Weight", 0.0, 2.0, 1.0)
        with target_weights[1]:
            w2 = st.number_input("Asset 2 Weight", 0.0, 2.0, 1.0)
            
        seed = st.number_input("Random Seed (optional)", value=None)

    if st.button("Run Simulation", type="primary"):
        with st.spinner("Running simulations..."):
            progress_bar = st.progress(0)
            
            # Convert parameters to simulation format
            params = {
                'num_simulations': num_simulations,
                'horizon_years': horizon_years,
                'rebalance_freq': rebalance_freq,
                'target_weights': [w1, w2],
                'sharpe1': sharpe1,
                'sharpe2': sharpe2,
                'sigma1': sigma1,
                'sigma2': sigma2,
                'rho': rho,
                'seed': seed
            }

            # Calculate derived parameters
            daily_drift = [
                (params['sharpe1'] * params['sigma1']) / 252,
                (params['sharpe2'] * params['sigma2']) / 252
            ]
            
            sigma1_daily = params['sigma1'] / np.sqrt(252)
            sigma2_daily = params['sigma2'] / np.sqrt(252)
            
            cov_matrix = np.array([
                [sigma1_daily**2, params['rho'] * sigma1_daily * sigma2_daily],
                [params['rho'] * sigma1_daily * sigma2_daily, sigma2_daily**2]
            ])
            
            freq_map = {
                'daily': 1,
                'weekly': 5,
                'monthly': 21,
                'quarterly': 63,
                'annual': 252
            }
            total_days = int(params['horizon_years'] * 252)
            freq = freq_map[params['rebalance_freq']]
            rebalance_days = set(range(freq-1, total_days, freq))
            
            sim_params = {
                'daily_drift': daily_drift,
                'cov_matrix': cov_matrix,
                'target_weights': params['target_weights'],
                'rebalance_days': rebalance_days,
                'total_days': total_days,
                'horizon_years': params['horizon_years'],
                'seed': params['seed'],
                'num_simulations': params['num_simulations']
            }

            # Run simulations with progress tracking
            results = run_simulation_wrapper(sim_params)
            progress_bar.empty()
            
            # Process results
            cagr_diff = np.array([r[0] for r in results])
            vol_diff = np.array([r[1] for r in results])
            sharpe_diff = np.array([r[2] for r in results])
            
            # Filter valid results
            valid_mask = ~np.isnan(cagr_diff)
            cagr_diff = cagr_diff[valid_mask]
            vol_diff = vol_diff[valid_mask]
            sharpe_diff = sharpe_diff[valid_mask]

            # Create plots
            plot_data = {
                'cagr_diff': cagr_diff,
                'vol_diff': vol_diff,
                'sharpe_diff': sharpe_diff
            }
            
            st.success("Simulation completed!")
            
        # Display results
        st.header("Simulation Results")
        
        # Create columns for metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Mean CAGR Difference", f"{np.mean(cagr_diff):.2%}")
            st.metric("Probability of Outperformance", 
                     f"{np.mean(cagr_diff > 0):.1%}")
            
        with col2:
            st.metric("Mean Underperformance", 
                     f"{np.mean(cagr_diff[cagr_diff < 0]):.2%}")
            st.metric("Mean Outperformance", 
                     f"{np.mean(cagr_diff[cagr_diff > 0]):.2%}")
            
        with col3:
            st.metric("Positive Outcome Confidence", 
                     f"{stats.percentileofscore(cagr_diff, 0):.1f}%")
            st.metric("95% CI Width", 
                     f"{np.percentile(cagr_diff, 95) - np.percentile(cagr_diff, 5):.2%}")

        # Create plots
        st.subheader("Performance Distribution")
        fig = plt.figure(figsize=(12, 6), dpi=300)
        sns.kdeplot(cagr_diff, fill=True, color="#3A6A9C", alpha=0.5)
        plt.axvline(0, color='#323A46', linestyle='--')
        plt.xlabel("CAGR Difference (Rebalanced - Non-Rebalanced)")
        plt.ylabel("Density")
        st.pyplot(fig, dpi=300)

        st.subheader("Performance Magnitude Analysis")
        magnitude_data = {
            'Mean': [np.mean(cagr_diff[cagr_diff < 0]), np.mean(cagr_diff[cagr_diff > 0])],
            'Median': [np.median(cagr_diff[cagr_diff < 0]), np.median(cagr_diff[cagr_diff > 0])]
        }
        
        fig = plt.figure(figsize=(12, 6), dpi=300)
        x = np.arange(2)
        width = 0.35
        
        # Create bars
        mean_bars = plt.bar(x - width/2, magnitude_data['Mean'], width, label='Mean', color="#3A6A9C")
        median_bars = plt.bar(x + width/2, magnitude_data['Median'], width, label='Median', color="#14CFA6")
        
        # Add value labels on top of each bar
        def add_labels(bars):
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.1%}',
                        ha='center', va='bottom')
        
        add_labels(mean_bars)
        add_labels(median_bars)
        
        plt.xticks(x, ['Underperformance', 'Outperformance'])
        plt.ylabel('CAGR Difference')
        plt.legend()
        plt.grid(True, alpha=0.3)
        st.pyplot(fig, dpi=300)

if __name__ == "__main__":
    main()