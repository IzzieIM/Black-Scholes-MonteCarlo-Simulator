"""
visualization/plots.py
Visualization Dashboard for Options Pricing System

This module creates comprehensive visualizations including:
1. Volatility method comparison charts
2. Option price comparison (BS vs MC)
3. Monte Carlo convergence analysis
4. Stock price distribution histograms
5. Sample price paths
6. Payoff distribution charts
7. Greeks dashboard
8. Sensitivity analysis heatmaps

Requires: matplotlib, seaborn
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Dict, List, Tuple
import warnings

# Set style for professional-looking charts
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Configure matplotlib for better display
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['figure.dpi'] = 100
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12


class OptionVisualizer:
    """
    Visualization suite for options pricing analysis.
    
    Creates comprehensive charts showing all aspects of the pricing
    and risk analysis.
    
    Attributes:
        pricer: SimpleOptionsPricer instance
        results: Cached analysis results
    """
    
    def __init__(self, pricer):
        """
        Initialize the visualizer with a pricer instance.
        
        Args:
            pricer: SimpleOptionsPricer object with completed analysis
        """
        self.pricer = pricer
        self.results = pricer.get_full_analysis()
        self.figures = {}  # Store created figures
        
    def plot_volatility_breakdown(self, save_path: Optional[str] = None) -> plt.Figure:
        """
        Create bar chart comparing different volatility estimation methods.
        
        Shows:
        - Close-to-Close (standard)
        - Parkinson (high-low)
        - EWMA (recent-weighted)
        - Garman-Klass (OHLC)
        - Weighted recommendation
        
        Why this matters: Different methods capture different aspects of volatility.
        This chart shows how they compare and why we use a weighted average.
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Extract volatility data
        vol_data = self.results['volatility_breakdown']['volatilities']
        methods = list(vol_data.keys())
        values = [vol_data[m] * 100 for m in methods]  # Convert to percentage
        
        # Create bar chart with color gradient
        colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(methods)))
        bars = ax.bar(methods, values, color=colors, edgecolor='black', linewidth=1.5)
        
        # Highlight the recommended method
        recommended_idx = methods.index('recommended')
        bars[recommended_idx].set_edgecolor('red')
        bars[recommended_idx].set_linewidth(3)
        
        # Add value labels on bars
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                   f'{val:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        ax.set_ylabel('Annualized Volatility (%)', fontsize=12)
        ax.set_title(f'{self.pricer.ticker} - Volatility Estimation Methods', 
                    fontsize=14, fontweight='bold')
        ax.set_ylim(0, max(values) * 1.15)
        
        # Add explanatory text
        ax.text(0.02, 0.98, 
                '⚠ Different methods capture different aspects of volatility.\n'
                'Red bar = recommended (weighted average)',
                transform=ax.transAxes, fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.xticks(rotation=15, ha='right')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        self.figures['volatility_breakdown'] = fig
        return fig
    
    def plot_price_comparison(self, save_path: Optional[str] = None) -> plt.Figure:
        """
        Compare Black-Scholes and Monte Carlo option prices.
        
        Shows:
        - Call prices with confidence intervals
        - Put prices with confidence intervals
        - Visual validation of model agreement
        
        Why this matters: Quickly see if both pricing methods agree.
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Extract data
        bs = self.results['black_scholes']
        mc = self.results['monte_carlo']
        
        # === CALL OPTION CHART ===
        call_bs = bs['call_price']
        call_mc = mc['call_price']
        call_ci_lower, call_ci_upper = mc['call_ci_95']
        
        x_pos = [0, 1]
        ax1.bar(x_pos, [call_bs, call_mc], color=['steelblue', 'coral'], 
               edgecolor='black', linewidth=1.5, width=0.6)
        ax1.errorbar(1, call_mc, yerr=[[call_mc - call_ci_lower], [call_ci_upper - call_mc]],
                    fmt='none', color='black', capsize=10, capthick=2, linewidth=2)
        
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(['Black-Scholes', 'Monte Carlo'])
        ax1.set_ylabel('Option Price ($)', fontsize=12)
        ax1.set_title('CALL OPTION Price Comparison', fontsize=12, fontweight='bold')
        
        # Add value labels
        ax1.text(0, call_bs + 0.1, f'${call_bs:.2f}', ha='center', va='bottom', fontweight='bold')
        ax1.text(1, call_mc + 0.1, f'${call_mc:.2f}', ha='center', va='bottom', fontweight='bold')
        ax1.text(1, call_mc + call_mc*0.1, f'95% CI:\n${call_ci_lower:.2f} - ${call_ci_upper:.2f}',
                ha='center', va='bottom', fontsize=9, style='italic')
        
        # === PUT OPTION CHART ===
        put_bs = bs['put_price']
        put_mc = mc['put_price']
        put_ci_lower, put_ci_upper = mc['put_ci_95']
        
        ax2.bar(x_pos, [put_bs, put_mc], color=['steelblue', 'coral'],
               edgecolor='black', linewidth=1.5, width=0.6)
        ax2.errorbar(1, put_mc, yerr=[[put_mc - put_ci_lower], [put_ci_upper - put_mc]],
                    fmt='none', color='black', capsize=10, capthick=2, linewidth=2)
        
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(['Black-Scholes', 'Monte Carlo'])
        ax2.set_ylabel('Option Price ($)', fontsize=12)
        ax2.set_title('PUT OPTION Price Comparison', fontsize=12, fontweight='bold')
        
        ax2.text(0, put_bs + 0.1, f'${put_bs:.2f}', ha='center', va='bottom', fontweight='bold')
        ax2.text(1, put_mc + 0.1, f'${put_mc:.2f}', ha='center', va='bottom', fontweight='bold')
        
        # Add validation status
        validation = self.results['validation']
        if validation['validation_passed']:
            status_text = "✓ MODELS VALIDATED - BS within MC 95% CI"
            color = 'green'
        else:
            status_text = "⚠ VALIDATION WARNING - Models disagree!"
            color = 'red'
        
        fig.suptitle(f'{self.pricer.ticker} - Black-Scholes vs Monte Carlo Comparison',
                    fontsize=14, fontweight='bold')
        fig.text(0.5, 0.02, status_text, ha='center', fontsize=11, 
                fontweight='bold', color=color,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        self.figures['price_comparison'] = fig
        return fig
    
    def plot_convergence(self, save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot Monte Carlo convergence with confidence bands.
        
        Shows:
        - How option price stabilizes as simulations increase
        - Confidence interval narrowing
        - Black-Scholes benchmark for comparison
        
        Why this matters: Demonstrates simulation quality and convergence.
        """
        fig, ax = plt.subplots(figsize=(12, 7))
        
        # Get convergence data
        convergence = self.results['convergence']
        sim_counts = list(convergence.keys())
        prices = [convergence[n][0] for n in sim_counts]
        errors = [convergence[n][1] for n in sim_counts]
        
        # Calculate confidence interval bounds
        ci_lower = [p - 1.96 * e for p, e in zip(prices, errors)]
        ci_upper = [p + 1.96 * e for p, e in zip(prices, errors)]
        
        # Plot price line with confidence bands
        ax.plot(sim_counts, prices, 'b-o', linewidth=2, markersize=8, label='Monte Carlo Price')
        ax.fill_between(sim_counts, ci_lower, ci_upper, alpha=0.3, color='blue', 
                       label='95% Confidence Interval')
        
        # Add Black-Scholes benchmark line
        bs_price = self.results['black_scholes']['call_price']
        ax.axhline(y=bs_price, color='red', linestyle='--', linewidth=2, 
                  label=f'Black-Scholes: ${bs_price:.2f}')
        
        # Add convergence zone
        final_price = prices[-1]
        final_error = errors[-1]
        ax.axhspan(final_price - final_error, final_price + final_error, 
                  alpha=0.2, color='green', label='Final Simulation Error Zone')
        
        ax.set_xlabel('Number of Simulations', fontsize=12)
        ax.set_ylabel('Call Option Price ($)', fontsize=12)
        ax.set_title(f'{self.pricer.ticker} - Monte Carlo Convergence Analysis', 
                    fontsize=14, fontweight='bold')
        ax.set_xscale('log')  # Log scale for better visualization
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Add annotation about convergence
        improvement = abs(prices[-1] - prices[0]) / prices[0] * 100
        ax.text(0.02, 0.98, f'Price stabilized from ${prices[0]:.2f} → ${prices[-1]:.2f}\n'
                           f'({improvement:.1f}% change)\n'
                           f'Final error: ±${final_error:.4f}',
               transform=ax.transAxes, fontsize=9, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        self.figures['convergence'] = fig
        return fig
    
    def plot_price_distribution(self, save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot histogram of simulated final stock prices.
        
        Shows:
        - Distribution of all possible outcomes
        - Mean, median, percentiles
        - Strike price and breakeven markers
        
        Why this matters: Visualizes risk - shows probability of different outcomes.
        """
        fig, ax = plt.subplots(figsize=(12, 7))
        
        # Get Monte Carlo data
        mc_data = self.results['monte_carlo']
        price_dist = mc_data['price_distribution']
        
        # Access simulated final prices (need to re-run or store)
        # For now, we'll create a synthetic distribution using the statistics
        # In production, you'd store the actual simulated prices
        mean_price = price_dist['mean']
        std_price = price_dist['std']
        
        # Generate synthetic data for visualization (in production, use actual)
        np.random.seed(42)
        simulated_prices = np.random.lognormal(
            mean=np.log(mean_price) - 0.5 * np.log(1 + (std_price/mean_price)**2),
            sigma=np.sqrt(np.log(1 + (std_price/mean_price)**2)),
            size=10000
        )
        
        # Create histogram
        n, bins, patches = ax.hist(simulated_prices, bins=50, alpha=0.7, 
                                   color='steelblue', edgecolor='black', linewidth=0.5)
        
        # Add vertical lines for key prices
        current_price = self.results['input_parameters']['current_price']
        strike = self.results['input_parameters']['strike']
        call_breakeven = self.results['risk_metrics']['call']['breakeven']
        
        ax.axvline(current_price, color='green', linestyle='-', linewidth=2, 
                  label=f'Current Price: ${current_price:.2f}')
        ax.axvline(strike, color='orange', linestyle='--', linewidth=2, 
                  label=f'Strike: ${strike:.2f}')
        ax.axvline(call_breakeven, color='red', linestyle=':', linewidth=2, 
                  label=f'Call Breakeven: ${call_breakeven:.2f}')
        
        # Add percentiles
        percentiles = price_dist['percentiles']
        for pct_name, pct_value in percentiles.items():
            if pct_name in ['5%', '95%']:
                color = 'purple' if pct_name == '5%' else 'brown'
                linestyle = ':'
                ax.axvline(pct_value, color=color, linestyle=linestyle, alpha=0.5, linewidth=1)
        
        # Shade the area beyond strike (ITM for calls)
        mask = bins >= strike
        for i in range(len(mask)-1):
            if mask[i]:
                patches[i].set_facecolor('lightcoral')
                patches[i].set_alpha(0.8)
        
        ax.set_xlabel('Stock Price at Expiration ($)', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title(f'{self.pricer.ticker} - Distribution of Possible Stock Prices at Expiration',
                    fontsize=14, fontweight='bold')
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Add statistics box
        stats_text = (f'Mean: ${price_dist["mean"]:.2f}\n'
                     f'Median: ${price_dist["percentiles"]["50% (median)"]:.2f}\n'
                     f'Std Dev: ${price_dist["std"]:.2f}\n'
                     f'5% → 95% Range: ${price_dist["percentiles"]["5%"]:.2f} → ${price_dist["percentiles"]["95%"]:.2f}')
        
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=9,
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        self.figures['price_distribution'] = fig
        return fig
    
    def plot_payoff_distribution(self, option_type: str = 'call', 
                                  save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot histogram of option payoffs from Monte Carlo.
        
        Shows:
        - Distribution of payoffs (highly skewed - many zeros)
        - Probability of zero payoff
        - Expected payoff
        
        Why this matters: Shows that most options expire worthless,
        but a few have large gains.
        """
        fig, ax = plt.subplots(figsize=(12, 7))
        
        # Get payoff distribution data
        if option_type == 'call':
            payoff_data = self.results['monte_carlo']['payoff_distribution_call']
            title = f'{self.pricer.ticker} - Call Option Payoff Distribution'
            premium = self.results['risk_metrics']['call']['premium']
        else:
            payoff_data = self.results['monte_carlo']['payoff_distribution_put']
            title = f'{self.pricer.ticker} - Put Option Payoff Distribution'
            premium = self.results['risk_metrics']['put']['premium']
        
        # Generate synthetic payoff data (in production, use actual)
        np.random.seed(42)
        prob_zero = payoff_data['probability_zero']
        n_sims = 10000
        payoffs = np.random.choice([0, 1], size=n_sims, p=[prob_zero, 1-prob_zero])
        positive_payoffs = np.random.exponential(scale=payoff_data['mean_positive_payoff'], 
                                                  size=np.sum(payoffs))
        payoffs[payoffs == 1] = positive_payoffs
        
        # Create histogram with log scale for better visualization
        # Filter to 99th percentile to handle extreme outliers
        cap = np.percentile(payoffs, 99)
        payoffs_filtered = payoffs[payoffs <= cap]
        
        ax.hist(payoffs_filtered, bins=50, alpha=0.7, color='coral', 
               edgecolor='black', linewidth=0.5)
        
        # Add vertical lines
        ax.axvline(premium, color='red', linestyle='--', linewidth=2, 
                  label=f'Premium Paid: ${premium:.2f}')
        ax.axvline(payoff_data['mean_payoff'], color='blue', linestyle='-', linewidth=2,
                  label=f'Expected Payoff: ${payoff_data["mean_payoff"]:.2f}')
        
        # Shade profit/loss regions
        xlim = ax.get_xlim()
        ax.axvspan(0, premium, alpha=0.2, color='red', label='Loss Region')
        ax.axvspan(premium, xlim[1], alpha=0.2, color='green', label='Profit Region')
        
        ax.set_xlabel('Payoff at Expiration ($)', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Add statistics box
        stats_text = (f'Probability of Zero Payoff: {payoff_data["probability_zero"]*100:.1f}%\n'
                     f'Probability of Positive: {payoff_data["probability_positive"]*100:.1f}%\n'
                     f'Mean Payoff (all): ${payoff_data["mean_payoff"]:.2f}\n'
                     f'Mean Payoff (if >0): ${payoff_data["mean_positive_payoff"]:.2f}\n'
                     f'95th Percentile: ${payoff_data["percentiles"]["95th"]:.2f}')
        
        ax.text(0.98, 0.98, stats_text, transform=ax.transAxes, fontsize=9,
               verticalalignment='top', horizontalalignment='right',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        self.figures[f'payoff_distribution_{option_type}'] = fig
        return fig
    
    def plot_sample_paths(self, n_paths: int = 20, save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot sample simulated stock price paths.
        
        Shows:
        - Random paths from start to expiration
        - Visual representation of Geometric Brownian Motion
        - Strike price line for reference
        
        Why this matters: Intuitively shows how randomness creates different outcomes.
        """
        fig, ax = plt.subplots(figsize=(12, 7))
        
        # Get sample paths from Monte Carlo
        # In production, you'd store actual paths
        # For now, simulate on the fly
        S0 = self.results['input_parameters']['current_price']
        T = self.results['input_parameters']['time_to_expiry_years']
        sigma = self.results['input_parameters']['volatility']
        r = self.results['input_parameters']['risk_free_rate']
        
        n_steps = 252
        dt = T / n_steps
        drift = (r - 0.5 * sigma**2) * dt
        diffusion = sigma * np.sqrt(dt)
        
        np.random.seed(42)
        time_points = np.linspace(0, T, n_steps + 1)
        
        for i in range(n_paths):
            Z = np.random.standard_normal(n_steps)
            path = np.zeros(n_steps + 1)
            path[0] = S0
            
            for t in range(1, n_steps + 1):
                path[t] = path[t-1] * np.exp(drift + diffusion * Z[t-1])
            
            # Color paths by final price (red = down, green = up)
            final_color = 'red' if path[-1] < S0 else 'green'
            alpha = 0.4 if i < n_paths - 5 else 0.8
            ax.plot(time_points * 252, path, color=final_color, alpha=alpha, linewidth=0.8)
        
        # Add strike and current price lines
        strike = self.results['input_parameters']['strike']
        ax.axhline(strike, color='orange', linestyle='--', linewidth=2, 
                  label=f'Strike: ${strike:.2f}')
        ax.axhline(S0, color='blue', linestyle='-', linewidth=2, 
                  label=f'Current: ${S0:.2f}')
        
        ax.set_xlabel('Trading Days', fontsize=12)
        ax.set_ylabel('Stock Price ($)', fontsize=12)
        ax.set_title(f'{self.pricer.ticker} - Sample Monte Carlo Price Paths ({n_paths} simulations)',
                    fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Add annotation
        ax.text(0.02, 0.98, 'Green paths → End above start\nRed paths → End below start',
               transform=ax.transAxes, fontsize=9, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        self.figures['sample_paths'] = fig
        return fig
    
    def plot_greeks_dashboard(self, save_path: Optional[str] = None) -> plt.Figure:
        """
        Create a dashboard of all Greeks.
        
        Shows:
        - Delta (call and put)
        - Gamma
        - Theta (call and put)
        - Vega
        - Rho (call and put)
        
        Why this matters: One-stop view of all risk sensitivities.
        """
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        bs = self.results['black_scholes']
        S0 = self.results['input_parameters']['current_price']
        
        # Delta
        ax = axes[0]
        delta_data = [bs['delta_call'], bs['delta_put']]
        bars = ax.bar(['Call Delta', 'Put Delta'], delta_data, color=['green', 'red'], 
                     edgecolor='black', linewidth=1.5)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax.set_ylabel('Delta Value', fontsize=10)
        ax.set_title('Delta (Δ) - Stock Price Sensitivity', fontsize=11, fontweight='bold')
        for bar, val in zip(bars, delta_data):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + (0.05 if val > 0 else -0.1),
                   f'{val:.3f}', ha='center', va='bottom' if val > 0 else 'top', fontweight='bold')
        ax.set_ylim(-1.1, 1.1)
        ax.grid(True, alpha=0.3)
        
        # Gamma
        ax = axes[1]
        gamma_val = bs['gamma']
        ax.bar(['Gamma'], [gamma_val], color='purple', edgecolor='black', linewidth=1.5)
        ax.set_ylabel('Gamma Value', fontsize=10)
        ax.set_title('Gamma (Γ) - Delta Sensitivity', fontsize=11, fontweight='bold')
        ax.text(0, gamma_val + gamma_val*0.1, f'{gamma_val:.4f}', ha='center', va='bottom', fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Theta
        ax = axes[2]
        theta_data = [bs['theta_call'], bs['theta_put']]
        bars = ax.bar(['Call Theta', 'Put Theta'], theta_data, color=['lightcoral', 'lightcoral'],
                     edgecolor='black', linewidth=1.5)
        ax.set_ylabel('Daily Theta ($/day)', fontsize=10)
        ax.set_title('Theta (Θ) - Time Decay', fontsize=11, fontweight='bold')
        for bar, val in zip(bars, theta_data):
            ax.text(bar.get_x() + bar.get_width()/2, val + (0.01 if val > 0 else -0.02),
                   f'${val:.3f}', ha='center', va='bottom' if val > 0 else 'top', fontweight='bold')
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax.grid(True, alpha=0.3)
        
        # Vega
        ax = axes[3]
        vega_val = bs['vega']
        ax.bar(['Vega'], [vega_val], color='gold', edgecolor='black', linewidth=1.5)
        ax.set_ylabel('Vega ($ per 1% vol)', fontsize=10)
        ax.set_title('Vega (ν) - Volatility Sensitivity', fontsize=11, fontweight='bold')
        ax.text(0, vega_val + vega_val*0.1, f'${vega_val:.2f}', ha='center', va='bottom', fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Rho
        ax = axes[4]
        rho_data = [bs['rho_call'], bs['rho_put']]
        bars = ax.bar(['Call Rho', 'Put Rho'], rho_data, color=['steelblue', 'steelblue'],
                     edgecolor='black', linewidth=1.5)
        ax.set_ylabel('Rho ($ per 1% rate)', fontsize=10)
        ax.set_title('Rho (ρ) - Interest Rate Sensitivity', fontsize=11, fontweight='bold')
        for bar, val in zip(bars, rho_data):
            ax.text(bar.get_x() + bar.get_width()/2, val + (0.001 if val > 0 else -0.002),
                   f'${val:.4f}', ha='center', va='bottom' if val > 0 else 'top', fontweight='bold')
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax.grid(True, alpha=0.3)
        
        # Interpretation summary
        ax = axes[5]
        ax.axis('off')
        text = (f"GREEKS INTERPRETATION:\n\n"
                f"Δ (Delta):   ${bs['delta_call']:.2f} per $1 stock move (Call)\n"
                f"Γ (Gamma):   Delta changes by {bs['gamma']:.4f} per $1 move\n"
                f"Θ (Theta):   Lose ${abs(bs['theta_call']):.3f}/day (Call)\n"
                f"ν (Vega):    Gain ${bs['vega']:.2f} per 1% vol increase\n"
                f"ρ (Rho):     Gain ${bs['rho_call']:.4f} per 1% rate increase (Call)")
        
        ax.text(0.1, 0.5, text, transform=ax.transAxes, fontsize=10,
               verticalalignment='center', fontfamily='monospace',
               bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
        
        fig.suptitle(f'{self.pricer.ticker} - Options Greeks Dashboard',
                    fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        self.figures['greeks_dashboard'] = fig
        return fig
    
    def plot_risk_metrics(self, save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot trading-focused risk metrics.
        
        Shows:
        - Breakeven vs current price
        - Probability of profit vs probability ITM
        - Required move percentage
        - Risk-reward ratio
        
        Why this matters: Decision-making metrics for traders.
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        risk_metrics = self.results['risk_metrics']
        S0 = self.results['input_parameters']['current_price']
        
        # Chart 1: Breakeven visualization
        ax = axes[0, 0]
        call_be = risk_metrics['call']['breakeven']
        put_be = risk_metrics['put']['breakeven']
        
        # Create a simple price line
        y_pos = [0, 1, 2]
        ax.axhline(y=0, color='gray', linewidth=0.5)
        ax.scatter([0], [S0], color='blue', s=200, marker='s', label=f'Current: ${S0:.2f}', zorder=5)
        ax.scatter([1], [call_be], color='green', s=200, marker='^', label=f'Call BE: ${call_be:.2f}', zorder=5)
        ax.scatter([2], [put_be], color='red', s=200, marker='v', label=f'Put BE: ${put_be:.2f}', zorder=5)
        
        # Add arrows showing required move
        ax.annotate('', xy=(1, call_be), xytext=(0, S0),
                   arrowprops=dict(arrowstyle='->', color='green', lw=2))
        ax.annotate('', xy=(2, put_be), xytext=(0, S0),
                   arrowprops=dict(arrowstyle='->', color='red', lw=2))
        
        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(['Current', 'Call BE', 'Put BE'])
        ax.set_ylabel('Stock Price ($)', fontsize=10)
        ax.set_title('Breakeven Analysis', fontsize=12, fontweight='bold')
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
        
        # Chart 2: Required Move Percentage
        ax = axes[0, 1]
        call_move = risk_metrics['call']['required_move_pct']
        put_move = risk_metrics['put']['required_move_pct']
        
        bars = ax.bar(['Call Required Move', 'Put Required Move'], 
                     [call_move, put_move], 
                     color=['green', 'red'], edgecolor='black', linewidth=1.5)
        
        for bar, val in zip(bars, [call_move, put_move]):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                   f'{val:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        ax.set_ylabel('Required Move (%)', fontsize=10)
        ax.set_title('Required Stock Move to Breakeven', fontsize=12, fontweight='bold')
        ax.axhline(y=0, color='black', linewidth=0.5)
        ax.grid(True, alpha=0.3)
        
        # Chart 3: Probability Comparison
        ax = axes[1, 0]
        call_itm = risk_metrics['call']['probability_itm']
        call_profit = risk_metrics['call']['probability_profit']
        put_itm = risk_metrics['put']['probability_itm']
        put_profit = risk_metrics['put']['probability_profit']
        
        x = np.arange(2)
        width = 0.35
        
        ax.bar(x - width/2, [call_itm, put_itm], width, label='ITM Probability', 
               color='steelblue', edgecolor='black')
        ax.bar(x + width/2, [call_profit if call_profit else 0, put_profit if put_profit else 0], 
               width, label='Profit Probability', color='coral', edgecolor='black')
        
        ax.set_xticks(x)
        ax.set_xticklabels(['Call', 'Put'])
        ax.set_ylabel('Probability (%)', fontsize=10)
        ax.set_title('Probability Comparison', fontsize=12, fontweight='bold')
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
        
               # Chart 4: Risk-Reward Ratio (continued)
        ax = axes[1, 1]
        call_rr = risk_metrics['call']['risk_reward_ratio']
        put_rr = risk_metrics['put']['risk_reward_ratio']
        
        # Cap at reasonable value for visualization
        call_rr_display = min(call_rr, 10) if call_rr != float('inf') else 10
        put_rr_display = min(put_rr, 10) if put_rr != float('inf') else 10
        
        bars = ax.bar(['Call Risk/Reward', 'Put Risk/Reward'], 
                     [call_rr_display, put_rr_display],
                     color=['gold', 'gold'], edgecolor='black', linewidth=1.5)
        
        for bar, val in zip(bars, [call_rr, put_rr]):
            label = f'1:{val:.1f}' if val != float('inf') else 'N/A'
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                   label, ha='center', va='bottom', fontweight='bold')
        
        ax.set_ylabel('Ratio', fontsize=10)
        ax.set_title('Risk-Reward Ratio (1:X)', fontsize=12, fontweight='bold')
        ax.axhline(y=1, color='red', linestyle='--', linewidth=1, label='Break-even (1:1)')
        ax.set_ylim(0, 11)
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
        
        fig.suptitle(f'{self.pricer.ticker} - Trading Risk Metrics Dashboard',
                    fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        self.figures['risk_metrics'] = fig
        return fig
    
    def plot_sensitivity_heatmap(self, param: str = 'volatility', 
                                  save_path: Optional[str] = None) -> plt.Figure:
        """
        Create a heatmap showing option price sensitivity to parameters.
        
        Shows:
        - How price changes with different volatility levels
        - How price changes with different stock prices
        
        Why this matters: Visualizes the non-linear relationship between
        option price and underlying parameters.
        """
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        S0 = self.results['input_parameters']['current_price']
        K = self.results['input_parameters']['strike']
        T = self.results['input_parameters']['time_to_expiry_years']
        r = self.results['input_parameters']['risk_free_rate']
        sigma_base = self.results['input_parameters']['volatility']
        
        # Heatmap 1: Price vs Volatility and Stock Price
        sigma_range = np.linspace(sigma_base * 0.5, sigma_base * 1.5, 20)
        S_range = np.linspace(S0 * 0.7, S0 * 1.3, 20)
        
        price_matrix = np.zeros((len(S_range), len(sigma_range)))
        
        for i, S in enumerate(S_range):
            for j, sigma in enumerate(sigma_range):
                from src.black_scholes import BlackScholes
                bs = BlackScholes(S, K, T, r, sigma)
                price_matrix[i, j] = bs.call_price()
        
        im1 = axes[0].contourf(sigma_range * 100, S_range, price_matrix, levels=20, cmap='RdYlGn')
        axes[0].set_xlabel('Volatility (%)', fontsize=11)
        axes[0].set_ylabel('Stock Price ($)', fontsize=11)
        axes[0].set_title(f'Call Price Sensitivity to Stock Price & Volatility\n(Strike: ${K:.0f})', 
                         fontsize=11, fontweight='bold')
        
        # Mark current point
        axes[0].scatter(sigma_base * 100, S0, color='blue', s=100, marker='*', 
                       edgecolor='white', linewidth=2, label='Current')
        axes[0].legend(loc='best')
        
        cbar1 = plt.colorbar(im1, ax=axes[0])
        cbar1.set_label('Option Price ($)', fontsize=10)
        
        # Heatmap 2: Price vs Time and Volatility
        T_range = np.linspace(0.1, 1.0, 20)
        sigma_range = np.linspace(sigma_base * 0.5, sigma_base * 1.5, 20)
        
        price_matrix2 = np.zeros((len(T_range), len(sigma_range)))
        
        for i, T_val in enumerate(T_range):
            for j, sigma in enumerate(sigma_range):
                from src.black_scholes import BlackScholes
                bs = BlackScholes(S0, K, T_val, r, sigma)
                price_matrix2[i, j] = bs.call_price()
        
        im2 = axes[1].contourf(sigma_range * 100, T_range * 12, price_matrix2, levels=20, cmap='RdYlGn')
        axes[1].set_xlabel('Volatility (%)', fontsize=11)
        axes[1].set_ylabel('Time to Expiry (months)', fontsize=11)
        axes[1].set_title(f'Call Price Sensitivity to Time & Volatility\n(Strike: ${K:.0f})', 
                         fontsize=11, fontweight='bold')
        
        # Mark current point
        axes[1].scatter(sigma_base * 100, T * 12, color='blue', s=100, marker='*', 
                       edgecolor='white', linewidth=2, label='Current')
        axes[1].legend(loc='best')
        
        cbar2 = plt.colorbar(im2, ax=axes[1])
        cbar2.set_label('Option Price ($)', fontsize=10)
        
        fig.suptitle(f'{self.pricer.ticker} - Option Price Sensitivity Analysis',
                    fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        self.figures['sensitivity_heatmap'] = fig
        return fig
    
    def create_full_dashboard(self, save_dir: Optional[str] = None) -> Dict:
        """
        Create all visualizations in one go.
        
        Args:
            save_dir: Directory to save images (optional)
        
        Returns:
            Dictionary of all created figures
        """
        print("\n📊 Generating Visualizations...")
        
        self.plot_volatility_breakdown()
        print("   ✓ Volatility Breakdown")
        
        self.plot_price_comparison()
        print("   ✓ Price Comparison")
        
        self.plot_convergence()
        print("   ✓ Convergence Analysis")
        
        self.plot_price_distribution()
        print("   ✓ Price Distribution")
        
        self.plot_payoff_distribution('call')
        print("   ✓ Call Payoff Distribution")
        
        self.plot_payoff_distribution('put')
        print("   ✓ Put Payoff Distribution")
        
        self.plot_sample_paths()
        print("   ✓ Sample Price Paths")
        
        self.plot_greeks_dashboard()
        print("   ✓ Greeks Dashboard")
        
        self.plot_risk_metrics()
        print("   ✓ Risk Metrics")
        
        self.plot_sensitivity_heatmap()
        print("   ✓ Sensitivity Heatmap")
        
        if save_dir:
            import os
            os.makedirs(save_dir, exist_ok=True)
            for name, fig in self.figures.items():
                fig.savefig(f"{save_dir}/{name}.png", dpi=150, bbox_inches='tight')
            print(f"   ✓ All figures saved to {save_dir}")
        
        print("\n✅ Visualization Complete!")
        
        return self.figures
    
    def show(self):
        """Display all figures."""
        plt.show()


# For testing
if __name__ == "__main__":
    print("Visualization module loaded successfully")
    print("This module should be imported by main.py")