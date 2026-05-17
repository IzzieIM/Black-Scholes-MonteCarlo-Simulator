"""
main.py
Main Application - Options Pricing & Risk Analysis System

This is the user-facing interface that orchestrates all components:
1. Fetch real-time market data (yfinance)
2. Calculate volatility from historical data
3. Price options using Black-Scholes (analytical)
4. Price options using Monte Carlo (numerical)
5. Validate results through comparison
6. Provide comprehensive risk analysis

USAGE:
    python main.py

Or as a module:
    from main import SimpleOptionsPricer
    pricer = SimpleOptionsPricer(ticker='AAPL', K=200, T=0.5, r=0.05)
    results = pricer.get_full_analysis()
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple
import warnings

# Import our modules
from src.volatility_calculator import VolatilityCalculator
from src.black_scholes import BlackScholes
from src.monte_carlo import MonteCarloPricer

# For visualization (will create in next file)
try:
    from visualization.plots import OptionVisualizer
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False
    print("Note: visualization module not yet available. Install when ready.")


class SimpleOptionsPricer:
    """
    Main interface for options pricing with auto-fetched market data.
    
    This class handles everything - just give it a ticker and basic parameters.
    
    Example:
        pricer = SimpleOptionsPricer(
            ticker='AAPL',
            K=200,
            T=0.5,
            r=0.05
        )
        results = pricer.get_full_analysis()
    """
    
    def __init__(
        self,
        ticker: str,
        K: float,
        T: float,
        r: float,
        volatility_lookback: int = 252,
        mc_simulations: int = 10000,
        mc_steps: int = 252,
        verbose: bool = True
    ):
        """
        Initialize the options pricer with auto-fetched market data.
        
        Args:
            ticker: Stock ticker symbol (e.g., 'AAPL', 'MSFT', 'SPY')
            K: Strike price
            T: Time to expiry in years (e.g., 0.5 = 6 months)
            r: Risk-free interest rate (e.g., 0.05 = 5%)
            volatility_lookback: Days of historical data for volatility (default 252)
            mc_simulations: Number of Monte Carlo simulations (default 10000)
            mc_steps: Time steps per simulation (default 252)
            verbose: Print progress updates (default True)
        """
        self.ticker = ticker.upper()
        self.K = K
        self.T = T
        self.r = r
        self.verbose = verbose
        
        # Step 1: Calculate volatility and get current price
        if self.verbose:
            print("\n" + "=" * 70)
            print(f"OPTIONS PRICING SYSTEM - {self.ticker}")
            print("=" * 70)
            print("\n📡 STEP 1: Fetching Market Data & Calculating Volatility")
            print("-" * 50)
        
        self.vol_calc = VolatilityCalculator(ticker, lookback_days=volatility_lookback)
        self.S = self.vol_calc.current_price
        self.sigma = self.vol_calc.get_recommended_volatility()
        
        # Store volatility details for analysis
        self.volatility_details = self.vol_calc.get_volatility_summary()
        
        if self.verbose:
            print(f"\n   ✅ Current Price: ${self.S:.2f}")
            print(f"   ✅ Recommended Volatility: {self.sigma*100:.1f}%")
        
        # Step 2: Initialize pricing models
        if self.verbose:
            print("\n📐 STEP 2: Initializing Pricing Engines")
            print("-" * 50)
        
        self.bs_model = BlackScholes(self.S, self.K, self.T, self.r, self.sigma)
        
        self.mc_model = MonteCarloPricer(
            self.S, self.K, self.T, self.r, self.sigma,
            n_simulations=mc_simulations,
            n_steps=mc_steps,
            random_seed=42  # For reproducibility
        )
        
        if self.verbose:
            print("   ✅ Black-Scholes Engine Ready")
            print(f"   ✅ Monte Carlo Engine Ready ({mc_simulations:,} simulations, {mc_steps} steps)")
        
        # Store results cache
        self._results_cache = None
    
    def get_black_scholes_results(self) -> Dict:
        """Get all Black-Scholes pricing and Greeks."""
        return self.bs_model.get_all_prices_and_greeks()
    
    def get_monte_carlo_results(self) -> Dict:
        """Get Monte Carlo pricing with confidence intervals."""
        call_price, call_se = self.mc_model.price_call()
        put_price, put_se = self.mc_model.price_put()
        
        return {
            'call_price': call_price,
            'call_std_error': call_se,
            'call_ci_95': (call_price - 1.96*call_se, call_price + 1.96*call_se),
            'put_price': put_price,
            'put_std_error': put_se,
            'put_ci_95': (put_price - 1.96*put_se, put_price + 1.96*put_se),
            'price_distribution': self.mc_model.get_price_distribution(),
            'payoff_distribution_call': self.mc_model.get_payoff_distribution('call'),
            'payoff_distribution_put': self.mc_model.get_payoff_distribution('put'),
            'probability_itm_call': self.mc_model.probability_of_profit('call'),
            'probability_itm_put': self.mc_model.probability_of_profit('put')
        }
    
    def validate_models(self) -> Dict:
        """
        Compare Black-Scholes and Monte Carlo for validation.
        
        Returns:
            Dictionary with validation results
        """
        bs_results = self.get_black_scholes_results()
        mc_results = self.get_monte_carlo_results()
        
        # Check if BS price falls within MC 95% CI
        call_bs = bs_results['call_price']
        call_ci_lower, call_ci_upper = mc_results['call_ci_95']
        call_valid = call_ci_lower <= call_bs <= call_ci_upper
        
        put_bs = bs_results['put_price']
        put_ci_lower, put_ci_upper = mc_results['put_ci_95']
        put_valid = put_ci_lower <= put_bs <= put_ci_upper
        
        return {
            'call': {
                'bs_price': call_bs,
                'mc_price': mc_results['call_price'],
                'mc_ci': (call_ci_lower, call_ci_upper),
                'within_ci': call_valid,
                'difference': call_bs - mc_results['call_price']
            },
            'put': {
                'bs_price': put_bs,
                'mc_price': mc_results['put_price'],
                'mc_ci': (put_ci_lower, put_ci_upper),
                'within_ci': put_valid,
                'difference': put_bs - mc_results['put_price']
            },
            'validation_passed': call_valid and put_valid
        }
    
    def get_convergence_analysis(self) -> Dict:
        """
        Analyze how Monte Carlo converges with more simulations.
        
        Shows the trade-off between accuracy and computation time.
        """
        return self.mc_model.convergence_analysis()
    
    def get_risk_metrics(self) -> Dict:
        """
        Calculate comprehensive risk metrics for trading decisions.
        
        Returns metrics that traders actually use:
        - Probability of profit (breakeven probability)
        - Risk-reward ratio
        - Maximum loss
        - Expected value
        - Sharpe ratio (simplified)
        """
        bs_results = self.get_black_scholes_results()
        mc_results = self.get_monte_carlo_results()
        
        # Call option metrics
        call_premium = bs_results['call_price']
        call_breakeven = self.bs_model.breakeven_price('call')
        
        # Probability of profit (requires stock to be above breakeven, not just ITM)
        # We estimate this from Monte Carlo distribution
        final_prices = self.mc_model._final_prices
        if final_prices is not None:
            call_profit_prob = np.sum(final_prices > call_breakeven) / len(final_prices)
            put_breakeven = self.bs_model.breakeven_price('put')
            put_profit_prob = np.sum(final_prices < put_breakeven) / len(final_prices)
        else:
            call_profit_prob = None
            put_profit_prob = None
        
        # Risk-reward ratio (max loss vs potential gain)
        # For long options, max loss = premium paid
        # Potential gain = theoretical max (using 95th percentile of payoffs)
        payoff_95th = mc_results['payoff_distribution_call']['percentiles']['95th']
        call_risk_reward = call_premium / payoff_95th if payoff_95th > 0 else float('inf')
        
        payoff_95th_put = mc_results['payoff_distribution_put']['percentiles']['95th']
        put_risk_reward = put_premium / payoff_95th_put if payoff_95th_put > 0 else float('inf')
        
        return {
            'call': {
                'premium': call_premium,
                'breakeven': call_breakeven,
                'required_move_pct': (call_breakeven / self.S - 1) * 100,
                'probability_itm': mc_results['probability_itm_call'] * 100,
                'probability_profit': call_profit_prob * 100 if call_profit_prob else None,
                'max_loss': call_premium,
                'max_loss_pct': (call_premium / self.S) * 100,
                'risk_reward_ratio': call_risk_reward,
                'expected_value': mc_results['payoff_distribution_call']['mean_payoff'] * np.exp(-self.r * self.T)
            },
            'put': {
                'premium': bs_results['put_price'],
                'breakeven': self.bs_model.breakeven_price('put'),
                'required_move_pct': (1 - self.bs_model.breakeven_price('put') / self.S) * 100,
                'probability_itm': mc_results['probability_itm_put'] * 100,
                'probability_profit': put_profit_prob * 100 if put_profit_prob else None,
                'max_loss': bs_results['put_price'],
                'max_loss_pct': (bs_results['put_price'] / self.S) * 100,
                'risk_reward_ratio': put_risk_reward,
                'expected_value': mc_results['payoff_distribution_put']['mean_payoff'] * np.exp(-self.r * self.T)
            }
        }
    
    def get_full_analysis(self) -> Dict:
        """
        Run complete analysis and return all results.
        
        This is the main method you'll use - it returns everything.
        """
        if self._results_cache is not None:
            return self._results_cache
        
        if self.verbose:
            print("\n🔬 STEP 3: Running Pricing Engines")
            print("-" * 50)
        
        # Get all results
        bs_results = self.get_black_scholes_results()
        mc_results = self.get_monte_carlo_results()
        validation = self.validate_models()
        convergence = self.get_convergence_analysis()
        risk_metrics = self.get_risk_metrics()
        
        # Compile everything
        results = {
            'input_parameters': {
                'ticker': self.ticker,
                'current_price': self.S,
                'strike': self.K,
                'time_to_expiry_years': self.T,
                'time_to_expiry_days': self.T * 365,
                'risk_free_rate': self.r,
                'volatility': self.sigma,
                'volatility_percent': f"{self.sigma*100:.1f}%"
            },
            'volatility_breakdown': self.volatility_details,
            'black_scholes': bs_results,
            'monte_carlo': mc_results,
            'validation': validation,
            'convergence': convergence,
            'risk_metrics': risk_metrics,
            'timestamp': datetime.now().isoformat()
        }
        
        self._results_cache = results
        
        if self.verbose:
            print("\n✅ Analysis Complete!")
            
            # Print summary
            print("\n" + "=" * 70)
            print("RESULTS SUMMARY")
            print("=" * 70)
            
            print(f"\n📊 {self.ticker} - ${self.S:.2f} (Current Price)")
            print(f"   Strike: ${self.K:.2f} | Expiry: {self.T*12:.1f} months")
            print(f"   Volatility: {self.sigma*100:.1f}% (weighted average)")
            
            print(f"\n💰 OPTION PRICES:")
            print(f"   Black-Scholes Call: ${bs_results['call_price']:.4f}")
            print(f"   Black-Scholes Put:  ${bs_results['put_price']:.4f}")
            print(f"   Monte Carlo Call:   ${mc_results['call_price']:.4f} ± {mc_results['call_std_error']:.4f}")
            print(f"   Monte Carlo Put:    ${mc_results['put_price']:.4f} ± {mc_results['put_std_error']:.4f}")
            
            print(f"\n✅ MODEL VALIDATION:")
            if validation['validation_passed']:
                print("   ✓ PASSED - Black-Scholes within Monte Carlo confidence intervals")
            else:
                print("   ⚠ WARNING - Models disagree! Check inputs and simulation quality")
            
            print(f"\n📐 KEY GREEKS (Black-Scholes):")
            print(f"   Delta (Call): {bs_results['delta_call']:.4f}  |  Gamma: {bs_results['gamma']:.4f}")
            print(f"   Delta (Put):  {bs_results['delta_put']:.4f}  |  Vega: ${bs_results['vega']:.2f} per 1% vol")
            print(f"   Theta (Call): ${bs_results['theta_call']:.4f}/day  |  Rho: ${bs_results['rho_call']:.4f} per 1% rate")
            
            print(f"\n📈 TRADING INSIGHTS:")
            rm = risk_metrics
            print(f"   Call Breakeven: ${rm['call']['breakeven']:.2f} ({rm['call']['required_move_pct']:.1f}% above current)")
            print(f"   Call Profit Probability: {rm['call']['probability_profit']:.1f}% (est.)")
            print(f"   Call Max Loss: ${rm['call']['max_loss']:.2f} ({rm['call']['max_loss_pct']:.1f}% of stock value)")
            print(f"   Put Breakeven: ${rm['put']['breakeven']:.2f} ({rm['put']['required_move_pct']:.1f}% below current)")
            
            print("\n" + "=" * 70)
        
        return results
    
    def print_detailed_report(self) -> None:
        """Print a beautifully formatted detailed report."""
        results = self.get_full_analysis()
        
        print("\n" + "=" * 80)
        print(f"COMPREHENSIVE OPTIONS ANALYSIS REPORT - {self.ticker}")
        print("=" * 80)
        
        # Section 1: Input Parameters
        print("\n📊 1. INPUT PARAMETERS")
        print("-" * 80)
        params = results['input_parameters']
        print(f"   Current Stock Price:     ${params['current_price']:.2f}")
        print(f"   Strike Price:            ${params['strike']:.2f}")
        print(f"   Time to Expiry:          {params['time_to_expiry_years']:.3f} years ({params['time_to_expiry_days']:.0f} days)")
        print(f"   Risk-Free Rate:          {params['risk_free_rate']*100:.2f}%")
        print(f"   Volatility (Recommended): {params['volatility_percent']}")
        
        # Section 2: Volatility Breakdown
        print("\n📈 2. VOLATILITY ESTIMATION BREAKDOWN")
        print("-" * 80)
        vol = results['volatility_breakdown']['volatilities']
        vol_pct = results['volatility_breakdown']['volatility_percent']
        print(f"   Close-to-Close:    {vol_pct['close_to_close']:>8}  (standard method)")
        print(f"   Parkinson (HL):    {vol_pct['parkinson']:>8}  (intraday efficient)")
        print(f"   EWMA (Recent):     {vol_pct['ewma']:>8}  (prioritizes recent data)")
        print(f"   Garman-Klass:      {vol_pct['garman_klass']:>8}  (OHLC efficient)")
        print(f"   ───────────────────────────────────")
        print(f"   RECOMMENDED:       {vol_pct['recommended']:>8}  ← Used for pricing")
        
        # Section 3: Option Prices
        print("\n💰 3. OPTION PRICES")
        print("-" * 80)
        bs = results['black_scholes']
        mc = results['monte_carlo']
        print(f"   BLACK-SCHOLES (Analytical):")
        print(f"      Call:  ${bs['call_price']:.4f}")
        print(f"      Put:   ${bs['put_price']:.4f}")
        print(f"\n   MONTE CARLO (Numerical - {self.mc_model.n_simulations:,} simulations):")
        print(f"      Call:  ${mc['call_price']:.4f} ± {mc['call_std_error']:.4f}")
        print(f"      Put:   ${mc['put_price']:.4f} ± {mc['put_std_error']:.4f}")
        print(f"      Call 95% CI: (${mc['call_ci_95'][0]:.4f}, ${mc['call_ci_95'][1]:.4f})")
        
        # Section 4: Greeks (Risk Sensitivities)
        print("\n📐 4. GREEKS - RISK SENSITIVITIES (Black-Scholes)")
        print("-" * 80)
        print(f"   Delta (Call):  {bs['delta_call']:.4f}  → Option moves ${bs['delta_call']:.2f} per $1 stock move")
        print(f"   Delta (Put):   {bs['delta_put']:.4f}  → Option moves ${bs['delta_put']:.2f} per $1 stock move")
        print(f"   Gamma:         {bs['gamma']:.4f}      → Delta changes by {bs['gamma']:.4f} per $1 stock move")
        print(f"   Theta (Call):  ${bs['theta_call']:.4f}/day → Daily time decay")
        print(f"   Theta (Put):   ${bs['theta_put']:.4f}/day → Daily time decay")
        print(f"   Vega:          ${bs['vega']:.2f}        → Price change per 1% volatility change")
        print(f"   Rho (Call):    ${bs['rho_call']:.4f}    → Price change per 1% interest rate change")
        print(f"   Rho (Put):     ${bs['rho_put']:.4f}     → Price change per 1% interest rate change")
        
        # Section 5: Monte Carlo Distribution Analysis
        print("\n📊 5. MONTE CARLO DISTRIBUTION ANALYSIS")
        print("-" * 80)
        dist = mc['price_distribution']
        print(f"   Stock Price at Expiration:")
        print(f"      Mean:     ${dist['mean']:.2f}")
        print(f"      Median:   {dist['percentiles']['50% (median)']:.2f}")
        print(f"      Std Dev:  ${dist['std']:.2f}")
        print(f"      5th Perc: ${dist['percentiles']['5%']:.2f}")
        print(f"      95th Perc:${dist['percentiles']['95%']:.2f}")
        print(f"      Skewness: {dist['skewness']:.3f} (positive = right tail heavier)")
        print(f"      Kurtosis: {dist['kurtosis']:.3f} (3.0 = normal distribution)")
        
        # Section 6: Payoff Analysis
        print("\n🎯 6. PAYOFF ANALYSIS")
        print("-" * 80)
        payoff_call = mc['payoff_distribution_call']
        payoff_put = mc['payoff_distribution_put']
        print(f"   CALL OPTION:")
        print(f"      Probability of Zero Payoff: {payoff_call['probability_zero']*100:.1f}%")
        print(f"      Probability of Positive:    {payoff_call['probability_positive']*100:.1f}%")
        print(f"      Average Payoff (if >0):     ${payoff_call['mean_positive_payoff']:.4f}")
        print(f"      95th Percentile Payoff:     ${payoff_call['percentiles']['95th']:.4f}")
        print(f"      Maximum Payoff:             ${payoff_call['max_payoff']:.2f}")
        
        print(f"\n   PUT OPTION:")
        print(f"      Probability of Zero Payoff: {payoff_put['probability_zero']*100:.1f}%")
        print(f"      Probability of Positive:    {payoff_put['probability_positive']*100:.1f}%")
        print(f"      Average Payoff (if >0):     ${payoff_put['mean_positive_payoff']:.4f}")
        print(f"      95th Percentile Payoff:     ${payoff_put['percentiles']['95th']:.4f}")
        
        # Section 7: Trading Insights
        print("\n💡 7. TRADING INSIGHTS & RISK METRICS")
        print("-" * 80)
        rm = results['risk_metrics']
        
        print(f"   CALL OPTION (Buying a Call):")
        print(f"      Premium to Pay:           ${rm['call']['premium']:.2f}")
        print(f"      Breakeven Price:          ${rm['call']['breakeven']:.2f}")
        print(f"      Required Stock Move:      {rm['call']['required_move_pct']:.1f}% UP")
        print(f"      Probability ITM:          {rm['call']['probability_itm']:.1f}%")
        print(f"      Probability of Profit:    {rm['call']['probability_profit']:.1f}% (est.)")
        print(f"      Maximum Loss:             ${rm['call']['max_loss']:.2f} (100% of premium)")
        print(f"      Risk/Reward (95th perc):  1:{rm['call']['risk_reward_ratio']:.1f}")
        
        print(f"\n   PUT OPTION (Buying a Put):")
        print(f"      Premium to Pay:           ${rm['put']['premium']:.2f}")
        print(f"      Breakeven Price:          ${rm['put']['breakeven']:.2f}")
        print(f"      Required Stock Move:      {rm['put']['required_move_pct']:.1f}% DOWN")
        print(f"      Probability ITM:          {rm['put']['probability_itm']:.1f}%")
        print(f"      Probability of Profit:    {rm['put']['probability_profit']:.1f}% (est.)")
        print(f"      Maximum Loss:             ${rm['put']['max_loss']:.2f} (100% of premium)")
        
        # Section 8: Validation
        print("\n✅ 8. MODEL VALIDATION")
        print("-" * 80)
        val = results['validation']
        if val['validation_passed']:
            print("   ✓ VALIDATION PASSED")
            print(f"   Black-Scholes call (${val['call']['bs_price']:.4f}) is within MC 95% CI")
            print(f"   Black-Scholes put (${val['put']['bs_price']:.4f}) is within MC 95% CI")
            print("   → Both models agree within simulation error")
        else:
            print("   ⚠ VALIDATION WARNING")
            print("   Black-Scholes and Monte Carlo differ significantly")
            print("   Consider: increasing simulations, checking inputs, or market inefficiencies")
        
        # Section 9: Convergence Analysis
        print("\n🔄 9. MONTE CARLO CONVERGENCE")
        print("-" * 80)
        print("   Simulations    Call Price    Std Error    95% CI Width")
        print("   ─────────────────────────────────────────────────────")
        for n, (price, se) in results['convergence'].items():
            ci_width = 2 * 1.96 * se
            print(f"   {n:<12}  ${price:<10.4f}  ±${se:<9.4f}  ${ci_width:.4f}")
        
        print("\n" + "=" * 80)
        print("END OF REPORT")
        print("=" * 80)
    
    def generate_warning_if_needed(self) -> Optional[str]:
        """Check for potential issues and return warning message."""
        warnings_list = []
        
        # Check if option is extremely OTM
        call_otm_pct = (self.K - self.S) / self.S * 100
        if call_otm_pct > 20:
            warnings_list.append(f"Call option is {call_otm_pct:.1f}% OTM - very low probability")
        
        # Check if volatility is extreme
        if self.sigma > 0.60:
            warnings_list.append(f"Very high volatility ({self.sigma*100:.1f}%) - options are expensive")
        
        # Check time to expiry
        if self.T < 0.08:  # Less than 1 month
            warnings_list.append("Very short expiry - high gamma risk, rapid time decay")
        
        # Check validation
        validation = self.validate_models()
        if not validation['validation_passed']:
            warnings_list.append("Model validation failed - results may be unreliable")
        
        return "\n".join(warnings_list) if warnings_list else None


def main():
    """Interactive command-line interface for the options pricing system."""
    print("\n" + "=" * 70)
    print("   REAL-TIME OPTIONS PRICING & RISK ANALYSIS SYSTEM")
    print("   Auto-Fetched Market Data | Black-Scholes + Monte Carlo")
    print("=" * 70)
    
    print("\n📋 Enter option parameters:")
    print("-" * 50)
    
    # Get user input with validation
    while True:
        ticker = input("   Stock Ticker (e.g., AAPL, MSFT, SPY): ").strip().upper()
        if ticker:
            break
        print("   ⚠ Please enter a valid ticker symbol")
    
    while True:
        try:
            K = float(input("   Strike Price ($): ").strip())
            if K > 0:
                break
            print("   ⚠ Strike price must be positive")
        except ValueError:
            print("   ⚠ Please enter a valid number")
    
    while True:
        try:
            T_input = input("   Time to Expiry (in months, e.g., 3 for 3 months): ").strip()
            T = float(T_input) / 12  # Convert months to years
            if 0 < T <= 5:
                break
            print("   ⚠ Time must be between 1 month and 5 years")
        except ValueError:
            print("   ⚠ Please enter a valid number")
    
    while True:
        try:
            r = float(input("   Risk-Free Rate (%, e.g., 5 for 5%): ").strip()) / 100
            if 0 <= r <= 0.20:
                break
            print("   ⚠ Rate must be between 0% and 20%")
        except ValueError:
            print("   ⚠ Please enter a valid number")
    
    # Optional advanced settings
    print("\n🔧 Advanced Settings (press Enter for defaults):")
    advanced = input("   Custom simulations? (default 10,000): ").strip()
    mc_sims = int(advanced) if advanced.isdigit() else 10000
    
    print("\n" + "=" * 70)
    
    try:
        # Initialize pricer
        pricer = SimpleOptionsPricer(
            ticker=ticker,
            K=K,
            T=T,
            r=r,
            mc_simulations=mc_sims,
            verbose=True
        )
        
        # Generate detailed report
        pricer.print_detailed_report()
        
        # Check for warnings
        warnings = pricer.generate_warning_if_needed()
        if warnings:
            print("\n⚠ WARNINGS:")
            print(warnings)
        
        # Ask about visualizations
        if VISUALIZATION_AVAILABLE:
            print("\n📊 Would you like to generate visualizations? (y/n)")
            if input().lower().startswith('y'):
                viz = OptionVisualizer(pricer)
                viz.create_full_dashboard()
                viz.show()
        
        # Export option
        print("\n💾 Export results to CSV? (y/n)")
        if input().lower().startswith('y'):
            results = pricer.get_full_analysis()
            df = pd.DataFrame([results['black_scholes']])
            filename = f"option_pricing_{ticker}_K{K}_T{T*12:.0f}m_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            df.to_csv(filename, index=False)
            print(f"   ✅ Results saved to {filename}")
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        print("   Please check your inputs and internet connection.")
        return


if __name__ == "__main__":
    main()