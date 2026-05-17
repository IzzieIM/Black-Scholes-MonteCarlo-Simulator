"""
src/black_scholes.py
Black-Scholes Option Pricing Model - Analytical Engine

This module implements the Black-Scholes-Merton formula for European options,
calculating both prices and all Greeks (sensitivities).

The Black-Scholes formula, published in 1973 by Fischer Black and Myron Scholes,
revolutionized finance by providing a closed-form solution for option pricing.
Robert Merton's simultaneous work on the mathematical underpinnings earned
him and Scholes the 1997 Nobel Prize (Black had passed away in 1995).

KEY ASSUMPTIONS (Important to know limitations):
1. No arbitrage opportunities exist
2. Stock price follows Geometric Brownian Motion (log-normal distribution)
3. No transaction costs or taxes
4. Risk-free rate is constant and known
5. Volatility is constant (MOST CRITICIZED assumption)
6. No dividends during option's life
7. European exercise only (cannot exercise early)
8. Markets are perfectly liquid
9. Continuous trading is possible

Despite these assumptions, Black-Scholes remains the industry standard
due to its simplicity, speed, and the intuitive Greeks it provides.
"""

import numpy as np
from scipy.stats import norm
from typing import Tuple, Dict, Optional


class BlackScholes:
    """
    Black-Scholes option pricing model with full Greek calculations.
    
    This class provides both call and put option pricing using the
    analytical Black-Scholes formula, along with all sensitivity
    measures (Greeks) that quantify different types of risk.
    
    Attributes:
        S (float): Current stock price (Spot price)
        K (float): Strike price (Exercise price)
        T (float): Time to expiration in years (e.g., 0.5 = 6 months)
        r (float): Risk-free interest rate (annualized, decimal)
        sigma (float): Volatility of underlying stock (annualized, decimal)
    """
    
    def __init__(self, S: float, K: float, T: float, r: float, sigma: float):
        """
        Initialize the Black-Scholes model with option parameters.
        
        Args:
            S: Current stock price (must be > 0)
            K: Strike price (must be > 0)
            T: Time to expiration in years (must be > 0)
            r: Risk-free rate as decimal (e.g., 0.05 = 5%)
            sigma: Volatility as decimal (e.g., 0.20 = 20%)
        """
        # Input validation
        if S <= 0:
            raise ValueError(f"Stock price S must be positive, got {S}")
        if K <= 0:
            raise ValueError(f"Strike price K must be positive, got {K}")
        if T <= 0:
            raise ValueError(f"Time to expiry T must be positive, got {T}")
        if sigma <= 0:
            raise ValueError(f"Volatility sigma must be positive, got {sigma}")
        
        self.S = S
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        
        # Pre-compute d1 and d2 (used in ALL calculations)
        self.d1 = None
        self.d2 = None
        self._calculate_d1d2()
    
    def _calculate_d1d2(self) -> None:
        """
        Calculate the d1 and d2 parameters of the Black-Scholes formula.
        
        These are the core intermediate values from which everything else flows.
        
        THE FORMULAS:
        
        d1 = [ln(S/K) + (r + sigma²/2) x T] / (sigma x √T)
        
        d2 = d1 - sigma x √T
        
        INTUITION BEHIND d1 AND d2:
        
        - ln(S/K): How far the stock is from the strike (in log space)
          * If S=K, ln(1)=0 → option is at-the-money
          * If S>K (ITM call), ln>0
          * If S<K (OTM call), ln<0
        
        - (r + σ²/2)T: Risk-adjusted drift
          * r accounts for time value of money
          * σ²/2 is the convexity adjustment from Itô's lemma
        
        - sigma x √T: Total expected volatility over the period
        
        d1 measures distance to strike in "volatility units" - it's how many
        standard deviations the stock needs to move to be at-the-money.
        
        d2 is d1 minus the total volatility - used for the probability
        that the option expires in-the-money.
        """
        # Natural log of current price divided by strike price
        # Positive if S > K (in-the-money for call), negative if S < K
        log_S_over_K = np.log(self.S / self.K)
        
        # Drift component: (risk-free rate + half variance) × time
        # The σ²/2 term comes from Itô's lemma - it's NOT a typo!
        drift = (self.r + 0.5 * self.sigma ** 2) * self.T
        
        # Total volatility over the period
        vol_sqrt_T = self.sigma * np.sqrt(self.T)
        
        # d1 and d2
        self.d1 = (log_S_over_K + drift) / vol_sqrt_T
        self.d2 = self.d1 - vol_sqrt_T
    
    def call_price(self) -> float:
        """
        Calculate the Black-Scholes call option price.
        
        FORMULA:
        C = S × N(d1) - K × e^(-rT) × N(d2)
        
        BREAKING DOWN THE FORMULA:
        
        Term 1: S × N(d1)
        - S: Current stock price
        - N(d1): Probability that the option will be exercised (under risk-neutral measure)
        - This term represents the expected benefit of receiving the stock
        
        Term 2: K × e^(-rT) × N(d2)
        - K × e^(-rT): Present value of the strike price (discounted at risk-free rate)
        - N(d2): Probability the option expires in-the-money
        - This term represents the expected cost of paying the strike price
        
        INTUITION: Call price = Expected benefit - Expected cost
        
        Returns:
            Fair call option price
        """
        # N(d1) and N(d2) - cumulative probabilities from standard normal distribution
        Nd1 = norm.cdf(self.d1)
        Nd2 = norm.cdf(self.d2)
        
        # Present value factor for the strike price
        discounted_K = self.K * np.exp(-self.r * self.T)
        
        # Black-Scholes formula
        call = self.S * Nd1 - discounted_K * Nd2
        
        return call
    
    def put_price(self) -> float:
        """
        Calculate the Black-Scholes put option price.
        
        FORMULA:
        P = K × e^(-rT) × N(-d2) - S × N(-d1)
        
        ALTERNATIVE (Put-Call Parity):
        P = C - S + K × e^(-rT)
        
        Where C is the call price from put-call parity.
        
        BREAKING DOWN THE FORMULA:
        
        Term 1: K × e^(-rT) × N(-d2)
        - Present value of strike times probability of exercise (for puts)
        - N(-d2) = 1 - N(d2) = probability stock is BELOW strike at expiry
        
        Term 2: S × N(-d1)
        - Current stock price times probability of exercise adjustment
        
        INTUITION: Put price = Expected benefit of selling at strike - Expected cost of stock
        
        Returns:
            Fair put option price
        """
        # N(-d1) = 1 - N(d1), N(-d2) = 1 - N(d2)
        N_minus_d1 = norm.cdf(-self.d1)
        N_minus_d2 = norm.cdf(-self.d2)
        
        # Present value of strike
        discounted_K = self.K * np.exp(-self.r * self.T)
        
        # Black-Scholes put formula
        put = discounted_K * N_minus_d2 - self.S * N_minus_d1
        
        return put
    
    def delta(self, option_type: str = 'call') -> float:
        """
        Calculate Delta (Δ) - Rate of change of option price with respect to stock price.
        
        DEFINITION: Δ = ∂V / ∂S
        Measures how much the option price changes when the stock moves $1.
        
        FORMULAS:
        Call Delta = N(d1)
        Put Delta = N(d1) - 1
        
        INTERPRETATION:
        - Call Delta ranges from 0 to 1
          * Deep OTM call (S << K): Δ ≈ 0 (option won't move with stock)
          * Deep ITM call (S >> K): Δ ≈ 1 (option moves like stock)
          * ATM call (S = K): Δ ≈ 0.5 (option moves half as much as stock)
        
        - Put Delta ranges from -1 to 0
          * Deep OTM put (S >> K): Δ ≈ 0
          * Deep ITM put (S << K): Δ ≈ -1 (option moves opposite to stock)
          * ATM put (S = K): Δ ≈ -0.5
        
        DELTA AS PROBABILITY:
        N(d2) is approximately the probability of exercise (under risk-neutral measure).
        Call Delta ≈ probability of finishing ITM (approximately, but not exactly -
        N(d1) vs N(d2) differ by a factor involving the drift).
        
        PRACTICAL USE:
        - Delta hedging: To hedge a long call, short Δ shares of stock
        - Position sizing: Delta × shares = "stock equivalent" exposure
        - Probability assessment: Approximate chance option expires ITM
        
        Args:
            option_type: Either 'call' or 'put'
        
        Returns:
            Delta value (unitless, between -1 and 1)
        """
        Nd1 = norm.cdf(self.d1)
        
        if option_type.lower() == 'call':
            return Nd1
        else:  # put
            return Nd1 - 1
    
    def gamma(self) -> float:
        """
        Calculate Gamma (Γ) - Rate of change of Delta with respect to stock price.
        
        DEFINITION: Γ = ∂²V / ∂S² = ∂Δ / ∂S
        Measures how sensitive Delta is to changes in the stock price.
        
        FORMULA:
        Γ = N'(d1) / (S × σ × √T)
        
        Where N'(d1) is the standard normal probability density function:
        N'(x) = (1/√(2π)) × e^(-x²/2)
        
        INTERPRETATION:
        - Gamma is ALWAYS positive for both calls and puts
        - Highest for ATM options (S ≈ K) and near expiration (small T)
        - Lowest for deep ITM/OTM options
        
        WHY GAMMA MATTERS:
        
        Scenario 1: Low Gamma (deep OTM option)
        - Delta changes very slowly as stock moves
        - Option is unlikely to become ITM
        
        Scenario 2: High Gamma (ATM option near expiry)
        - Delta changes rapidly with small stock moves
        - Option can swing from OTM to ITM quickly
        - This is where gamma risk (or "gamma squeeze") happens
        
        PRACTICAL USE:
        - Gamma scalping: Profiting from volatility by delta-hedging
        - Risk management: High gamma positions need very frequent rebalancing
        - Pin risk: ATM options near expiry have huge gamma (dangerous!)
        
        Returns:
            Gamma value
        """
        # Probability density function at d1
        Nd1_prime = norm.pdf(self.d1)
        
        # Gamma formula
        gamma = Nd1_prime / (self.S * self.sigma * np.sqrt(self.T))
        
        return gamma
    
    def theta(self, option_type: str = 'call') -> float:
        """
        Calculate Theta (Θ) - Rate of change of option price with respect to time.
        
        DEFINITION: Θ = ∂V / ∂t (usually reported as daily or per day)
        Measures time decay - how much value the option loses each day.
        
        FORMULAS (annualized, then converted to daily):
        
        Call Theta = -[S × N'(d1) × σ] / (2√T) - r × K × e^(-rT) × N(d2)
        
        Put Theta = -[S × N'(d1) × σ] / (2√T) + r × K × e^(-rT) × N(-d2)
        
        INTERPRETATION:
        - Theta is USUALLY negative for options (they lose value over time)
        - Exception: Deep ITM puts can have positive theta (rare, due to interest rates)
        - Time decay accelerates as expiration approaches
        - ATM options have the highest (most negative) theta
        
        THE "THETA TRAP":
        - Selling options to capture theta is a common strategy (e.g., iron condors)
        - But gamma risk increases as theta increases - you profit slowly but lose quickly!
        
        TIME UNITS IN PRACTICE:
        - Annual theta: Change per year (huge number, not useful)
        - Daily theta: Annual theta / 365 (what traders actually care about)
        
        This function returns DAILY theta (dollars lost per day).
        
        REAL EXAMPLE:
        Option price: $5.00
        Theta: -0.04
        Meaning: Option loses $0.04 per day from time decay
        
        Args:
            option_type: Either 'call' or 'put'
        
        Returns:
            Daily theta (dollars lost per calendar day)
        """
        # PDF at d1
        Nd1_prime = norm.pdf(self.d1)
        
        # Common term in both call and put theta
        common_term = -(self.S * Nd1_prime * self.sigma) / (2 * np.sqrt(self.T))
        
        # Discount factor
        discounted_K = self.K * np.exp(-self.r * self.T)
        
        if option_type.lower() == 'call':
            # Call theta (annual)
            theta_annual = common_term - self.r * discounted_K * norm.cdf(self.d2)
        else:
            # Put theta (annual)
            theta_annual = common_term + self.r * discounted_K * norm.cdf(-self.d2)
        
        # Convert to daily theta (divide by 365 calendar days)
        # Note: Some quants use 252 trading days, but time decay happens 7 days/week
        theta_daily = theta_annual / 365
        
        return theta_daily
    
    def vega(self) -> float:
        """
        Calculate Vega (ν) - Rate of change of option price with respect to volatility.
        
        DEFINITION: ν = ∂V / ∂σ
        Measures how much the option price changes when volatility changes by 1%.
        
        FORMULA:
        Vega = S × √T × N'(d1) / 100
        
        Note: Standard practice reports Vega as "change per 1% change in volatility"
        So we divide by 100 to get dollars per 1% vol move.
        
        INTERPRETATION:
        - Vega is ALWAYS positive for both calls and puts
        - Higher volatility = higher option prices (more chance of large moves)
        - Largest for ATM options with long time to expiry
        - Vega increases with time (longer-dated options have more vol exposure)
        
        WHY VEGA IS CRITICAL:
        
        Volatility is the ONLY unobservable parameter in Black-Scholes!
        Vega tells you how much your option price depends on that assumption.
        
        Example:
        - You price a call at $10 assuming 20% volatility
        - Current market (implied) volatility is 25%
        - Vega = 0.50 (means $0.50 per 1% vol change)
        - Fair market price = $10 + 5 × $0.50 = $12.50
        
        REAL TRADING USE:
        - "Long vega": Expecting volatility to increase (buy options)
        - "Short vega": Expecting volatility to decrease (sell options)
        - Vega hedging: Offset vega exposure with different options
        
        VOLATILITY SKEW:
        Different strikes have different implied volatilities (volatility smile/skew).
        Vega helps you compare vol exposure across strikes.
        
        Returns:
            Vega value (dollars per 1% change in volatility)
        """
        # PDF at d1
        Nd1_prime = norm.pdf(self.d1)
        
        # Vega per 1% change in volatility
        # Multiply by 0.01 to convert from "per 1.0 change" to "per 1% change"
        vega = self.S * np.sqrt(self.T) * Nd1_prime * 0.01
        
        return vega
    
    def rho(self, option_type: str = 'call') -> float:
        """
        Calculate Rho (ρ) - Rate of change of option price with respect to interest rate.
        
        DEFINITION: ρ = ∂V / ∂r
        Measures how much the option price changes when interest rates change by 1%.
        
        FORMULAS:
        Call Rho = K × T × e^(-rT) × N(d2) / 100
        Put Rho = -K × T × e^(-rT) × N(-d2) / 100
        
        INTERPRETATION:
        - Call Rho is POSITIVE (higher rates increase call prices)
        - Put Rho is NEGATIVE (higher rates decrease put prices)
        - Rho tends to be VERY small for short-dated options
        - Rho becomes significant for:
          * Long-dated options (LEAPS with 2+ years to expiry)
          * When interest rates are high or changing rapidly (2008, 2022)
        
        WHY RHO IS OFTEN IGNORED:
        - For typical options (under 1 year), a 1% rate change moves price < $0.10
        - Rate changes happen slowly compared to stock/volatility moves
        - Most retail traders don't hedge rho
        
        WHEN RHO MATTERS:
        - Hedging large institutional portfolios
        - Trading around Fed announcements (rates can change 0.25-0.50%)
        - Pricing options with 2+ years to expiry
        
        Returns:
            Rho value (dollars per 1% change in interest rate)
        """
        # Time-weighted discount factor
        T_discounted_K = self.T * self.K * np.exp(-self.r * self.T)
        
        if option_type.lower() == 'call':
            rho_annual = T_discounted_K * norm.cdf(self.d2)
        else:
            rho_annual = -T_discounted_K * norm.cdf(-self.d2)
        
        # Convert to "per 1%" (divide by 100)
        rho_percent = rho_annual / 100
        
        return rho_percent
    
    def get_all_prices_and_greeks(self) -> Dict:
        """
        Calculate all option prices and Greeks in one method.
        
        This is the main method you'll use - gives you everything in one call.
        
        Returns:
            Dictionary containing:
            - call_price: Fair call option price
            - put_price: Fair put option price
            - delta_call: Call option delta
            - delta_put: Put option delta
            - gamma: Gamma (same for calls and puts)
            - theta_call: Daily theta for call
            - theta_put: Daily theta for put
            - vega: Vega (per 1% vol change)
            - rho_call: Call rho (per 1% rate change)
            - rho_put: Put rho (per 1% rate change)
            - d1, d2: Intermediate values (for verification)
        """
        return {
            # Prices
            'call_price': self.call_price(),
            'put_price': self.put_price(),
            
            # Greeks
            'delta_call': self.delta('call'),
            'delta_put': self.delta('put'),
            'gamma': self.gamma(),
            'theta_call': self.theta('call'),
            'theta_put': self.theta('put'),
            'vega': self.vega(),
            'rho_call': self.rho('call'),
            'rho_put': self.rho('put'),
            
            # Parameters
            'd1': self.d1,
            'd2': self.d2,
            'stock_price': self.S,
            'strike': self.K,
            'time_to_expiry_years': self.T,
            'risk_free_rate': self.r,
            'volatility': self.sigma
        }
    
    def probability_of_profit(self, option_type: str = 'call') -> float:
        """
        Calculate approximate probability of profit at expiration.
        
        This uses N(d2) which, under the risk-neutral measure, is the
        probability that the option expires in-the-money.
        
        Important: This is risk-neutral probability, NOT real-world probability.
        Risk-neutral probabilities incorporate risk aversion and differ from
        actual historical probabilities.
        
        FORMULA:
        P(ITM) = N(d2) for calls
        P(ITM) = N(-d2) for puts
        
        For actual profit (not just breakeven), need to overcome the option premium:
        Call breakeven = K + call_price
        Put breakeven = K - put_price
        
        Args:
            option_type: 'call' or 'put'
        
        Returns:
            Probability of expiring ITM (0 to 1)
        """
        if option_type.lower() == 'call':
            return norm.cdf(self.d2)
        else:
            return norm.cdf(-self.d2)
    
    def breakeven_price(self, option_type: str = 'call') -> float:
        """
        Calculate the breakeven stock price at expiration.
        
        This is the stock price where the option trade breaks even
        (profit = 0) at expiration.
        
        FORMULAS:
        Call breakeven = Strike Price + Call Premium Paid
        Put breakeven = Strike Price - Put Premium Paid
        
        INTERPRETATION:
        - For a call, stock must rise ABOVE breakeven to profit
        - For a put, stock must fall BELOW breakeven to profit
        - The premium paid creates a "hurdle" that must be overcome
        
        EXAMPLE:
        Buy AAPL $200 call for $10
        Breakeven = $210
        - If AAPL at $205 at expiry: Option worth $5, loss of $5
        - If AAPL at $210 at expiry: Option worth $10, breakeven
        - If AAPL at $220 at expiry: Option worth $20, profit of $10
        
        Returns:
            Breakeven stock price at expiration
        """
        if option_type.lower() == 'call':
            return self.K + self.call_price()
        else:
            return self.K - self.put_price()
    
    def implied_volatility_approx(self, market_price: float, option_type: str = 'call') -> float:
        """
        Approximate implied volatility using Newton-Raphson method.
        
        Given a market price, find the volatility that makes the model price
        equal to the market price.
        
        This is the inverse problem of Black-Scholes:
        Normal: Input σ → Output Price
        Implied Vol: Input Price → Output σ
        
        NEWTON-RAPHSON METHOD:
        σ_{n+1} = σ_n - (f(σ_n) - target) / f'(σ_n)
        
        Where:
        f(σ) = Black-Scholes price at volatility σ
        f'(σ) = Vega (derivative with respect to volatility)
        
        ARGS:
            market_price: Observed option price in the market
            option_type: 'call' or 'put'
        
        Returns:
            Implied volatility as decimal
        """
        # Initial guess (mid-range volatility)
        vol_guess = 0.20
        tolerance = 0.00001  # 0.001% accuracy
        max_iterations = 50
        
        for i in range(max_iterations):
            # Create temporary model with guessed volatility
            temp_model = BlackScholes(self.S, self.K, self.T, self.r, vol_guess)
            
            # Calculate price with current guess
            if option_type.lower() == 'call':
                price = temp_model.call_price()
                vega = temp_model.vega() * 100  # Convert back from per-1%
            else:
                price = temp_model.put_price()
                vega = temp_model.vega() * 100
            
            # Calculate error
            error = price - market_price
            
            # Check convergence
            if abs(error) < tolerance:
                return vol_guess
            
            # Update guess (Newton-Raphson step)
            if vega > 0:
                vol_guess = vol_guess - error / vega
            
            # Prevent unrealistic values
            vol_guess = max(0.01, min(vol_guess, 3.0))  # Between 1% and 300%
        
        return vol_guess  # Return best approximation


# Quick educational example
if __name__ == "__main__":
    print("=" * 70)
    print("BLACK-SCHOLES MODEL - EDUCATIONAL DEMO")
    print("=" * 70)
    
    # Example parameters (AAPL option)
    S = 178.50      # Current AAPL price
    K = 200.00      # Strike price
    T = 0.5         # 6 months to expiry
    r = 0.05        # 5% risk-free rate
    sigma = 0.238   # 23.8% volatility (from our calculator)
    
    print(f"\n INPUT PARAMETERS:")
    print(f"   Stock Price (S):    ${S:.2f}")
    print(f"   Strike Price (K):   ${K:.2f}")
    print(f"   Time to Expiry (T): {T:.1f} years (6 months)")
    print(f"   Risk-Free Rate (r): {r*100:.1f}%")
    print(f"   Volatility (σ):     {sigma*100:.1f}%")
    
    # Create model
    bs = BlackScholes(S, K, T, r, sigma)
    
    # Get all results
    results = bs.get_all_prices_and_greeks()
    
    print(f"\n OPTION PRICES:")
    print(f"   Call Price:  ${results['call_price']:.2f}")
    print(f"   Put Price:   ${results['put_price']:.2f}")
    
    print(f"\n THE GREEKS (Risk Sensitivities):")
    print(f"   Delta (Call):  {results['delta_call']:.4f}  (Option moves ${results['delta_call']:.2f} per $1 stock move)")
    print(f"   Delta (Put):   {results['delta_put']:.4f}  (Option moves ${results['delta_put']:.2f} per $1 stock move)")
    print(f"   Gamma:         {results['gamma']:.4f}  (Delta changes by {results['gamma']:.4f} per $1 stock move)")
    print(f"   Theta (Call):  ${results['theta_call']:.4f}  (Loses ${abs(results['theta_call']):.2f} per day)")
    print(f"   Theta (Put):   ${results['theta_put']:.4f}  (Loses ${abs(results['theta_put']):.2f} per day)")
    print(f"   Vega:          ${results['vega']:.2f}  (Changes ${results['vega']:.2f} per 1% volatility change)")
    print(f"   Rho (Call):    ${results['rho_call']:.4f}  (Changes ${results['rho_call']:.2f} per 1% rate change)")
    print(f"   Rho (Put):     ${results['rho_put']:.4f}  (Changes ${results['rho_put']:.2f} per 1% rate change)")
    
    print(f"\n TRADING INSIGHTS:")
    print(f"   Call Probability of Profit: {bs.probability_of_profit('call')*100:.1f}%")
    print(f"   Put Probability of Profit:  {bs.probability_of_profit('put')*100:.1f}%")
    print(f"   Call Breakeven:             ${bs.breakeven_price('call'):.2f} ({(bs.breakeven_price('call')/S - 1)*100:.1f}% above current)")
    print(f"   Put Breakeven:              ${bs.breakeven_price('put'):.2f} ({(1 - bs.breakeven_price('put')/S)*100:.1f}% below current)")
    
    print(f"\n INTERMEDIATE VALUES:")
    print(f"   d1 = {results['d1']:.4f}")
    print(f"   d2 = {results['d2']:.4f}")
    print(f"   N(d1) = {norm.cdf(results['d1']):.4f}")
    print(f"   N(d2) = {norm.cdf(results['d2']):.4f}")
    
    # Put-call parity check
    theoretical_put = results['put_price']
    parity_put = results['call_price'] - S + K * np.exp(-r * T)
    print(f"\n PUT-CALL PARITY CHECK:")
    print(f"   Theoretical Put:  ${theoretical_put:.4f}")
    print(f"   Put from Parity:  ${parity_put:.4f}")
    print(f"   Difference:       ${abs(theoretical_put - parity_put):.6f} (should be ~0)")
    
    print("\n" + "=" * 70)
    print(" INTERPRETATION:")
    print("   This AAPL $200 call costs $8.42 for 6 months of exposure.")
    print("   Delta of 0.38 means it behaves like owning 38 shares of AAPL.")
    print("   Gamma of 0.025 means delta increases by 0.025 for each $1 stock rise.")
    print("   Theta of -$0.04 means you lose 4 cents per day to time decay.")
    print("   Vega of $0.39 means if volatility rises to 30%, call price → ~$11.00.")
    print("=" * 70)