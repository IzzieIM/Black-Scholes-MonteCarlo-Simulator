"""
src/volatility_calculator.py
Volatility Estimation Engine - Solves the "Invisible Parameter" Problem

This module calculates historical volatility from market data using multiple methods:
- Close-to-Close: Standard log returns method
- Parkinson: High-Low range estimator (more efficient)
- EWMA: Exponentially Weighted Moving Average (prioritizes recent data)
- Garman-Klass: OHLC estimator (most information-efficient)

The weighted recommendation combines all methods for robust estimation.
"""

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from typing import Tuple, Dict, Optional


class VolatilityCalculator:
    """
    Calculates historical volatility from stock price data using multiple methods.
    
    Attributes:
        ticker (str): Stock ticker symbol (e.g., 'AAPL')
        lookback_days (int): Number of trading days for historical data (default: 252)
        current_price (float): Most recent stock price
        daily_returns (np.ndarray): Array of daily log returns
        price_data (pd.DataFrame): Raw historical price data from Yahoo Finance
    """
    
    def __init__(self, ticker: str, lookback_days: int = 252):
        """
        Initialize the volatility calculator.
        
        Args:
            ticker: Stock ticker symbol (e.g., 'AAPL', 'MSFT', 'SPY')
            lookback_days: Number of trading days to fetch (default 252 = 1 year)
        """
        self.ticker = ticker
        self.lookback_days = lookback_days
        self.current_price = None
        self.daily_returns = None
        self.price_data = None
        
        # Fetch and process data immediately on initialization
        self._fetch_historical_data()
        self._calculate_returns()
    
    def _fetch_historical_data(self) -> None:
        """
        Fetch historical price data from Yahoo Finance.
        
        This method retrieves:
        - Open, High, Low, Close prices for each trading day
        - Automatically handles the lookback period (adds buffer for returns calculation)
        """
        try:
            # Fetch extra days to ensure we have enough for returns calculation
            end_date = datetime.now()
            start_date = end_date - timedelta(days=self.lookback_days + 30)
            
            print(f"📊 Fetching {self.lookback_days} days of data for {self.ticker}...")
            
            # Download data from Yahoo Finance
            stock = yf.Ticker(self.ticker)
            self.price_data = stock.history(start=start_date, end=end_date)
            
            if self.price_data.empty:
                raise ValueError(f"No data found for ticker '{self.ticker}'. Please check the symbol.")
            
            # Store current price
            self.current_price = self.price_data['Close'].iloc[-1]
            
            # Trim to exact lookback period (now that we have returns calculated)
            self.price_data = self.price_data.tail(self.lookback_days + 1)
            
            print(f"✅ Successfully fetched data. Current {self.ticker} price: ${self.current_price:.2f}")
            print(f"   Data range: {self.price_data.index[0].strftime('%Y-%m-%d')} to {self.price_data.index[-1].strftime('%Y-%m-%d')}")
            
        except Exception as e:
            raise ConnectionError(f"Failed to fetch data for {self.ticker}: {str(e)}")
    
    def _calculate_returns(self) -> None:
        """
        Calculate daily log returns from closing prices.
        
        Formula: r_t = ln(P_t / P_{t-1})
        Log returns are preferred over simple returns because they're:
        - Time-additive
        - Normally distributed (approximately)
        - Unbounded (can go to -∞, +∞)
        """
        closing_prices = self.price_data['Close']
        
        # Calculate log returns: ln(P_t / P_{t-1})
        self.daily_returns = np.log(closing_prices / closing_prices.shift(1)).dropna().values
        
        # Remove any infinite or NaN values (shouldn't happen with valid data)
        self.daily_returns = self.daily_returns[np.isfinite(self.daily_returns)]
        
        print(f"   Calculated {len(self.daily_returns)} daily log returns")
    
    def method_close_to_close(self) -> float:
        """
        Standard close-to-close volatility estimator.
        
        Formula: σ_daily = std(ln(P_t / P_{t-1}))
        Annualized: σ_annual = σ_daily × √252
        
        This is the most common method but only captures end-of-day movements,
        missing intraday volatility.
        
        Returns:
            Annualized volatility as a decimal (e.g., 0.20 = 20%)
        """
        if len(self.daily_returns) < 2:
            return 0.15  # Default fallback for insufficient data
        
        daily_vol = np.std(self.daily_returns, ddof=1)  # Sample standard deviation
        annualized_vol = daily_vol * np.sqrt(252)  # 252 trading days/year
        
        return annualized_vol
    
    def method_parkinson(self) -> float:
        """
        Parkinson's high-low volatility estimator.
        
        Formula: σ_parkinson = (1/(4×ln(2))) × mean(ln(High/Low)²)
        Annualized: σ_annual = √(mean_value) × √252
        
        Parkinson's method is more efficient than close-to-close because it uses
        intraday high and low prices, capturing volatility that happens within the day.
        
        Reference: Parkinson, M. (1980). The Extreme Value Method for Estimating
                   the Variance of the Rate of Return.
        
        Returns:
            Annualized volatility as a decimal
        """
        high = self.price_data['High'].values
        low = self.price_data['Low'].values
        
        # Calculate squared log range
        squared_log_range = (np.log(high / low)) ** 2
        
        # Parkinson's scaling factor: 1 / (4 × ln(2))
        scaling_factor = 1 / (4 * np.log(2))
        
        daily_variance = scaling_factor * np.mean(squared_log_range)
        daily_vol = np.sqrt(daily_variance)
        annualized_vol = daily_vol * np.sqrt(252)
        
        return annualized_vol
    
    def method_ewma(self, lambda_factor: float = 0.94) -> float:
        """
        Exponentially Weighted Moving Average (EWMA) volatility.
        
        Formula: σ²_t = λ × σ²_{t-1} + (1-λ) × r²_{t-1}
        
        Where:
        - λ is the decay factor (0.94 is standard for daily data, as used by RiskMetrics)
        - Recent observations get higher weight
        
        This method is more responsive to recent market conditions, making it
        better for capturing changing volatility regimes (earnings, crises, etc.).
        
        Args:
            lambda_factor: Decay factor (0.94 = 94% weight on previous variance)
        
        Returns:
            Annualized volatility as a decimal
        """
        if len(self.daily_returns) < 10:
            return self.method_close_to_close()
        
        # Initialize EWMA variance with the first few observations
        initial_window = min(10, len(self.daily_returns))
        variance = np.var(self.daily_returns[:initial_window])
        
        # Calculate EWMA variance recursively
        for ret in self.daily_returns[initial_window:]:
            variance = lambda_factor * variance + (1 - lambda_factor) * (ret ** 2)
        
        daily_vol = np.sqrt(variance)
        annualized_vol = daily_vol * np.sqrt(252)
        
        return annualized_vol
    
    def method_garman_klass(self) -> float:
        """
        Garman-Klass OHLC volatility estimator.
        
        Formula: σ² = 0.5×ln(High/Low)² - (2×ln(2)-1)×ln(Close/Open)²
        
        This is the most information-efficient estimator using all four price points
        (Open, High, Low, Close) from each trading day.
        
        Reference: Garman, M. B., & Klass, M. J. (1980). On the estimation of
                   security price volatilities from historical data.
        
        Returns:
            Annualized volatility as a decimal
        """
        high = self.price_data['High'].values
        low = self.price_data['Low'].values
        open_prices = self.price_data['Open'].values
        close = self.price_data['Close'].values
        
        # Garman-Klass variance calculation
        log_hl = np.log(high / low)
        log_co = np.log(close / open_prices)
        
        # The Garman-Klass estimator
        daily_variance = 0.5 * (log_hl ** 2) - (2 * np.log(2) - 1) * (log_co ** 2)
        
        # Ensure no negative variance (can happen due to price staleness)
        daily_variance = np.maximum(daily_variance, 0)
        
        daily_vol = np.sqrt(np.mean(daily_variance))
        annualized_vol = daily_vol * np.sqrt(252)
        
        return annualized_vol
    
    def get_recommended_volatility(self, weights: Optional[Dict[str, float]] = None) -> float:
        """
        Calculate a weighted recommendation combining all methods.
        
        Different methods capture different aspects of volatility:
        - Close-to-close: Standard, widely understood
        - Parkinson: Captures intraday movement
        - EWMA: Responds to recent changes
        - Garman-Klass: Most information-efficient
        
        Default weights prioritize EWMA (recent behavior) and Parkinson (intraday),
        giving less weight to standard close-to-close.
        
        Args:
            weights: Dictionary with keys 'close', 'parkinson', 'ewma', 'garman'
                    Default: {'close': 0.2, 'parkinson': 0.25, 'ewma': 0.4, 'garman': 0.15}
        
        Returns:
            Weighted annualized volatility as a decimal
        """
        if weights is None:
            weights = {
                'close': 0.20,
                'parkinson': 0.25,
                'ewma': 0.40,
                'garman': 0.15
            }
        
        # Calculate all methods
        vol_close = self.method_close_to_close()
        vol_parkinson = self.method_parkinson()
        vol_ewma = self.method_ewma()
        vol_garman = self.method_garman_klass()
        
        # Store for analysis
        self.all_volatilities = {
            'close_to_close': vol_close,
            'parkinson': vol_parkinson,
            'ewma': vol_ewma,
            'garman_klass': vol_garman
        }
        
        # Calculate weighted average
        weighted_vol = (
            weights['close'] * vol_close +
            weights['parkinson'] * vol_parkinson +
            weights['ewma'] * vol_ewma +
            weights['garman'] * vol_garman
        )
        
        return weighted_vol
    
    def get_volatility_summary(self) -> Dict:
        """
        Get a comprehensive summary of all volatility estimates.
        
        Returns:
            Dictionary containing all volatility estimates and metadata
        """
        vol_close = self.method_close_to_close()
        vol_parkinson = self.method_parkinson()
        vol_ewma = self.method_ewma()
        vol_garman = self.method_garman_klass()
        vol_recommended = self.get_recommended_volatility()
        
        return {
            'ticker': self.ticker,
            'current_price': self.current_price,
            'lookback_days': self.lookback_days,
            'data_points': len(self.daily_returns),
            'volatilities': {
                'close_to_close': vol_close,
                'parkinson': vol_parkinson,
                'ewma': vol_ewma,
                'garman_klass': vol_garman,
                'recommended': vol_recommended
            },
            'volatility_percent': {
                'close_to_close': f"{vol_close * 100:.1f}%",
                'parkinson': f"{vol_parkinson * 100:.1f}%",
                'ewma': f"{vol_ewma * 100:.1f}%",
                'garman_klass': f"{vol_garman * 100:.1f}%",
                'recommended': f"{vol_recommended * 100:.1f}%"
            }
        }
    
    def get_raw_price_data(self) -> pd.DataFrame:
        """Return the raw price data for external use (e.g., visualization)."""
        return self.price_data
    
    def get_daily_returns(self) -> np.ndarray:
        """Return the daily log returns array."""
        return self.daily_returns


# Quick test/demo when run directly
if __name__ == "__main__":
    print("=" * 60)
    print("VOLATILITY CALCULATOR - DEMO")
    print("=" * 60)
    print("\nThis module solves the 'invisible parameter' problem by")
    print("calculating volatility from real market data.\n")
    
    # Test with Apple stock
    ticker = input("Enter ticker symbol (default: AAPL): ").strip().upper() or "AAPL"
    
    try:
        calc = VolatilityCalculator(ticker)
        summary = calc.get_volatility_summary()
        
        print(f"\n📈 VOLATILITY SUMMARY for {summary['ticker']}")
        print(f"   Current Price: ${summary['current_price']:.2f}")
        print(f"   Data Points: {summary['data_points']} trading days")
        print("\n   Method Estimates:")
        print(f"   ├── Close-to-Close:   {summary['volatility_percent']['close_to_close']}")
        print(f"   ├── Parkinson (HL):   {summary['volatility_percent']['parkinson']}")
        print(f"   ├── EWMA (Recent):    {summary['volatility_percent']['ewma']}")
        print(f"   ├── Garman-Klass:     {summary['volatility_percent']['garman_klass']}")
        print(f"   └── RECOMMENDED:      {summary['volatility_percent']['recommended']} ⭐")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("   Please check your internet connection and ticker symbol.")