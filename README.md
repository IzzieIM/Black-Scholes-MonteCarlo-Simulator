# COMPREHENSIVE README.md FOR OPTIONS PRICING SYSTEM

```markdown
# 📈 Real-Time Options Pricing & Risk Analysis System

---

## 🎯 Project Overview

This is a **production-grade options pricing system** that automatically fetches real market data and prices options using both analytical (Black-Scholes) and numerical (Monte Carlo) methods. The system requires only a company ticker symbol and basic option parameters from the user.

### The Core Philosophy

> **"Black-Scholes provides one elegant answer in an ideal world; Monte Carlo shows the messy reality of all possible outcomes. Together, they create a robust pricing and risk management engine."**

### What Makes This Project Unique?

| Feature | Traditional Tools | This System |
|---------|------------------|--------------|
| Data Input | Manual price entry | Auto-fetches from Yahoo Finance |
| Volatility | User must estimate | Calculated from 4 methods automatically |
| Pricing Method | Single model | Dual validation (BS + MC) |
| Risk Metrics | Basic or none | All 5 Greeks + confidence intervals |
| Validation | None | Cross-validation between models |
| Cost | Expensive ($1000s/month) | Free and open-source |

---

## 📊 System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      USER INPUT (Minimal)                     │
│  • Ticker Symbol (e.g., 'AAPL')                             │
│  • Strike Price (K)                                          │
│  • Time to Expiry (T)                                        │
│  • Risk-Free Rate (r)                                        │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│              DATA ACQUISITION LAYER                           │
│  • Yahoo Finance API (yfinance)                             │
│  • Auto-fetches real-time stock price (S)                   │
│  • Downloads 1 year of historical data                      │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│            VOLATILITY CALCULATION ENGINE                      │
│  • Close-to-Close (20% weight)                              │
│  • Parkinson (25% weight)                                    │
│  • EWMA (40% weight)                                         │
│  • Garman-Klass (15% weight)                                 │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│                 PRICING ENGINES                               │
│  ┌─────────────────────┐    ┌─────────────────────────────┐ │
│  │  BLACK-SCHOLES       │    │  MONTE CARLO                 │ │
│  │  • Exact formula      │    │  • 10,000+ paths             │ │
│  │  • Instant (<0.01s)   │    │  • Confidence intervals      │ │
│  │  • All 5 Greeks       │    │  • Convergence analysis      │ │
│  └─────────────────────┘    └─────────────────────────────┘ │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│                   OUTPUT LAYER                                │
│  • Fair option price          • Price distribution           │
│  • All Greeks                 • Convergence plots            │
│  • 95% Confidence Intervals   • Risk metrics                 │
└─────────────────────────────────────────────────────────────┘
```

---

## ✨ Key Features

### 1. **Auto-Market Data Integration**
- No manual price entry needed
- Fetches real-time stock prices from Yahoo Finance
- Downloads historical data for volatility calculation

### 2. **Multi-Method Volatility Estimation**
Solves the "invisible parameter" problem using 4 methods:

| Method | Description | Best For |
|--------|-------------|----------|
| **Close-to-Close** | Standard log returns method | Benchmark comparison |
| **Parkinson** | High-Low range estimator | Intraday volatility capture |
| **EWMA** | Exponentially weighted (λ=0.94) | Recent market conditions |
| **Garman-Klass** | OHLC estimator | Most information-efficient |

### 3. **Dual Pricing Engines**

#### Black-Scholes (Analytical)
- Exact closed-form solution
- Instantaneous calculation
- All 5 Greeks (Delta, Gamma, Theta, Vega, Rho)

#### Monte Carlo (Numerical)
- 10,000+ simulated price paths
- Confidence intervals (95%)
- Convergence analysis
- Distribution of outcomes

### 4. **Comprehensive Risk Metrics**
- **Probability ITM** - Chance of expiring in-the-money
- **Probability of Profit** - Chance of positive return
- **Breakeven Analysis** - Required stock move for profit
- **Risk/Reward Ratios** - Premium vs potential gain
- **Maximum Loss** - Premium paid

### 5. **Model Validation**
- Automatic comparison between BS and MC
- Confidence interval checking
- Validation pass/fail reporting

### 6. **Professional Visualizations**
- Volatility method comparison
- Price convergence charts
- Distribution histograms
- Greeks dashboard
- Sensitivity heatmaps
- Sample price paths

---

## 🚀 Installation

### Prerequisites

```bash
Python 3.8 or higher
pip package manager
Internet connection (for data fetching)
```

### Step 1: Clone or Download

```bash
git clone https://github.com/IzzieIM/Black-Scholes-MonteCarlo-Simulator
```

### Step 2: Create Virtual Environment (Recommended)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Mac/Linux
python -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

**Dependencies Explained:**

| Package | Version | Purpose |
|---------|---------|---------|
| numpy | ≥1.21.0 | Numerical computations |
| scipy | ≥1.7.0 | Statistical functions |
| pandas | ≥1.3.0 | Data manipulation |
| yfinance | ≥0.2.0 | Market data fetching |
| matplotlib | ≥3.4.0 | Visualizations |
| seaborn | ≥0.11.0 | Statistical plots |

---

## 💻 Usage Guide

### Quick Start - Interactive Mode

```bash
python main.py
```

Then follow the prompts:

```
📋 Enter option parameters:
--------------------------------------------------
   Stock Ticker (e.g., AAPL, MSFT, SPY): AAPL
   Strike Price ($): 200
   Time to Expiry (in months, e.g., 3 for 3 months): 6
   Risk-Free Rate (%, e.g., 5 for 5%): 5

🔧 Advanced Settings (press Enter for defaults):
   Custom simulations? (default 10,000): [Enter]
```

### Programmatic Usage

```python
from main import SimpleOptionsPricer

# Create pricer with auto-fetched data
pricer = SimpleOptionsPricer(
    ticker='AAPL',    # Just the ticker - price auto-fetched!
    K=200,            # Strike price
    T=0.5,            # 6 months (0.5 years)
    r=0.05            # 5% risk-free rate
)

# Get complete analysis
results = pricer.get_full_analysis()

# Print detailed report
pricer.print_detailed_report()

# Access specific values
print(f"Call Price: ${results['black_scholes']['call_price']:.2f}")
print(f"Delta: {results['black_scholes']['delta_call']:.4f}")
print(f"Vega: ${results['black_scholes']['vega']:.2f} per 1% vol")
```

### Generate Visualizations

```python
from visualization.plots import OptionVisualizer

# Create visualizer
viz = OptionVisualizer(pricer)

# Generate all plots
viz.create_full_dashboard()

# Display
viz.show()

# Save to files
viz.create_full_dashboard(save_dir='output_plots')
```

---

## 📖 Understanding the Output

### Sample Output Explained

```
💰 OPTION PRICES:
   Black-Scholes Call: $105.20
   Black-Scholes Put:  $0.03
   Monte Carlo Call:   $105.13 ± 0.33
   Monte Carlo Put:    $0.03 ± 0.01
```

| Component | Meaning |
|-----------|---------|
| Black-Scholes | Exact theoretical price |
| Monte Carlo | Simulated price ± standard error |
| ± Value | 1 standard deviation of simulation error |

### The Greeks Explained

| Greek | Symbol | What It Measures | Typical Range |
|-------|--------|------------------|---------------|
| **Delta** | Δ | Price change per $1 stock move | 0 to 1 (call), -1 to 0 (put) |
| **Gamma** | Γ | Delta change per $1 stock move | 0 to 0.1+ |
| **Theta** | Θ | Daily time decay | -$1 to $0 |
| **Vega** | ν | Price change per 1% vol change | $0 to $1 |
| **Rho** | ρ | Price change per 1% rate change | -$0.10 to $0.10 |

### Probability Metrics

| Metric | Calculation | Interpretation |
|--------|-------------|----------------|
| **Probability ITM** | N(d2) for calls | Chance option has intrinsic value |
| **Probability of Profit** | Simulated | Chance of positive return after premium |

---

## 📁 Project Structure

```
options_pricing_system/
│
├── src/                              # Core modules
│   ├── __init__.py                   # Package initialization
│   ├── volatility_calculator.py     # Multi-method volatility estimation
│   ├── black_scholes.py             # Black-Scholes model + Greeks
│   └── monte_carlo.py               # Monte Carlo simulation engine
│
├── visualization/                    # Visualization suite
│   ├── __init__.py
│   └── plots.py                      # Dashboard and charts
│
├── main.py                           # Main application & CLI
├── requirements.txt                  # Dependencies
├── README.md                         # Documentation
└── LICENSE                           # MIT License
```

---

## 🧪 Testing

### Run Included Test Suite

```bash
python test_dummy.py
```

### Test Different Scenarios

```python
# Test 1: At-the-money option
pricer = SimpleOptionsPricer('AAPL', K=300, T=0.5, r=0.05)

# Test 2: Deep OTM call
pricer = SimpleOptionsPricer('AAPL', K=350, T=0.5, r=0.05)

# Test 3: Short expiry
pricer = SimpleOptionsPricer('AAPL', K=200, T=0.05, r=0.05)

# Test 4: High volatility stock
pricer = SimpleOptionsPricer('TSLA', K=250, T=0.5, r=0.05)
```

### Expected Output Ranges

| Ticker | Current Price | Volatility | ATM Call (6M) |
|--------|--------------|------------|---------------|
| AAPL | $165-190 | 20-30% | $8-15 |
| MSFT | $400-430 | 18-25% | $25-35 |
| TSLA | $160-200 | 40-60% | $30-50 |
| SPY | $500-550 | 12-18% | $20-30 |

---

## 📊 Sample Results

### Case Study: AAPL $200 Call (6 months)

**Input:**
```
Ticker: AAPL
Strike: $200
Expiry: 6 months
Rate: 5%
```

**Output:**
```
Current Price: $300.23
Volatility: 21.8%

Call Price: $105.20
Delta: 0.998 (moves almost 1:1 with stock)
Theta: -$0.027/day (slow time decay)
Vega: $0.01 per 1% vol (low vol sensitivity)

Probability ITM: 99.7%
Breakeven: $305.20 (only 1.7% above current)
```

### Interpretation

This is a **deep in-the-money call option**:
- Almost certain to be exercised (99.7% probability)
- Behaves like owning 100 shares
- Small time decay
- Low volatility sensitivity
- Small required move for profit (1.7%)

---

## 🔬 Financial Mathematics Background

### Black-Scholes Formula

```
C = S·N(d₁) - K·e^{-rT}·N(d₂)

where:
d₁ = [ln(S/K) + (r + σ²/2)T] / (σ√T)
d₂ = d₁ - σ√T
```

### Geometric Brownian Motion (Monte Carlo)

```
S_{t+Δt} = S_t × exp((r - σ²/2)Δt + σ√Δt × Z)

where Z ~ N(0,1)
```

### Volatility Estimators

**Parkinson:**
```
σ² = (1/(4·ln(2))) × E[ln(H/L)²]
```

**Garman-Klass:**
```
σ² = 0.5·ln(H/L)² - (2·ln(2)-1)·ln(C/O)²
```

**EWMA:**
```
σ²_t = λ·σ²_{t-1} + (1-λ)·r²_{t-1}
```

---

## ⚠️ Limitations & Assumptions

### Black-Scholes Assumptions
1. No arbitrage opportunities
2. Stock follows Geometric Brownian Motion
3. No transaction costs or taxes
4. Constant risk-free rate
5. **Constant volatility** (most criticized)
6. No dividends
7. European exercise only

### Monte Carlo Limitations
1. Simulation error (mitigated by multiple runs)
2. Computationally intensive
3. Requires pseudo-random number generator

### Practical Considerations
- Historical volatility ≠ future volatility
- Real markets have fat tails (Black-Scholes assumes normal)
- Dividends not modeled (can be added)
- American options not supported (European only)

---

## 🛠️ Troubleshooting

### Common Issues & Solutions

| Issue | Solution |
|-------|----------|
| `No data found for ticker` | Check ticker symbol, internet connection |
| `ImportError: No module named 'src'` | Run from project root directory |
| `SSL: CERTIFICATE_VERIFY_FAILED` (Mac) | Run Install Certificates.command |
| Monte Carlo too slow | Reduce `n_simulations` or install numba |
| Negative option price | Check inputs (volatility too high?) |

### Debug Mode

```python
# Enable verbose output
pricer = SimpleOptionsPricer(..., verbose=True)

# Test individual components
from src.volatility_calculator import VolatilityCalculator
calc = VolatilityCalculator('AAPL')
print(calc.get_volatility_summary())
```

---

## 🎓 Learning Outcomes

By exploring this project, you'll understand:

### Quantitative Finance
- Options theory and pricing
- Black-Scholes model derivation
- Risk-neutral valuation
- Greeks and risk management

### Stochastic Processes
- Geometric Brownian Motion
- Random walks and diffusion
- Itô's lemma application

### Numerical Methods
- Monte Carlo simulation
- Convergence analysis
- Variance reduction (antithetic variates)

### Statistical Analysis
- Confidence intervals
- Standard error estimation
- Distribution analysis (skewness, kurtosis)

### Software Engineering
- Modular design patterns
- API integration (yfinance)
- Object-oriented programming
- Data visualization

---

## 🚀 Future Enhancements

### Planned Features
- [ ] Dividend support
- [ ] American option pricing (binomial tree)
- [ ] Implied volatility surface
- [ ] Greeks for exotic options
- [ ] Real-time streaming data
- [ ] Web interface (Streamlit/Flask)
- [ ] Database storage for historical analysis
- [ ] Strategy backtesting engine
- [ ] Export to Excel/CSV with formatting
- [ ] Multi-threaded Monte Carlo
