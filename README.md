# Options Pricing & Greeks Calculator

A small Black–Scholes–Merton calculator for European calls and puts, with an interactive Streamlit interface.

## Run

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

## Model and conventions

Inputs include spot, strike, days to expiry, annual continuously compounded risk-free rate, annual volatility, and annual continuous dividend yield. The core in `pricing.py` returns price, Delta, Gamma, Vega, Theta, and Rho. Vega and Rho are scaled per one percentage point; Theta is per calendar day. The implied-volatility solver uses SciPy's bracketed Brent method and checks no-arbitrage bounds first.

`put_call_parity_residual` and `finite_difference_delta` are included as simple independent consistency checks for callers. The app shows the expiry payoff alongside model value and plots each Greek over a range of underlying prices.
