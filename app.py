import numpy as np
import plotly.graph_objects as go
import streamlit as st

from pricing import black_scholes, implied_volatility

st.set_page_config(page_title="Options Pricing & Greeks", page_icon="📈", layout="wide")
st.title("Options Pricing & Greeks Calculator")
st.caption("European options · Black–Scholes–Merton · annualized inputs")

with st.sidebar:
    st.header("Contract inputs")
    option_type = st.radio("Option type", ["call", "put"], horizontal=True)
    spot = st.number_input("Spot price", min_value=0.01, value=100.0, step=1.0)
    strike = st.number_input("Strike price", min_value=0.01, value=100.0, step=1.0)
    days = st.number_input("Days to expiry", min_value=0, value=90, step=1)
    rate_pct = st.number_input("Risk-free rate (%)", value=4.0, step=0.25)
    vol_pct = st.number_input("Volatility (%)", min_value=0.0, value=25.0, step=1.0)
    dividend_pct = st.number_input("Dividend yield (%)", value=0.0, step=0.25)
    time = days / 365.0
    rate, vol, dividend = rate_pct / 100.0, vol_pct / 100.0, dividend_pct / 100.0

result = black_scholes(spot, strike, time, rate, vol, option_type, dividend)
labels = ["Option price", "Delta", "Gamma", "Vega / 1%", "Theta / day", "Rho / 1%"]
values = [result.price, result.delta, result.gamma, result.vega, result.theta, result.rho]
cols = st.columns(6)
for col, label, value in zip(cols, labels, values):
    col.metric(label, f"{value:.5f}" if label != "Option price" else f"{value:.4f}")

st.subheader("Payoff and option value at expiry")
spot_grid = np.linspace(max(0.01, spot * 0.5), spot * 1.5, 160)
payoff = np.maximum(spot_grid - strike, 0) if option_type == "call" else np.maximum(strike - spot_grid, 0)
fig = go.Figure()
fig.add_trace(go.Scatter(x=spot_grid, y=payoff, name="Expiry payoff", line=dict(dash="dash")))
if time > 0 and vol > 0:
    values_grid = [black_scholes(float(s), strike, time, rate, vol, option_type, dividend).price for s in spot_grid]
    fig.add_trace(go.Scatter(x=spot_grid, y=values_grid, name="Current option value", line=dict(width=3)))
fig.add_vline(x=strike, line_dash="dot", annotation_text="Strike")
fig.add_vline(x=spot, line_dash="dot", line_color="gray", annotation_text="Spot")
fig.update_layout(xaxis_title="Underlying price at expiry / valuation", yaxis_title="Value per share", hovermode="x unified", legend=dict(orientation="h"))
st.plotly_chart(fig, use_container_width=True)

st.subheader("Greeks across underlying prices")
greek_fig = go.Figure()
for greek, label in [("delta", "Delta"), ("gamma", "Gamma"), ("vega", "Vega / 1%"), ("theta", "Theta / day"), ("rho", "Rho / 1%")]:
    series = [getattr(black_scholes(float(s), strike, time, rate, vol, option_type, dividend), greek) for s in spot_grid]
    greek_fig.add_trace(go.Scatter(x=spot_grid, y=series, name=label))
greek_fig.update_layout(xaxis_title="Underlying price", yaxis_title="Greek value", hovermode="x unified", legend=dict(orientation="h"))
st.plotly_chart(greek_fig, use_container_width=True)

with st.expander("Implied volatility"):
    market_price = st.number_input("Observed option price", min_value=0.0, value=float(result.price), step=0.1, key="market_price")
    if st.button("Solve implied volatility"):
        try:
            iv = implied_volatility(market_price, spot, strike, time, rate, option_type, dividend)
            st.success(f"Implied volatility: {iv:.2%}")
        except ValueError as exc:
            st.error(str(exc))

with st.expander("Model conventions"):
    st.write("European exercise. Rates, volatility, and dividend yield are continuously compounded and annualized. Vega and rho are shown per 1 percentage point; theta is per calendar day. The chart shows undiscounted expiry payoff and model value today.")
