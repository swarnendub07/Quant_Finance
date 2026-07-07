#### README

This repository contains Python implementations of quantitative finance models and numerical methods.

- `Pricing_option_commodities.py`: 
  - Prices **European** and **American put options** on commodities using the **Cox–Ross–Rubinstein (CRR) Binomial Tree** model.
  - Implements the **Black–Scholes pricing formula** for European commodity options with convenience yield.
  - Compares the convergence of the binomial model to the Black–Scholes solution.
  - Studies the effect of **convenience yield** on European and American option prices.
  - Computes and visualizes the **early-exercise region** for American put options.
  - Computes and plots the **early-exercise boundary** as a function of time.
  - Analyzes the impact of convenience yield on the early-exercise boundary
  - Investigates the effect of a **time-varying (seasonal) convenience yield** on the early-exercise boundary.

- `Implied_volatility`:
  - Prices European put option under a geometric Brownian motion (GBM) model with a deterministic, time-dependent volatility:
  - Compares analytical and Monte Carlo option prices 
  - The implied volatility curve is generated across different strike prices

