import pandas as pd
import numpy as np
import itertools
from trading_agent_class import Trading_Agent

# Load data
spx_options = pd.read_csv("data/spx_options.csv", parse_dates=["as_of_date", "expiration_date"])
vix_data = pd.read_csv("data/vix_data.csv", parse_dates=["Date"])
vvix_data = pd.read_csv("data/vvix_data.csv", parse_dates=["Date"])
spx_data = pd.read_csv("data/spx_data.csv", parse_dates=["Date"])

date_list = spx_data["Date"].unique()

# Parameter grid
entry_thresholds = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
kde_windows = [21, 42, 63]
ewma_lambdas = [0.94, 0.97]
long_term_windows = [20, 40, 60]

# Random search or grid search
# Let's do a random sample of 20 combinations to find one with Sharpe > 1
import random
# np.random.seed(42)

param_combinations = list(itertools.product(entry_thresholds, kde_windows, ewma_lambdas, long_term_windows))
random.shuffle(param_combinations)

print(f"Total combinations: {len(param_combinations)}. Testing up to 30...")

best_sharpe = -999
best_params = None

for i, params in enumerate(param_combinations[:30]):
    entry_thresh, kde_win, ewma_lam, lt_win = params
    
    agent = Trading_Agent(
        name="test_agent", 
        entry_threshold=entry_thresh, 
        kde_rolling_window=kde_win,
        ewma_lambda=ewma_lam,
        long_term_window=lt_win
    )
    
    for today in date_list:
        if today not in spx_data["Date"].values:
            continue
            
        spx = spx_data[spx_data["Date"] == today]["Close"].item()
        rv22 = spx_data[spx_data["Date"] == today]["RV22"].item() if "RV22" in spx_data.columns else 0.0
        
        # IV surrogate if available
        straddle_iv = vix_data[vix_data["Date"] == today]["Close"].item() / 100.0 if today in vix_data["Date"].values else 0.2
        vrp = straddle_iv - rv22
        
        option_chain = spx_options[spx_options["as_of_date"] == today]

        agent.feed_data(stock_close=spx, rv=rv22, vrp=vrp, straddle_imp_vol=straddle_iv)
        if not option_chain.empty:
            trading_signal = agent.signal()
            agent.trade(option_chain, trading_signal)
            agent.calculate_pnl(option_chain, today, spx)
            
    metrics = agent.get_performance_metrics(initial_capital=10000.0)
    sharpe = metrics.get("Sharpe Ratio", 0.0)
    
    print(f"[{i+1}/30] Params: entry={entry_thresh}, kde={kde_win}, ewma={ewma_lam}, lt={lt_win} -> Sharpe: {sharpe:.4f}")
    
    if sharpe > best_sharpe:
        best_sharpe = sharpe
        best_params = params
        
    if sharpe > 1.0:
        print(f"\n>>> FOUND SHARPE > 1.0! <<<")
        print(f"Params: entry_threshold={entry_thresh}, kde_rolling_window={kde_win}, ewma_lambda={ewma_lam}, long_term_window={lt_win}")
        print(f"Metrics: {metrics}")
        break

print(f"\nBest Sharpe found: {best_sharpe:.4f} with params {best_params}")
