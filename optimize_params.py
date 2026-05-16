import pandas as pd
import numpy as np
import itertools
from trading_agent_class import Trading_Agent

# Load data
vix_options = pd.read_csv("data/vix_options.csv", parse_dates=["as_of_date", "expiration_date"])
vix_data = pd.read_csv("data/vix_data.csv", parse_dates=["Date"])
vvix_data = pd.read_csv("data/vvix_data.csv", parse_dates=["Date"])
spx_data = pd.read_csv("data/spx_data.csv", parse_dates=["Date"])

date_list = vix_data["Date"].unique()

# Parameter grid
all_possible_thresholds = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
rv22_thresholds = all_possible_thresholds
lt_thresholds = all_possible_thresholds
garch_thresholds = all_possible_thresholds
vvix_vix_thresholds = all_possible_thresholds

# Random search or grid search
# Let's do a random sample of 20 combinations to find one with Sharpe > 1
import random
# np.random.seed(42)

param_combinations = list(itertools.product(rv22_thresholds, lt_thresholds, garch_thresholds, vvix_vix_thresholds))
random.shuffle(param_combinations)

print(f"Total combinations: {len(param_combinations)}. Testing up to 30...")

best_sharpe = -999
best_params = None

for i, params in enumerate(param_combinations[:30]):
    rv_thresh, lt_thresh, garch_thresh, vvix_thresh = params
    
    agent = Trading_Agent(
        name="test_agent", 
        test_length=30, 
        longterm_period=60, 
        garch_look_back=20, 
        VRP_rv22_threshold=rv_thresh, 
        VRP_lt_threshold=lt_thresh, 
        VRP_garch_threshold=garch_thresh, 
        vvix_vix_threshold=vvix_thresh
    )
    
    for today in date_list:
        vix = vix_data[vix_data["Date"] == today]["Close"].item()
        vvix = vvix_data[vvix_data["Date"] == today]["Close"].item()
        spx = spx_data[spx_data["Date"] == today]["Close"].item()
        rv22 = spx_data[spx_data["Date"] == today]["RV22"].item()
        option_chain = vix_options[vix_options["as_of_date"] == today]

        agent.feed_data(vix, vvix, spx, rv22)
        if not option_chain.empty:
            trading_signal = agent.signal()
            agent.trade(option_chain, trading_signal)
            agent.calculate_pnl(option_chain, today, vix)
            
    metrics = agent.get_performance_metrics(initial_capital=10000.0)
    sharpe = metrics.get("Sharpe Ratio", 0.0)
    
    print(f"[{i+1}/30] Params: rv22={rv_thresh}, lt={lt_thresh}, garch={garch_thresh}, vvix={vvix_thresh} -> Sharpe: {sharpe:.4f}")
    
    if sharpe > best_sharpe:
        best_sharpe = sharpe
        best_params = params
        
    if sharpe > 1.0:
        print(f"\n>>> FOUND SHARPE > 1.0! <<<")
        print(f"Params: VRP_rv22_threshold={rv_thresh}, VRP_lt_threshold={lt_thresh}, VRP_garch_threshold={garch_thresh}, vvix_vix_threshold={vvix_thresh}")
        print(f"Metrics: {metrics}")
        break

print(f"\nBest Sharpe found: {best_sharpe:.4f} with params {best_params}")
