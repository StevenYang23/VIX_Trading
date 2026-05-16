from memory_class import memory
import numpy as np
import pandas as pd

class Trading_Agent:
    def __init__(self, name, test_length, longterm_period, garch_look_back, VRP_rv22_threshold, VRP_lt_threshold, VRP_garch_threshold):
        self.name = name
        self.longterm_period = longterm_period
        self.test_length = test_length
        self.memory = memory(longterm_period, garch_look_back)
        self.VRP_rv22_threshold = VRP_rv22_threshold
        self.VRP_lt_threshold = VRP_lt_threshold
        self.VRP_garch_threshold = VRP_garch_threshold

        self.long_positions = []
        self.short_positions = []

        self.realized_long_pnl = 0.0
        self.realized_short_pnl = 0.0

        self.cum_pnl_history = []
        self.cum_long_pnl_history = []
        self.cum_short_pnl_history = []
    
    def feed_data(self, vix,vvix,spx,rv22):
        self.memory.memorize(vix,vvix,spx,rv22)

    def signal(self):
        vix_test     = self.memory.vix[-self.test_length:]
        vvix_test = self.memory.vvix[-self.test_length:]
        spx_test = self.memory.spx[-self.test_length:]
        rv22_test = self.memory.rv22[-self.test_length:]
        rv_long_term_test = self.memory.rv_long_term[-self.test_length:]
        garch_test = self.memory.garch_forecast[-self.test_length:]
        VRP_rv22_test = self.memory.VRP_rv22[-self.test_length:]
        VRP_lt_test = self.memory.VRP_lt[-self.test_length:]
        VRP_garch_test = self.memory.VRP_garch[-self.test_length:]

        z_score_rv22 = (VRP_rv22_test - np.mean(VRP_rv22_test)) / np.std(VRP_rv22_test)
        latest_z_score_rv22 = z_score_rv22[-1]
        if latest_z_score_rv22 >= self.VRP_rv22_threshold:
            rv22_signal = 1
        elif latest_z_score_rv22 <= -self.VRP_rv22_threshold:
            rv22_signal = -1
        else:
            rv22_signal = 0

        z_score_lt = (VRP_lt_test - np.mean(VRP_lt_test)) / np.std(VRP_lt_test)
        latest_z_score_lt = z_score_lt[-1]
        if latest_z_score_lt >= self.VRP_lt_threshold:
            lt_signal = 1
        elif latest_z_score_lt <= -self.VRP_lt_threshold:
            lt_signal = -1
        else:
            lt_signal = 0
            
        z_score_garch = (VRP_garch_test - np.mean(VRP_garch_test)) / np.std(VRP_garch_test)
        latest_z_score_garch = z_score_garch[-1]
        if latest_z_score_garch >= self.VRP_garch_threshold:
            garch_signal = 1
        elif latest_z_score_garch <= -self.VRP_garch_threshold:
            garch_signal = -1
        else:
            garch_signal = 0
            
        return rv22_signal + lt_signal + garch_signal
    
    def search_option(self, option_chain, delta, ttm, option_type):
        # Filter by option type and ensure it has a valid close price
        chain_filtered = option_chain[
            (option_chain["contract_type"].str.lower() == option_type.lower()) & 
            (option_chain["close"].notna())
        ]
        
        if chain_filtered.empty:
            return None
            
        days_to_maturity = (chain_filtered["expiration_date"] - chain_filtered["as_of_date"]).dt.days
        closest_ttm = days_to_maturity.iloc[(days_to_maturity - ttm).abs().argmin()]
        closest_ttm_options = chain_filtered[days_to_maturity == closest_ttm]
        
        # Use absolute delta to handle puts properly (puts have negative delta)
        closest_idx = (closest_ttm_options["delta"].abs() - abs(delta)).abs().argmin()
        closest_option = closest_ttm_options.iloc[closest_idx]
        
        return {
            "ticker": closest_option["ticker"],
            "entry_price": closest_option["close"],
            "last_price": closest_option["close"],
            "expiration_date": closest_option["expiration_date"],
            "strike_price": closest_option["strike_price"],
            "contract_type": closest_option["contract_type"].lower()
        }

    def trade(self, option_chain, trading_signal):
        d50_t22_c = self.search_option(option_chain, delta=0.5, ttm=22, option_type='call')
        d50_t22_p = self.search_option(option_chain, delta=0.5, ttm=22, option_type='put')
        
        if trading_signal >= 1 and d50_t22_c is not None:
            self.long_positions.append(d50_t22_c)
        elif trading_signal <= -1 and d50_t22_p is not None:
            self.short_positions.append(d50_t22_p)

    def calculate_pnl(self, option_chain, today, vix_close):
        unrealized_long = 0.0
        unrealized_short = 0.0
        
        today_ts = pd.Timestamp(today)
        
        # Process long positions
        for i in range(len(self.long_positions) - 1, -1, -1):
            pos = self.long_positions[i]
            
            # Check if option has expired
            if today_ts >= pos["expiration_date"]:
                # Calculate expiration payoff
                if pos["contract_type"] == "call":
                    payoff = max(0.0, vix_close - pos["strike_price"])
                else:
                    payoff = max(0.0, pos["strike_price"] - vix_close)
                
                # Realize the PnL and remove position
                self.realized_long_pnl += (payoff - pos["entry_price"])
                self.long_positions.pop(i)
                continue
                
            # If not expired, find it in today's chain
            target_opt = option_chain[option_chain["ticker"] == pos["ticker"]]
            if not target_opt.empty:
                current_price = target_opt["close"].iloc[-1]
                if pd.notna(current_price):
                    pos["last_price"] = current_price
            
            unrealized_long += (pos["last_price"] - pos["entry_price"])

        total_long_pnl = self.realized_long_pnl + unrealized_long
        self.cum_long_pnl_history.append(total_long_pnl)
        
        # Process short positions
        for i in range(len(self.short_positions) - 1, -1, -1):
            pos = self.short_positions[i]
            
            if today_ts >= pos["expiration_date"]:
                if pos["contract_type"] == "call":
                    payoff = max(0.0, vix_close - pos["strike_price"])
                else:
                    payoff = max(0.0, pos["strike_price"] - vix_close)
                
                # For short positions, PnL is Entry Price - Payoff
                self.realized_short_pnl += (pos["entry_price"] - payoff)
                self.short_positions.pop(i)
                continue
                
            target_opt = option_chain[option_chain["ticker"] == pos["ticker"]]
            if not target_opt.empty:
                current_price = target_opt["close"].iloc[-1]
                if pd.notna(current_price):
                    pos["last_price"] = current_price
            
            # For short positions, unrealized PnL is Entry Price - Current Price
            unrealized_short += (pos["entry_price"] - pos["last_price"])
                
        total_short_pnl = self.realized_short_pnl + unrealized_short
        self.cum_short_pnl_history.append(total_short_pnl)
        
        self.cum_pnl_history.append(total_long_pnl + total_short_pnl)

    def get_cum_pnl(self):
        return self.cum_pnl_history

    def get_cum_long_pnl(self):
        return self.cum_long_pnl_history

    def get_cum_short_pnl(self):
        return self.cum_short_pnl_history