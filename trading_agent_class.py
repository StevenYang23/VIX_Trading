from memory_class import memory
import numpy as np
import pandas as pd
import math

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
        self.force_close = False
    
    def feed_data(self, vix,vvix,spx,rv22):
        self.memory.memorize(vix,vvix,spx,rv22)

    @staticmethod
    def _norm_cdf(x):
        return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

    @classmethod
    def _kde_percentile(cls, x):
        arr = np.asarray(x, dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size < 2:
            return np.nan

        x0 = float(arr[-1])
        sample = arr[:-1] if arr.size > 2 else arr
        n = sample.size
        if n < 2:
            return np.nan

        std = float(np.std(sample, ddof=1))
        if (not np.isfinite(std)) or std < 1e-12:
            # Degenerate case: fallback to empirical percentile
            return float(np.mean(sample <= x0))

        # Silverman's rule-of-thumb bandwidth
        h = 1.06 * std * (n ** (-1.0 / 5.0))
        h = max(float(h), 1e-6)

        z = (x0 - sample) / h
        cdf_vals = np.array([cls._norm_cdf(float(v)) for v in z], dtype=float)
        p = float(np.mean(cdf_vals))
        return min(max(p, 0.0), 1.0)

    @classmethod
    def _kde_cdf_signal(cls, x):
        p = cls._kde_percentile(x)
        if not np.isfinite(p):
            return 0

        # map percentile p in [0, 1] to continuous signal in [-1, 1]
        return float(2.0 * p - 1.0)

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

        rv22_signal = self._kde_cdf_signal(VRP_rv22_test)
        lt_signal = self._kde_cdf_signal(VRP_lt_test)
        garch_signal = self._kde_cdf_signal(VRP_garch_test)

        # Keep compatibility with existing notebook setup:
        # threshold >= 100 means "disable this factor".
        active_signals = []
        if float(self.VRP_rv22_threshold) < 100.0:
            active_signals.append(rv22_signal)
        if float(self.VRP_lt_threshold) < 100.0:
            active_signals.append(lt_signal)
        if float(self.VRP_garch_threshold) < 100.0:
            active_signals.append(garch_signal)

        if len(active_signals) == 0:
            return 0.0

        return float(np.mean(active_signals))
    
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
        # Core regime rule:
        # - short vol when signal > 0.8
        # - long vol when signal < -0.8
        # - close when signal in [-0.1, 0.1]
        if -0.1 <= trading_signal <= 0.1:
            self.force_close = True
            return

        self.force_close = False
        if trading_signal <= -0.8 and d50_t22_c is not None:
            self.long_positions.append(d50_t22_c)
        elif trading_signal >= 0.8 and d50_t22_c is not None:
            self.short_positions.append(d50_t22_c)

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

        # Force-close in neutral regime using marked price of the day.
        if self.force_close:
            for pos in self.long_positions:
                self.realized_long_pnl += (pos["last_price"] - pos["entry_price"])
            for pos in self.short_positions:
                self.realized_short_pnl += (pos["entry_price"] - pos["last_price"])
            self.long_positions.clear()
            self.short_positions.clear()
            total_long_pnl = self.realized_long_pnl
            total_short_pnl = self.realized_short_pnl
            self.force_close = False

        self.cum_pnl_history.append(total_long_pnl + total_short_pnl)

    def get_cum_pnl(self):
        return self.cum_pnl_history

    def get_cum_long_pnl(self):
        return self.cum_long_pnl_history

    def get_cum_short_pnl(self):
        return self.cum_short_pnl_history