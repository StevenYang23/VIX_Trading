from memory_class import memory
import numpy as np
import pandas as pd
import math

class Trading_Agent:
    def __init__(self, name, test_length, longterm_period, garch_look_back, VRP_rv22_threshold, VRP_lt_threshold, VRP_garch_threshold, vvix_vix_threshold):
        self.name = name
        self.longterm_period = longterm_period
        self.test_length = test_length
        self.memory = memory(longterm_period, garch_look_back)
        self.VRP_rv22_threshold = VRP_rv22_threshold
        self.VRP_lt_threshold = VRP_lt_threshold
        self.VRP_garch_threshold = VRP_garch_threshold
        self.vvix_vix_threshold = vvix_vix_threshold

        self.long_positions = []
        self.short_positions = []

        self.realized_long_pnl = 0.0
        self.realized_short_pnl = 0.0

        self.cum_pnl_history = []
        self.cum_long_pnl_history = []
        self.cum_short_pnl_history = []
        self.daily_log_returns = []
        self.cum_log_returns = []
        self.current_exposure = 0.0
        self.previous_cum_pnl = 0.0
        
        self.force_close_longs = False
        self.force_close_shorts = False
    
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
        vix_test = self.memory.vix[-self.test_length:]
        vvix_test = self.memory.vvix[-self.test_length:]
        spx_test = self.memory.spx[-self.test_length:]
        rv22_test = self.memory.rv22[-self.test_length:]
        rv_long_term_test = self.memory.rv_long_term[-self.test_length:]
        garch_test = self.memory.garch_forecast[-self.test_length:]
        VRP_rv22_test = self.memory.VRP_rv22[-self.test_length:]
        VRP_lt_test = self.memory.VRP_lt[-self.test_length:]
        VRP_garch_test = self.memory.VRP_garch[-self.test_length:]
        vvix_vix_test = self.memory.vvix_vix[-self.test_length:]

        def get_discrete_signal(data_series, threshold):
            if float(threshold) >= 100.0:
                return 0.0
            kde_val = self._kde_cdf_signal(data_series)
            if kde_val > float(threshold):
                return -1.0  # High VRP -> Short Vol (VIX is overvalued)
            elif kde_val < -float(threshold):
                return 1.0   # Low VRP -> Long Vol (VIX is undervalued)
            return 0.0

        rv22_signal = get_discrete_signal(VRP_rv22_test, self.VRP_rv22_threshold)
        lt_signal = get_discrete_signal(VRP_lt_test, self.VRP_lt_threshold)
        garch_signal = get_discrete_signal(VRP_garch_test, self.VRP_garch_threshold)
        vvix_vix_signal = get_discrete_signal(vvix_vix_test, self.vvix_vix_threshold)

        total_signal = rv22_signal + lt_signal + garch_signal + vvix_vix_signal

        # Extreme VVIX override
        current_vvix = vvix_test[-1]
        if current_vvix > 110:
            # High VVIX -> panic -> VIX likely to drop -> favor SHORT (-1.5)
            total_signal -= 1.5

        return total_signal
    
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
        d50_t22_p = self.search_option(option_chain, delta=-0.5, ttm=22, option_type='put')
        
        self.force_close_longs = False
        self.force_close_shorts = False

        # - close when signal is strictly between -1 and 1 (e.g., 0, 0.5, -0.5)
        if -1 < trading_signal < 1:
            self.force_close_longs = True
            self.force_close_shorts = True
            return

        # signal >= 1 means LONG vol: Buy Call
        if trading_signal >= 1:
            self.force_close_shorts = True  # Reversal: close short positions if any
            if d50_t22_c is not None:
                self.long_positions.append(d50_t22_c)
                
        # signal <= -1 means SHORT vol: Buy Put
        elif trading_signal <= -1:
            self.force_close_longs = True   # Reversal: close long positions if any
            if d50_t22_p is not None:
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
        
        # Process short positions (which are actually long Puts)
        for i in range(len(self.short_positions) - 1, -1, -1):
            pos = self.short_positions[i]
            
            if today_ts >= pos["expiration_date"]:
                if pos["contract_type"] == "call":
                    payoff = max(0.0, vix_close - pos["strike_price"])
                else:
                    payoff = max(0.0, pos["strike_price"] - vix_close)
                
                # Since we BOUGHT the put, PnL is Payoff - Entry Price
                self.realized_short_pnl += (payoff - pos["entry_price"])
                self.short_positions.pop(i)
                continue
                
            target_opt = option_chain[option_chain["ticker"] == pos["ticker"]]
            if not target_opt.empty:
                current_price = target_opt["close"].iloc[-1]
                if pd.notna(current_price):
                    pos["last_price"] = current_price
            
            # Since we BOUGHT the put, unrealized PnL is Current Price - Entry Price
            unrealized_short += (pos["last_price"] - pos["entry_price"])
                
        total_short_pnl = self.realized_short_pnl + unrealized_short

        # Force-close in neutral regime or reversals
        if self.force_close_longs:
            for pos in self.long_positions:
                self.realized_long_pnl += (pos["last_price"] - pos["entry_price"])
            self.long_positions.clear()
            total_long_pnl = self.realized_long_pnl
            self.force_close_longs = False

        if self.force_close_shorts:
            for pos in self.short_positions:
                self.realized_short_pnl += (pos["last_price"] - pos["entry_price"])
            self.short_positions.clear()
            total_short_pnl = self.realized_short_pnl
            self.force_close_shorts = False

        current_cum_pnl = total_long_pnl + total_short_pnl
        self.cum_pnl_history.append(current_cum_pnl)
        self.cum_long_pnl_history.append(total_long_pnl)
        self.cum_short_pnl_history.append(total_short_pnl)
        
        daily_pnl = current_cum_pnl - self.previous_cum_pnl
        self.previous_cum_pnl = current_cum_pnl

        cost = self.current_exposure

        if cost > 0:
            # Safeguard against log(0)
            value_after_pnl = max(1e-6, cost + daily_pnl)
            daily_log_ret = np.log(value_after_pnl / cost)
        else:
            daily_log_ret = 0.0

        self.daily_log_returns.append(daily_log_ret)
        
        current_cum_ret = sum(self.daily_log_returns)
        self.cum_log_returns.append(current_cum_ret)

        # Update exposure for tomorrow (cost for next day)
        new_exposure = 0.0
        for pos in self.long_positions:
            new_exposure += pos["last_price"]
        for pos in self.short_positions:
            new_exposure += pos["last_price"]
        self.current_exposure = new_exposure

    def get_cum_pnl(self):
        return self.cum_pnl_history

    def get_cum_long_pnl(self):
        return self.cum_long_pnl_history

    def get_cum_short_pnl(self):
        return self.cum_short_pnl_history

    def get_log_returns(self, initial_capital=10000.0):
        if not self.cum_pnl_history:
            return [], [], []
            
        cum_pnl = np.array(self.cum_pnl_history)
        cum_long = np.array(self.cum_long_pnl_history)
        cum_short = np.array(self.cum_short_pnl_history)
        
        # Cumulative Log Return = ln(Current Value / Initial Capital)
        # Using max(1e-6) to prevent log(0) or log(negative)
        val_total = np.maximum(1e-6, initial_capital + cum_pnl)
        val_long = np.maximum(1e-6, initial_capital + cum_long)
        val_short = np.maximum(1e-6, initial_capital + cum_short)
        
        log_ret_total = np.log(val_total / initial_capital)
        log_ret_long = np.log(val_long / initial_capital)
        log_ret_short = np.log(val_short / initial_capital)
        
        return log_ret_total, log_ret_long, log_ret_short

    def get_performance_metrics(self, initial_capital=10000.0):
        if not self.cum_pnl_history:
            return {}
            
        cum_pnl = np.array(self.cum_pnl_history)
        net_value = initial_capital + cum_pnl
        
        # Calculate daily simple returns
        daily_returns = np.zeros_like(net_value)
        daily_returns[1:] = np.diff(net_value) / net_value[:-1]
        
        # Annual Return (CAGR)
        days = len(daily_returns)
        if days > 0:
            # Using max(0, ...) to prevent complex numbers if net_value goes negative
            final_value = max(1e-6, net_value[-1])
            annual_return = (final_value / initial_capital) ** (252 / days) - 1
        else:
            annual_return = 0.0
            
        # Annual Volatility
        annual_volatility = np.std(daily_returns) * np.sqrt(252)
        
        # Sharpe Ratio (assuming risk-free rate = 0)
        if annual_volatility > 0:
            sharpe_ratio = (np.mean(daily_returns) * 252) / annual_volatility
        else:
            sharpe_ratio = 0.0
            
        # Sortino Ratio
        negative_returns = daily_returns[daily_returns < 0]
        downside_std = np.std(negative_returns) * np.sqrt(252) if len(negative_returns) > 0 else 0.0
        if downside_std > 0:
            sortino_ratio = (np.mean(daily_returns) * 252) / downside_std
        else:
            sortino_ratio = 0.0
            
        # Max Drawdown
        running_max = np.maximum.accumulate(net_value)
        drawdowns = (net_value - running_max) / running_max
        max_drawdown = np.min(drawdowns)
        
        return {
            "Sharpe Ratio": sharpe_ratio,
            "Sortino Ratio": sortino_ratio,
            "Annual Return": annual_return,
            "Annual Volatility": annual_volatility,
            "Max Drawdown": max_drawdown
        }

    def get_daily_log_returns(self):
        return self.daily_log_returns

    def get_cum_log_returns(self):
        return self.cum_log_returns