from memory_class import memory
import numpy as np
import pandas as pd
import math


class Trading_Agent:
    def __init__(
        self,
        name,
        test_length,
        longterm_period,
        garch_look_back,
        VRP_rv22_threshold,
        VRP_lt_threshold,
        VRP_garch_threshold,
        vvix_vix_threshold,
        initial_balance=4000.0,
        k=4.0,
        contract_multiplier=100,
    ):
        self.name = name
        self.longterm_period = longterm_period
        self.test_length = test_length
        self.memory = memory(longterm_period, garch_look_back)
        self.VRP_rv22_threshold = VRP_rv22_threshold
        self.VRP_lt_threshold = VRP_lt_threshold
        self.VRP_garch_threshold = VRP_garch_threshold
        self.vvix_vix_threshold = vvix_vix_threshold

        # Capital management
        self.initial_balance = float(initial_balance)
        self.balance = float(initial_balance)
        self.k = max(float(k), 1.0)
        self.contract_multiplier = max(int(contract_multiplier), 1)

        # Position buckets
        # long_positions  => buy calls (long vol)
        # short_positions => buy puts  (short vol expression)
        self.long_positions = []
        self.short_positions = []

        # PnL trackers
        self.realized_long_pnl = 0.0
        self.realized_short_pnl = 0.0

        self.cum_pnl_history = []
        self.cum_long_pnl_history = []
        self.cum_short_pnl_history = []
        self.balance_history = []
        self.equity_history = []

        self.force_close_longs = False
        self.force_close_shorts = False

    def feed_data(self, vix, vvix, spx, rv22):
        self.memory.memorize(vix, vvix, spx, rv22)

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
            return float(np.mean(sample <= x0))

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
            return 0.0
        return float(2.0 * p - 1.0)

    def signal(self):
        vvix_test = self.memory.vvix[-self.test_length:]
        VRP_rv22_test = self.memory.VRP_rv22[-self.test_length:]
        VRP_lt_test = self.memory.VRP_lt[-self.test_length:]
        VRP_garch_test = self.memory.VRP_garch[-self.test_length:]
        vvix_vix_test = self.memory.vvix_vix[-self.test_length:]

        def get_discrete_signal(data_series, threshold):
            if float(threshold) >= 100.0:
                return 0.0
            kde_val = self._kde_cdf_signal(data_series)
            if kde_val > float(threshold):
                return -1.0  # High VRP -> Short Vol (buy put)
            if kde_val < -float(threshold):
                return 1.0   # Low VRP -> Long Vol (buy call)
            return 0.0

        rv22_signal = get_discrete_signal(VRP_rv22_test, self.VRP_rv22_threshold)
        lt_signal = get_discrete_signal(VRP_lt_test, self.VRP_lt_threshold)
        garch_signal = get_discrete_signal(VRP_garch_test, self.VRP_garch_threshold)
        vvix_vix_signal = get_discrete_signal(vvix_vix_test, self.vvix_vix_threshold)

        total_signal = rv22_signal + lt_signal + garch_signal + vvix_vix_signal

        if vvix_test:
            current_vvix = vvix_test[-1]
            if current_vvix > 110:
                total_signal -= 1.5

        return float(total_signal)

    def search_option(self, option_chain, delta, ttm, option_type):
        chain_filtered = option_chain[
            (option_chain["contract_type"].str.lower() == option_type.lower())
            & (option_chain["close"].notna())
        ]

        if chain_filtered.empty:
            return None

        days_to_maturity = (chain_filtered["expiration_date"] - chain_filtered["as_of_date"]).dt.days
        closest_ttm = days_to_maturity.iloc[(days_to_maturity - ttm).abs().argmin()]
        closest_ttm_options = chain_filtered[days_to_maturity == closest_ttm]

        closest_idx = (closest_ttm_options["delta"].abs() - abs(delta)).abs().argmin()
        closest_option = closest_ttm_options.iloc[closest_idx]

        entry_price = float(closest_option["close"])
        if (not np.isfinite(entry_price)) or entry_price <= 0.0:
            return None

        return {
            "ticker": closest_option["ticker"],
            "entry_price": entry_price,
            "last_price": entry_price,
            "expiration_date": pd.Timestamp(closest_option["expiration_date"]),
            "strike_price": float(closest_option["strike_price"]),
            "contract_type": closest_option["contract_type"].lower(),
        }

    def _open_position(self, bucket, option):
        if option is None:
            return

        contract_cost = option["entry_price"] * self.contract_multiplier
        if contract_cost <= 0.0:
            return

        allocation = self.balance / self.k
        qty = int(allocation // contract_cost)
        if qty <= 0:
            return

        invested = qty * contract_cost
        self.balance -= invested

        pos = dict(option)
        pos["qty"] = qty
        pos["invested"] = invested
        bucket.append(pos)

    def trade(self, option_chain, trading_signal):
        d50_t22_c = self.search_option(option_chain, delta=0.5, ttm=22, option_type="call")
        d50_t22_p = self.search_option(option_chain, delta=-0.5, ttm=22, option_type="put")

        self.force_close_longs = False
        self.force_close_shorts = False

        if -1 < trading_signal < 1:
            self.force_close_longs = True
            self.force_close_shorts = True
            return

        if trading_signal >= 1:
            self.force_close_shorts = True
            self._open_position(self.long_positions, d50_t22_c)
        elif trading_signal <= -1:
            self.force_close_longs = True
            self._open_position(self.short_positions, d50_t22_p)

    def _update_position_marks(self, positions, option_chain):
        for pos in positions:
            target_opt = option_chain[option_chain["ticker"] == pos["ticker"]]
            if not target_opt.empty:
                current_price = target_opt["close"].iloc[-1]
                if pd.notna(current_price) and np.isfinite(float(current_price)) and float(current_price) >= 0.0:
                    pos["last_price"] = float(current_price)

    def _settle_expired_positions(self, positions, today_ts, vix_close, is_long_bucket):
        realized_delta = 0.0
        for i in range(len(positions) - 1, -1, -1):
            pos = positions[i]
            if today_ts >= pos["expiration_date"]:
                if pos["contract_type"] == "call":
                    payoff = max(0.0, float(vix_close) - pos["strike_price"])
                else:
                    payoff = max(0.0, pos["strike_price"] - float(vix_close))

                proceeds = payoff * pos["qty"] * self.contract_multiplier
                pnl = (payoff - pos["entry_price"]) * pos["qty"] * self.contract_multiplier

                self.balance += proceeds
                realized_delta += pnl
                positions.pop(i)

        if is_long_bucket:
            self.realized_long_pnl += realized_delta
        else:
            self.realized_short_pnl += realized_delta

    def _liquidate_positions(self, positions, is_long_bucket):
        realized_delta = 0.0
        for pos in positions:
            last_price = float(pos["last_price"])
            proceeds = last_price * pos["qty"] * self.contract_multiplier
            pnl = (last_price - pos["entry_price"]) * pos["qty"] * self.contract_multiplier

            self.balance += proceeds
            realized_delta += pnl

        positions.clear()

        if is_long_bucket:
            self.realized_long_pnl += realized_delta
        else:
            self.realized_short_pnl += realized_delta

    def _compute_unrealized(self, positions):
        unrealized = 0.0
        for pos in positions:
            unrealized += (pos["last_price"] - pos["entry_price"]) * pos["qty"] * self.contract_multiplier
        return float(unrealized)

    def _positions_market_value(self):
        total = 0.0
        for pos in self.long_positions:
            total += pos["last_price"] * pos["qty"] * self.contract_multiplier
        for pos in self.short_positions:
            total += pos["last_price"] * pos["qty"] * self.contract_multiplier
        return float(total)

    def calculate_pnl(self, option_chain, today, vix_close):
        today_ts = pd.Timestamp(today)

        self._update_position_marks(self.long_positions, option_chain)
        self._update_position_marks(self.short_positions, option_chain)

        self._settle_expired_positions(self.long_positions, today_ts, vix_close, is_long_bucket=True)
        self._settle_expired_positions(self.short_positions, today_ts, vix_close, is_long_bucket=False)

        if self.force_close_longs:
            self._liquidate_positions(self.long_positions, is_long_bucket=True)
            self.force_close_longs = False

        if self.force_close_shorts:
            self._liquidate_positions(self.short_positions, is_long_bucket=False)
            self.force_close_shorts = False

        unrealized_long = self._compute_unrealized(self.long_positions)
        unrealized_short = self._compute_unrealized(self.short_positions)

        total_long_pnl = self.realized_long_pnl + unrealized_long
        total_short_pnl = self.realized_short_pnl + unrealized_short

        equity = self.balance + self._positions_market_value()
        current_cum_pnl = equity - self.initial_balance

        self.balance_history.append(float(self.balance))
        self.equity_history.append(float(equity))
        self.cum_pnl_history.append(float(current_cum_pnl))
        self.cum_long_pnl_history.append(float(total_long_pnl))
        self.cum_short_pnl_history.append(float(total_short_pnl))

    def get_balance_history(self):
        return self.balance_history

    def get_equity_history(self):
        return self.equity_history

    def get_cum_pnl(self):
        return self.cum_pnl_history

    def get_cum_long_pnl(self):
        return self.cum_long_pnl_history

    def get_cum_short_pnl(self):
        return self.cum_short_pnl_history

    def get_long_short_pnl(self):
        return {
            "long_vol_buy_call_pnl": float(self.realized_long_pnl),
            "short_vol_buy_put_pnl": float(self.realized_short_pnl),
        }

    def get_log_returns(self, initial_capital=None):
        if not self.cum_pnl_history:
            return [], [], []

        base = float(self.initial_balance if initial_capital is None else initial_capital)
        cum_pnl = np.array(self.cum_pnl_history, dtype=float)
        cum_long = np.array(self.cum_long_pnl_history, dtype=float)
        cum_short = np.array(self.cum_short_pnl_history, dtype=float)

        val_total = np.maximum(1e-6, base + cum_pnl)
        val_long = np.maximum(1e-6, base + cum_long)
        val_short = np.maximum(1e-6, base + cum_short)

        log_ret_total = np.log(val_total / base)
        log_ret_long = np.log(val_long / base)
        log_ret_short = np.log(val_short / base)

        return log_ret_total, log_ret_long, log_ret_short

    def get_performance_metrics(self, initial_capital=None):
        if not self.equity_history:
            return {}

        base = float(self.initial_balance if initial_capital is None else initial_capital)
        equity = np.array(self.equity_history, dtype=float)

        if equity.size < 2:
            return {
                "Sharpe Ratio": 0.0,
                "Sortino Ratio": 0.0,
                "Annual Return": 0.0,
                "Annual Volatility": 0.0,
                "Max Drawdown": 0.0,
            }

        prev = np.maximum(1e-6, equity[:-1])
        daily_returns = np.diff(equity) / prev

        mean_daily = float(np.mean(daily_returns))
        annual_volatility = float(np.std(daily_returns) * np.sqrt(252))

        if annual_volatility > 0:
            sharpe_ratio = (mean_daily * 252) / annual_volatility
        else:
            sharpe_ratio = 0.0

        negative_returns = daily_returns[daily_returns < 0]
        downside_std = float(np.std(negative_returns) * np.sqrt(252)) if negative_returns.size > 0 else 0.0
        if downside_std > 0:
            sortino_ratio = (mean_daily * 252) / downside_std
        else:
            sortino_ratio = 0.0

        years = max(equity.size / 252.0, 1e-6)
        final_value = max(1e-6, float(equity[-1]))
        annual_return = (final_value / base) ** (1.0 / years) - 1.0

        running_max = np.maximum.accumulate(equity)
        drawdowns = (equity - running_max) / np.maximum(1e-6, running_max)
        max_drawdown = float(np.min(drawdowns))

        return {
            "Sharpe Ratio": float(sharpe_ratio),
            "Sortino Ratio": float(sortino_ratio),
            "Annual Return": float(annual_return),
            "Annual Volatility": float(annual_volatility),
            "Max Drawdown": float(max_drawdown),
        }
