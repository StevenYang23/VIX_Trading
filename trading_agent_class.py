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
        Max_holding=2,
        w_rv22=0.35,
        w_lt=0.15,
        w_garch=0.35,
        w_vvix_vix=0.15,
        entry_threshold=2.0,
        exit_threshold=1.0,
        vvix_short_gate=110.0,
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
        self.max_holding = max(int(Max_holding), 1)
        self.signal_weights = {
            "rv22": float(w_rv22),
            "lt": float(w_lt),
            "garch": float(w_garch),
            "vvix_vix": float(w_vvix_vix),
        }
        self.entry_threshold = float(entry_threshold)
        self.exit_threshold = float(exit_threshold)
        self.vvix_short_gate = float(vvix_short_gate)
        # Hysteresis state: -1 short-vol regime, 0 neutral, +1 long-vol regime
        self._signal_state = 0

        # New spread-level position engine
        # Each element is a spread dict, each spread has:
        # - purpose: long_vol / short_vol
        # - strategy_name: bucket name
        # - qty: number of spreads
        # - legs: [{ticker, side, entry_price, last_price, strike_price, contract_type, expiration_date}, ...]
        self.strategy_positions = []
        # Explicit list requested: tracks purpose of each opened spread
        self.position_purposes = []

        # PnL trackers
        self.realized_long_pnl = 0.0
        self.realized_short_pnl = 0.0

        self.cum_pnl_history = []
        self.cum_long_pnl_history = []
        self.cum_short_pnl_history = []
        self.balance_history = []
        self.equity_history = []

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
                return -1.0  # High VRP -> Short Vol
            if kde_val < -float(threshold):
                return 1.0   # Low VRP -> Long Vol
            return 0.0

        rv22_signal = get_discrete_signal(VRP_rv22_test, self.VRP_rv22_threshold)
        lt_signal = get_discrete_signal(VRP_lt_test, self.VRP_lt_threshold)
        garch_signal = get_discrete_signal(VRP_garch_test, self.VRP_garch_threshold)
        vvix_vix_signal = get_discrete_signal(vvix_vix_test, self.vvix_vix_threshold)

        # Weighted composite signal (replaces equal-vote sum)
        weighted_signal = (
            self.signal_weights["rv22"] * rv22_signal
            + self.signal_weights["lt"] * lt_signal
            + self.signal_weights["garch"] * garch_signal
            + self.signal_weights["vvix_vix"] * vvix_vix_signal
        )
        total_signal = 4.0 * weighted_signal

        # Extra VVIX override signal
        if vvix_test:
            current_vvix = vvix_test[-1]
            if current_vvix > self.vvix_short_gate:
                total_signal -= 1.5
                # Regime gate: under stress VVIX, block NEW short-vol impulse.
                if total_signal < 0:
                    total_signal = 0.0

        # Hysteresis layer (entry/exit separated to reduce churn)
        if self._signal_state == 0:
            if total_signal >= self.entry_threshold:
                self._signal_state = 1
            elif total_signal <= -self.entry_threshold:
                self._signal_state = -1
        elif self._signal_state == 1:
            if total_signal <= self.exit_threshold:
                self._signal_state = 0
        else:  # self._signal_state == -1
            if total_signal >= -self.exit_threshold:
                self._signal_state = 0

        if self._signal_state == 0:
            return 0.0
        if self._signal_state > 0:
            return float(max(total_signal, self.entry_threshold))
        return float(min(total_signal, -self.entry_threshold))

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

    def _strategy_from_signal(self, trading_signal):
        if trading_signal >= 3:
            return {
                "purpose": "long_vol",
                "strategy_name": "bull_call_spread_20_10",
                "legs": [
                    {"delta": 0.2, "option_type": "call", "side": +1},
                    {"delta": 0.1, "option_type": "call", "side": -1},
                ],
            }
        if trading_signal >= 2:
            return {
                "purpose": "long_vol",
                "strategy_name": "bull_call_spread_50_30",
                "legs": [
                    {"delta": 0.5, "option_type": "call", "side": +1},
                    {"delta": 0.3, "option_type": "call", "side": -1},
                ],
            }
        if trading_signal >= 1:
            return {
                "purpose": "long_vol",
                "strategy_name": "digital_bull_call_spread_atm",
                "legs": [
                    {"delta": 0.5, "option_type": "call", "side": +1},
                    {"delta": 0.4, "option_type": "call", "side": -1},
                ],
            }
        if trading_signal <= -3:
            return {
                "purpose": "short_vol",
                "strategy_name": "bear_call_spread_20_10",
                "legs": [
                    {"delta": 0.2, "option_type": "call", "side": -1},
                    {"delta": 0.1, "option_type": "call", "side": +1},
                ],
            }
        if trading_signal <= -2:
            return {
                "purpose": "short_vol",
                "strategy_name": "bear_call_spread_50_30",
                "legs": [
                    {"delta": 0.5, "option_type": "call", "side": -1},
                    {"delta": 0.3, "option_type": "call", "side": +1},
                ],
            }
        if trading_signal <= -1:
            return {
                "purpose": "short_vol",
                "strategy_name": "digital_bear_call_spread_atm",
                "legs": [
                    {"delta": 0.5, "option_type": "call", "side": -1},
                    {"delta": 0.4, "option_type": "call", "side": +1},
                ],
            }
        return None
        
    def _build_spread(self, option_chain, strategy_def, ttm=45):
        legs = []
        for leg_def in strategy_def["legs"]:
            opt = self.search_option(
                option_chain,
                delta=leg_def["delta"],
                ttm=ttm,
                option_type=leg_def["option_type"],
            )
            if opt is None:
                return None

            leg = dict(opt)
            leg["side"] = int(leg_def["side"])  # +1 long, -1 short
            legs.append(leg)

        expiration_date = min(leg["expiration_date"] for leg in legs)

        return {
            "purpose": strategy_def["purpose"],
            "strategy_name": strategy_def["strategy_name"],
            "qty": 0,
            "legs": legs,
            "expiration_date": expiration_date,
        }

    def _spread_width_per_spread(self, spread):
        call_strikes = [leg["strike_price"] for leg in spread["legs"] if leg["contract_type"] == "call"]
        put_strikes = [leg["strike_price"] for leg in spread["legs"] if leg["contract_type"] == "put"]

        if len(call_strikes) >= 2:
            return (max(call_strikes) - min(call_strikes)) * self.contract_multiplier
        if len(put_strikes) >= 2:
            return (max(put_strikes) - min(put_strikes)) * self.contract_multiplier
        return 0.0

    def _open_spread(self, spread):
        if len(self.strategy_positions) >= self.max_holding:
            return False

        # Per-spread opening cashflow:
        # side +1 long -> pay premium (negative cashflow)
        # side -1 short -> receive premium (positive cashflow)
        open_cash_flow_per_spread = 0.0
        for leg in spread["legs"]:
            open_cash_flow_per_spread += -leg["side"] * leg["entry_price"] * self.contract_multiplier

        net_debit_per_spread = max(-open_cash_flow_per_spread, 0.0)
        net_credit_per_spread = max(open_cash_flow_per_spread, 0.0)
        spread_width_per_spread = self._spread_width_per_spread(spread)

        # Requested risk sizing model:
        # - long spread: use net debit
        # - short spread: use max loss = spread width - credit
        if spread["purpose"] == "long_vol":
            risk_per_spread = net_debit_per_spread
        else:
            risk_per_spread = max(spread_width_per_spread - net_credit_per_spread, 0.0)

        if risk_per_spread <= 0.0:
            return False

        allocation = self.balance / self.k
        qty = int(allocation // risk_per_spread)
        if qty <= 0:
            return False

        self.balance += open_cash_flow_per_spread * qty
        spread["qty"] = qty
        spread["risk_per_spread"] = float(risk_per_spread)
        spread["open_cash_flow_per_spread"] = float(open_cash_flow_per_spread)
        self.strategy_positions.append(spread)
        self.position_purposes.append(spread["purpose"])
        return True

    def _update_leg_mark(self, leg, option_chain):
        target_opt = option_chain[option_chain["ticker"] == leg["ticker"]]
        if not target_opt.empty:
            current_price = target_opt["close"].iloc[-1]
            if pd.notna(current_price) and np.isfinite(float(current_price)) and float(current_price) >= 0.0:
                leg["last_price"] = float(current_price)

    def _refresh_spread_marks(self, spread, option_chain):
        for leg in spread["legs"]:
            self._update_leg_mark(leg, option_chain)

    def _intrinsic_value(self, leg, vix_close):
        strike = leg["strike_price"]
        if leg["contract_type"] == "call":
            return max(0.0, float(vix_close) - strike)
        return max(0.0, strike - float(vix_close))

    def _close_spread(self, spread, option_chain=None, use_expiry=False, vix_close=None):
        qty = spread["qty"]
        close_cash_flow = 0.0
        spread_pnl = 0.0

        for leg in spread["legs"]:
            if use_expiry:
                val = self._intrinsic_value(leg, vix_close)
            else:
                if option_chain is not None:
                    self._update_leg_mark(leg, option_chain)
                val = float(leg["last_price"])

            # Close cash flow: side * value * qty * multiplier
            # long(+1): receive +value, short(-1): pay -value
            close_cash_flow += leg["side"] * val * qty * self.contract_multiplier

            # PnL: side * (close - entry)
            spread_pnl += leg["side"] * (val - leg["entry_price"]) * qty * self.contract_multiplier

        self.balance += close_cash_flow

        if spread["purpose"] == "long_vol":
            self.realized_long_pnl += spread_pnl
        else:
            self.realized_short_pnl += spread_pnl

    def _close_positions_by_purpose(self, purpose, option_chain):
        kept = []
        for spread in self.strategy_positions:
            if spread["purpose"] == purpose:
                self._close_spread(spread, option_chain=option_chain, use_expiry=False)
            else:
                kept.append(spread)
        self.strategy_positions = kept
        self.position_purposes = [s["purpose"] for s in self.strategy_positions]

    def _close_all_positions(self, option_chain):
        for spread in self.strategy_positions:
            self._close_spread(spread, option_chain=option_chain, use_expiry=False)
        self.strategy_positions = []
        self.position_purposes = []

    def _unrealized_pnl_split(self):
        unreal_long = 0.0
        unreal_short = 0.0

        for spread in self.strategy_positions:
            qty = spread["qty"]
            spread_unreal = 0.0
            for leg in spread["legs"]:
                spread_unreal += leg["side"] * (leg["last_price"] - leg["entry_price"]) * qty * self.contract_multiplier

            if spread["purpose"] == "long_vol":
                unreal_long += spread_unreal
            else:
                unreal_short += spread_unreal

        return float(unreal_long), float(unreal_short)

    def _positions_market_value(self):
        total = 0.0
        for spread in self.strategy_positions:
            qty = spread["qty"]
            for leg in spread["legs"]:
                total += leg["side"] * leg["last_price"] * qty * self.contract_multiplier
        return float(total)

    def trade(self, option_chain, trading_signal):
        """
        Signal regime mapping:
        3 <= s: close short_vol purpose, open long_vol 20/10 call spread
        2 <= s < 3: close short_vol purpose, open long_vol 50/30 call spread
        1 <= s < 2: close short_vol purpose, open long_vol atm digital-like spread
       -1 <= s < 1: close all
       -2 <= s < -1: close long_vol purpose, open short_vol atm digital-like spread
       -3 <= s < -2: close long_vol purpose, open short_vol 50/30 call spread
             s < -3: close long_vol purpose, open short_vol 20/10 call spread
        """
        target = self._strategy_from_signal(trading_signal)

        if target is None:
            self._close_all_positions(option_chain)
            return

        opposite = "short_vol" if target["purpose"] == "long_vol" else "long_vol"
        self._close_positions_by_purpose(opposite, option_chain)

        # If same strategy already exists, keep it (no re-open churn).
        active_same = any(
            s["purpose"] == target["purpose"] and s["strategy_name"] == target["strategy_name"]
            for s in self.strategy_positions
        )
        if active_same:
            return

        # Close same-purpose but different strategy, then open new one.
        self._close_positions_by_purpose(target["purpose"], option_chain)
        spread = self._build_spread(option_chain, target, ttm=45)
        if spread is not None:
            self._open_spread(spread)

    def calculate_pnl(self, option_chain, today, vix_close):
        today_ts = pd.Timestamp(today)

        # Mark to market all open spreads.
        for spread in self.strategy_positions:
            self._refresh_spread_marks(spread, option_chain)

        # Settle expired spreads.
        kept = []
        for spread in self.strategy_positions:
            if today_ts >= spread["expiration_date"]:
                self._close_spread(spread, use_expiry=True, vix_close=vix_close)
            else:
                kept.append(spread)

        self.strategy_positions = kept
        self.position_purposes = [s["purpose"] for s in self.strategy_positions]

        unrealized_long, unrealized_short = self._unrealized_pnl_split()

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
            "short_vol_call_spread_pnl": float(self.realized_short_pnl),
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
