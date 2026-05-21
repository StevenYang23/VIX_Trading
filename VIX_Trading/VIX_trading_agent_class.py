import numpy as np
import pandas as pd
import math

class Trading_Agent:
    _MIN_SPREAD_OBS = 20
    _EWMA_INIT_WINDOW = 20
    _EWMA_REFIT_EVERY = 5
    _EWMA_LAMBDA = 0.94
    _DEFAULT_FORECAST_DAYS = 30

    def __init__(
        self,
        name="Agent_Vote",
        entry_threshold=0.8,
        kde_rolling_window=63,
        ewma_lambda=0.94,
        long_term_window=40,
        initial_balance=4000.0,
        k=4.0,
        contract_multiplier=100,
        Max_holding=2,
    ):
        self.name = name
        self.entry_threshold = float(entry_threshold)
        self.kde_rolling_window = max(int(kde_rolling_window), 3)
        self.long_term_window = max(int(long_term_window), 5)
        self._ewma_lambda = float(ewma_lambda)
        
        self.initial_balance = float(initial_balance)
        self.balance = float(initial_balance)
        self.k = max(float(k), 1.0)
        self.contract_multiplier = max(int(contract_multiplier), 1)
        self.max_holding = max(int(Max_holding), 1)
        
        self.strategy_positions = []
        self.position_purposes = []
        self.trade_purpose_history = []

        self.realized_long_pnl = 0.0
        self.realized_short_pnl = 0.0

        self.cum_pnl_history = []
        self.cum_long_pnl_history = []
        self.cum_short_pnl_history = []
        self.balance_history = []
        self.equity_history = []
        
        # Methodology variables
        self.trading_days_per_year = 252
        self._rv_buf = []
        self._vrp_history = []
        self._rv_buf_max = max(self.long_term_window * 4, 512)
        self._longterm_spread_history = []
        self._ewma_spread_history = []
        self._stock_close_buf = []
        self._h = np.nan
        self._days_since_fit = self._EWMA_REFIT_EVERY

    @staticmethod
    def _norm_cdf(x):
        return 0.5 * (1.0 + math.erf(x / np.sqrt(2.0)))

    @classmethod
    def _kde_cdf_signal(cls, values):
        arr = np.asarray(values, dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size < 3:
            return np.nan

        x0 = float(arr[-1])
        sample = arr[:-1]
        n = sample.size
        if n < 2:
            return np.nan

        std = float(np.std(sample, ddof=1))
        if (not np.isfinite(std)) or std < 1e-12:
            return float(2.0 * np.mean(sample <= x0) - 1.0)

        h = 1.06 * std * (n ** (-1.0 / 5.0))
        h = max(float(h), 1e-6)
        z = (x0 - sample) / h
        cdf_vals = np.array([cls._norm_cdf(float(v)) for v in z], dtype=float)
        p = float(np.mean(cdf_vals))
        p = min(max(p, 0.0), 1.0)
        return float(2.0 * p - 1.0)

    def _fit_ewma(self):
        px = np.asarray(self._stock_close_buf[-(self._EWMA_INIT_WINDOW + 1) :], dtype=float)
        px = px[np.isfinite(px)]
        if px.size < self._EWMA_INIT_WINDOW + 1:
            return False

        log_ret = np.diff(np.log(px))
        log_ret = log_ret[np.isfinite(log_ret)]
        if log_ret.size < 20:
            return False

        log_ret_pct = log_ret * 100.0
        var = float(np.var(log_ret_pct))
        if not np.isfinite(var) or var <= 0.0:
            return False
        lam = float(self._ewma_lambda)
        for r in log_ret_pct:
            var = lam * var + (1.0 - lam) * (r * r)
        if not np.isfinite(var) or var <= 0.0:
            return False
        self._h = float(var)
        self._days_since_fit = 0
        return True

    def _update_ewma_h(self, daily_log_return):
        if np.isnan(self._h):
            return
        eps_pct = daily_log_return * 100.0
        lam = float(self._ewma_lambda)
        self._h = lam * self._h + (1.0 - lam) * (eps_pct**2)

    def _ewma_rv_annualized(self, horizon_days):
        if np.isnan(self._h) or self._h <= 0:
            return np.nan
        avg_daily_var_pct = float(self._h)
        sigma_daily = np.sqrt(avg_daily_var_pct) / 100.0
        return sigma_daily * np.sqrt(self.trading_days_per_year)

    def feed_data(self, stock_close, rv, vrp, straddle_imp_vol, days_to_strike=30):
        curr_spot = stock_close
        if pd.notna(curr_spot):
            self._stock_close_buf.append(float(curr_spot))

        if len(self._stock_close_buf) >= 2:
            ret = np.log(self._stock_close_buf[-1] / self._stock_close_buf[-2])
            if np.isfinite(ret) and not np.isnan(self._h):
                self._update_ewma_h(ret)
            self._days_since_fit += 1

        if (
            self._days_since_fit >= self._EWMA_REFIT_EVERY
            and len(self._stock_close_buf) >= self._EWMA_INIT_WINDOW + 1
        ):
            self._fit_ewma()
            
        self.current_stock_close = stock_close
        self.current_rv = rv
        self.current_vrp = vrp
        self.current_iv = straddle_imp_vol
        self.current_days_to_strike = days_to_strike

    def signal(self):
        curr_iv = self.current_iv
        curr_rv = self.current_rv
        vrp = self.current_vrp

        if pd.isna(curr_iv) or not np.isfinite(curr_iv):
            return 0.0
        if pd.isna(curr_rv) or not np.isfinite(curr_rv):
            return 0.0

        if len(self._rv_buf) < self.long_term_window:
            self._rv_buf.append(float(curr_rv))
            if len(self._rv_buf) > self._rv_buf_max:
                self._rv_buf = self._rv_buf[-self._rv_buf_max :]
            return 0.0

        rv_signal = 0
        if pd.notna(vrp) and np.isfinite(vrp):
            self._vrp_history.append(float(vrp))
            if len(self._vrp_history) >= self._MIN_SPREAD_OBS:
                signal_window = np.asarray(
                    self._vrp_history[-self.kde_rolling_window :], dtype=float
                )
                if np.sum(np.isfinite(signal_window)) >= 3:
                    kde_signal = self._kde_cdf_signal(signal_window)
                    if np.isfinite(kde_signal):
                        if kde_signal > self.entry_threshold:
                            rv_signal = -1
                        elif kde_signal < -self.entry_threshold:
                            rv_signal = 1

        window_rv = self._rv_buf[-self.long_term_window :]
        long_term_mean = float(np.mean(window_rv))
        longterm_spread = float(curr_iv) - long_term_mean
        ewma_rv = self._ewma_rv_annualized(self.current_days_to_strike)
        ewma_spread = (
            float(curr_iv) - float(ewma_rv) if np.isfinite(ewma_rv) else np.nan
        )

        self._longterm_spread_history.append(longterm_spread)
        if np.isfinite(ewma_spread):
            self._ewma_spread_history.append(ewma_spread)

        self._rv_buf.append(float(curr_rv))
        if len(self._rv_buf) > self._rv_buf_max:
            self._rv_buf = self._rv_buf[-self._rv_buf_max :]

        longterm_signal = 0
        if len(self._longterm_spread_history) >= self._MIN_SPREAD_OBS:
            spread_arr = np.asarray(
                self._longterm_spread_history[-self.kde_rolling_window :], dtype=float
            )
            if np.sum(np.isfinite(spread_arr)) >= 3:
                longterm_kde = self._kde_cdf_signal(spread_arr)
                if np.isfinite(longterm_kde):
                    if longterm_kde > self.entry_threshold:
                        longterm_signal = -1
                    elif longterm_kde < -self.entry_threshold:
                        longterm_signal = 1

        ewma_signal = 0
        if len(self._ewma_spread_history) >= self._MIN_SPREAD_OBS and np.isfinite(ewma_spread):
            ewma_arr = np.asarray(
                self._ewma_spread_history[-self.kde_rolling_window :], dtype=float
            )
            if np.sum(np.isfinite(ewma_arr)) >= 3:
                ewma_kde = self._kde_cdf_signal(ewma_arr)
                if np.isfinite(ewma_kde):
                    if ewma_kde > self.entry_threshold:
                        ewma_signal = -1
                    elif ewma_kde < -self.entry_threshold:
                        ewma_signal = 1

        overall_vote = rv_signal + longterm_signal + ewma_signal
        return float(overall_vote)

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

    def _chain_spot(self, option_chain):
        if "VIX_close" not in option_chain.columns:
            return np.nan
        spot = option_chain["VIX_close"].dropna()
        if spot.empty:
            return np.nan
        return float(spot.iloc[0])

    def search_option_by_strike(self, option_chain, strike, ttm, option_type):
        chain_filtered = option_chain[
            (option_chain["contract_type"].str.lower() == option_type.lower())
            & (option_chain["close"].notna())
        ]

        if chain_filtered.empty:
            return None

        days_to_maturity = (chain_filtered["expiration_date"] - chain_filtered["as_of_date"]).dt.days
        closest_ttm = days_to_maturity.iloc[(days_to_maturity - ttm).abs().argmin()]
        closest_ttm_options = chain_filtered[days_to_maturity == closest_ttm]

        strike_matches = closest_ttm_options[
            closest_ttm_options["strike_price"] == float(strike)
        ]
        if strike_matches.empty:
            return None

        closest_option = strike_matches.iloc[0]
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

    def _atm_call_strikes(self, option_chain, ttm, spread_width):
        spot = self._chain_spot(option_chain)
        if not np.isfinite(spot):
            return None, None

        chain_filtered = option_chain[
            (option_chain["contract_type"].str.lower() == "call")
            & (option_chain["close"].notna())
        ]
        if chain_filtered.empty:
            return None, None

        days_to_maturity = (chain_filtered["expiration_date"] - chain_filtered["as_of_date"]).dt.days
        closest_ttm = days_to_maturity.iloc[(days_to_maturity - ttm).abs().argmin()]
        closest_ttm_options = chain_filtered[days_to_maturity == closest_ttm]
        if closest_ttm_options.empty:
            return None, None

        dist = (closest_ttm_options["strike_price"] - spot).abs()
        lower_strike = float(closest_ttm_options.iloc[dist.argmin()]["strike_price"])
        upper_strike = lower_strike + float(spread_width)
        return lower_strike, upper_strike

    def _resolve_strike_for_leg(self, option_chain, leg_def, strategy_def, ttm):
        spread_width = float(strategy_def.get("spread_width", 1.0))
        lower_strike, upper_strike = self._atm_call_strikes(
            option_chain, ttm, spread_width
        )
        if lower_strike is None or upper_strike is None:
            return None

        role = leg_def.get("strike_role")
        if role == "atm":
            return lower_strike
        if role == "atm_plus_width":
            return upper_strike
        return None

    def _strategy_from_signal(self, trading_signal):
        if trading_signal < -1:
            return {
                "purpose": "short_vol",
                "strategy_name": "digital_bear_call_spread",
                "ttm": 30,
                "spread_width": 1.0,
                "legs": [
                    {"strike_role": "atm", "option_type": "call", "side": -1},
                    {"strike_role": "atm_plus_width", "option_type": "call", "side": +1},
                ],
            }
        return None

    def _build_spread(self, option_chain, strategy_def, ttm=None):
        if ttm is None:
            ttm = int(strategy_def.get("ttm", 45))
        else:
            ttm = int(ttm)
        legs = []
        for leg_def in strategy_def["legs"]:
            if "strike_role" in leg_def:
                strike = self._resolve_strike_for_leg(
                    option_chain, leg_def, strategy_def, ttm
                )
                if strike is None:
                    return None
                opt = self.search_option_by_strike(
                    option_chain,
                    strike=strike,
                    ttm=ttm,
                    option_type=leg_def["option_type"],
                )
            else:
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
        self.trade_purpose_history.append(spread["purpose"])
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

    def _intrinsic_value(self, leg, underlying_close):
        strike = leg["strike_price"]
        if leg["contract_type"] == "call":
            return max(float(underlying_close) - strike, 0.0)  # max(S-K, 0)
        return max(strike - float(underlying_close), 0.0)  # max(K-S, 0)

    def _close_spread(self, spread, option_chain=None, use_expiry=False, underlying_close=None):
        qty = spread["qty"]
        close_cash_flow = 0.0
        spread_pnl = 0.0

        for leg in spread["legs"]:
            if use_expiry:
                val = self._intrinsic_value(leg, underlying_close)
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
        s > 1: close short_vol only (no new trades)
        -1 <= s <= 1: close all
        s < -1: open digital_bear_call_spread (30 DTE, 1.0 width at spot)
        """
        if trading_signal > 1:
            self._close_positions_by_purpose("short_vol", option_chain)
            return

        target = self._strategy_from_signal(trading_signal)

        if target is None:
            self._close_all_positions(option_chain)
            return

        self._close_positions_by_purpose("long_vol", option_chain)

        active_same = any(
            s["purpose"] == target["purpose"] and s["strategy_name"] == target["strategy_name"]
            for s in self.strategy_positions
        )
        if active_same:
            return

        self._close_positions_by_purpose(target["purpose"], option_chain)
        spread = self._build_spread(option_chain, target)
        if spread is not None:
            self._open_spread(spread)

    def calculate_pnl(self, option_chain, today, underlying_close):
        today_ts = pd.Timestamp(today)

        # Mark to market all open spreads.
        for spread in self.strategy_positions:
            self._refresh_spread_marks(spread, option_chain)

        # Settle expired spreads.
        kept = []
        for spread in self.strategy_positions:
            if today_ts >= spread["expiration_date"]:
                self._close_spread(spread, use_expiry=True, underlying_close=underlying_close)
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
            "long_vol_pnl": float(self.realized_long_pnl),
            "short_vol_bear_call_pnl": float(self.realized_short_pnl),
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
