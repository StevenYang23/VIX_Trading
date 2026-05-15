import numpy as np
from warnings import catch_warnings, simplefilter

try:
    from arch import arch_model
except ImportError:
    arch_model = None


class GARCHModel:
    def __init__(self, look_back=20):
        self.look_back = look_back
        self._REFIT_EVERY = 5
        
        self._stock_close_buf = []
        self._omega = np.nan
        self._alpha = np.nan
        self._gamma = np.nan
        self._beta = np.nan
        self._nu = np.nan
        self._h = np.nan
        self._days_since_fit = self._REFIT_EVERY  # Force fit on first available opportunity
        self.trading_days_per_year = 252

    def _fit_garch(self):
        px = np.asarray(self._stock_close_buf[-(self.look_back + 1) :], dtype=float)
        px = px[np.isfinite(px)]
        if px.size < self.look_back + 1:
            return False

        log_ret = np.diff(np.log(px))
        log_ret = log_ret[np.isfinite(log_ret)]
        
        # Need at least a few points to even attempt fitting
        if log_ret.size < min(10, self.look_back):
            return False

        log_ret_pct = log_ret * 100.0

        if arch_model is None:
            var = float(np.var(log_ret_pct))
            for r in log_ret_pct:
                var = 0.94 * var + 0.06 * r * r
            self._omega = 0.0
            self._alpha = 0.06
            self._gamma = 0.0
            self._beta = 0.94
            self._nu = np.nan
            self._h = var
            self._days_since_fit = 0
            return True

        try:
            with catch_warnings():
                simplefilter("ignore")
                model = arch_model(
                    log_ret_pct,
                    mean="Zero",
                    vol="GARCH",
                    p=1,
                    o=1,
                    q=1,
                    dist="t",
                    rescale=False,
                )
                fit = model.fit(disp="off")
            self._omega = float(fit.params.get("omega", 0.0))
            self._alpha = float(fit.params.get("alpha[1]", 0.06))
            self._gamma = float(fit.params.get("gamma[1]", 0.0))
            self._beta = float(fit.params.get("beta[1]", 0.94))
            self._nu = float(fit.params.get("nu", np.nan))
            cond_vol = fit.conditional_volatility
            self._h = (
                float(cond_vol.iloc[-1]) ** 2
                if len(cond_vol) > 0
                else float(np.var(log_ret_pct))
            )
            self._days_since_fit = 0
            return True
        except Exception:
            return False

    def _update_h(self, daily_log_return):
        if np.isnan(self._omega):
            return
        eps_pct = daily_log_return * 100.0
        asym = self._gamma if eps_pct < 0 else 0.0
        self._h = self._omega + (self._alpha + asym) * eps_pct**2 + self._beta * self._h

    def _forecast_math(self, horizon=22):
        if np.isnan(self._h) or self._h <= 0:
            return [np.nan] * horizon, np.nan

        h_step = float(self._h)
        omega = float(self._omega) if np.isfinite(self._omega) else 0.0
        alpha = float(self._alpha) if np.isfinite(self._alpha) else 0.0
        gamma = float(self._gamma) if np.isfinite(self._gamma) else 0.0
        beta = float(self._beta) if np.isfinite(self._beta) else 0.0
        
        # Expected value of asymmetric term is 0.5 * gamma
        phi = alpha + beta + 0.5 * gamma
        phi = min(max(phi, 0.0), 1.2)

        h_path = []
        for _ in range(horizon):
            h_step = omega + phi * h_step
            if not np.isfinite(h_step) or h_step <= 0:
                h_path.extend([np.nan] * (horizon - len(h_path)))
                break
            h_path.append(h_step)

        # Convert variance path to annualized volatility path
        vol_path = []
        for h_val in h_path:
            if np.isnan(h_val):
                vol_path.append(np.nan)
            else:
                vol_path.append((np.sqrt(h_val) / 100.0) * np.sqrt(self.trading_days_per_year))

        # Average volatility is calculated from the average variance over the period
        valid_h = [h for h in h_path if not np.isnan(h)]
        if not valid_h:
            avg_vol = np.nan
        else:
            avg_daily_var_pct = float(np.mean(valid_h))
            avg_vol = (np.sqrt(avg_daily_var_pct) / 100.0) * np.sqrt(self.trading_days_per_year)

        return vol_path, avg_vol

    def forecast(self, current_price):
        """
        Updates the model with a new daily close price, checks if refit is needed, 
        and returns the 22-day forecast.
        
        Args:
            current_price (float): The latest daily close price.
            
        Returns:
            tuple: (vol_path, avg_vol)
                - vol_path (list): 22-day forecasted annualized volatility path.
                - avg_vol (float): Average annualized volatility over the 22 days.
        """
        if current_price is None or np.isnan(current_price):
            return [np.nan] * 22, np.nan

        self._stock_close_buf.append(float(current_price))

        # Update the conditional variance incrementally
        if len(self._stock_close_buf) >= 2:
            ret = np.log(self._stock_close_buf[-1] / self._stock_close_buf[-2])
            if np.isfinite(ret) and not np.isnan(self._omega):
                self._update_h(ret)
            self._days_since_fit += 1

        # Check if model requires refit
        if (
            self._days_since_fit >= self._REFIT_EVERY
            and len(self._stock_close_buf) >= self.look_back + 1
        ):
            self._fit_garch()

        return self._forecast_math(horizon=22)
