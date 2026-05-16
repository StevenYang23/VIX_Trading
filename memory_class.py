import numpy as np
from garch_model import GARCHModel

class memory:
    def __init__(self, longterm_period, garch_look_back):
        self.longterm_period = longterm_period
        self.garch_look_back = garch_look_back
        self.spx = []
        self.vix = []
        self.vvix = []
        self.rv22 = []
        self.VRP_rv22 = []
        self.VRP_lt = []
        self.VRP_garch = []
        self.rv_long_term = []
        self.garch_forecast = []
        self.garch_model = GARCHModel(look_back=garch_look_back)

    @staticmethod
    def _to_pct_vol(x):
        """
        Normalize annualized vol to percentage points.
        - decimal input: 0.15 -> 15.0
        - pct-point input: 15.0 -> 15.0
        """
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return np.nan
        x = float(x)
        if not np.isfinite(x):
            return np.nan
        return x * 100.0 if abs(x) <= 3.0 else x


    def memorize(self, vix, vvix, spx, rv22):
        rv22_pct = self._to_pct_vol(rv22)
        self.vix.append(vix)
        self.vvix.append(vvix)
        self.spx.append(spx)
        self.rv22.append(rv22_pct)
        self.VRP_rv22.append(vix - rv22_pct)
        
        if len(self.rv22) >= self.longterm_period:
            rv_long_term = np.mean(self.rv22[-self.longterm_period:])
            self.rv_long_term.append(rv_long_term)
            self.VRP_lt.append(vix - rv_long_term)
        else:
            self.rv_long_term.append(np.nan)
            self.VRP_lt.append(np.nan)

        # Always call forecast so GARCH buffer warms up from day 1.
        _, avg_vol = self.garch_model.forecast(spx)
        avg_vol_pct = self._to_pct_vol(avg_vol)
        if len(self.spx) >= self.garch_look_back + 1 and np.isfinite(avg_vol_pct):
            self.garch_forecast.append(avg_vol_pct)
            self.VRP_garch.append(vix - avg_vol_pct)
        else:
            self.garch_forecast.append(np.nan)
            self.VRP_garch.append(np.nan)
