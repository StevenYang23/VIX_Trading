from collections import deque
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


    def memorize(self, vix, vvix, spx, rv22):
        self.vix.append(vix)
        self.vvix.append(vvix)
        self.spx.append(spx)
        self.rv22.append(rv22)
        self.VRP_rv22.append(vix - rv22)
        
        if len(self.rv22) >= self.longterm_period:
            rv_long_term = np.mean(self.rv22[-self.longterm_period:])
            self.rv_long_term.append(rv_long_term)
            self.VRP_lt.append(vix - rv_long_term)
        else:
            self.rv_long_term.append(np.nan)
            self.VRP_lt.append(np.nan)

        if len(self.spx) >= self.garch_look_back + 1:
            vol_path, avg_vol = self.garch_model.forecast(spx)
            self.garch_forecast.append(avg_vol)
            self.VRP_garch.append(vix - avg_vol)
        else:
            self.garch_forecast.append(np.nan)
            self.VRP_garch.append(np.nan)
