from collections import deque
import numpy as np

class memory:
    def __init__(self, longterm_period):
        self.longterm_period = longterm_period
        self.spx = []
        self.vix = []
        self.vvix = []
        self.rv22 = []
        self.VRP_rv22 = []
        self.VRP_lt = []
        self.rv_long_term = []

    def memorize(self, vix, vvix, spx, rv22):
        self.vix.append(vix)
        self.vvix.append(vvix)
        self.spx.append(spx)
        self.rv22.append(rv22)
        self.VRP_rv22.append(vix-rv22)
        if len(self.rv22) >= self.longterm_period:
            rv_long_term = np.mean(self.rv22[-self.longterm_period:])
            self.rv_long_term.append(rv_long_term)
            self.VRP_lt.append(vix-rv_long_term)