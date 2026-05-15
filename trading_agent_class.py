from memory_class import memory
import numpy as np

class Trading_Agent:
    def __init__(self, name, test_length, longterm_period, garch_look_back):
        self.name = name
        self.longterm_period = longterm_period
        self.test_length = test_length
        self.memory = memory(longterm_period, garch_look_back)
    
    def feed_data(self, vix,vvix,spx,rv22):
        self.memory.memorize(vix,vvix,spx,rv22)

    def signal(self, VRP_rv22_threshold, VRP_lt_threshold, VRP_garch_threshold):
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
        if z_score_rv22 >= VRP_rv22_threshold:
            rv22_signal = 1
        elif z_score_rv22 <= -VRP_rv22_threshold:
            rv22_signal = -1
        else:
            rv22_signal = 0

        z_score_lt = (VRP_lt_test - np.mean(VRP_lt_test)) / np.std(VRP_lt_test)
        if z_score_lt >= VRP_lt_threshold:
            lt_signal = 1
        elif z_score_lt <= -VRP_lt_threshold:
            lt_signal = -1
        else:
            lt_signal = 0
            
        z_score_garch = (VRP_garch_test - np.mean(VRP_garch_test)) / np.std(VRP_garch_test)
        if z_score_garch >= VRP_garch_threshold:
            garch_signal = 1
        elif z_score_garch <= -VRP_garch_threshold:
            garch_signal = -1
        else:
            garch_signal = 0
            
        return rv22_signal + lt_signal + garch_signal



        