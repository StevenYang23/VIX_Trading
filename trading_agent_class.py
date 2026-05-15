from memory_class import memory
import numpy as np



class Trading_Agent:
    def __init__(self, name, test_length, longterm_period):
        self.name = name
        self.longterm_period = longterm_period
        self.test_length = test_length
        self.memory = memory(longterm_period)
    
    def feed_data(self, vix,vvix,spx,rv22):
        self.memory.memorize(vix,vvix,spx,rv22)

    def signal(self):
        vix_test     = self.memory.vix[-self.test_length:]
        vvix_test = self.memory.vvix[-self.test_length:]
        spx_test = self.memory.spx[-self.test_length:]
        rv22_test = self.memory.rv22[-self.test_length:]
        VRP_rv22_test = self.memory.VRP_rv22[-self.test_length:]
        VRP_lt_test = self.memory.VRP_lt[-self.test_length:]
        rv_long_term_test = self.memory.rv_long_term[-self.test_length:]

        
        