from memory_class import memory
import numpy as np

class Trading_Agent:
    def __init__(self, name, test_length, longterm_period, garch_look_back, VRP_rv22_threshold, VRP_lt_threshold, VRP_garch_threshold):
        self.name = name
        self.longterm_period = longterm_period
        self.test_length = test_length
        self.memory = memory(longterm_period, garch_look_back)
        self.VRP_rv22_threshold = VRP_rv22_threshold
        self.VRP_lt_threshold = VRP_lt_threshold
        self.VRP_garch_threshold = VRP_garch_threshold

        self.long_position_symbols = []
        self.short_position_symbols = []
        self.long_position_price = []
        self.short_position_price = []

        self.daily_pnl_history = []
        self.daily_long_pnl_history = []
        self.daily_short_pnl_history = []
    
    def feed_data(self, vix,vvix,spx,rv22):
        self.memory.memorize(vix,vvix,spx,rv22)

    def signal(self):
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
        if z_score_rv22 >= self.VRP_rv22_threshold:
            rv22_signal = 1
        elif z_score_rv22 <= -self.VRP_rv22_threshold:
            rv22_signal = -1
        else:
            rv22_signal = 0

        z_score_lt = (VRP_lt_test - np.mean(VRP_lt_test)) / np.std(VRP_lt_test)
        if z_score_lt >= self.VRP_lt_threshold:
            lt_signal = 1
        elif z_score_lt <= -self.VRP_lt_threshold:
            lt_signal = -1
        else:
            lt_signal = 0
            
        z_score_garch = (VRP_garch_test - np.mean(VRP_garch_test)) / np.std(VRP_garch_test)
        if z_score_garch >= self.VRP_garch_threshold:
            garch_signal = 1
        elif z_score_garch <= -self.VRP_garch_threshold:
            garch_signal = -1
        else:
            garch_signal = 0
            
        return rv22_signal + lt_signal + garch_signal
    
    def search_option(self, option_chain, delta, ttm, option_type):
        chain_filtered = option_chain[option_chain["contract_type"].str.lower() == option_type.lower()]
        if chain_filtered.empty:
            print(f"No {option_type} options found")
            return None
        days_to_maturity = (chain_filtered["expiration_date"] - chain_filtered["as_of_date"]).dt.days
        closest_ttm = days_to_maturity.iloc[(days_to_maturity - ttm).abs().argmin()]
        closest_ttm_options = chain_filtered[days_to_maturity == closest_ttm]
        closest_idx = (closest_ttm_options["delta"] - delta).abs().argmin()
        closest_option = closest_ttm_options.iloc[closest_idx]
        return closest_option["ticker"], closest_option["close"]

    def trade(self, option_chain, trading_signal):
        d50_t22_c = self.search_option(option_chain, delta = 0.5, ttm = 22, option_type = 'call')
        if trading_signal >= 1 and d50_t22_c is not None:
            self.long_position_symbols.append(d50_t22_c[0])
            self.long_position_price.append(d50_t22_c[1])


    def calculate_pnl(self, option_chain):
        long_sum = 0
        short_sum = 0
        net_sum = 0
        for i in range(len(self.long_position_symbols)):
            symbol = self.long_position_symbols[i]
            cost = self.long_position_price[i]
            target_opt = option_chain[option_chain["ticker"] == symbol]
            pnl = target_opt["close"].iloc[-1] - cost
            long_sum += pnl

            if target_opt["expiration_date"] - target_opt["as_of_date"] <= 0:
                self.long_position_symbols.pop(i)
                self.long_position_price.pop(i)
            
        self.daily_long_pnl_history.append(long_sum)
        for i in range(len(self.short_position_symbols)):
            symbol = self.short_position_symbols[i]
            cost = self.short_position_price[i]
            target_opt = option_chain[option_chain["ticker"] == symbol]
            pnl = target_opt["close"].iloc[-1] - cost
            short_sum += pnl

            if target_opt["expiration_date"] - target_opt["as_of_date"] <= 0:
                self.short_position_symbols.pop(i)
                self.short_position_price.pop(i)
        self.daily_short_pnl_history.append(short_sum)
        net_sum = long_sum + short_sum
        self.daily_pnl_history.append(net_sum)

    def get_cum_pnl(self):
        return np.cumsum(self.daily_pnl_history)

    def get_cum_long_pnl(self):
        return np.cumsum(self.daily_long_pnl_history) 

    def get_cum_long_pnl(self):
        return np.cumsum(self.daily_short_pnl_history)   