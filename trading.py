# Gymnasium documentation
# https://gymnasium.farama.org/introduction/create_custom_env/#check-environment-validity

# Paper: Practical Deep Reinforcement Learning Approach for Stock Trading
# https://arxiv.org/pdf/1811.07522

from typing import Optional
import numpy as np
import gymnasium as gym
from gymnasium.spaces import Dict, Box, MultiDiscrete
import pandas as pd

class Trading(gym.Env):
    
    # pass df with only wanted stock
    def __init__(self, data, init_cash, descriptors):
        # static atributes
        self.data_df = data
        self.stock_names = self.data_df.columns.get_level_values(0).unique() # also columns in df
        self.descriptors = descriptors    
        self.day_id = 0
        self.days = len(self.data_df)
        self.init_cash = init_cash
        
        # dinamic atributes
        self.cash = self.init_cash
        self.last_portfolio_price = init_cash
        self.holdings = [0 for _ in self.stock_names]
        self.stock_prices = [0 for _ in self.stock_names]
        
        self.actions = [0,1,2,3]
        self.actions_dim = [len(self.actions) for _ in self.stock_names] 
        self.nb_actions = len(self.actions) * len(self.stock_names)
        
        #self.holdings = { name:0 for name in self.stock_names} 
        
        # self.observation_space = Dict({
        # "price_history":Box(
        #     low=0, 
        #     high=np.inf,
        #     shape=(),
        #     dtype=np.float32),
        # "holdings": Dict({
        #     "cash": Box(low=0, high=np.inf,shape=(),dtype=np.float32),
        #     "stock1": Dict({
        #         "holding_stock":Box(low=0, high=np.inf,shape=(),dtype=np.int32),
        #         "price_for_one_stock": Box(low=0, high=np.inf,shape=(),dtype=np.float32)
        #         }),
        #     "stock2": Dict({
        #         "holding_stock":Box(low=0, high=np.inf,shape=(),dtype=np.int32),
        #         "price_for_one_stock": Box(low=0, high=np.inf,shape=(),dtype=np.float32)
        #         })
        # }),
        # "portfolio_price":Box(low=0, high=np.inf,shape=(),dtype=np.float32)
        # })
        
        self.observation_space = Dict({
        "descriptors":Box(
            low=0, 
            high=np.inf,
            shape=(len(self.stock_names), len(descriptors)),
            dtype=np.float32),
        "cash": Box(low=0, high=np.inf, shape=(),dtype=np.float32),    
        "holdings": Box(low=0, high=np.inf, shape=(len(self.stock_names),), dtype=np.int32),
        "portfolio_price":Box(low=0, high=np.inf, shape=(len(self.stock_names),), dtype=np.float32)
        })
        #self.observation_space_flatten_size = len(self.stock_names) * descriptors + len(self.stock_names)*2 + 2
        
        self.action_space = MultiDiscrete(self.actions_dim)

    def _get_info(self):
        return {
            "average": "whatever"
        }
        
    def _get_obs(self):
        # return {
        #     "price_history":self.data_df[(self.window_size-self.window_end_id):self.window_end_id],
        #     "cash":self.cash,
        #     "holding":self.holding,
        #     "portfolio_price":self.portfolio_price
        # }
        # return {
        #     "price_history": self.data_df.xs('Close', level=1, axis=1).iloc[self.day_id],#self.data_df.iloc[self.day_id],
        #     "holdings": {
        #         "cash": self.cash,
        #         "stock1": {
        #             "holding_stock":self.holdings[self.stock_names[0]],
        #             "price_for_one_stock": self.data_df[self.stock_names[0]].iloc[self.day_id]['Open'] 
        #             },
        #         "stock2": {
        #             "holding_stock":self.holdings[self.stock_names[1]],
        #             "price_for_one_stock": self.data_df[self.stock_names[1]].iloc[self.day_id]['Open'] 
        #             }
        # },}
        # "portfolio_price": self.cash + sum(self.holdings[stock_name] * self.data_df[stock_name].iloc[self.day_id]['Open'] for stock_name in self.stock_names)
        # #self.holdings[self.stock_names[0]]*self.data_df[self.stock_names[0]][self.window_end_id]['Open'] + 
        #self.holdings[self.stock_names[1]]*self.data_df[self.stock_names[1]][self.window_end_id]['Open']
        #print(f"Hi: {self.data_df.loc[:, pd.IndexSlice[:, self.descriptors]].iloc[self.day_id]}")
        return {
            "descriptors": self.data_df.loc[:, pd.IndexSlice[:, self.descriptors]].iloc[self.day_id].values,#self.data_df.xs('Close', level=1, axis=1).iloc[self.day_id],
            "cash": self.cash,                
            "holdings":self.holdings,
            "portfolio_price": self.cash + sum(self.holdings[stock_id] * self.stock_prices[stock_id] for stock_id in range(len(self.holdings)))
        }
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        
        super().reset(seed=seed)

        self.day_id = 0
        self.cash = self.init_cash
        
        self.holdings = [0 for _ in self.stock_names]
        
        observation = self._get_obs()
        info = self._get_info()
        
        return observation, info
    
    def step(self, action): 
        invalid_action = False
        
        for stock_id, stock_action in enumerate(action):
            stock_name = self.stock_names[stock_id]
            #print(self.stock_names)
            #print(self.data_df[stock_name])
            #print(self.data_df[stock_name].iloc[self.day_id])
            
            #TODO ['Open']
            stock_price = self.data_df[stock_name].iloc[self.day_id]['Open']
            self.stock_prices[stock_id] = stock_price
            
            if stock_action == 1: # small buy
                if self.__possible_buy(self.cash, 10, stock_price): 
                    self.holdings[stock_id] += 10 
                    self.cash -= 10*stock_price  
                else:
                    invalid_action = True
            elif stock_action == 2: # small sell
                if self.__possible_sell(10, self.holdings[stock_id]):
                    self.holdings[stock_id] -= 10
                    self.cash += 10*stock_price
                else:
                    invalid_action = True
            elif stock_action == 3: # sell all
                self.cash += self.holdings[stock_id]*stock_price
                self.holdings[stock_id] = 0 
            
        observation = self._get_obs()
        info = self._get_info()
        
        self.day_id += 1
        terminated = False
        if self.day_id >= self.days:
            terminated = True
        truncated = False
        
        if invalid_action:
            reward = -1
        else:
            #reward = (observation['portfolio_price'] - self.last_portfolio_price) / self.last_portfolio_price
            reward = np.log(observation['portfolio_price'] / (observation['portfolio_price']-1))
        reward = np.clip(reward, -1, 1)
        
        self.last_portfolio_price = observation['portfolio_price']

        return observation, reward, terminated, truncated, info
    
    def __possible_buy(self, cash, amount, price_for_one):
        if cash < amount * price_for_one:
            return False
        return True
    
    def __possible_sell(self, sell_amount, hold_amount):
        if sell_amount>hold_amount:
            return False
        return True

