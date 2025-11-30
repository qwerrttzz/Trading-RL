# gym documentation
# https://gym.farama.org/introduction/create_custom_env/#check-environment-validity

# Paper: Practical Deep Reinforcement Learning Approach for Stock Trading
# https://arxiv.org/pdf/1811.07522

from typing import Optional
import numpy as np
import gym 
from gym.spaces import Dict, Box, MultiDiscrete

class Trading(gym.Env):

    def __init__(self, data, init_cash):
        # static atributes
        self.data_df = data
        self.stock_names = self.data_df.columns.get_level_values(0).unique() # also columns in df
        self.day_id = 0
        self.days = len(self.data_df)
        
        self.init_cash = init_cash
        # dinamic atributes
        self.cash = init_cash
        self.last_portfolio_price = init_cash
        self.holdings = { name:0 for name in self.stock_names} 
        
        
        self.observation_space = Dict({
        "price_history":Box(
            low=0, 
            high=np.inf,
            shape=(),
            dtype=np.float32),
        "holdings": Dict({
            "cash": Box(low=0, high=np.inf,shape=(),dtype=np.float32),
            "stock1": Dict({
                "holding_stock":Box(low=0, high=np.inf,shape=(),dtype=np.int32),
                "price_for_one_stock": Box(low=0, high=np.inf,shape=(),dtype=np.float32)
                }),
            "stock2": Dict({
                "holding_stock":Box(low=0, high=np.inf,shape=(),dtype=np.int32),
                "price_for_one_stock": Box(low=0, high=np.inf,shape=(),dtype=np.float32)
                })
        }),
        "portfolio_price":Box(low=0, high=np.inf,shape=(),dtype=np.float32)
        })
        
        self.action_space = MultiDiscrete([6, 6])

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
        #     "price_history": self.data_df.iloc[self.day_id],
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
        # },
        # "portfolio_price": self.cash + sum(self.holdings[stock_name] * self.data_df[stock_name].iloc[self.day_id]['Open'] for stock_name in self.stock_names)
        # #self.holdings[self.stock_names[0]]*self.data_df[self.stock_names[0]][self.window_end_id]['Open'] + 
        #self.holdings[self.stock_names[1]]*self.data_df[self.stock_names[1]][self.window_end_id]['Open']
        #}
        return np.array([self.data_df.iloc[self.day_id], self.cash, 
                        self.holdings[self.stock_names[0]], self.data_df[self.stock_names[0]].iloc[self.day_id]['Open'],
                        self.holdings[self.stock_names[1]], self.data_df[self.stock_names[1]].iloc[self.day_id]['Open'],
                        self.cash + sum(self.holdings[stock_name] * self.data_df[stock_name].iloc[self.day_id]['Open'] for stock_name in self.stock_names)])
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        
        super().reset(seed=seed)

        self.day_id = 0
        self.cash = self.init_cash
        
        self.holdings = { name: 0 for name in self.stock_names }
        
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
            
            stock_price = self.data_df[stock_name].iloc[self.day_id]['Open']
            
            if stock_action == 1: # small buy
                if self.__possible_buy(self.cash, 10, stock_price): 
                    self.holdings[stock_name] += 10 
                    self.cash -= 10*stock_price  
                else:
                    invalid_action = True
            elif stock_action == 2: # big buy
                if self.__possible_buy(self.cash, 50, stock_price):
                    self.holdings[stock_name] += 50
                    self.cash -= 50*stock_price
                else:
                    invalid_action = True
            elif stock_action == 3: # small sell
                if self.__possible_sell(10, self.holdings[stock_name]):
                    self.holdings[stock_name] -= 10
                    self.cash += 10*stock_price
                else:
                    invalid_action = True
            elif stock_action == 4: # big sell
                if self.__possible_sell(50, self.holdings[stock_name]):
                
                    self.holdings[stock_name] -= 50
                    self.cash += 10*stock_price
                else:
                    invalid_action = True
            elif stock_action == 5: # sell all
                self.cash += self.holdings[stock_name]*stock_price
                self.holdings[stock_name] = 0 
            
                
        
        #time_window = self.data_df[(self.window_size-self.window_end_id):self.window_end_id]
        
        

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
            reward = (observation['portfolio_price'] - self.last_portfolio_price) / self.last_portfolio_price
        
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

