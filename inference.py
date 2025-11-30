import pandas as pd
from trading import Trading
from gymnasium.spaces import flatten
import numpy as np
import os
import json
import itertools
import joblib

class Trader():
    def __init__(self, init_cash, descriptors,stock_names, validation_data_path, results_dir):
        self.aproximator_model = None
        self.validation_data_path = validation_data_path
        self.results_dir=results_dir
        self.env = Trading(data=self.__prepare_data(stock_names), init_cash=init_cash, descriptors = descriptors)
        self.stock_names=stock_names
        self.descriptors = descriptors
        
        self.NN_aproximator=False
        self.tree_aproximator=False
        self.self_action_combinations = list(itertools.product(self.env.actions, repeat=len(self.env.stock_names)))
        self.action_map = {0:"H",1:"B",2:"S",3:"SA"}
        
    def set_NN_aproximator(self, model):
        self.NN_aproximator = True
        self.tree_aproximator = False
        
        self.aproximator_model = model
        
    
    def set_tree_aproximator(self, model):
        self.NN_aproximator = False
        self.tree_aproximator = True
        
        self.aproximator_model = model
        
    
    def __prepare_data(self, stock_names):
        # načítanie
        joined_data = pd.DataFrame()
        former_stock_name = "" 
        for stock_name in stock_names:
            stock_data = pd.read_csv(f"{self.validation_data_path}/{stock_name}.csv")
            stock_data["Date"] = pd.to_datetime(stock_data["Date"]) 
            stock_data = stock_data.set_index("Date")
            stock_data.columns = pd.MultiIndex.from_product([[stock_name], stock_data.columns])
                
            if joined_data.empty:
                joined_data = stock_data
            else:
                joined_data = joined_data.join(stock_data, how="inner", lsuffix=f"_{former_stock_name}", rsuffix=f"_{stock_name}")
        
        return joined_data

    def trade(self):
        os.makedirs(self.results_dir, exist_ok=False)
        results = []
        state_current,reward,done,_,_ = self.env.step(self.env.action_space.sample())
        results.append
        while not done:      
            
            state_current_flattened = flatten(self.env.observation_space,state_current)
            state_current_flattened = state_current_flattened[np.newaxis, :]
                
            if self.NN_aproximator:
                Q_values = self.aproximator_model.predict(state_current_flattened)
            if self.tree_aproximator:
                Q_values = self.aproximator_model.predict([np.concatenate((flatten(self.env.observation_space,state_current),action)) for action in itertools.product(self.env.actions, self.env.actions)])
                
            best_action_id = np.argmax(Q_values)
            
            current_action = self.self_action_combinations[best_action_id]
            state_new,_,done,_,_ = self.env.step(self.self_action_combinations[best_action_id])
            
            state_current = state_new
            results.append(self.__save_step(state_new,self.stock_names,self.descriptors,best_action_id))

        self.__save_data_json(results, self.results_dir)
    def __save_data_json(self, data_list, dir_name):
        filepath= os.path.join(dir_name, "results.json")
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data_list, f, indent=4, ensure_ascii=False,cls=NumpyEncoder)    
        except IOError as e:
            print(f"Chyba pri zápise do súboru: {e}")
    
    def __save_step(self, observation, stock_names, descriptors, action):
        nb_stock = len(stock_names)
        nb_descriptors = len(descriptors)
        inputs = {
            "portfolio_value":observation["portfolio_price"],
            "cash":observation["cash"]
        }
        input_holdings = {f"holds_{n}":h for (n,h) in zip(stock_names,observation["holdings"])}
        input_descriptors = {f"{stock_names[i%nb_stock]}_{descriptors[i//nb_descriptors]}":d for i,d in enumerate(observation["descriptors"])}
        
        inputs = inputs | input_holdings | input_descriptors
        
        
        step = {"state":inputs,
                "chosen_action":self.__action_to_readable(self.self_action_combinations[action],stock_names),
                "chosen_action_id":action
                }
        
        return step
    
    def __action_to_readable(self,action,stock_names):
        return [f"{self.action_map[action]}_{stock_names[stock_id]}" for stock_id,action in enumerate(action)]

    def load_tree(self,file_path):
        return joblib.load(file_path)
    
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer): # Chytí np.int64
            return int(obj)
        if isinstance(obj, np.floating): # Chytí np.float64
            return float(obj)
        return json.JSONEncoder.default(self, obj)