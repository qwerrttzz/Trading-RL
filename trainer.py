# creater trainer with adaptable environment and saveable progress
# do rsi and moving average stuff
# better logs with pictures (reward, descriptors, portfolio price and holdings last)
import itertools
import keras
from keras import layers
from keras.models import load_model
import json
import pandas as pd
import os
from trading import Trading
import logging
import random
from gymnasium.spaces import flatten
from tensorflow.keras.models import clone_model
import numpy as np
import time
import pandas_ta as ta
from sklearn.ensemble import ExtraTreesRegressor
import joblib

os.environ["KERAS_BACKEND"] = "tensorflow"

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer): # Chytí np.int64
            return int(obj)
        if isinstance(obj, np.floating): # Chytí np.float64
            return float(obj)
        return json.JSONEncoder.default(self, obj)
    
class Trainer():
    def __init__(self, data_dir):
        self.data_dir=data_dir
        
        pass
       
    def __create_q_model(self, input_dim, num_actions):
        return keras.Sequential([
            layers.Input(shape=(input_dim,)),
            layers.Dense(128, activation="relu"),
            layers.Dense(128, activation="relu"),
            layers.Dense(num_actions, activation="linear")
        ])
        
    def __prepare_data(self, stock_names):
        # načítanie
        joined_data = pd.DataFrame()
        former_stock_name = "" 
        for stock_name in stock_names:
            stock_data = pd.read_csv(f"{self.data_dir}/{stock_name}.csv")
            stock_data["Date"] = pd.to_datetime(stock_data["Date"]) 
            stock_data = stock_data.set_index("Date")
            stock_data.columns = pd.MultiIndex.from_product([[stock_name], stock_data.columns])
                
            if joined_data.empty:
                joined_data = stock_data
            else:
                joined_data = joined_data.join(stock_data, how="inner", lsuffix=f"_{former_stock_name}", rsuffix=f"_{stock_name}")
        
        return joined_data

    def train(self, dir_name, stock_names, descriptors, uptrain = "", max_steps=100000000, init_cash=10000, save=False):
        os.makedirs(dir_name, exist_ok=False)
        self.__create_log_file(dir_name)
        
        data = self.__prepare_data(stock_names)
        env = Trading(data=data, init_cash=init_cash, descriptors = descriptors)
        
        # descriptors for each stock + holding and price of each stock + portfolio price + cash 
        observation_dim = (len(stock_names) * len(descriptors)) + (len(stock_names)) + 2
        # 6 actions for each stock
        action_dim = len(stock_names) * 4 # tu by som dal env.action_dim
        print(f"Action dimension={action_dim}")
        actor_model = self.__get_model(uptrain, observation_dim, action_dim)
        #critic_model = self.__get_model(uptrain, observation_dim, action_dim)
        critic_model = clone_model(actor_model)
        critic_model.set_weights(actor_model.get_weights())

        if actor_model is None or critic_model is None:
            return None
        
        self.__DQN_training_loop(dir_name,actor_model,critic_model, env, descriptors, 2, 0.99, max_steps=max_steps,save=save)
    
    
        
    
    
    
    
    #
    #                   /--> a21 |                         tQ(s2,a21) |                        Q(s2,a21) 
    # |s1| -- a11 --> s2 --> a22 | critic -- generate -->  tQ(s2,a22) | actor -- generate -->  Q(s2,a22)    
    #                   \--> a23 |                         tQ(s2,a23) |                        Q(s2,a23) 
    #
    # Q(s1,a11) = reward + gamma * max(actor.generate())
    # target_Q(s1,a11) = reward + gamma * max(critic.generate())
    #
    #actor_learn = loss(target_Q(s1,a11) - Q(s1,a11))
    def __DQN_training_loop(self,dir_name, actor_model,critic_model, env, descriptors, nb_episodes, gamma, max_steps, save):    
        # toto by som dal do env
        self.self_action_combinations = list(itertools.product(env.actions, repeat=len(env.stock_names)))
        self.action_map = {0:"H",1:"B",2:"S",3:"SA"}
        #print("HI",self_action_combinations)
        for episode in range(1,nb_episodes+1):
            env.reset()
            done = False
            score = 0
            day=0
            
            # first day state_prew doesnt exist yet
            current_action=env.action_space.sample()
            state_current,reward,_,_,_ = env.step(current_action)
            Q_prev=0
            target_update_freq = 10
            step = 0
            
            #saved_states = {"states":[], "action_values":[],"rewards":[], "actor_q_values":[], "critic_q_values":[]}
            saved_steps = []
            # we have next state and reward given by taking an action
            while not done:      
                start = time.perf_counter()
                step+=1
                
                state_current_flattened = flatten(env.observation_space,state_current)
                state_current_flattened = state_current_flattened[np.newaxis, :]
                
                # predict next q values base on current state(approximate v())
                Q_values = actor_model.predict(state_current_flattened,verbose=0)[0]
                #Q_values_target = critic_model.predict(state,verbose=0)[0]
                
                # evaluate current state
                current_state_Q = reward + gamma * max(Q_values)
                
                # best_action = [np.argmax(stock_Q_values[0]),np.argmax(stock_Q_values[1])]
                # pick actions(for each stock) based on policy(can pick action with Q value or random action)
                
                #best_action_id, action_by_stock = self.__mixed_policy(best_action_id, env)
                if random.random()>0.5:
                    best_action_id = np.argmax(Q_values)
                else:
                    best_action_id = random.randint(0, env.nb_actions-1)
                
                # perform picked action for each stock and thus get new state
                current_action = self.self_action_combinations[best_action_id]
                state_new,reward,done,_,_ = env.step(self.self_action_combinations[best_action_id])
                day+=1
                
                transition = (state_current, best_action_id, reward, state_new)
                
                state_new_flattened = flatten(env.observation_space,state_new)
                state_new_flattened = state_new_flattened[np.newaxis, :]
                
                # compute target value
                target_model_Q_values = critic_model.predict(state_new_flattened,verbose=0)[0]
                target_value = transition[2] + gamma * np.max(target_model_Q_values)
                
                # create target vector by replacing target value in q values vektor
                targets = Q_values.copy()
                targets[best_action_id] = target_value 
                
                # train model
                loss = actor_model.train_on_batch(state_current_flattened, targets)
                score +=loss
                end = time.perf_counter()
                
                logging.info(f"episode:{episode}, training_time:{end - start}, day:{day}, reward:{reward}, loss:{loss}, state:{state_current}")
                if save:
                    saved_steps.append(self.__save_step(state_current, env.stock_names, descriptors, Q_values, target_model_Q_values,reward,current_state_Q,best_action_id))
                
                if step % target_update_freq == 0:
                    critic_model.set_weights(actor_model.get_weights())
                    
                if step > max_steps:
                    done=True 
                
        print(f'Episode {episode}: Score:{score}')
        
        if save:
            #print(saved_states["states"])
            self.__save_data_json(saved_steps,dir_name)
        
        model_path = os.path.join(dir_name, "NN_model.h5")
        actor_model.save(model_path)

        return actor_model
    
    def __save_step(self, observation, stock_names, descriptors, Q,target,reward,Q_value,action):
        nb_stock = len(stock_names)
        nb_descriptors = len(descriptors)
        inputs = {
            "portfolio_value":observation["portfolio_price"],
            "cash":observation["cash"]
        }
        input_holdings = {f"holds_{n}":h for (n,h) in zip(stock_names,observation["holdings"])}
        input_descriptors = {f"{stock_names[i%nb_stock]}_{descriptors[i//nb_descriptors]}":d for i,d in enumerate(observation["descriptors"])}
        
        inputs = inputs | input_holdings | input_descriptors
        
        outputs_Q = {f"{self.__action_to_readable(a,stock_names)}":q for (a,q) in zip(self.self_action_combinations, Q)}
        outputs_target = {f"{self.__action_to_readable(a,stock_names)}":q for (a,q) in zip(self.self_action_combinations, target)}
        
        step = {"inputs":inputs,
                "outputs_Q":outputs_Q,
                "outputs_target": outputs_target,
                "reward": reward,
                "Q": Q_value,
                "max_Q_id": np.argmax(list(outputs_Q.values())),
                "chosen_action":self.__action_to_readable(self.self_action_combinations[action],stock_names),
                "chosen_action_id":action
                }
        
        return step

    def __action_to_readable(self,action,stock_names):
        return [f"{self.action_map[action]}_{stock_names[stock_id]}" for stock_id,action in enumerate(action)]
    def __save_data_json(self, data_list, dir_name):
        filepath= os.path.join(dir_name, "states.json")
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data_list, f, indent=4, ensure_ascii=False,cls=NumpyEncoder)    
        except IOError as e:
            print(f"Chyba pri zápise do súboru: {e}")
        
    def __check_model_dimensions(self, model, input_dim, ouput_dim):
        pass
    
    def __get_model(self, uptrain, observation_dim, action_dim):
        if uptrain == "":
            model = self.__create_q_model(observation_dim, action_dim)
            model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-3),
            loss='mse'   # pre Q-learning typicky mean squared error
            )
        else:    
            model = load_model(uptrain)
            if self.__check_model_dimensions(model, observation_dim, action_dim):
                print("error: loaded model dim doesnt fit the dataset for uptraining")
                return None
            
        return model

    def __create_log_file(self, dir_name):
        log_path = os.path.join(dir_name, "training.log")

        logging.basicConfig(
            filename=log_path,
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            filemode='w'
        )
     
    def __mixed_policy(self, Q_values, env):
        if random.random()>0.5:
            return np.argmax(Q_values)
            
        
        return env.action_space.sample()
    
    def fQi_extremeTree(self,save_dir,T, init_cash,stock_names,descriptors, save=True):
        
        tree = ExtraTreesRegressor()
        gamma = 0.99
        data = self.__prepare_data(stock_names)
        self.env = Trading(data=data, init_cash=init_cash, descriptors = descriptors)
        
        self.self_action_combinations = list(itertools.product(self.env.actions, repeat=len(self.env.stock_names)))
        self.action_map_2 = {f"{list(action)}":idx for idx,action in enumerate(self.self_action_combinations)}
        print(self.action_map_2)
        self.action_map = {0:"H",1:"B",2:"S",3:"SA"}
        self.self_action_combinations = list(itertools.product(self.env.actions, repeat=len(self.env.stock_names)))
        
        saved_steps = []
        D, train = self.__generate_fQi_dataset(self.env,data,init_cash,descriptors)
        tree.fit(train, [reward for (_,_,_,reward) in D])
        
        for i in range(0,T-1):
            Q_targets = np.zeros(len(D))
            for idx, (s, a,s_next,r) in enumerate(D):
                # extra tree as Q(s,a) aproximator - for combination of state and action generate q value
                next_actions_Q_values = tree.predict([np.concatenate((flatten(self.env.observation_space,s_next),action)) for action in itertools.product(self.env.actions, self.env.actions)])
                target = r + gamma * max(next_actions_Q_values)
                Q_targets[idx] = target
                saved_steps.append(self.__save_step(s, self.env.stock_names, self.env.descriptors, [0], [0],r,target,self.action_map_2[f"{[int(x) for x in a]}"]))
            

            tree.fit(train,Q_targets)

        os.makedirs(save_dir, exist_ok=False)
        if save:
            self.__save_data_json(saved_steps,save_dir)
            
        joblib.dump(tree, os.path.join(save_dir, "tree_model.joblib"))
        
        return tree

    def __generate_fQi_dataset(self,env,data,init_cash,descriptors,multiplicator=1):
        D = []
        train = []
        init_fit = []
        for i in range(0, multiplicator):
            env.reset()
            
            action=env.action_space.sample()
            state_current,reward,done,_,_ = env.step(action)
            done=False
            
            while not done:
                action=env.action_space.sample()
                state_next,reward,done,_,_ = env.step(action)
                D.append((state_current, action, state_next, reward))
                train.append(np.concatenate((flatten(env.observation_space,state_current),action)))
        
        #train = np.zeros(env.days, self.observation_space_flatten_size + len(env.stock_names))
        
        return D, train
