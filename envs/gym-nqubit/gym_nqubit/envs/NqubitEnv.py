

import gym
from gym import error, spaces, utils
from gym.utils import seeding

import numpy as np
import time
import HB 
import HP
import math
import scipy.sparse.linalg
import random
import copy
import datetime
import measure
# version = '0.0.1'
class NqubitEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    
    def __init__(self):
        super(NqubitEnv, self).__init__()

        self.action_value = ['0','1+', '1-', '2+', '2-','3+','3-','4+','4-','5+','5-','6+','6-']

        self.action_space = spaces.Discrete(13)
        self.observation_space = spaces.Box(low = -float('inf'), high = float('inf'), shape=(6, ),dtype = np.float32)

        self.nbits = 5 # n 
        self.action_delta = 0.01  # delta
        self.T = 1.6344  # T 
        self.g = 1e-2 # g
        self.Numbers = [[49, 7, 7, 3],[57, 3, 19, 3],[69, 3, 23, 3],[87, 3, 29, 3]]
        self.Hb, self.Hp_array = self.MakeMatrix(self.nbits, self.Numbers) # Hb, Hp_array

        self.time_interval = np.linspace(0, T, 1000) # split into 1000 timesteps   t
        self.delta = self.time_interval/self.T  # t/T
        self.done = False


        self.state = None  # s
        self.Pi = np.pi


    def step(self, action):
        current_obs = self.state

        if action % 2 == 1 :
            self.state[int((action-1)/2)] += self.action_delta
        elif action == 0:
            pass # NOOP
        else:
            self.state[int(action/2 - 1)] -= self.action_delta

        ## path is the constraint of the state
        path = self.delta + np.sum([self.state[i] * np.sin((i+1)* self.Pi * self.delta)for i in range(self.observation_space.shape[0])])


        strictly_increasing = all(x<=y for x,y in zip(path,path[1:]))


        if strictly_increasing == 0:
			
			reward = measure.CalcuFidelity(self.n, current_obs, self.Hb, self.Hp_array, self.T, self.g) 
		else:
			
			reward = measure.CalcuFidelity(self.n, self.state , self.Hb, self.Hp_array, self.T, self.g) 

        if (reward >= -1.0):
            self.done = True
			

        return self.state, reward, self.done, {}

    def reset(self):
        self.state = np.zeros(shape= (6, ), dtype=np.float32)
        self.done = False
        return self.state

    def MakeMatrix(self, n , Numbers):
	    lenthNumbers = len(Numbers)
	    Hp_array = np.zeros((lenthNumbers,2**n),dtype = float)
	    Hb=HB.HB(n)
	    for i in range(len(Numbers)):
		    number = Numbers[i]
		    fact = HP.Factorization(number[0],number[1],number[2],number[3])
	    	Hp_array[i][:]=fact.Hamiltonian_matrix()
	    return Hb,Hp_array



