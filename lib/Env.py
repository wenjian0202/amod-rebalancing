from lib.Agents import *
from lib.Demand import *

import gym
from gym import spaces

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.policy import EpsGreedyQPolicy
from rl.memory import SequentialMemory

class RebalancingEnv(gym.Env):
    """
    RebalancingEnv is the environment class for DQN
    Attributes:
        amod: AMoD system to train
        N: input grid of N*N cell
        c: length of the side of the cell
        dT: time interval for training
        penalty: penalty of rebalancing a vehicle
        action_space: action space
        state: the system state
        input_dim: input dimension
    """ 
    def __init__(self, amod):
        self.amod = amod
        self.amods = []
        self.N = 5
        self.c = 0.1
        self.dT = 5.0
        self.penalty = -0.02
        self.action_space = spaces.Discrete(5)
        self.state = np.zeros((2,self.N,self.N))
        self.input_dim = 2*self.N*self.N
        self.setp_count = 0
        self.Q_count = 0.0
        
    def step(self, action):
        self.setp_count += 1
        lat_, lng_ = self.get_vehicle_location()
        self.act(action)
        self.amod.dispatch_at_time(self.amod.T + self.dT)
        self.amods.append( copy.deepcopy(self.amod) )
        reward = 0 if action == 0 else self.penalty
        if not self.is_vehicle_idle():
            lat, lng = self.get_vehicle_location()
            tlat, tlng = self.get_target_location()
            t = get_distance(lat, lng, tlat, tlng) / self.get_vehicle_speed()
            t_ = get_distance(lat_, lng_, tlat, tlng) / self.get_vehicle_speed()
            reward += t_ - t
            self.Q_count += t_ - t
            while not self.is_vehicle_idle():
                self.amod.dispatch_at_time(self.amod.T + self.dT)
                self.amods.append( copy.deepcopy(self.amod) )
        print("step %d: Q: %2f; action: %s; reward: %.2f" % (self.setp_count, self.Q_count,
                                                     "noop" if action == 0 else 
                                                     "right" if action == 1 else
                                                     "left" if action == 2 else
                                                     "up" if action == 3 else
                                                     "down" if action == 4 else "error!", reward))
        self.update_state()
        return self.state, reward, False, {}
    
    def act(self, action, i=0):
        v = self.amod.vehs[i]
        v.jobs.clear()
        self.amod.act(v, action, self.c)
    
    def reset(self):
        self.update_state()
        self.amods.append( copy.deepcopy(self.amod) )
        return self.state
    
    def get_vehicle(self, i=0):
        return self.amod.vehs[i]

    def get_vehicle_location(self, i=0):
        return self.amod.vehs[i].get_location()
    
    def get_target_location(self, i=0):
        return self.amod.vehs[i].get_target_location()
    
    def get_vehicle_speed(self, i=0):
        return self.amod.vehs[i].S
    
    def is_vehicle_idle(self, i=0):
        return self.amod.vehs[i].is_idle()
    
    def update_state(self):
        self.state = self.amod.get_state(self.get_vehicle(), self.N, self.c)