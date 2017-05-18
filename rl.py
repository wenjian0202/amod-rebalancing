from lib.Agents import *
from lib.Demand import *
from lib.Env import *

import gym
from gym import spaces

env = RebalancingEnv(AMoD(5, demand, V=20))

nb_actions = env.action_space.n
input_shape = (1,) + env.state.shape
input_dim = env.input_dim

model = Sequential()
model.add(Flatten(input_shape=input_shape))
model.add(Dense(128, activation='relu'))
model.add(Dense(nb_actions, activation='linear'))

memory = SequentialMemory(limit=50000, window_length=1)
policy = EpsGreedyQPolicy()
dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=100,
               target_model_update=1e-2, policy=policy)
dqn.compile(Adam(lr=1e-3), metrics=['mae'])

dqn.fit(env, nb_steps=100, visualize=False, verbose=2)

# dqn.save_weights('dqn_weights.h5f', overwrite=True)