from lib.Agents import *
from lib.Demand import *
from lib.Env import *
import csv

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

runtime = []
waittime = []
vehtime = []
servtime = []
rebltime = []

for _ in range(1):
    amod = AMoD(5, demand, V=20)
    dqn.load_weights('dqn_weights.h5f')
    
    stime = time.time()
    amods = []
    for T in range(0,10000,10):
        amod.dispatch_at_time(T)
        if np.remainder(T, 10) == 0:
            amod.rebalance("rl", dqn)
        if np.remainder(T, 100) == 0:
            print("run %d, t = %.0f" % (_, T) )
        amods.append( copy.deepcopy(amod) )
    etime = time.time()    
    rt = etime - stime
    runtime.append(rt)
    
    wt = 0.0
    vt = 0.0
    c_wt = 0
    c_vt = 0
    for r in amod.reqs:
        if r.Tp > -1:
            wt += (r.Tp - r.Tr)
            c_wt += 1
            if r.Td > -1:
                vt += (r.Td - r.Tp)
                c_vt += 1
    wt = wt / c_wt
    vt = vt / c_vt
    waittime.append(wt)
    vehtime.append(vt)
    
    st = 0.0
    rt = 0.0
    for v in amod.vehs:
        st += v.Ts
        rt += v.Tr
    st = st / amod.V
    rt = rt / amod.V
    servtime.append(st)
    rebltime.append(rt)

f = open('results.csv', 'a');
writer = csv.writer(f)
writer.writerow(runtime)
writer.writerow(waittime)
writer.writerow(vehtime)
writer.writerow(servtime)
writer.writerow(rebltime)
f.close()
