import copy
import time
from collections import deque
from queue import PriorityQueue
import numpy as np
from numpy import unravel_index
from matplotlib import pyplot as plt
from matplotlib import animation
from IPython.display import HTML
from scipy.optimize import linprog

def get_distance(lat1, lng1, lat2, lng2):
    return np.sqrt( (lat1 - lat2)**2 + (lng1 - lng2)**2 )

class Veh(object):
    """ 
    Veh is a class for vehicles
    Attributes:
        id: sequential unique id
        T: system time
        Ts: accumulated service time (including pickup and dropoff)
        Tr: accumulated rebalancing time
        lat: current latitude
        lng: current longitude
        tlat: target (end of route) latitude
        tlng: target (end of route) longitude
        K: capacity
        n: number of passengers on board
        jobs: a list of jobs in the format of (request id, pickup or dropoff, target lat, target lng)
        S: speed
    """ 
    def __init__(self, id, T=0.0, lat=0.5, lng=0.5, K=1, S=0.01):
        self.id = id
        self.T = T
        self.Ts = 0.0
        self.Tr = 0.0
        self.lat = lat
        self.lng = lng
        self.tlat = lat
        self.tlng = lng
        self.K = K
        self.n = 0
        self.jobs = deque([])
        self.S = S
        
    def get_location(self):
        return self.lat, self.lng
    
    def get_target_location(self):
        return self.tlat, self.tlng
    
    def move_to_location(self, lat, lng):
        self.lat = lat
        self.lng = lng
        
    def is_idle(self):
        if len(self.jobs) == 0:
            return True
        elif len(self.jobs) == 1 and self.jobs[0][0] == -1 and self.jobs[0][1] == 0:
            return True
        else:
            return False
        
    def is_rebalancing(self):
        if len(self.jobs) == 0:
            return False
        elif len(self.jobs) == 1 and self.jobs[0][0] == -1 and self.jobs[0][1] == 0:
            return True
        else:
            return False
        
    def move_to_time(self, T):
        dT = T - self.T
        assert dT >= 0
        done = []
        while dT > 0 and len(self.jobs) > 0:
            j = self.jobs[0]
            d = get_distance(self.lat, self.lng, j[2], j[3])
            t = d / self.S
            if t < dT:
                dT -= t
                self.T += t
                if j[0] == -1 and j[1] == 0:
                    self.Tr += t
                elif j[0] >= 0 and j[1] == 1 or j[1] == -1:
                    self.Ts += t
                self.n += j[1]
                self.move_to_location(j[2], j[3])
                done.append( (j[0], j[1], self.T) )
                self.jobs.popleft()
            else:
                pct = dT / t
                if j[0] == -1 and j[1] == 0:
                    self.Tr += dT
                elif j[0] >= 0 and j[1] == 1 or j[1] == -1:
                    self.Ts += dT
                lat_ = self.lat + pct * (j[2] - self.lat)
                lng_ = self.lng + pct * (j[3] - self.lng)
                self.move_to_location(lat_, lng_)
                break
        self.T = T
        return done 
    
    def draw(self):
        plt.plot(self.lat, self.lng, 'blue' if self.id == 0 else 'black', marker='o')
        slat = self.lat
        slng = self.lng
        for l in range(len(self.jobs)):
            elat = self.jobs[l][2]
            elng = self.jobs[l][3]
            plt.plot([slat, elat], [slng, elng],
                     'grey' if self.is_rebalancing() else 'black',
                     linestyle=':')
            slat = elat
            slng = elng
                        
    def __str__(self):
        str =  "veh %d at (%.7f, %.7f) when t = %.3f; occupancy = %d/%d" % (
            self.id, self.lat, self.lng, self.T, self.n, self.K)
        str +=  "\n  in which service time = %.3f; rebalancing time = %.3f" % (
            self.Ts, self.Tr)
        for j in self.jobs:
            str += "\n    %s: req %d at (%.7f, %.7f)" % ("pickup" if j[1] > 0 else "dropoff", j[0], j[2], j[3])
        return str
    

class Req(object):
    """ 
    Req is a class for requests
    Attributes:
        id: sequential unique id
        Tr: request time
        Tp: pickup time
        Td: dropoff time
        olat: origin latitude
        olng: origin longitude
        dlat: destination latitude
        dlng: destination longitude
    """
    def __init__(self, id, Tr, olat=0.115662, olng=51.374282, dlat=0.089282, dlng=51.350675):
        self.id = id
        self.Tr = Tr
        self.Tp = -1.0
        self.Td = -1.0
        self.olat = olat
        self.olng = olng
        self.dlat = dlat
        self.dlng = dlng
    
    def get_origin(self):
        return (self.olat, self.olng)
    
    def get_destination(self):
        return (self.dlat, self.dlng)
    
    def draw(self):
        plt.plot(self.olat, self.olng, 'red', marker='s')
        plt.plot(self.dlat, self.dlng, 'red', marker='*')
        plt.plot([self.olat, self.dlat], [self.olng, self.dlng], 'red', linestyle='--', dashes=(0.5,1.5))
    
    def __str__(self):
        str =  "req %d from (%.7f, %.7f) to (%.7f, %.7f) at t = %.3f \n" % (
            self.id, self.olat, self.olng, self.dlat, self.dlng, self.Tr)
        str +=  "  it was picked up at t = %.3f and dropped off at t = %.3f" % (
            self.Tp, self.Td) if self.Td != -1.0 else "  it was picked up at t = %.3f and is en route" % (
            self.Tp) if self.Tp != -1.0 else "  it's still waiting for pick-up"
        return str
    

class AMoD(object):
    """
    AMoD is the class for the AMoD system
    Attributes:
        T: system time at current state
        D: average arrival interval (sec)
        demand: demand matrix
        V: number of vehicles
        K: capacity of vehicles
        vehs: the list of vehicles
        N: number of requests
        reqs: the list of requests
        queue: requests in the queue
    """ 
    def __init__(self, D, demand, V=10, K=1):
        self.T = 0.0
        self.D = D
        self.demand = demand
        self.V = V
        self.K = K
        self.vehs = []
        for i in range(V):
            self.vehs.append( Veh(i, lat=np.random.rand(), lng=np.random.rand(), K=K) )
        self.N = 0
        self.reqs = []
        self.queue = deque([])
        
    def generate_request(self):
        dt = self.D * np.random.exponential()
        rand = np.random.rand()
        for d in self.demand:
            if d[4] > rand:
                req = Req(0 if self.N == 0 else self.reqs[-1].id+1,
                          dt if self.N == 0 else self.reqs[-1].Tr+dt,
                          d[1], d[0], d[3], d[2])
                break
        self.N += 1
        return req
        
    def generate_requests_to_time(self, T):
        if self.N == 0:
            req = self.generate_request()
            self.reqs.append(req)
        while self.reqs[-1].Tr <= T:
            req = self.generate_request()
            self.queue.append(self.reqs[-1])
            self.reqs.append(req)
        assert self.N == len(self.reqs)
        
    def dispatch_at_time(self, T):
        for v in self.vehs:
            done = v.move_to_time(T)
            for d in done:
                if d[1] == 1:
                    self.reqs[ d[0] ].Tp = d[2]
                elif d[1] == -1:
                    self.reqs[ d[0] ].Td = d[2]
        self.generate_requests_to_time(T)
        self.T = T
        self.assign()
        
    def assign(self):
        l = len(self.queue)
        for i in range(l):
            req = self.queue.popleft()
            if not self.insert_nearest(req):
                self.queue.append(req)
        
    def insert_nearest(self, req):
        d_ = np.inf
        v_ = None
        for v in self.vehs:
            if v.is_idle():
                d = get_distance(v.lat, v.lng, req.olat, req.olng)
                if d < d_:
                    d_ = d
                    v_ = v
        if v_ == None:
            return False
        else:
            v_.jobs.clear()
            v_.jobs.append( (req.id, 1, req.olat, req.olng) )
            v_.jobs.append( (req.id, -1, req.dlat, req.dlng) )
            v_.tlat = req.dlat
            v_.tlng = req.dlng
            return True  
        
    def rebalance(self, str, dnq):
        if str == "optimal":
            self.rebalance_optimal()
        elif str == "approx":
            self.rebalance_approx()
        elif str == "rl":
            self.rebalance_rl(dnq)
        
    def rebalance_optimal(self):
        N = 5
        D = np.zeros((N,N))
        for d in self.demand:
            D[int(d[0]*N)][int(d[1]*N)] += 1
        V = np.zeros((N,N))
        for v in self.vehs:
            if v.is_idle():
                V[int(v.lat*N)][int(v.lng*N)] += 1
        C = np.zeros((N,N,N,N))
        B = np.zeros((N,N,N,N))
        X_lb = np.zeros((N,N,N,N))
        X_ub = np.zeros((N,N,N,N))
        for i1, j1, i2, j2 in np.ndindex((N,N,N,N)):
            C[i1][j1][i2][j2] = np.sqrt( (i1-i2)**2 + (j1-j2)**2 )
            B[i1][j1][i2][j2] = D[i2][j2]
            X_ub[i1][j1][i2][j2] = V[i1][j1]
        A_ub = np.zeros((N,N,N,N,N,N))
        b_ub = np.zeros((N,N))
        A_eq = np.zeros((N,N,N,N,N,N))
        b_eq = np.zeros((N,N))
        for i, j in np.ndindex((N,N)):
            for i_, j_ in np.ndindex((N,N)):
                A_ub[i][j][i_][j_][i][j] = 1
                A_eq[i][j][i][j][i_][j_] = 1
            b_eq[i][j] = V[i][j]
            b_ub[i][j] = 1
        C = C.reshape(N*N*N*N)
        B = B.reshape(N*N*N*N)
        X_lb = X_lb.reshape(N*N*N*N)
        X_ub = X_ub.reshape(N*N*N*N)
        A_ub = A_ub.reshape((N*N,N*N*N*N))
        b_ub = b_ub.reshape(N*N)
        A_eq = A_eq.reshape((N*N,N*N*N*N))
        b_eq = b_eq.reshape(N*N)
        res = linprog(C-B, A_eq=A_eq, b_eq=b_eq, A_ub=A_ub, b_ub=b_ub,
                      bounds=(np.transpose([X_lb, X_ub])), options={"maxiter": 5000, "disp": True})
        if res.success:
            x = res.x.reshape((N,N,N,N))
            for v in self.vehs:
                if v.is_idle():
                    i = int(v.lat*N)
                    j = int(v.lng*N)
                    (i_, j_) = unravel_index(x[i][j].argmax(), x[i][j].shape)
                    v.tlat = (i_+np.random.rand())/N
                    v.tlng = (j_+np.random.rand())/N
                    v.jobs.clear()
                    v.jobs.append( (-1, 0, v.tlat, v.tlng) )
                    x[i][j][i_][j_] -= 1
            assert np.max(x) == 0               
        
    def rebalance_approx(self):
        N = 5
        B = np.zeros((N,N))
        for d in self.demand:
            B[int(d[0]*N)][int(d[1]*N)] += 1
        for v in self.vehs:
            if v.is_idle():
                C = np.zeros((N,N))
                for i, j in np.ndindex((N,N)):
                    C[i][j] = np.power((int(v.lat*N)-i)**2 + (int(v.lng*N)-j)**2, 1/2)
                CB = C - B
                (i, j) = unravel_index(CB.argmin(), CB.shape)
#                 B[i][j] = 0 if B[i][j] < 1 else np.log(B[i][j])
                v.tlat = (i+np.random.rand())/N
                v.tlng = (j+np.random.rand())/N
                v.jobs.clear()
                v.jobs.append( (-1, 0, v.tlat, v.tlng) ) 
    
    def rebalance_rl(self, dqn):
        N = 5
        c = 0.1
        for v in self.vehs:
            if v.is_idle():
                v.jobs.clear()
                state = self.get_state(v, N, c)
                action = dqn.forward(state)
                self.act(v, action, c)
                    
    def get_state(self, v, N, c):
        state = np.zeros((2,N,N))
        lat, lng = v.get_location()
        for d in self.demand:
            for i, j in np.ndindex((N,N)):
                if d[0] > lat+(i-N/2)*c and d[0] < lat+(i+1-N/2)*c and d[1] > lng+(j-N/2)*c and d[1] < lng+(j+1-N/2)*c:
                    state[0][i][j] += 1
                    break
        for v_ in self.vehs:
            for i, j in np.ndindex((N,N)):
                 if v_.is_idle():
                    lat_, lng_ = v_.get_location()
                    if lat_ > lat+(i-N/2)*c and lat_ < lat+(i+1-N/2)*c and lng_ > lng+(j-N/2)*c and lng_ < lng+(j+1-N/2)*c:
                        state[1][i][j] += 1
                        break
        return state
    
    def act(self, v, action, c):
        if action == 0:
            return
        if action == 1:
            v.tlat = min(v.lat + c, 1)
            v.tlng = v.lng
            v.jobs.append( (-1, 0, v.tlat, v.tlng) )
        elif action == 2:
            v.tlat = max(v.lat - c, 0)
            v.tlng = v.lng
            v.jobs.append( (-1, 0, v.tlat, v.tlng) )
        elif action == 3:
            v.tlat = v.lat
            v.tlng = min(v.lng + c, 1)
            v.jobs.append( (-1, 0, v.tlat, v.tlng) )
        elif action == 4:
            v.tlat = v.lat
            v.tlng = max(v.lng - c, 0)
            v.jobs.append( (-1, 0, v.tlat, v.tlng) )
    
    def draw(self):
        plt.figure(figsize=(8,8))
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        for v in self.vehs:
            v.draw()
        for r in self.queue:
            r.draw()
        plt.show()
        
    def __str__(self):
        str = "AMoD system: %d vehicles of capacity %d; %.1f trips/h" % (self.V, self.K, 3600/self.D)
        str += "\n  at t = %.3f, %d requests, in which %d in queue" % ( self.T, self.N-1, len(self.queue) )
        for r in self.queue:
            str += "\n    " + r.__str__()
        return str
    
def draw(amods):
    def init():
        for i, veh in enumerate(vehs):
            veh.set_data(amods[0].vehs[i].lat, amods[0].vehs[i].lng)
        return routes, vehs,

    def animate(n):
        for i, veh in enumerate(vehs):
            veh.set_data(amods[n].vehs[i].lat, amods[n].vehs[i].lng)
        for i, route in enumerate(routes):
            lats = [amods[n].vehs[i].lat]
            lngs = [amods[n].vehs[i].lng]
            for j in amods[n].vehs[i].jobs:
                lats.append(j[2])
                lngs.append(j[3])
            route.set_data(lats, lngs)
            route.set_color('blue' if i == 0 and amods[n].vehs[i].is_rebalancing()
                            else 'grey' if amods[n].vehs[i].is_rebalancing() else 'black')
        return routes, vehs,
    
    fig = plt.figure(figsize=(8,8))
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    routes = []
    vehs = []
    for v in amods[0].vehs:
        routes.append( plt.plot([], [], 'black', linestyle=':', linewidth=1)[0] )
        vehs.append( plt.plot([], [], 'blue' if v.id == 0 else 'black', marker='o')[0] )
    anim = animation.FuncAnimation(fig, animate, init_func=init, frames=len(amods), interval=40)
    return anim