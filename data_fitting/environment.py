import numpy as np
import math
import random
import copy
from scipy import integrate

class environment():
    def __init__(self):
        self.b = 3.846e-9
        self.ml = 0.05

        self.L_max = 1e7

        # tumor's increase effect on Treg
        

        self.u =1e5 # tumor will make the Treg accumulation when it is larger than mr*u/(f-mr)

        # Treg's decrease effect on CD8+T cell
        
        # MHC's effects on active CD8+T proliferation
        self.k1 =1e4#

        # CD8+T killing effect
        self.d = 2.34  # 1e9 CD8+T can kill 5cm tumor
        self.l = 2
        self.s = 8.39e-2

        # CD8+T's lysed tumor's recruitment
#         self.jl = 0.1 # r*(jc+jl)>ml+h

        self.k2 =1e4   #

        self.gamma_i = 9* 10 ** (-1)

        self.e = math.e
        self.jl=0

    def fit_pre_treatment(self, mu, r, jc, R_L_prop, h, q,a,ar_ratio,t1):
        self.a=a
        self.ar=ar_ratio*a
        self.mu = mu
        self.r = r
        self.jc = jc
        
        self.t1=t1
        self.h = h
        self.q = q
        self.R_L_prop = R_L_prop
        # initial value
        x_ = self.gen_state()

        return x_

    def state_eq(self, t, x):
        Ts = x[0]
        Tr = x[1]
        L = x[2]
        T = Ts + Tr
        R_L_prop=self.R_L_prop
        
        Dl = self.d * (L * (1 - self.mu) / T) ** self.l / (self.s + (L * (1 - self.mu) / T) ** self.l)

        dts = max(self.a * Ts * (1 - self.b * (Ts + 0.7 * Tr)) - 1, 0) - max(Dl * Ts - 1, 0)

        dtr = max(self.ar * Tr * (1 - self.b * (0.9 * Ts + Tr)) - 1, 0) - max(Dl * Tr - 1, 0)

        dl = -max(self.ml * L - 1, 0) - max(self.q * T * L / (self.u + T) - 1, 0) + max(
            self.r * L * (1 - L / self.L_max) * (
                        self.jc*T / (self.k1 +  T) + self.jl * Dl * T / (self.k2 + Dl * T)) - 1, 0) - max(
            self.h*R_L_prop*L - 1, 0)
        return np.array([dts, dtr, dl])
    def update_fitted_para(self,h,q):
        self.h = h
        self.q = q
    def update_fitted_jl(self,jl):
        self.jl = jl
        
    def run(self,mu,r,jc,R_L_prop,t1,a,ar_ratio):

        self.mu=mu
        self.r = r
        self.jc = jc
        self.a=a
        self.ar=a*ar_ratio
        self.R_L_prop=R_L_prop
        self.t1=t1
         # initial value
        x_ = self.gen_state()
        
        return x_
    
    def gen_state(self):

        t0, t1 =0,self.t1    # start and end
        interval=t1*24+1
        t = np.linspace(t0, t1,interval)  # the points of evaluation of solution
        x0 = [5e4, 4e1, 1e2]    # initial value
        x = np.zeros((len(t), len(x0)))   # array for solution
        x[0, :] = x0

        r = integrate.ode(self.state_eq).set_integrator("dopri5")
        r.set_initial_value(x0, t0)
        for i in range(1, t.size):
            x[i, :] = r.integrate(t[i])
        return x
    
    def state_eq_ici(self, t, x):
        Ts = x[0]
        Tr = x[1]
        L = x[2]
        I = x[3]
        T = Ts + Tr
        R_L_prop=self.R_L_prop
        mu = self.mu*(1-I)
        
        Dl = self.d * (L * (1 - mu) / T) ** self.l / (self.s + (L * (1 - mu) / T) ** self.l)

        dts = max(self.a * Ts * (1 - self.b * (Ts + 0.7 * Tr)) - 1, 0) - max(Dl * Ts - 1, 0)

        dtr = max(self.ar * Tr * (1 - self.b * (0.9 * Ts + Tr)) - 1, 0) - max(Dl * Tr - 1, 0)

        dl = -max(self.ml * L - 1, 0) - max(self.q * T * L / (self.u + T) - 1, 0) + max(self.r * L * (1 - L / self.L_max) * (self.jc*T / (self.k1 +  T) + self.jl * Dl * T / (self.k2 + Dl * T)) - 1, 0) - max(self.h*R_L_prop*L - 1, 0)

        di = -max(self.gamma_i * I - 1e-10, 0)
        
        return np.array([dts, dtr, dl,di])
    
    def step(self, x_begin,mu,r,jc,R_L_prop,jl):
        
        self.mu=mu
        self.r = r
        self.jc = jc
        self.jl=jl

        self.R_L_prop=R_L_prop
        self.t1=400
        
        r_=integrate.ode(self.state_eq_ici).set_integrator("dopri5", max_step=1/20,nsteps=1e5)
        r_.set_initial_value(x_begin,0)
        
        x=r_.integrate(21)

        return x




