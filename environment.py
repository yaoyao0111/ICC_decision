import numpy as np
import math
import random
import copy
from scipy import integrate

# environment simulator of TIME formation without treatment intervension.
class environment():
    def __init__(self):
        self.b = 3.846e-9

        self.ml = 0.05

        self.L_max = 1e7

        self.u = 1e5
        self.k1 = 1e4

        # CD8+T killing effect
        self.d = 2.34  # 1e9 CD8+T can kill 5cm tumor
        self.l = 2
        self.s = 8.39e-2

        self.k2 = 1e4  #

        self.e = math.e

        self.h = 0.01
        self.q = 0.03
        self.j2 = 0.11

    def state_eq(self, t, x):
        Ts = x[0]
        Tr = x[1]
        L = x[2]
        T = Ts + Tr
        R_L_prop = self.R_L_prop

        Dl = self.d * (L * (1 - self.mu) / T) ** self.l / (self.s + (L * (1 - self.mu) / T) ** self.l)

        dts = max(self.a * Ts * (1 - self.b * (Ts + 0.7 * Tr)) - 1, 0) - max(Dl * Ts - 1, 0)

        dtr = max(self.ar * Tr * (1 - self.b * (0.9 * Ts + Tr)) - 1, 0) - max(Dl * Tr - 1, 0)

        dl = -max(self.ml * L - 1, 0) - max(self.q * T * L / (self.u + T) - 1, 0) + max(
            self.r * L * (1 - L / self.L_max) * (
                    self.j1 * T / (self.k1 + T) + self.j2 * Dl * T / (self.k2 + Dl * T)) - 1, 0) - max(
            self.h * R_L_prop * L - 1, 0)

        return np.array([dts, dtr, dl])


    def run(self, mu, r, jc, R_L_prop,a,ar_ratio,t1):
        self.mu = mu
        self.r = r
        self.j1 = jc
        self.R_L_prop = R_L_prop
        self.a= a
        self.ar = a * ar_ratio
        self.t1= t1

        x_ = self.gen_state()

        return x_

    def gen_state(self):
        t0, t1 = 0, self.t1  # start and end
        interval = t1 * 24 + 1
        t = np.linspace(t0, t1, interval)
        x0 = [5e4, 4e1, 1e2]
        x = np.zeros((len(t), len(x0)))
        x[0, :] = x0

        r = integrate.ode(self.state_eq).set_integrator("dopri5")
        r.set_initial_value(x0, t0)
        for i in range(1, t.size):
            x[i, :] = r.integrate(t[i])
        return x

    def featurize(self, state):
        cell_l = state['cell'][:3]
        state_fea = np.log(cell_l).tolist()
        state_fea.append(cell_l[2]*1e3/(cell_l[1]+cell_l[0]))

        try:
            state_fea.append(state['mu'][0])
            state_fea.append(state['r'][0])
        except:
            state_fea.append(state['mu'])
            state_fea.append(state['r'])
        return state_fea


# environment simulator of ICC treatment
class env_step():
    def __init__(self,mu,r,jc,R_L_prop,a,ar_ratio,begin_tumor):
        self.a = a
        self.ar = self.a * ar_ratio
        self.b = 3.846e-9

        self.ml = 0.05

        self.L_max = 1e7

        self.u = 1e5
        self.k1 = 1e4

        # CD8+T killing effect
        self.d = 2.34  #
        self.l = 2
        self.s = 8.39e-2

        self.k2 = 1e4  #
        self.k3 = 1e4
        self.j2= 0.11
        self.j3= 0.033

        self.e = math.e

        # cytotoxic drugs' killing effect
        self.Kt = 6e-1
        self.Kl = 6e-1

        self.gamma_i = 9 * 10 ** (-1)
        self.gamma_m = 9 * 10 ** (-1)
        self.q=0.03
        self.h=0.01

        self.e = math.e
        self.p = 0.2
        self.interval = 21 * 24 + 1
        self.week_num=4
        self.clock=1
        self. r_ = integrate.ode(self.state_eq_icc).set_integrator("dopri5", max_step=1/20,nsteps=1e5)
        self.begin_tumor=begin_tumor
        self.ici_cost=0
        self.mu = mu
        self.r = r
        self.j1 = jc
        self.R_L_prop = R_L_prop

    def re_set(self):
        self.ici_cost = 0
        self.clock = 1

    def step(self, state, u1, u2):
        next_state = copy.deepcopy(state)
        x_begin = state['cell']

        x_begin[3]=u1
        x_begin[4]=u2
        self. r_.set_initial_value(x_begin,0)
        x=self.r_.integrate(21)
        x_end = x
        tumor_begin=x_begin[0]+x_begin[1]
        tumor_end=x_end[0]+x_end[1]
        tumor_reduction = (tumor_begin-tumor_end) / self.begin_tumor

        if u1>0:
            cost=0.04
            self.ici_cost+=0.04
        else:
            cost=0
            self.ici_cost += 0
        if self.clock >=self.week_num or tumor_end<1e2 or tumor_end>2.6e8:
            done = 1
            if tumor_end<1e2:
                goal_reached= 1+(4-self.clock)*0.1

            elif tumor_end>2.6e8:
                goal_reached=-1-self.ici_cost
            else:
                goal_reached,x= self.cal_six_cycles(x_end,u1)
                # next_state['six_cycles_cell']=x
        else:
            done = 0
            next_state['six_cycles_cell']=0
            goal_reached = 0

        self.clock+=1

        reward = tumor_reduction+ goal_reached
        next_state['cell'] = x_end

        return next_state, reward, done



    def cal_six_cycles(self,x_begin,u1):
        tumor_begin = x_begin[0] + x_begin[1]
        cost=0
        if u1==0:
            self.r_.set_initial_value(x_begin, 0)
            x = self.r_.integrate(126)
        else:
            for i in range(6):
                cost += 0.04
                self.ici_cost += 0.04
                x_begin[3] = 1
                self.r_.set_initial_value(x_begin, 0)
                x = self.r_.integrate(21)
                x_begin=x
                if x[0] + x[1]<1e2:
                    break
        x_end=x
        tumor_end = x_end[0] + x_end[1]
        tumor_reduction = (tumor_begin - tumor_end) / self.begin_tumor
        if tumor_end<1e2:

            tumor_reduction+= 1
        elif tumor_end>2.6e8:
            tumor_reduction += -1-self.ici_cost
        else:
            tumor_reduction -=self.ici_cost
        tumor_reduction=max(tumor_reduction, -1)
        return tumor_reduction,x_end

    def step_tra(self,t1, cell, u1, u2):
        x_begin = cell
        x_begin[3] = u1
        x_begin[4] = u2

        t0 = 0
        self.interval =t1* 24 + 1
        t = np.linspace(t0, t1, self.interval)
        x = np.zeros((len(t), len(x_begin)))
        x[0, :] = x_begin
        r = integrate.ode(self.state_eq_icc).set_integrator("dopri5")
        r.set_initial_value(x_begin, t0)
        for i in range(1, t.size):
            x[i, :] = r.integrate(t[i])
        return t, x

    def state_eq_icc(self, t,x):
        Ts = x[0]
        Tr = x[1]
        L = x[2]
        I = x[3]
        M = x[4]
        T = Ts + Tr
        R_L_prop=self.R_L_prop
        mu = min(self.mu * (1 + self.p * (1 - self.e ** (-M))) * (1 - I ), 1)
        Dl = self.d * (L * (1 - mu) / T) ** self.l / (self.s + (L * (1 - mu) / T) ** self.l)
        Dt = self.Kt * (1 - self.e ** (-M))
        dts = max(self.a * Ts * (1 - self.b * (Ts + 0.7 * Tr)) - 1, 0) - max(Dl * Ts - 1, 0) - max(Dt * Ts - 1, 0)

        dtr = max(self.ar * Tr * (1 - self.b * (0.9 * Ts + Tr)) - 1, 0) - max(Dl * Tr - 1, 0)

        dl = -max(self.ml * L - 1, 0) - max(self.q * T * L / (self.u + T) - 1, 0) + max(self.r * L * (1 - L / self.L_max) * (self.j1 * T / (self.k1 + T) + self.j2 * Dl * T / (self.k2 + Dl * T) + self.j3 * Dt * Ts / (self.k3 + Dt * Ts)) - 1, 0) - max(self.h * R_L_prop * L - 1, 0) - max(self.Kl * (1 - self.e ** (-M)) * L - 1, 0)
        di = -max(self.gamma_i * I - 1e-10, 0)
        dm = -max(self.gamma_m * M - 1e-10, 0)

        return np.array([dts, dtr, dl, di, dm])





