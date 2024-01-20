import random
import torch
import numpy as np

from collections import deque, namedtuple

Transition = namedtuple('Transition', ('states','actions','a_logprob','rewards','next_states','dones'))

class Buffer_init(object):
    def __init__(self,cell_dim,size):
        self.ptr=0
        self.cell_dim=cell_dim
        self.size=size
        self.cell = np.zeros(( self.size,cell_dim))

        self.r = np.zeros((self.size, 1))
        self.jc = np.zeros((self.size, 1))
        self.mu = np.zeros((self.size, 1))
        self.R_L_prop= np.zeros((self.size, 1))
        self.a = np.zeros((self.size, 1))
        self.ar_ratio = np.zeros((self.size, 1))
        self.t1 = np.zeros((self.size, 1))

    def add(self, cell, r, jc, mu, R_L_prop,a,ar_ratio,t1):
        self.cell[self.ptr] = cell
        self.r[self.ptr] = r
        self.jc[self.ptr] = jc
        self.mu[self.ptr] = mu
        self.R_L_prop[self.ptr] = R_L_prop
        self.a[self.ptr] = a
        self.ar_ratio[self.ptr] = ar_ratio
        self.t1[self.ptr] = t1

        self.ptr = (self.ptr + 1) % self.size
    def get_(self,ind):
        return (
            self.cell[ind],
            self.r[ind] ,
            self.jc[ind] ,
            self.mu[ind] ,
            self.R_L_prop[ind],
            self.a[ind],
            self.ar_ratio[ind],
            self.t1[ind]
        )
    def sample(self):
        ind = np.random.randint(0, self.size)
        return (
            self.cell[ind],
            self.r[ind],
            self.jc[ind],
            self.mu[ind],
            self.R_L_prop[ind],
            self.a[ind],
            self.ar_ratio[ind],
            self.t1[ind]
        )

    def save(self, save_folder):
        np.save(f"{save_folder}_cell.npy", self.cell)
        np.save(f"{save_folder}_r.npy", self.r)
        np.save(f"{save_folder}_jc.npy",self.jc)
        np.save(f"{save_folder}_mu.npy", self.mu)
        np.save(f"{save_folder}_R_L_prop.npy", self.R_L_prop)
        np.save(f"{save_folder}_a.npy", self.a)
        np.save(f"{save_folder}_ar_ratio.npy", self.ar_ratio)
        np.save(f"{save_folder}_t1.npy", self.t1)

    def load(self, save_folder):
        self.ind_last = 1
        self.cell = np.load(f"{save_folder}_cell.npy")
        self.r = np.load(f"{save_folder}_r.npy")
        self.jc = np.load(f"{save_folder}_jc.npy")
        self.mu = np.load(f"{save_folder}_mu.npy")
        self.R_L_prop = np.load(f"{save_folder}_R_L_prop.npy")
        self.a = np.load(f"{save_folder}_a.npy")
        self.ar_ratio = np.load(f"{save_folder}_ar_ratio.npy")
        self.t1 = np.load(f"{save_folder}_t1.npy")
        self.size = len(self.r)
        print(f"Replay Buffer loaded with {self.size} elements.")

class ReplayBuffer_off(object):
    def __init__(self, state_dim, action_dim):
        self.max_size = int(1e5)
        self.size = 0
        self.s = np.zeros((self.max_size, state_dim))
        self.a = np.zeros((self.max_size, action_dim))
        self.r = np.zeros((self.max_size, 1))
        self.s_ = np.zeros((self.max_size, state_dim))
        self.dw = np.zeros((self.max_size, 1))


    def store(self, s, a, r, s_, dw):
        self.s[self.size] = s
        self.a[self.size] = a
        self.r[self.size] = r
        self.s_[self.size] = s_
        self.dw[self.size] = dw
        self.size =self.size + 1 # Record the number of  transitions

    def sample(self, batch_size):
        index = np.random.choice(self.size, size=batch_size)  # Randomly sampling
        batch_s = torch.tensor(self.s[index], dtype=torch.float)
        batch_a = torch.tensor(self.a[index], dtype=torch.float)
        batch_r = torch.tensor(self.r[index], dtype=torch.float)
        batch_s_ = torch.tensor(self.s_[index], dtype=torch.float)
        batch_dw = torch.tensor(self.dw[index], dtype=torch.float)
        return batch_s, batch_a, batch_r, batch_s_, batch_dw

    def save(self, save_folder):
        np.save(f"{save_folder}_s.npy", self.s)
        np.save(f"{save_folder}_a.npy",self.a )
        np.save(f"{save_folder}_r.npy", self.r)
        np.save(f"{save_folder}_s_.npy",self.s_)
        np.save(f"{save_folder}_dw.npy", self.dw)
        print(f"Replay Buffer saved with {self.size} elements.")

    def delete(self,ind_from,ind_to):
        self.s =np.delete(self.s, np.arange(ind_from, ind_to), axis=0)
        self.a= np.delete(self.a, np.arange(ind_from, ind_to), axis=0)
        self.r = np.delete(self.r, np.arange(ind_from, ind_to), axis=0)
        self.s_ = np.delete(self.s_, np.arange(ind_from, ind_to), axis=0)
        self.dw= np.delete(self.dw, np.arange(ind_from, ind_to), axis=0)
        if self.size>ind_to:
            self.size-=(ind_to-ind_from)
        print(f"Replay Buffer has {self.size} elements after delete.")

    def load(self, save_folder,size):
        self.s = np.load(f"{save_folder}_s.npy")
        self.a= np.load(f"{save_folder}_a.npy")
        self.r = np.load(f"{save_folder}_r.npy")
        self.s_ = np.load(f"{save_folder}_s_.npy")
        self.dw = np.load(f"{save_folder}_dw.npy")

        self.size =size
        print(f"Replay Buffer loaded with {self.size} elements.")

