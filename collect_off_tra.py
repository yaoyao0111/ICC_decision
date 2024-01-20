from replay_buffer import *
import pandas as pd
from util import *
from tqdm import trange


if __name__ == '__main__':
    env = environment()

    jc_scale = 6

    num_seed_train = 200
    cell_dim = 3

    save_path_train = './buffer/Train_buffer'
    init_patient_buffer = Buffer_init(cell_dim, num_seed_train)
    init_patient_buffer.load(save_path_train)

    # create init training data
    replay_buffer = ReplayBuffer_off(6, 2)
    test_u_l = [(1, 1), (0, 1), (1, 0), (0, 2), (1, 2), (0, 5), (1, 5)]
    save_path_init = './buffer/Train_buffer_init'

    # generate trajectories of fixed ICC schedules and store them to warm start the training of DDPG agent.
    for  u1_,u2_ in test_u_l:
        print(u1_,u2_)
        for i in trange(100,200):
            cell, r_, j1, mu,R_L_prop, a ,ar_ratio,t1= init_patient_buffer.get_(i)
            state = {}
            state['cell'] = np.append(cell, [0, 0])
            state['mu'] = mu
            state['grow_rate'] = a
            state['r'] = r_
            env_s = env_step(mu, r_, j1, R_L_prop,a,ar_ratio, sum(state['cell'][:2]))
            done = False
            reward_=0
            while not done:
                s = env.featurize(state)
                next_state, r, done = env_s.step(state, u1_, u2_)
                state=next_state
                s_ = env.featurize(next_state)
                if u1_==0:
                    a=[-1,u2_-2.5]
                else:
                    a=[1,u2_-2.5]
                reward_+=r
                replay_buffer.store(s, a, r, s_, done)
    replay_buffer.save(save_path_init)