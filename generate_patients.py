import numpy as np, matplotlib.pyplot as plt

import scipy.interpolate as ip
from scipy import integrate
import matplotlib.pyplot as plt
# import torch
from matplotlib import cm

from sklearn.metrics import mean_absolute_error as MAE
from environment import *
from replay_buffer import *
import time
from replay_buffer import *
import pandas as pd
from util import *
from tqdm import trange


if __name__ == '__main__':
    env = environment()

    jc_scale=6
    # create buffer for training patients
    tcga_all = pd.read_csv('data/tcga_all_train.csv')
    num_seed_train = 200
    cell_dim = 3

    save_path_train = './buffer/Train_buffer'
    init_patient_buffer = Buffer_init(cell_dim, num_seed_train)

    # # # store the information of training patients
    for i in trange(num_seed_train):
        mu = tcga_all.loc[i, 'mu']
        r = tcga_all.loc[i, 'r']
        jc = tcga_all.loc[i, 'mhc_norm']
        jc_ = (1 - np.exp(-jc * jc_scale)) / (1 + np.exp(-jc * jc_scale))
        R_L_prop = tcga_all.loc[i, 'Treg_pro']
        tumor = tcga_all.loc[i, 'tumor_size']
        ar_ratio = tcga_all.loc[i, 'ar_ratio']
        a = tcga_all.loc[i, 'a']
        t1 = int(tcga_all.loc[i, 'time'])
        x_ = env.run (mu, r, jc_, R_L_prop,a,ar_ratio,t1)
        j = cal_time(x_, tumor)
        cells=x_[j]
        t1=int(j/24)
        cells[2]=(cells[1]+cells[0])* tcga_all.loc[i, 'tcd8_pop']
        init_patient_buffer.add(cells,  r, jc, mu, R_L_prop,a,ar_ratio,t1)
    init_patient_buffer.save(save_path_train)


    # create buffer for testing patients
    tcga_all = pd.read_csv('data/tcga_all_test.csv')
    num_seed_test = 30


    save_path_train = './buffer/Test_buffer'
    num_rand=30
    init_patient_buffer = Buffer_init(cell_dim, num_seed_test*num_rand)

    # store the information of testing patients
    for i in trange(num_seed_test):

        mu = tcga_all.loc[i, 'mu']
        r = tcga_all.loc[i, 'r']
        jc = tcga_all.loc[i, 'mhc_norm']
        jc_ = (1 - np.exp(-jc * jc_scale)) / (1 + np.exp(-jc * jc_scale))
        R_L_prop = tcga_all.loc[i, 'Treg_pro']
        tumor = tcga_all.loc[i, 'tumor_size']
        ar_ratio = tcga_all.loc[i, 'ar_ratio']
        a = tcga_all.loc[i, 'a']
        t1 = int(tcga_all.loc[i, 'time'])
        x_ = env.run (mu, r, jc_, R_L_prop,a,ar_ratio,t1)
        j = cal_time(x_, tumor)
        cells=x_[j]
        t1=int(j/24)
        cells[2]=(cells[1]+cells[0])* tcga_all.loc[i, 'tcd8_pop']
        init_patient_buffer.add(cells,  r, jc, mu, R_L_prop,a,ar_ratio,t1)
    init_patient_buffer.save(save_path_train)




