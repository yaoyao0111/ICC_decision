from environment import *
from replay_buffer import *

from sklearn.metrics import mean_absolute_error as MAE
from replay_buffer import *
import pandas as pd
from util import *
from tqdm import trange

from ddpg import *
def cal_reward(state,env_s,ici,chemo):
    rand_num=1
    rand_re_l=[]
    for rand in range(rand_num):
        env_s.re_set()
        done = False
        re = []
        while not done:
            next_state, reward, done = env_s.step(state, ici, chemo)
            state = next_state
            re.append(reward)
        rand_re_l.append(sum(re))
    return np.mean(rand_re_l)

def cal_reward_agent(state,env_s,agent):
    rand_num=1
    rand_re_l=[]
    u1=[]
    u2=[]
    for rand in range(rand_num):
        env_s.re_set()

        done = False
        re = []
        while not done:
            s = env.featurize(state)
            a = agent.choose_action(s)
            if a[0] > 0:
                ici = 1
            else:
                ici = 0
            chemo = a[1] + 2.5

            u1.append(ici)
            u2.append(chemo)

            next_state, reward, done = env_s.step(state, ici, chemo)
            state = next_state
            re.append(reward)
        print(u1)
        print(u2)
        rand_re_l.append(sum(re))
    return np.mean(rand_re_l)

if __name__ == '__main__':
    num_seed = 30
    cell_dim = 3
    action_dim = 2
    state_dim = 6
    max_action = 2.5
    num_test_patient=30
    save_path = './buffer/Test_buffer'
    init_patient_buffer = Buffer_init(cell_dim, num_seed*num_test_patient)
    agent = torch.load('./DRL_agent.pth')

    env = environment()
    # #
    u_ = []
    reward = []
    max_action = 2.5
    noise_std = 0.1 * max_action
    replay_buffer = ReplayBuffer_off(6, 2)

    init_patient_buffer.load(save_path)
    reward_l = {}
    test_u_l = [(1, 0), (0, 1), (1, 1), (0, 2), (1, 2),(0, 3), (1, 3), (0, 4), (1, 4),(0, 5), (1, 5)]

    jc_scale= 6

    for ici, chemo in test_u_l:
        key_name = str(ici) + '_' + str(chemo)
        reward_l[key_name] = []
    reward_l['agent'] = []
    for patient_id in trange(30):
    # for patient_id in [6,23,21]:
        cell, r, jc, mu, R_L_prop, a, ar_ratio, t1 = init_patient_buffer.get_(patient_id)
        print(cell, r, jc, mu, R_L_prop, a, ar_ratio, t1 )
        state = {}
        state['cell'] = np.append(cell, [0, 0])
        state['mu'] = mu
        state['r'] = r
        jc_ = (1 - np.exp(-jc * jc_scale)) / (1 + np.exp(-jc * jc_scale))
        state['jc'] = jc
        state['R_L_prop'] = R_L_prop

        env_s = env_step(mu, r, jc_, R_L_prop, a, ar_ratio, sum(state['cell'][:2]))

        re = cal_reward_agent(state, env_s, agent)
        reward_l['agent'].append(re)
        print('agent', re)

        for ici, chemo in test_u_l:
            re = cal_reward(state, env_s, ici, chemo)
            key_name = str(ici) + '_' + str(chemo)
            reward_l[key_name].append(re)
            print(key_name,re)
    reward_=pd.DataFrame(reward_l)
    reward_.to_csv('data/reward_agent.csv')
    for key in reward_l.keys():
        re_l = reward_l[key]
        print(key, np.mean(re_l))