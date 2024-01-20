from environment import *
import pandas as pd
import math
import os
import matplotlib.pyplot as plt

from ddpg import *
from tqdm import tqdm
from ddpg import *
from replay_buffer import *
import os
import time

def tournament(env,agent,i):
    jc_scale = 6
    cell, r, jc, mu, R_L_prop, a,ar_ratio,t1 = init_patient_buffer.get_(i)
    state = {}
    state['cell'] = np.append(cell, [0, 0])
    state['mu'] = mu
    state['r'] = r
    jc_ = (1 - np.exp(-jc * jc_scale)) / (1 + np.exp(-jc * jc_scale))
    state['jc'] = jc
    state['R_L_prop'] = R_L_prop
    env_s = env_step(mu, r, jc_, R_L_prop,a, ar_ratio, sum(state['cell'][:2]))
    done = False
    state_record = []
    re = []
    u1 = []
    u2 = []
    while not done:
        s = env.featurize(state)
        a = agent.choose_action(s)
        if a[0] > 0:
            ici = 1
        else:
            ici = 0
        u1.append(ici)
        chemo = a[1] + 2.5
        u2.append(chemo)
        next_state, reward, done = env_s.step(state, ici, chemo)
        state = next_state
        re.append(reward)
    print(u1)
    print(u2)
    return 0
if __name__ == '__main__':
    env = environment()
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device(
        "cpu")

    state_dim=6
    action_dim=2
    epsilon=0.2
    entropy_coef=0.01
    max_action=2.5
    num_seed = 1000
    cell_dim = 3
    save_path = './buffer/Train_buffer'
    init_patient_buffer = Buffer_init(cell_dim, num_seed)
    init_patient_buffer.load(save_path)
    # off_relay_buffer
    replay_buffer = ReplayBuffer_off(6, 2)
    save_path = './buffer/Train_buffer_init'
    replay_buffer.load(save_path, 2680)
    agent = DDPG(state_dim, action_dim, max_action)
    # agent_ = torch.load('./final.pth')
    # agent=copy.deepcopy(agent_)
    noise_std = 0.1 * max_action  # the std of Gaussian noise for exploration
    max_train_epochs = 2e3  # Maximum number of training steps

    update_freq = 10
    evaluate_freq = 1e2
    evaluate_num = 0
    evaluate_rewards = []  # Record the rewards during the evaluating
    total_epochs = 0
    return_list=[]
    max_reward=0
    jc_scale=6
    for _ in range(update_freq*50):
        agent.learn(replay_buffer)
    while total_epochs < max_train_epochs:

        for i in tqdm(range(200)):
            cell, r_, jc, mu, R_L_prop,a, ar_ratio,t1 = init_patient_buffer.get_(i)
            state = {}
            state['cell'] = np.append(cell, [0, 0])
            state['mu'] = mu
            state['r'] = r_
            jc_ = (1 - np.exp(-jc * jc_scale)) / (1 + np.exp(-jc * jc_scale))
            state['jc'] = jc
            state['R_L_prop'] = R_L_prop
            env_s = env_step(mu, r_, jc_, R_L_prop, a,ar_ratio, sum(state['cell'][:2]))
            done = False
            reward_sum=0
            total_epochs += 1
            while not done:
                s = env.featurize(state)
                a = agent.choose_action(s)
                a = (a + np.random.normal(0, noise_std, size=action_dim)).clip(-max_action, max_action)
                if a[0] > 0:
                    ici = 1
                else:
                    ici = 0
                chemo=a[1]+2.5
                next_state, r, done = env_s.step(state, ici,chemo)

                s_ = env.featurize(next_state)
                replay_buffer.store(s, a, r, s_, done)  # Store the transition
                reward_sum+=r
                state=next_state
            return_list.append(reward_sum)

            if total_epochs%50==0:
                for ind in range(5):
                    tournament(env, agent,ind)
                print(np.mean(return_list[-50:]))
                if np.mean(return_list[-50:]) > max_reward:
                    torch.save(agent, './high.pth')
                    max_reward = np.mean(return_list[-50:])
                # replay_buffer.save(save_path)
                torch.save(agent, './final.pth')

            if total_epochs % update_freq == 0:
                for _ in range(update_freq*5):
                    agent.learn(replay_buffer)
