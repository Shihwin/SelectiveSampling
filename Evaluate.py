import os
import glob
import numpy as np


import torch
# args
from utils.SACArgument import get_args
# Log
from tensorboardX import SummaryWriter
from tqdm import tqdm
# Env
import gym
# Buffer
# Agent
from agents.SACAgent import SACAgent


args = get_args()
device = torch.device("cuda:0")
env = gym.make(args.env_id)
log_dir = './results/Ant-v2/SAC/'

obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.shape[0]
act_limit = env.action_space.high[0]

agent = SACAgent(obs_dim, act_dim, act_limit, args, device)
agent.model.load_state_dict(torch.load(os.path.join(log_dir, 'sac_model.dump')))

test_episodes = 10

for i in range(test_episodes):
    obs = env.reset()
    done = False
    while not done :
        env.render()
        action = agent.get_action(obs, deterministic = True)
        next_obs, reward, done, info = env.step(action)
        obs = next_obs

env.close()


