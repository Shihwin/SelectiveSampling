import os
import glob
import numpy as np
import json

import sys
#sys.path = ['/home/pami/Desktop/Continuous_Algo','/home/pami/Desktop/','/home/pami/Desktop/Continuous_Algo', '/home/pami/anaconda3/envs/RL/lib/python37.zip', '/home/pami/anaconda3/envs/RL/lib/python3.7','/home/pami/anaconda3/envs/RL/lib/python3.7/lib-dynload','/home/pami/anaconda3/envs/RL/lib/python3.7/site-packages']
#sys.path = ['/home/pami/Desktop/Continuous_Algo', '/opt/ros/melodic/lib/python2.7/dist-packages', '/home/pami/anaconda3/envs/RL/lib/python37zip', '/home/pami/anaconda3/envs/RL/lib/python3.7', '/home/pami/anaconda3/envs/RL/lib/python3.7/lib-dynload', '/home/pami/anacond3/envs/RL/lib/python3.7/site-packages', '/home/pami/Desktop/', '/home/pami/Desktop/Continuous_Algo']
sys.path.insert(0,'/home/pami/Desktop/') 
sys.path.insert(0,'/home/pami/Desktop/Continuous_Algo')
print(sys.path)

import torch
# args
from Continuous_Algo.utils.SACArgument import get_args
# Log
from tensorboardX import SummaryWriter, writer
from tqdm import tqdm

# Env
import gym
# Buffer
from Continuous_Algo.buffer.DDPGBuffer import DDPGReplayBuffer
from Continuous_Algo.buffer.ClusterBuffer import  buffer_2_cluster_buffer, expert_2_cluster_buffer
# Agent
from Continuous_Algo.agents.SACAgent import SACAgent


from sklearn.cluster import KMeans

CLUSTER_NUMBER = 50


'''
Running Statics Mean/Std Remember to add in PPO
'''

if __name__ == '__main__':
    args = get_args()
    sac_dict = vars(args)
    # specific args
    torch.set_num_threads(1)
    device = torch.device("cuda:{}".format(args.device) if args.cuda else "cpu")
    
    # log_dir = os.path.join(prefix , args.env_id + '/' + args.algo + 'car_sac')
    try:
        # os.makedirs(log_dir)
        os.makedirs(args.res_dir)
    except OSError:
        # files = glob.glob(os.path.join(log_dir, '*.pami12')) + glob.glob(os.path.join(log_dir, '*.dump'))
        files = glob.glob(os.path.join(args.res_dir, '*.pami12')) + glob.glob(os.path.join(args.res_dir, '*.dump'))
        for f in files:
            os.remove(f)

    # tensorboardX
    # writer = SummaryWriter(log_dir)
    writer = SummaryWriter(args.res_dir)

    # save seed ....
    with open(os.path.join(args.res_dir, 'params.json'), 'w') as f:
        f.write(json.dumps(sac_dict, ensure_ascii=False, indent=4, separators=(',',':')))

    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Env & Extract Info from envs (obs_shape, act_shape)
    # gym.seed = args.seed
    env = gym.make(args.env_id)
    test_env = gym.make(args.env_id)

    env.seed(args.seed)
    test_env.seed(args.seed)

    # Assumed to Use Mujoco (Which can be writtened into Function For Different Envs
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    act_limit = env.action_space.high[0] 

    # Buffer (input nd.array output nd.array)
    buffer = DDPGReplayBuffer(obs_dim, act_dim, args.buffer_size)
    on_policy_buffer = DDPGReplayBuffer(obs_dim, act_dim, args.buffer_size)

    # Agent
    agent = SACAgent(obs_dim, act_dim, act_limit, args, device, writer)

    # Training Loop
    obs = env.reset()
    episode_reward = 0. 
    episode_length = 0
    num_episodes = 0
    
    for step in tqdm(range(args.total_env_steps)): # total env steps -> 100000

        # update cluster buffer
        if ((step + 1) % 1000 == 0 and args.reward_form == "implicit"):
            # off-policy
            off_state_action, off_rews_buffer = buffer_2_cluster_buffer(buffer, args.cluster_off_policy_buffer_size, device)
        
            # expert
            if (args.cluster_expert_policy_buffer_size > 0):
                file_name = os.path.join(args.experts_dir, "trajs_{}.pt".format(args.env_id.split('-')[0].lower()))
                all_trajectories = torch.load(file_name)
                expert_state_action, expert_rews_buffer = expert_2_cluster_buffer(all_trajectories, args.cluster_expert_policy_buffer_size, device)
            else:
                expert_state_action, expert_rews_buffer = buffer_2_cluster_buffer(buffer, 0, device)

            # on-policy 
            on_state_action, on_rews_buffer = buffer_2_cluster_buffer(on_policy_buffer, args.cluster_on_policy_buffer_size, device)
        
            # integrate
            state_action = torch.cat((off_state_action, expert_state_action, on_state_action), dim=0)
            rews_buffer = torch.cat((off_rews_buffer, expert_rews_buffer, on_rews_buffer), dim=0)

            state_action_kmeans = KMeans(n_clusters=CLUSTER_NUMBER)
            # obs_kmeans.fit(obs_buffer.cpu())
            # y_kmeans = obs_kmeans.predict(obs_buffer)
            state_action_kmeans.fit(state_action)
            y_kmeans = state_action_kmeans.predict(state_action)
            rews_buffer_dict = {}
            mean_rews_dict = {}
            for i in range(CLUSTER_NUMBER):
                idxs = torch.from_numpy(np.argwhere(y_kmeans == i)).squeeze(-1)
                rews_buffer_dict[i] = torch.as_tensor(rews_buffer[idxs], dtype = torch.float32, device = device).squeeze(-2)

                mean_rews_dict[i] = rews_buffer_dict[i].mean()


        # Exploration or Exploitation
        if (step < args.random_steps): # random steps -> 10000
            action = env.action_space.sample()
        else:
            if (args.reward_form == "original"):
                action = agent.get_action(obs , act_dim)
            elif (args.reward_form == "explicit"):
                action = agent.explicit_get_action(obs, env, act_dim)
            elif (args.reward_form == "implicit"):
                action = agent.implicit_get_action(obs, test_env, act_dim, state_action_kmeans, mean_rews_dict)

        # Excute
        next_obs, reward, done, info = env.step(action)
        episode_reward += reward
        episode_length += 1

        # TimeLimit or Real Done 
        done = False if episode_length == env._max_episode_steps else done

        # store
        buffer.store(obs, action, reward, next_obs, done)
        on_policy_buffer.store(obs, action, reward, next_obs, done)

        obs = next_obs

        # if Real Done
        if done:
            num_episodes += 1
            writer.add_scalar('Train_Episode_Reward', episode_reward, num_episodes)
            writer.add_scalar('Train_Episode_Length', episode_length, num_episodes)
            obs = env.reset()
            episode_reward = 0
            episode_length = 0

        # update 
        if step > args.learn_start_steps:
            if (step + 1) % args.update_every_steps == 0:
                value_loss, policy_loss = agent.update(buffer, step, env)
                # writer.add_scalar('policy_loss', policy_loss, step)
                # writer.add_scalar('value_loss', value_loss, step)
            
        # on policy degree:
        if ((step + 1) % args.on_policy_degree == 0):
            on_policy_buffer = DDPGReplayBuffer(obs_dim, act_dim, args.buffer_size)

        # log state & action
        # if (step + 1) % 100 == 0:
            #  writer.add_scalars('state_value', {'s' + str(i) : obs[i] for i in range(obs_dim)}, step)
            #  writer.add_scalars('action_value', {'a' + str(j): action[j] for j in range(act_dim)}, step)

        # test_agent & save model
        if (step + 1) % (args.total_env_steps/100) == 0:
            avg_reward = 0.
            test_episodes = 5
            for _ in range(test_episodes):
                obs, done, ep_rew, ep_len = test_env.reset(), False, 0.0, 0
                while not (done or (ep_len == test_env._max_episode_steps)):
                    if (args.reward_form == "original"):
                        action = agent.get_action(obs, deterministic=True)
                    elif (args.reward_form == "explicit"):
                        action = agent.explicit_get_action(obs, test_env, act_dim, deterministic=True)
                    elif (args.reward_form == "implicit"):
                        action = agent.implicit_get_action(obs, test_env, act_dim, state_action_kmeans, mean_rews_dict, deterministic=True)
                    next_obs, reward, done, info = test_env.step(action)
                    ep_rew += reward
                    ep_len += 1
                    obs = next_obs
                    
                avg_reward += ep_rew
            
            avg_reward /= test_episodes
            # save_model
            # torch.save(agent.model.state_dict(), os.path.join(log_dir, 'sac_model.dump'))
            torch.save(agent.model.state_dict(), os.path.join(args.res_dir, 'sac_model.dump'))
            writer.add_scalar('Test_Episode_Reward', avg_reward, step)

    
    env.close()
    test_env.close()
