from argparse import Action
import math
import re
from sklearn.base import ClusterMixin
import torch
import numpy as np
from networks.sac_model import GaussianPolicyMLPCritic
import torch.optim as optim
import torch.nn.functional as F

from sklearn.cluster import KMeans
'''
declare_networks()
declare_optimizer()
prepare_minibatch()
compute_value_loss()
compute_policy_loss()
update()
'''

ACTION_BUFFER_SIZE = 30
ACTION_CLUSTER_NUMBER = 3

class SACAgent(object):
    def __init__(self, obs_dim, act_dim, act_limit, args, device, writer=None):
        super().__init__()
        # get the variable
        self.args = args
        self.device = device
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.act_limit = act_limit
        self.writer = writer

        self.declare_networks()
        self.declare_optimizers()

        # self.auto_tune_alpha = args.auto_tune_alpha
        # if self.auto_tune_alpha:
        #     self.target_entropy = -torch.prod(torch.Tensor(self.act_dim))

    def declare_networks(self):
        self.model = GaussianPolicyMLPCritic(self.obs_dim, self.act_dim, self.args.hidden_size, self.act_limit)
        self.target_model = GaussianPolicyMLPCritic(self.obs_dim, self.act_dim, self.args.hidden_size, self.act_limit)
        self.target_model.load_state_dict(self.model.state_dict()) # hard update initial
        for param in self.target_model.parameters():
            param.requires_grad = False
        self.model = self.model.to(self.device)
        self.target_model = self.target_model.to(self.device)

    def declare_optimizers(self):
        self.policy_optimizer = optim.Adam(self.model.policy.parameters(), lr = self.args.policy_lr)
        self.critic_optimizer = optim.Adam(self.model.critic.parameters(), lr = self.args.value_lr)

    def get_cost_pendulum(self, action, env):
        # 这里评估动作价值，看的不是当前状态的theta、thetadot、action
        # 看的是当前的action，下一个状态的theta、thetadot、action

        # max_speed = 8
        # g = 10.0
        # m = 1.
        # l = 1.
        # dt = .05

        th, thdot = env.state
        action = np.array([action[0]])
        action = np.clip(action, -env.max_torque, env.max_torque)[0]

        newthdot = thdot + (-3 * env.g / (2 * env.l) * np.sin(th + np.pi) + 3. / (env.m * env.l ** 2) * action) * env.dt
        newth = th + newthdot * env.dt
        newthdot = np.clip(newthdot, env.max_speed, env.max_speed)

        newth = ((newth + np.pi) % (2 * np.pi)) - np.pi # angle_normalize
        new_cost = newth ** 2 + .1 * newthdot ** 2 + .001 * (action ** 2)

        # new_cost 恒 > 0
        reward = -new_cost

        return reward # 恒 < 0, 但越大越好
    
    def get_cost_mountain_car_continuous(self, action, env):

        position = env.state[0]
        velocity = env.state[1]
        force = min(max(action[0], env.min_action), env.max_action)

        velocity += force * env.power - 0.0025 * math.cos(3 * position)
        if (velocity > env.max_speed): velocity = env.max_speed
        if (velocity < -env.max_speed): velocity 
        position += velocity
        if (position > env.max_position): position = env.max_position
        if (position < env.min_position): position = env.min_position
        if (position == env.min_position and velocity < 0): velocity = 0

        done = bool(
            position >= env.goal_position and velocity >= env.goal_velocity
        )

        # 因为要算是不是done了，所以要算前面很多东西
        reward = 0
        if done:
            reward = 100.0
        # reward -= math.pow(action[0], 2) * 0.1
        reward = reward - (math.pow(action[0], 2) * 0.1) - 0.1 * pow((position - env.goal_position), 2)
        

        return reward # 越大越好

    def implicit_get_action(self, obs, env, act_dim, kmeans, mean_rews_dict, deterministic = False):
        obs = torch.as_tensor(obs, dtype = torch.float32, device = self.device).unsqueeze(0)
        with torch.no_grad():
            # action, _ = self.model.policy.forward(obs, env, deterministic)

            # select good action from here
            actions = torch.Tensor(ACTION_BUFFER_SIZE, act_dim)

            if (deterministic):
                action = self.model.policy.get_action(obs, deterministic)
                action = action.detach().cpu().numpy()[0]
                return action
            
            for i in range(ACTION_BUFFER_SIZE): # default ACTION_BUFFER_SIZE = 30
                # key step here -- to estimate the reward value.
                return_value = self.model.policy.get_action(obs, deterministic)[0]
                actions[i] = return_value
            
            reward = torch.Tensor(ACTION_BUFFER_SIZE)

            # obs_category = kmeans.predict(obs)
            obs_vec = obs
            obs_vec = obs.expand(ACTION_BUFFER_SIZE, obs.shape[1]).cpu()
            state_action = torch.cat((obs_vec, actions), dim=1)

            # 一共有两个聚类，
            # 第一个聚类：这个聚类是用来评估reward的，是参数传递进来的
            y_kmeans = kmeans.predict(state_action)

            # 第二个聚类：这个聚类选出哪一类
            action_kmeans = KMeans(n_clusters=ACTION_CLUSTER_NUMBER)
            action_kmeans.fit(actions)
            action_kmeans = action_kmeans.predict(actions)

            for i in range(ACTION_BUFFER_SIZE):
                reward[i] = mean_rews_dict[y_kmeans[i]] # 这里是第一个聚类！！！！！！
            
            
            # mean 指的是reward的mean，而不是action的means！！！！！！！
            # mean = torch.zeros(ACTION_CLUSTER_NUMBER, act_dim) 
            mean = torch.zeros(ACTION_CLUSTER_NUMBER) 
            count = torch.zeros(ACTION_CLUSTER_NUMBER)
             
            for i in range(ACTION_BUFFER_SIZE):
                for j in range(ACTION_CLUSTER_NUMBER):
                    if (action_kmeans[i] == j):
                        mean[j] += reward[i]
                        count[j] += 1
            
            for j in range(ACTION_CLUSTER_NUMBER):
                mean[j] = mean[j] / count[j]
            
            mean_max = torch.max(mean, 0)
            mean_max_index = mean_max[1].item()

            action_buffer = torch.Tensor(count[mean_max_index].int().item(), act_dim)
            action_buffer_ct = 0
            for i in range(ACTION_BUFFER_SIZE): # 30
                if (action_kmeans[i] == mean_max_index):
                    action_buffer[action_buffer_ct] = actions[i]
                    action_buffer_ct += 1

            selected_action = action_buffer[np.random.randint(0, action_buffer_ct, 1)]

        selected_action = selected_action.detach().cpu().numpy()[0]
        
        return selected_action

    def explicit_get_action(self, obs, env, act_dim, deterministic = False):
        '''
        input: obs is numpy.array with shape (obs_dim, ), transfer into (1, obs_dim)
        output: action is tensor with shape (1, obs_dim), trasfer into numpy.array with shape (act_dim, )
        '''
        obs = torch.as_tensor(obs, dtype = torch.float32, device = self.device).unsqueeze(0)
        with torch.no_grad():
            # action, _ = self.model.policy.forward(obs, env, deterministic)

            # select good action from here
            actions = torch.Tensor(ACTION_BUFFER_SIZE, act_dim)

            if (deterministic):
                action = self.model.policy.get_action(obs, deterministic)
                action = action.detach().cpu().numpy()[0]
                return action
            
            for i in range(ACTION_BUFFER_SIZE): # default ACTION_BUFFER_SIZE = 30
                # key step here -- to estimate the reward value.
                return_value = self.model.policy.get_action(obs, deterministic)[0]
                actions[i] = return_value

            kmeans = KMeans(n_clusters=ACTION_CLUSTER_NUMBER)
            kmeans.fit(actions)
            y_kmeans = kmeans.predict(actions)
            # centers = kmeans.cluster_centers_ # -1.7、-1.0、-0.01、0.81、1.69
            
            reward = torch.Tensor(ACTION_BUFFER_SIZE)
            for i in range(ACTION_BUFFER_SIZE):
                if (env.spec.id == 'MountainCarContinuous-v0'):
                    reward[i] = self.get_cost_mountain_car_continuous(actions[i], env)
                elif (env.spec.id == 'Pendulum-v0'):
                    reward[i] = self.get_cost_pendulum(actions[i], env)

                # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                # tp1, reward[i], tp2, tp3 = env.step(actions[i])
            
            mean = torch.zeros(ACTION_CLUSTER_NUMBER)
            count = torch.zeros(ACTION_CLUSTER_NUMBER)
             
            for i in range(ACTION_BUFFER_SIZE):
                for j in range(ACTION_CLUSTER_NUMBER):
                    if (y_kmeans[i] == j):
                        mean[j] += reward[i]
                        count[j] += 1
            
            for j in range(ACTION_CLUSTER_NUMBER):
                mean[j] = mean[j] / count[j]
            
            mean_max = torch.max(mean, 0)
            mean_max_index = mean_max[1].item()

            action_buffer = torch.Tensor(count[mean_max_index].int().item(), act_dim)
            action_buffer_ct = 0
            for i in range(ACTION_BUFFER_SIZE):
                if (y_kmeans[i] == mean_max_index):
                    action_buffer[action_buffer_ct] = actions[i]
                    action_buffer_ct += 1

            selected_action = action_buffer[np.random.randint(0, action_buffer_ct, 1)]

            # 从buffer里面选，用k-means, pca, tsne. 用sklearn很方便（聚类）
            # 有20000个动作，假设聚了5类，就可以用state action 后面对reward算一个均值
            # 预测一下动作属于哪一类，把reward高的动作挑出来。

            # 30个动作，对30个动作做聚类，聚成3类

            # 用sklearn里面的层次聚类
            # 不一定每一个step都要从buffer里面选，可以每1000个step更新一次buffer
            # 用policy的点是用policy采样了30个动作，然后用buffer来评估这30个动作
            # 用buffer的点是对buffer里面的state和action一起做层次聚类，然后看policy sample出来的
            # 样本是属于哪一类，然后去看这一类的mean reward（已经在buffer里面有记录了）

            # replay buffer到底是怎么用的buffer
            # SAC q-value是怎么更新的

        selected_action = selected_action.detach().cpu().numpy()[0]
        # action = action.detach().cpu().numpy()[0] # 这句话是什么意思？？
        return selected_action


    def get_action(self, obs, deterministic = False):
        '''
        input: obs is numpy.array with shape (obs_dim, ), transfer into (1, obs_dim)
        output: action is tensor with shape (1, obs_dim), trasfer into numpy.array with shape (act_dim, )
        '''
        obs = torch.as_tensor(obs, dtype = torch.float32, device= self.device).unsqueeze(0)
        with torch.no_grad():
            action, _ = self.model.policy(obs, deterministic)
        action = action.detach().cpu().numpy()[0]
        return action    


    def prepare_minibatch(self, buffer):
        idxs = np.random.randint(0, buffer.size, self.args.batch_size)
        obs_batch = torch.as_tensor(buffer.obs_buf[idxs], dtype = torch.float32, device = self.device)
        acts_batch = torch.as_tensor(buffer.acts_buf[idxs], dtype = torch.float32, device = self.device)
        rews_batch = torch.as_tensor(buffer.rew_buf[idxs], dtype = torch.float32, device= self.device).unsqueeze(-1)
        next_obs_batch = torch.as_tensor(buffer.next_obs_buf[idxs], dtype = torch.float32, device= self.device)
        done_batch = torch.as_tensor(buffer.done_buf[idxs], dtype = torch.float32, device= self.device).unsqueeze(-1)

        return obs_batch, acts_batch, rews_batch, next_obs_batch, done_batch


    def compute_value_loss(self, batch_data, env):
        obs, act, rew, next_obs, done = batch_data
        q1_value, q2_value = self.model.critic(obs, act)

        with torch.no_grad():
            next_acts, next_probs = self.model.policy(next_obs, env)
            # next_acts, next_probs batch
            
            target_q1_value , target_q2_value = self.target_model.critic(next_obs, next_acts)
            
            
            target_q_value = torch.min(target_q1_value, target_q2_value)
            
           
            target_update = rew + self.args.gamma * (1 - done) * (target_q_value - self.args.alpha * next_probs)
            

        loss1 = F.mse_loss(q1_value, target_update)
        loss2 = F.mse_loss(q2_value, target_update)
        value_loss = loss1 + loss2

        return value_loss

    def compute_policy_loss(self, batch_data, env):
        obs, _, _, _, _ = batch_data
        act, log_prob = self.model.policy(obs, env)
        q1_value, q2_value = self.model.critic(obs, act)
        q_value = torch.min(q1_value, q2_value)

        policy_loss = (-q_value + self.args.alpha * log_prob).mean()
        
        return policy_loss

    def update(self, buffer, step, env):
        value_loss_log = []
        policy_loss_log = []
        
        for i in range(self.args.updates_per_step):
            batch_data = self.prepare_minibatch(buffer) # on gpu-device 

            # SARS'
            # not A'
            value_loss = self.compute_value_loss(batch_data, env)
            value_loss_log.append(value_loss.detach().cpu().numpy())

            self.critic_optimizer.zero_grad()
            value_loss.backward()
            self.critic_optimizer.step()
        

            if (i + 1) % self.args.policy_delay == 0:
                policy_loss = self.compute_policy_loss(batch_data, env)
                policy_loss_log.append(policy_loss.detach().cpu().numpy())
                

                for param in self.model.critic.parameters():
                    param.requires_grad = False
                
                self.policy_optimizer.zero_grad()
                policy_loss.backward()
                self.policy_optimizer.step()

                for param in self.model.critic.parameters():
                    param.requires_grad = True

            if (i + 1) % self.args.target_update_freq == 0:
                with torch.no_grad():
                    for param, target_parma in zip(self.model.parameters(), self.target_model.parameters()):
                        target_parma.data.mul_(self.args.polyak)
                        target_parma.data.add_((1-self.args.polyak) * param.data)
        
        return np.mean(value_loss_log), np.mean(policy_loss_log)


