import numpy as np
import torch


def buffer_2_cluster_buffer(buffer, cluster_buffer_size, device):
    idxs = np.random.randint(0, buffer.size, cluster_buffer_size)
    obs_buffer = torch.as_tensor(buffer.obs_buf[idxs], dtype = torch.float32, device = device)
    acts_buffer = torch.as_tensor(buffer.acts_buf[idxs], dtype = torch.float32, device = device)
    rews_buffer = torch.as_tensor(buffer.rew_buf[idxs], dtype = torch.float32, device = device).unsqueeze(-1)
    state_action = torch.cat((obs_buffer, acts_buffer), dim=1).cpu()
    return state_action, rews_buffer

def expert_2_cluster_buffer(all_trajectories, cluster_buffer_size, device):
    buffer_size = all_trajectories['states'].shape[0]
    idxs = np.random.randint(0, buffer_size, cluster_buffer_size)
    # 存在device的问题，还没修改，现在的device恒为cpu，但是上面的off policy的device默认是gpu的
    obs_buffer = all_trajectories['states'][idxs] 
    acts_buffer = all_trajectories['actions'][idxs]
    rews_buffer = torch.as_tensor(all_trajectories['rewards'][idxs], dtype = torch.float32, device = device).unsqueeze(-1)
    state_action = torch.cat((obs_buffer, acts_buffer), dim=1)
    return state_action, rews_buffer