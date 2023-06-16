import numpy as np
import torch

def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)


class DDPGReplayBuffer(object):
    
    '''
    Simple Buffer For DDPG , Store the Experience In Cpu Memory
    '''

    def __init__(self, obs_dim, act_dim, buffer_size):
        super(DDPGReplayBuffer, self).__init__()
        # s,a,r,s_, done
        
        self.obs_buf = np.zeros(combined_shape(buffer_size, obs_dim), dtype=np.float32)
        
        self.acts_buf = np.zeros(combined_shape(buffer_size, act_dim), dtype=np.float32)

        self.rew_buf = np.zeros(buffer_size, dtype=np.float32)

        self.next_obs_buf = np.zeros(combined_shape(buffer_size, obs_dim), dtype=np.float32)

        self.done_buf = np.zeros(buffer_size, dtype=np.uint8)
        
        # pointer, if buffer is not full , buffer_size
        self.pointer, self.size, self.max_size = 0, 0, buffer_size

    def store(self, obs, act, rew, next_obs, done):

        self.obs_buf[self.pointer] = obs
        self.next_obs_buf[self.pointer] = next_obs

        self.acts_buf[self.pointer] = act
        self.rew_buf[self.pointer] = rew
        self.done_buf[self.pointer] = done

        self.pointer = (self.pointer+1) % self.max_size


        self.size = min(self.size+1, self.max_size)

    def refresh(self):
        super(DDPGReplayBuffer, self).__init__()
        # s,a,r,s_, done
        
        self.obs_buf = np.zeros(combined_shape(buffer_size, obs_dim), dtype=np.float32)
        
        self.acts_buf = np.zeros(combined_shape(buffer_size, act_dim), dtype=np.float32)

        self.rew_buf = np.zeros(buffer_size, dtype=np.float32)

        self.next_obs_buf = np.zeros(combined_shape(buffer_size, obs_dim), dtype=np.float32)

        self.done_buf = np.zeros(buffer_size, dtype=np.uint8)
        
        # pointer, if buffer is not full , buffer_size
        self.pointer, self.size, self.max_size = 0, 0, buffer_size

'''
    # sample buffer from cpu memory , return batch tensor
    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(obs=self.obs_buf[idxs],
                     next_obs=self.next_obs_buf[idxs],
                     act=self.acts_buf[idxs],
                     rew=self.rew_buf[idxs],
                     done=self.done_buf[idxs])

        return {k: torch.as_tensor(v, dtype=torch.float32) for k,v in batch.items()}
'''