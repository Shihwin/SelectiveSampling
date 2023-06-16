from gym.envs.registration import register

register(
      id='nqubit-v0',
      entry_point='gym_nqubit.envs:NqubitEnv',
  )