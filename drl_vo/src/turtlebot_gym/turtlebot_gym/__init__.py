from gym.envs.registration import register

# Register drl_nav env 
register(
  id='drl-nav-v0',
  entry_point='turtlebot_gym.envs:DRLNavEnv'
  )


