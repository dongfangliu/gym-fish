from gym.envs.registration import register

register(
    id='fish-v0',
    entry_point='gym_fish.envs:FishEnv',
)
