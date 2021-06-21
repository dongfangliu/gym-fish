from gym.envs.registration import register

register(
    id='fish-basic-v0',
    entry_point='gym_fish.envs:FishEnvBasic',
)
register(
    id='fish-vel-v0',
    entry_point='gym_fish.envs:FishEnvVel',
)

