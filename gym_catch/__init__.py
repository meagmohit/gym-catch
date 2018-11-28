from gym.envs.registration import register

register(
    id='catch-v0',
    entry_point='gym_catch.atari:CatchEnv',
)

register(
    id='CatchNoFrameskip-v4',
    entry_point='gym_catch.atari:CatchEnv',
    #kwargs={'obs_type': 'image', 'frameskip': 1}, # A frameskip of 1 means we get every frame
    #max_episode_steps=10,
    #nondeterministic=False,
)
