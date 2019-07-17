import gym
import gym_catch
import time
import numpy as np

env = gym.make('CatchNoFrameskip-v3')
env.reset()
action_idx = [0, 1, 2]
actions = [0, 1, -1]	# 0 means stay, 1 mean right i.e. 1, 2 means -1 i.e. left
p_err = 0.5
speed = 1.5 # in seconds
for _ in range(1):
	done = False
	env.reset()
	while not done:
		state = env.unwrapped._state
		[ball_row, ball_col, bar_col] = state
		if ball_col == bar_col:	# correct action is stay
			action = np.random.choice(action_idx, p=[1-p_err, p_err/2.0, p_err/2.0])
		elif ball_col > bar_col: #correct action is move right
			action = np.random.choice(action_idx, p=[p_err/2.0, 1-p_err, p_err/2.0])
		else:
			action = np.random.choice(action_idx, p=[p_err/2.0, p_err/2.0, 1-p_err])
		(obs, reward, done, info) =  env.step(action) # take a random action
		print actions[action], info['internal_state'], reward
		env.render()
		time.sleep(speed)

print env.unwrapped._score
env.close()
