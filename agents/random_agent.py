import gym
import gym_catch
env = gym.make('catch-v0')
env.reset()
for _ in range(1):
	done = False
	env.reset()
	while not done:
		env.render()
		(obs, reward, done, info) =  env.step(env.action_space.sample()) # take a random action
                print reward
env.close()
