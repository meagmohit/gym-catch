import gym
from gym import error, spaces, utils
from gym.utils import seeding

import numpy as np
import matplotlib.pyplot as plt

# Atari pixel configurations
atari_height = 210
atari_width = 160
atari_channels = 3

# Configuration
screen_height = 10
screen_width = 10
bar_width = 2

# Actions
action_left = 0
action_stay = 1
action_right = 2

score_max = 20

class ALEInterface(object):
    def __init__(self):
      self.xyz = 0

    def lives(self):
      return 0

class CatchEnv(gym.Env):

    #class ale(gym.Env):
    #    def __init__(self):
    #        self.total_lives = None
    #
    #	@classmethod
    #    def lives(cls):
    #        return 0
    #metadata = {'render.modes': ['human']}
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self):
        self._action_set = np.array([0,3,4],dtype=np.int32)
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(low=0, high=255, shape=(atari_height,atari_width,3), dtype=np.uint8)
        self.screen_dims = [screen_height, screen_width] #Black and White
        self.atari_dims = (atari_height, atari_width, atari_channels)
        self.viewer = None
        self.score = 0.0
        self.ale = ALEInterface()
        self.reset()

    # Act by taking an action # return observation (object), reward (float), done (boolean) and info (dict)
    def step(self, action):
        assert self.action_space.contains(action)   # makes sure the action is valid
        # Updating the state
        [ball_row, ball_col, bar_col] = self.state
        bar_col = min(max(0, bar_col + action - 1), screen_width - bar_width)
        ball_row = ball_row + 1
        self.state = [ball_row, ball_col, bar_col]
        # Generating the rewards
        #reward = 0.0
        if (ball_row == screen_height -1):
            if (ball_col >= bar_col) and (ball_col <= bar_col+bar_width):
                self.score = self.score + 1.0
            else:
                self.score = self.score - 1.0
            self.reset()
        # Generate the done (boolean)
        done = False
        if (abs(self.score)==score_max):
            done = True
        #return self.state, reward, done, None
        return self._get_observation(), self.score, done, {"ale.lives": 0}

    def reset(self):
        ball_row = 0
        ball_col = np.random.randint(screen_width)    # picks b/w 0 to screen_width-1 (both inclusive)
        bar_col = np.random.randint(screen_width - bar_width)
        self.state = [ball_row, ball_col, bar_col]
        #return self.state#
        return self._get_observation()

    def _get_observation(self):
        img = 255*np.ones(self.atari_dims, dtype=np.uint8) # White screen
        pixel_in_row = int(atari_height/screen_height)
        pixel_in_col = int(atari_width/screen_width)
        [ball_row, ball_col, bar_col] = self.state
        bar_row = screen_height-1
        img[ball_row*pixel_in_row:(ball_row+1)*pixel_in_row, ball_col*pixel_in_col:(ball_col+1)*pixel_in_col, 1:3] = 0    # Ball in Red
        img[bar_row*pixel_in_row:(bar_row+1)*pixel_in_row, bar_col*pixel_in_col:(bar_col+1+bar_width)*pixel_in_col, 0:2] = 0    # Bar in Blue
        return img

    def render(self, mode='human', close=False):
        img = self._get_observation()
        if mode == 'rgb_array':
            return img
        #return np.array(...) # return RGB frame suitable for video
        elif mode is 'human':
            #... # pop up a window and render
            from gym.envs.classic_control import rendering
            if self.viewer is None:
                self.viewer = rendering.SimpleImageViewer()
            self.viewer.imshow(img)
            return self.viewer.isopen
            #plt.imshow(img)
            #plt.show()
        else:
            super(CatchEnv, self).render(mode=mode) # just raise an exception

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

    def get_action_meanings(self):
        return [ACTION_MEANING[i] for i in self._action_set]

    def seed(self, seed=None):
        self.np_random, seed1 = seeding.np_random(seed)
        # Derive a random seed. This gets passed as a uint, but gets
        # checked as an int elsewhere, so we need to keep it below
        # 2**31.
        seed2 = seeding.hash_seed(seed1 + 1) % 2**31
        # Empirically, we need to seed before loading the ROM.
        #self.ale.setInt(b'random_seed', seed2)
        #self.ale.loadROM(self.game_path)
        return [seed1, seed2]


ACTION_MEANING = {
    0 : "NOOP",
    1 : "FIRE",
    2 : "UP",
    3 : "RIGHT",
    4 : "LEFT",
    5 : "DOWN",
    6 : "UPRIGHT",
    7 : "UPLEFT",
    8 : "DOWNRIGHT",
    9 : "DOWNLEFT",
    10 : "UPFIRE",
    11 : "RIGHTFIRE",
    12 : "LEFTFIRE",
    13 : "DOWNFIRE",
    14 : "UPRIGHTFIRE",
    15 : "UPLEFTFIRE",
    16 : "DOWNRIGHTFIRE",
    17 : "DOWNLEFTFIRE",
}
