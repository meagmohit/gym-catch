import gym
from gym import error, spaces, utils
from gym.utils import seeding

import numpy as np
import matplotlib.pyplot as plt

# Libraires for sending external stimulations over TCP port
import sys
import socket
from time import time, sleep



class ALEInterface(object):
    def __init__(self):
      self.lives_left = 0

    def lives(self):
      return 0 #self.lives_left

class CatchEnv(gym.Env):

    metadata = {'render.modes': ['human', 'rgb_array'], 'video.frames_per_second' : 50}

    def __init__(self, grid_size=(105,20), bar_size=5, total_balls=10, tcp_tagging=False, tcp_port=15361):
        
        # Atari-platform related parameters
        self.atari_dims = (210,160,3)		# Specifies standard atari resolution
        (self.atari_height, self.atari_width, self.atari_channels) = self.atari_dims

        #  Game-related paramteres
        self.screen_height = grid_size[0]
        self.screen_width = grid_size[1]
        self.screen_dims = [self.screen_height, self.screen_width]
        self.bar_width = bar_size
        self.actions = [0, 1, -1]
        self.score = 0.0
        self.total_balls = total_balls
        self.current_balls = 0
        
        # Gym-related variables [must be defined]
        self._action_set = np.array([0,3,4],dtype=np.int32)
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(low=0, high=255, shape=(self.atari_height, self.atari_width, 3), dtype=np.uint8)
        self.viewer = None
        
        # Code for TCP Tagging
        self.tcp_tagging = tcp_tagging
        if (self.tcp_tagging):
            self.host = '127.0.0.1'
            self.port = tcp_port
            self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.s.connect((self.host, self.port))

        # Methods
        self.ale = ALEInterface()
        self.seed()
        self.reset()

    # Act by taking an action # return observation (object), reward (float), done (boolean) and info (dict)
    def step(self, action):
        assert self.action_space.contains(action)   # makes sure the action is valid
        
        # Updating the state, state is hidden from observation
        [ball_row, ball_col, bar_col] = self.state
        current_action = self.actions[action]
        bar_col = min(max(0, bar_col + current_action), self.screen_width - self.bar_width)
        ball_row = ball_row + 1
        self.state = [ball_row, ball_col, bar_col]
        
        # Generating the rewards
        reward = 0.0
        if (ball_row == self.screen_height -1):
            if (ball_col >= bar_col) and (ball_col <= bar_col+self.bar_width):
                reward = 1.0
            else:
                reward = -1.0
            self.score = self.score + reward
            ball_row = 0
            ball_col = np.random.randint(self.screen_width)
            self.state = [ball_row, ball_col, bar_col]
            self.current_balls = self.current_balls + 1
        
        # Generate the done (boolean)
        done = False
        if (self.current_balls>=self.total_balls):
            done = True

        # Sending the external stimulation over TCP port
        if self.tcp_tagging:
            padding=[0]*8
            event_id = [ball_row, ball_col, bar_col, action, 0, 0, 0, 0]
            timestamp=list(self.to_byte(int(time()*1000), 8))
            self.s.sendall(bytearray(padding+event_id+timestamp))
        
        return self._get_observation(), reward, done, {"ale.lives": self.ale.lives()}

    def reset(self):
        self.score = 0.0
        self.current_balls = 0
        ball_row = 0
        ball_col = np.random.randint(self.screen_width)    # picks b/w 0 to screen_width-1 (both inclusive)
        bar_col = np.random.randint(self.screen_width - self.bar_width)
        self.state = [ball_row, ball_col, bar_col]
        return self._get_observation()

    def _get_observation(self):
        img = np.zeros(self.atari_dims, dtype=np.uint8) # Black screen
        pixel_in_row = int(self.atari_height/self.screen_height)
        pixel_in_col = int(self.atari_width/self.screen_width)
        [ball_row, ball_col, bar_col] = self.state
        bar_row = self.screen_height-1
        img[ball_row*pixel_in_row:(ball_row+1)*pixel_in_row, ball_col*pixel_in_col:(ball_col+1)*pixel_in_col, 0:3] = 255    # Ball in white
        img[bar_row*pixel_in_row:(bar_row+1)*pixel_in_row, bar_col*pixel_in_col:(bar_col+self.bar_width)*pixel_in_col, 0:3] = 255    # Bar in while
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
        if self.tcp_tagging:
            self.s.close()

    def get_action_meanings(self):
        return [ACTION_MEANING[i] for i in self._action_set]
 
    @property
    def _n_actions(self):
        return len(self._action_set)

    def seed(self, seed=None):
        self.np_random, seed1 = seeding.np_random(seed)
        # Derive a random seed. This gets passed as a uint, but gets
        # checked as an int elsewhere, so we need to keep it below
        # 2**31.
        seed2 = seeding.hash_seed(seed1 + 1) % 2**31
        return [seed1, seed2]

    # A function for TCP_tagging in openvibe
    # transform a value into an array of byte values in little-endian order.
    def to_byte(self, value, length):
        for x in range(length):
            yield value%256
            value//=256


    def get_keys_to_action(self):
        KEYWORD_TO_KEY = {
            'UP':      ord('w'),
            'DOWN':    ord('s'),
            'LEFT':    ord('a'),
            'RIGHT':   ord('d'),
            'FIRE':    ord(' '),
        }

        keys_to_action = {}

        for action_id, action_meaning in enumerate(self.get_action_meanings()):
            keys = []
            for keyword, key in KEYWORD_TO_KEY.items():
                if keyword in action_meaning:
                    keys.append(key)
            keys = tuple(sorted(keys))

            assert keys not in keys_to_action
            keys_to_action[keys] = action_id

        return keys_to_action


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
