# gym-catch
A simple catch game (with atari rendering) in the gym OpenAI environment

## Installation instructions
----------------------------

Requirements: gym with atari dependency

```shell
git clone https://github.com/meagmohit/gym-catch
cd gym-catch
python setup.py install
```

```python
import gym
import gym_catch
env = gym.make('catch-v0') # The other option is 'CatchNoFrameskip-v4'
env.render()
```

## References
-------------
