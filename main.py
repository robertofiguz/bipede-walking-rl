from gym import Env
from gym.spaces import Discrete, Box
import numpy as np
import random

class TestEnvironment(Env):
    def __init__(self, angle_increment = 1):
        self.angle_increment = angle_increment
        self.action_space = Discrete(3) # this will chenge the joint actions by maintaining, increasing or decreasing the angle. multiply angle static increment by sample action
        self.observation_space = Box(low=np.array([-100]), high=np.array([100])) # why is this a numpy array?
        self.state = 0
        self.episode_lenght = 60
    def step(self, action):
        self.state += action*self.angle_increment
        self.episode_lenght -= 1
    
        if self.state<20 and self.state>-20:
            reward = 1
        else:
            reward = -1
        
        done = False

        if self.episode_lenght<=0:
            done = True
        self.state += random.randint(-1,1)
        info = {}

        return self.state, reward, done, info

    def reset(self):
        self.state = 0
        self.episode_lenght = 60
        return self.state

env = TestEnvironment()
episodes = 10
print("states ", env.action_space.n)
for episode in range(1, episodes+1):
    state = env.reset()
    done = False
    score = 0

    while not done:
        action = env.action_space.sample()
        n_state, reward, done, info = env.step(action)
        score += reward
    print('Episode:{} score:{} state:{}'.format(episode, score, n_state))