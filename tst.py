# Import the gym module

import gym
import sys
import time

from __builtin__ import raw_input

if __name__ == "__main__":

    # Create a breakout environment -------------------------------------------
    env = gym.make('BreakoutDeterministic-v4')
    env.reset()

    env.reset()
    for _ in range(1000):
        env.render()
        # action = env.action_space.sample()
        action = int(raw_input("prompt"))
        x_t1, reward, is_done, misc = env.step(action)  # take a random action
        print(action, reward, is_done, misc)
