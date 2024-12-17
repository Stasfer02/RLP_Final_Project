"""
Main file.

Here, we initialize the (wrapped) Environment, DQN agent and perform the training loop.
"""
import numpy as np
import matplotlib as plt

import gymnasium as gym
import minigrid

from stable_baselines3 import DQN
from algorithms.NGU_system import NGU_env_wrapper

def main():

    env = gym.make('MiniGrid-Empty-16x16-v0')

    # wrap our env with the NGU reward system
    wrapped_env = NGU_env_wrapper(env)

    # initialize the (multi-layered-perceptron) DQN agent, based on the wrapped env.
    dqn_agent = DQN('MlpPolicy', wrapped_env,learning_rate=0.001,verbose=0)



if __name__ == "__main__":
    main()
    pass