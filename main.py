"""
Main file.

Here, we initialize the (wrapped) Environment, DQN agent and perform the training loop.
"""
import numpy as np
import matplotlib as plt
import logging

import gymnasium as gym
import minigrid

from stable_baselines3 import DQN
from algorithms.NGU_system import NGU_env_wrapper
from gymnasium.utils.env_checker import check_env

def main():
    logging.basicConfig(level=logging.INFO, format='%(message)s')

    logging.info("Creating the env ...")
    env = gym.make('MiniGrid-Empty-5x5-v0')
    logging.info(f"Environment created: {env.__str__}")

    # wrap our env with the NGU reward system
    wrapped_env = NGU_env_wrapper(env)
    check_env(wrapped_env)
    #logging.info(f"Wrapped the environment with NGU.")

    # initialize the (multi-layered-perceptron) DQN agent, based on the wrapped env.
    dqn_agent = DQN('MlpPolicy', env,learning_rate=0.001,verbose=0)



if __name__ == "__main__":
    main()
    pass