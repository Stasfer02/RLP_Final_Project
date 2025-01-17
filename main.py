"""
Main file.

HALLOO

Here, we initialize the (wrapped) Environment, DQN agent and perform the training loop.
"""
import os
import numpy as np
import matplotlib as plt
import logging

import utils

import gymnasium as gym
import minigrid
from minigrid.wrappers import FlatObsWrapper

from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy
from algorithms.NGU_system import NGU_env_wrapper
from gymnasium.utils.env_checker import check_env

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S")

    # initialize instance of the environment for training and evaluation
    logging.info("Creating the environments ...")
    env_train = gym.make('MiniGrid-Empty-5x5-v0')
    env_eval = gym.make('MiniGrid-Empty-5x5-v0')

    # flatten the observation space for SB3 approval, because it does not support dict-type observations.
    env_train = FlatObsWrapper(env_train)
    env_eval = FlatObsWrapper(env_eval)

    # wrap our training env with the NGU reward system, check the validity of the environment.
    """
    logging.info("Wrapping training env with NGU system ...")
    env_train = NGU_env_wrapper(env_train)
    logging.debug(f"\n --- obs space: {env_train.observation_space} \n --- action space: {env_train.action_space}")
    #check_env(env_train) -> this fails because there is randomness in the env (probably due to the NN's involved.)
    """

    data_rewards = []      # 2D array

    for i in range(0,5):
        # initialize the DQN agent for 5 seperate runs.
        dqn_agent = DQN('MlpPolicy', env_train ,learning_rate=0.0001,verbose=0)
        logging.info(f"Successfully created DQN agent.")

        custom_callback = utils.CustomCallback()
        # let the DQN agent learn on the wrapped env, but use the custom callback (non-wrapped env) for evaluation
        dqn_agent.learn(100000, callback=custom_callback)

        rewards = custom_callback.get_results()
        data_rewards.append(rewards)
    
    means, stds = utils.calculate_means_stds(data_rewards)

    utils.create_plot(means, stds, f"{os.path.dirname(__file__)}/data/SB3_Empty_LR0_001.png")

if __name__ == "__main__":
    main()
    pass