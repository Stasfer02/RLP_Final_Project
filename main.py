"""
Main file.

HALLOO

Here, we initialize the (wrapped) Environment, DQN agent and perform the training loop.
"""
import os
from random import gauss
import numpy as np
import logging

import utils

import gymnasium as gym
import minigrid
from minigrid.wrappers import FlatObsWrapper

from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy

from algorithms.NGU_system import NGU_env_wrapper, DoWhaM_agent
from scipy.ndimage import gaussian_filter1d
from gymnasium.utils.env_checker import check_env

import matplotlib.pyplot as plt

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S")

    # initialize instance of the environment for training and evaluation
    logging.info("Creating the environments ...")
    env_base = gym.make('MiniGrid-Empty-5x5-v0')
    # flatten the observation space for SB3 approval, because it does not support dict-type observations.
    env_base = FlatObsWrapper(env_base)

    seeds = [0,1,2,3,4]
    """
    first run with NGU system
    """
    # wrap our training env with the NGU reward system, check the validity of the environment.
    logging.info("Wrapping training env with NGU system ...")
    env_train_1 = NGU_env_wrapper(env_base,beta=0.2, useNGU=True, useDoWhaM=False)

    data_rewards = []      # 2D array of rewards over the multiple runs
    for i in range(0,2):
        # initialize the DQN agent for 5 seperate runs.
        dqn_agent = DQN('MlpPolicy', env_train_1 ,learning_rate=0.0001,verbose=0,seed=seeds[i])
        logging.info(f"Successfully created DQN agent number {i}.")

        custom_callback = utils.CustomCallback_updated(env_train_1)
        # let the DQN agent learn on the wrapped env, but use the custom callback (non-wrapped env) for evaluation
        dqn_agent.learn(100000, callback=custom_callback)

        rewards = custom_callback.get_results()
        data_rewards.append(rewards)
    
    means_1, stds_1 = utils.calculate_means_stds(data_rewards)
    

    """
    second run
    """
    logging.info("Wrapping training env with NGU system ...")
    env_train_2 = NGU_env_wrapper(env_base,eta=40, useNGU=False, useDoWhaM=True)
    data_rewards = []      # 2D array of rewards over the multiple runs
    for i in range(0,2):
        # initialize the DQN agent for 5 seperate runs.
        dqn_agent = DQN('MlpPolicy', env_train_2 ,learning_rate=0.0001,verbose=0,seed=seeds[i])
        logging.info(f"Successfully created DQN agent number {i}.")

        custom_callback = utils.CustomCallback_updated(env_train_2)
        # let the DQN agent learn on the wrapped env, but use the custom callback (non-wrapped env) for evaluation
        dqn_agent.learn(100000, callback=custom_callback)

        rewards = custom_callback.get_results()
        data_rewards.append(rewards)
    
    means_2, stds_2 = utils.calculate_means_stds(data_rewards)
    
    """
    third run 
    """
    logging.info("Wrapping training env with NGU system ...")
    env_train_3 = NGU_env_wrapper(env_base, beta=0.4, useNGU=True, useDoWhaM=False)
    data_rewards = []      # 2D array of rewards over the multiple runs
    for i in range(0,2):
        # initialize the DQN agent for 5 seperate runs.
        dqn_agent = DQN('MlpPolicy', env_train_3 ,learning_rate=0.0001,verbose=0,seed=seeds[i])
        logging.info(f"Successfully created DQN agent number {i}.")

        custom_callback = utils.CustomCallback_updated(env_train_3)
        # let the DQN agent learn on the wrapped env, but use the custom callback (non-wrapped env) for evaluation
        dqn_agent.learn(100000, callback=custom_callback)

        rewards = custom_callback.get_results()
        data_rewards.append(rewards)
    
    means_3, stds_3 = utils.calculate_means_stds(data_rewards)
    

    """
    fourth run: standard DQN
    """
    logging.info("Creating standard DQN without wrapping env")
    env_train_4 = env_base
    data_rewards = []      # 2D array of rewards over the multiple runs
    for i in range(0,5):
        # initialize the DQN agent for 5 seperate runs.
        dqn_agent = DQN('MlpPolicy', env_train_4 ,learning_rate=0.0001,verbose=0,seed=seeds[i])
        logging.info(f"Successfully created DQN agent number {i}.")

        custom_callback = utils.CustomCallback()
        # let the DQN agent learn on the wrapped env, but use the custom callback (non-wrapped env) for evaluation
        dqn_agent.learn(100000, callback=custom_callback)

        rewards = custom_callback.get_results()
        data_rewards.append(rewards)
    
    means_4, stds_4 = utils.calculate_means_stds(data_rewards)
    
    # plotting
    x_values_1 = np.arange(len(means_1))
    x_values_2 = np.arange(len(means_2))
    x_values_3 = np.arange(len(means_3))
    x_values_4 = np.arange(len(means_4))

    # apply smoothening
    means_1 = gaussian_filter1d(np.array(means_1), sigma=5)
    means_2 = gaussian_filter1d(np.array(means_2), sigma=5)
    means_3 = gaussian_filter1d(np.array(means_3), sigma=5)
    means_4 = gaussian_filter1d(np.array(means_4), sigma=5)

    plt.figure(figsize=(20,10))
    # line 1
    plt.plot(x_values_4, means_4,color="blue",linewidth= 0.5, label="standard DQN")
    plt.fill_between(x_values_4, means_4 - stds_4, means_4 + stds_4, color="blue", alpha= 0.15)
    # line 2
    plt.plot(x_values_1, means_1,color="red",linewidth= 0.5, label=r"DQN + NGU ($\beta$=0.2")
    plt.fill_between(x_values_1, means_1 - stds_1, means_1 + stds_1, color="red", alpha= 0.15)
    # line 3
    plt.plot(x_values_2, means_2,color="red",linewidth= 0.5, label=r"DQN + NGU with $\beta$ = 0.3")
    plt.fill_between(x_values_2, means_2 - stds_2, means_2 + stds_2, color="red", alpha= 0.15)
    # line 4
    plt.plot(x_values_3, means_3,color="green",linewidth= 0.5, label=r"DQN + NGU with $\beta$ = 0.4")
    plt.fill_between(x_values_3, means_3 - stds_3, means_3 + stds_3, color="green", alpha= 0.15)
    
    plt.legend(fontsize= 20, loc="lower right")
    plt.xlabel(fontsize= 20, xlabel="Episodes")
    plt.ylabel(fontsize = 20, ylabel="Reward")

    plt.tick_params(axis='both', which='major', labelsize=20)

    plt.savefig(f"{os.path.dirname(__file__)}/data/1M_doorkey_DQN_vs_DoWhaM_LR_0_0005.pdf")
    plt.close()

if __name__ == "__main__":
    main()
    pass