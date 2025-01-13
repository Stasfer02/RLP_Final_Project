"""
Main file.

HALLOO

Here, we initialize the (wrapped) Environment, DQN agent and perform the training loop.
"""
import numpy as np
import matplotlib as plt
import logging

import gymnasium as gym
import minigrid

from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import EvalCallback
from algorithms.NGU_system import NGU_env_wrapper
from gymnasium.utils.env_checker import check_env

def main():
    logging.basicConfig(level=logging.DEBUG, format='%(message)s')

    # initialize instance of the environment for training and evaluation
    logging.info("Creating the environments ...")
    env_train = gym.make('MiniGrid-Empty-5x5-v0')
    env_eval = gym.make('MiniGrid-Empty-5x5-v0')
    logging.info(f"Environments created.")

    # wrap our training env with the NGU reward system, check the validity of the environment. 
    env_train = NGU_env_wrapper(env_train)
    #check_env(env_train)
    logging.info(f"Wrapped the training environment with NGU. Observation space: {env_train.observation_space}")

    # initialize the (multi-layered-perceptron) DQN agent, based on the wrapped env.
    dqn_agent = DQN('MlpPolicy', env_train,learning_rate=0.001,verbose=0)

    # setup the evaluation callback with the normal env (not extended with NGU). 
    # taken from here: https://stable-baselines3.readthedocs.io/en/master/guide/callbacks.html
    # we store the logs for debugging and interpretability
    eval_callback = EvalCallback(
        env_eval,
        eval_freq= 1000,
        deterministic= True,
        render= False,                      # set to true for visual
        best_model_save_path= './logs/',
        log_path= './logs/'
    )

    # let the DQN agent learn on the wrapped env, but use the custom callback (non-wrapped env) for evaluation
    #dqn_agent.learn(10000, callback= eval_callback)



if __name__ == "__main__":
    main()
    pass