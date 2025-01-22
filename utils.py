"""
additional functions for tracking/plotting data.

We create a custom callback function to keep track of rewards/timesteps per run.
"""


from stable_baselines3.common.callbacks import BaseCallback
import numpy as np
import matplotlib.pyplot as plt

from scipy.ndimage import gaussian_filter1d

class CustomCallback(BaseCallback):
    def __init__(self):
        """
        initialize with memory for rewards and timesteps
        """
        super().__init__(verbose=0)
        
        self.rewards = []
    
    def _on_step(self) -> bool:
        # Check if an episode has finished and get the reward and amount of timesteps
        if "episode" in self.locals["infos"][0]:

            episode_reward = self.locals["infos"][0]["episode"]["r"]
            
            # store in memory
            self.rewards.append(episode_reward)

        return True  # Continue training
    
    def get_results(self):
        return self.rewards
    

def calculate_means_stds(data_rewards):
    """
    input is our 2D array with different lengths for the reward arrays. This depends on how many timesteps the DQN took to perform the episodes.
    We take the minimum length for which all the simulations contain data and evaluate the means of those.

        
    # we first find the maximum length
    max_length = max(len(x) for x in data_rewards)

    # then we fill the missing values with NaN values that will be ignored for mean calculation
    padded_rewards = np.array([r + [np.nan] * (max_length - len(r)) for r in data_rewards])
    means = np.nanmean(padded_rewards, axis=0)
    stds = np.nanstd(padded_rewards, axis=0)


    """

    min_length = min(len(x) for x in data_rewards)

    shortened_data = np.array([sim[:min_length] for sim in data_rewards])

    # calculate the mean and std values
    means = np.mean(shortened_data, axis=0)
    stds = np.std(shortened_data, axis=0)

    return means, stds


def create_plot(means, stds, storage_path) -> None:
    x_values = np.arange(len(means))

    # apply smoothening
    means = gaussian_filter1d(means, sigma=2)

    plt.figure(figsize=(20,10))
    plt.plot(x_values, means,color="blue",linewidth= 0.5, label="Mean reward")
    plt.fill_between(x_values, means - stds, means + stds, color="blue", alpha= 0.25)
    plt.xlabel("Episodes")
    plt.ylabel("Reward")
    plt.savefig(storage_path)
    plt.close()