"""
The NGU reward system.

We will implement the NGU reward system by building a custom wrapper function for the Gymnasium environment. 
A detailed explanation on these wrapper functions can be found here:
https://gymnasium.farama.org/api/wrappers/


In our wrapper function, we extend the normal environment reward with an instrinsic reward (and DoWhaM addition). 
More specifically, we rewrite the step function which is normally given as: (taken from: https://gymnasium.farama.org/_modules/gymnasium/core/#Wrapper.step)

    def step(
        self, action: WrapperActType
    ) -> tuple[WrapperObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        return self.env.step(action)

Of which the arguments and returns are given as:

        Args:
            action (ActType): an action provided by the agent to update the environment state.

        Returns:
            observation (ObsType): An element of the environment's :attr:`observation_space` as the next observation due to the agent actions.
                An example is a numpy array containing the positions and velocities of the pole in CartPole.
            reward (SupportsFloat): The reward as a result of taking the action.
            terminated (bool): Whether the agent reaches the terminal state (as defined under the MDP of the task)
                which can be positive or negative. An example is reaching the goal state or moving into the lava from
                the Sutton and Barto Gridworld. If true, the user needs to call :meth:`reset`.
            truncated (bool): Whether the truncation condition outside the scope of the MDP is satisfied.
                Typically, this is a timelimit, but could also be used to indicate an agent physically going out of bounds.
                Can be used to end the episode prematurely before a terminal state is reached.
                If true, the user needs to call :meth:`reset`.
            info (dict): Contains auxiliary diagnostic information (helpful for debugging, learning, and logging).
                This might, for instance, contain: metrics that describe the agent's performance state, variables that are
                hidden from observations, or individual reward terms that are combined to produce the total reward.
                In OpenAI Gym <v26, it contains "TimeLimit.truncated" to distinguish truncation and termination,
                however this is deprecated in favour of returning terminated and truncated variables.
            done (bool): (Deprecated) A boolean value for if the episode has ended, in which case further :meth:`step` calls will
                return undefined results. This was removed in OpenAI Gym v26 in favor of terminated and truncated attributes.
                A done signal may be emitted for different reasons: Maybe the task underlying the environment was solved successfully,
                a certain timelimit was exceeded, or the physics simulation has entered an invalid state.
"""

from typing import TYPE_CHECKING, Any, SupportsFloat, List
from scipy.spatial.distance import euclidean

import gymnasium as gym
import numpy as np
from gymnasium.core import WrapperActType, WrapperObsType, ObsType, ActType


class NGU_env_wrapper(gym.Wrapper):

    def __init__(self, env: gym.Env[ObsType, ActType], beta:float =0.001):
        """
        initialize the wrapper.
        
        Use the super function to inherit the init method from the standard gym Wrapper.
        Store "beta", our meta-controller
        """
        super.__init__(env)
        self.beta = beta

        self.intrinsic_agent = intrinsic_agent()
        self.DoWhaM_agent = DoWhaM_agent()

    def step(self, action: WrapperActType) -> tuple[WrapperObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        """
        The updated step function. The framework is taken from the standard step function. 

        Arguments:
        - the action taken

        We store the current state and then perform a step in the initialized environment with that action, and store the returns accordingly.
        We preserve everything except for the reward, which we will process by performing our NGU addition.

        """

        # the current state is the observation space
        current_state = self.env.observation_space()

        # take a step in the environment adn store the returns
        next_state, extrinsic_reward, terminated, truncated, info = self.env.step(action)

        # get the intrinsic reward
        intrinsic_reward = self.intrinsic_agent.get_reward(next_state)

        # get the DoWhaM reward
        DoWhaM_reward = self.DoWhaM_agent.get_reward(current_state, next_state)

        # calculate the total reward
        total_reward = extrinsic_reward + self.beta * intrinsic_reward + DoWhaM_reward

        return next_state, total_reward, terminated, truncated, info

    
    def _calc_DoWhaM(self) -> float:
        """
        TODO
        Private method for calculating the DoWhaM additional reward.
        """

        return 0

class intrinsic_agent:
    """
    agent for calculating the intrinsic reward
    """
    
    def __init__(self, alfa: float= 0.1):
        """
        initialize with the alfa (scaling factor for similarity) parameter.

        Create the episodic memory dictionary in which we will store our past experiences.

        TODO: Do we want to specify a certain length for this memory and consider the last x episodes?
        """
        self.episodic_memory: List[WrapperObsType] = []

    def get_reward(self, state: WrapperObsType) -> float:
        """
        calculate the intrinsic reward for some action.
        """

        # first we append the current state to the episodic memory
        self.episodic_memory.append(state)

        # then we calculate it's similarity to the entire set of past states in memory
        total_similarity = 0
        for prev_state in self.episodic_memory:
            similarity = self._calc_Euclidean_distance(state, prev_state)
            total_similarity += similarity
        
        return 1 / np.sqrt(total_similarity + 1e-8) # adding a small value to the root to avoid dividing by 0.
    
    def _calc_Euclidean_distance(self, state: WrapperObsType, prev_state: WrapperObsType):
        """
        We evaluate similarity using the Euclidean distance function. 
        While this is generally best for vector representations, we think it will suffice for this instance of image representation.
        Mainly because the representations are not too complicated and similar states have very similar pixelated images.

        We use the ".flatten()" operator to transform our 3D representations of states (observation space) to a one-dimensional one.
        Now we can evaluate the euclidean distance.
        """
        return euclidean(state.flatten(), prev_state.flatten())

class DoWhaM_agent:
    """
    DoWhaM additional reward.
    """

    def __init__(self):
        
        pass

    def get_reward(self, state: WrapperObsType, next_state: WrapperObsType):
        """
        TODO
        actual implementation
        """
        if state == next_state:
            return 1
        return 0