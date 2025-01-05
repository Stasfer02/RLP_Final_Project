"""
The NGU reward system.

We will implement the NGU reward system by building a custom wrapper function for the Gymnasium environment. 
A detailed explanation on these wrapper functions can be found here:
https://gymnasium.farama.org/api/wrappers/


In our wrapper function, we extend the normal environment reward with an instrinsic reward (and DoWhaM addition). 
Most importantly, we rewrite the step method which is normally given as: (taken from: https://gymnasium.farama.org/_modules/gymnasium/core/#Wrapper.step)

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
from gymnasium.core import WrapperActType, WrapperObsType, ObsType, ActType, Any


class NGU_env_wrapper(gym.Wrapper):
    """
    Wrapper class for a(ny) gymnasium environment to add the NGU reward system.
    Initially built on the "Simple" and "dynamic-obstacles" environments.
    """
    def __init__(self, env: gym.Env[ObsType, ActType], beta:float =0.001, alpha:float= 0.1, eta:float = 40, L:float = 5.0):
        """
        initialize the wrapper.
        
        The arguments are:
        env: The Gymnasium environment.
        beta: the meta-controller to balance extrinsic and intrinsic rewards.
        alpha: the scaling factor within the intrinsic reward.
        eta: the decay rate for the DoWhaM reward.
        L: reward scaling factor: for scaling the life-long novelty reward and the episodic reward. standard value = 5 (as mentioned in the paper)
        """
        super().__init__(env)
        self.beta = beta

        # keep track of previous state for DoWhaM
        self.previous_state = None

        # initialize the additional reward agents with the hyperparameters.
        self.intrinsic_agent = intrinsic_agent(alpha, L)
        self.DoWhaM_agent = DoWhaM_agent(eta)

    def step(self, action: WrapperActType) -> tuple[WrapperObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        """
        The updated step function. The framework is taken from the standard step function. 

        Arguments:
        - the action taken

        We store the current state and then perform a step in the initialized environment with that action, and store the returns accordingly.
        We preserve everything except for the reward, which we will process by performing our NGU addition.
        """

        # take a step in the environment adn store the returns
        next_state, extrinsic_reward, terminated, truncated, info = self.env.step(action)

        # get the intrinsic and DoWhaM rewards
        intrinsic_reward = self.intrinsic_agent.get_reward(next_state)
        DoWhaM_reward = self.DoWhaM_agent.get_reward(self.previous_state, next_state)

        # calculate the total reward
        total_reward = extrinsic_reward + self.beta * intrinsic_reward + DoWhaM_reward

        # set the previous state to this state for next step, needed for DoWham
        self.previous_state = next_state

        return next_state, total_reward, terminated, truncated, info
    
    def reset(self, *, seed:int = None, options: dict[str, Any] = None) -> tuple[WrapperObsType, dict[str, Any]]:
        """
        Inherited reset method. 
        Forward the seeding/options to the env reset and return them after updating the previous state var.
        """
        observation, info =  self.env.reset(seed=seed,options=options)

        # when resetting (= initializing) the env. Update the previous state for DoWhaM
        self.previous_state = observation

        return observation, info


class intrinsic_agent:
    """
    agent for calculating the intrinsic reward

    TODO:
    Now we need to replace this with the system from the paper, using an embedding network and k-nearest for the episodic memory.
    And a random and prediction network for the life-long module.
    """
    
    def __init__(self, alpha: float, L: float):
        """
        initialize with the alfa (scaling factor for similarity) parameter.

        Create the episodic memory dictionary in which we will store our past experiences.

        TODO: Do we want to specify a certain length for this memory and consider the last x episodes?
        """
        self.alpha = alpha
        self.L = L
        self.episodic_memory: List[WrapperObsType] = []

    def get_reward(self, state: WrapperObsType) -> float:
        """
        calculate the intrinsic reward for some action.
        The episodic reward is scaled based on the life-long reward. as read in chapter 2 of the fundamental paper.
        """

        intrinsic_reward = self._episodic_reward * np.min( [np.max( [self._life_long_reward, 1] ), self.L] )
        return intrinsic_reward
    
    def _episodic_reward(self):
        """
        calculate the episodic reward using the embedding network and k-nearest neighbours.
        TODO: implement
        """

        reward = None
        return reward

    def _life_long_reward(self):
        """
        calculate the life-long (life = one episode) reward. 
        
        """
        reward = None
        return reward


class DoWhaM_agent:
    """
    DoWhaM additional reward.

    TODO:
    reset method?
    """

    def __init__(self, eta:float):
        """
        eta is the decay rate parameter. From the paper, we took a base value of 40. We will later tune this. appendix A.3 in paper: https://arxiv.org/pdf/2105.09992

        We need to keep track of two dictionaries:
        1. The amount of times an action has been taken
        2. The amount of times an action has been "effective"
        """
        self.eta:float = eta

        self.count_memory: dict = {}
        self.effect_memory: dict = {}

    def get_reward(self, state: WrapperObsType, action: WrapperActType, next_state: WrapperObsType) -> float:
        """
        getting the DoWhaM reward.
        """

        # get the action count U^H
        action_count = self._get_action_count(action)
        action_effect = self._get_action_effect(action)

        if np.array_equal(state, next_state):
            # the state has not been changed, so reward is 0
            return 0
        else:
            # effective state change occurred
            self._put_action_effect(action)
            
            # calculate bonus score
            B = ( self.eta ** (1 - (action_effect/action_count)) - 1) / (self.eta - 1)
            # calculate total episodes
            N = sum(self.count_memory.values())

            return B / np.sqrt(N)

    def _get_action_count(self, action: WrapperActType) -> int:
        """
        return and update/create the total count of the action. 
        """
        if action in self.count_memory:
            # retrieve the count and increment it for this iteration.
            self.count_memory[action] += 1
            action_count = self.count_memory[action]
        else:
            # action is new, 
            self.count_memory[action] = 1
            action_count = 1

        return action_count
    
    def _get_action_effect(self, action: WrapperActType) -> int:
        """
        return the effectiveness count of the action.
        """
        if action in self.effect_memory:
            return self.effect_memory[action]
        else:
            return 0
    
    def _put_action_effect(self, action: WrapperActType) -> None:
        """
        Increase/create effectiveness score for the action.
        """
        if action in self.effect_memory:
            self.effect_memory[action] += 1
        else:
            self.effect_memory[action] = 1