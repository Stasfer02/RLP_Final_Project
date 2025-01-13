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
from sklearn.neighbors import NearestNeighbors
import gymnasium as gym
import numpy as np
import logging

import tensorflow as tf
from keras import models, layers

from gymnasium.core import WrapperActType, WrapperObsType, ObsType, ActType, Any


class NGU_env_wrapper(gym.Wrapper):
    """
    Wrapper class for a(ny) gymnasium environment to add the NGU reward system.
    Initially built on the "Simple" and "dynamic-obstacles" environments.
    """
    def __init__(self, env: gym.Env[ObsType, ActType], beta:float =0.001, alpha:float= 0.1, eta:float = 40, L:float = 5.0, k:int = 10):
        """
        initialize the wrapper.
        
        The arguments are:
        env: The Gymnasium environment.
        beta: the meta-controller to balance extrinsic and intrinsic rewards.
        alpha: the scaling factor within the intrinsic reward.
        eta: the decay rate for the DoWhaM reward.
        L: reward scaling factor: for scaling the life-long novelty reward and the episodic reward. standard value = 5 (as mentioned in the paper)
        k: amount of k-nearest neighbours for the embedding network.
        """
        super().__init__(env)
        self.beta = beta

        # keep track of previous state for DoWhaM
        self.previous_state = None

        # initialize the additional reward agents with the hyperparameters.
        # to the intrinsic agent, we pass alpha, L and the observation- and action spaces.
        self.intrinsic_agent = intrinsic_agent(alpha, L, k, env.observation_space, env.action_space)
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
        intrinsic_reward = self.intrinsic_agent.get_reward(self.previous_state, next_state)
        DoWhaM_reward = self.DoWhaM_agent.get_reward(self.previous_state, action, next_state)

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
    - We need to preprocess the observation to a format that can be handled by the embedding network. The format also needs to be specified and forwarded on creation of the embedding network.
    - implement the life-long module
    """
    
    def __init__(self, alpha: float, L: float, k: int, obs_space, action_space):
        """
        initialize with the alfa (scaling factor for similarity) parameter.
        L is the scaling factor for the life-long reward.
        k is the amount of nearest neighbours

        TODO:
        input/output shapes for the embedding network? how to implement.
        """

        self.alpha = alpha
        self.L = L
        self.k = k

        # TODO what will the input- and output shape be based on the obs_space? we later need to preprocess these
        input_shape = (7,7,3)

        # create the embedding network
        self.embedding_network = self._create_embedding_network(input_shape, action_space.n)

        # k-nearest neighbours memory storage
        self.KNN = NearestNeighbors(n_neighbors= self.k, algorithm= 'auto')
        self.KNN_memory = np.empty(action_space.n,dtype=float)
    
    def _preproc_obs(self, obs_space):
        """
        Preprocess the observation, which is now ('direction', 'image', 'mission') to something that can be embedded. 
        TODO: how to include the direction vector. now we only take the image.
        """

        # ??? use one-hot encoder to expand the direction variable. now it is not just a single value (like: 1) but [0, 1, 0, 0]
        direction = obs_space['direction']
        onehot_encoded_dir = np.eye(4)[direction]

        img = obs_space['image']
        normalized_img = img / 255    # make sure values are between 0 and 1

        # TODO how should we add the direction?
        #processed_obs = np.concatenate([onehot_encoded_dir, normalized_img])
        #logging.debug(f"INTRINSIC AGENT: Processed observation {processed_obs}")

        # temporary sol: just the image
        processed_obs = normalized_img
        return processed_obs

    def _create_embedding_network(self, input_shape, output_shape):
        """
        We need to create a siamese embedding network, following the structure that can be found in
        Appendix H.1 of the paper. 

        TODO; TESTING
        """
        logging.debug("INTRINSIC AGENT: Creating the embedding network...")

        def create_partial_network(input_shape):
            # create part of the siamese network (twice)
            # kernel and stride values are fit to our env with a much lower-dimensional input space (compared to the env in the paper)
            inputs = layers.Input(shape= input_shape)
            logging.debug(f"{inputs.shape} input dimensions")

            x = layers.Conv2D(32, kernel_size= 3, strides= 2, activation='relu',padding='same')(inputs)
            logging.debug(f"{x.shape} dimensions after first Conv layer")

            x = layers.Conv2D(64, kernel_size= 2, strides= 1, activation='relu')(x)
            logging.debug(f"{x.shape} dimensions after second Conv layer")

            x = layers.Conv2D(64, kernel_size= 2, strides= 1, activation='relu')(x)
            logging.debug(f"{x.shape} dimensions after third Conv layer")

            # now flatten the 3D vector into 1D for the fully connected layer
            x = layers.Flatten()(x)
            x = layers.Dense(32, activation='relu')(x)
            logging.debug(f"{x.shape} final dimensions after connecting siamese networks through the flattened, dense layer")
            # return the entire model
            return models.Model(inputs, x)

        # the input for the siamese components are 2 similar inputs for t and t +1
        input_A = layers.Input(input_shape)
        input_B = layers.Input(input_shape)

        # create the siamese networks
        siamese_A = create_partial_network(input_shape)         # for state at timestep t
        siamese_B = create_partial_network(input_shape)         # for state at timestep t+1

        # process the inputs through both siamese components
        processed_A = siamese_A(input_A)
        processed_B = siamese_B(input_B)

        # merge them
        merged_model = layers.Concatenate()([processed_A, processed_B])

        # finish with fully connected layers (last layer has the dimensions of the output, this is different for the paper as the environment is different.)
        x = layers.Dense(128, activation='relu')(merged_model)
        x = layers.Dense(output_shape, activation='softmax')(x)
        logging.debug(f"{x.shape} Dimensions after connecting siamese networks")
        # return the entire model
        return models.Model(inputs=[input_A, input_B], outputs= x)

    def get_reward(self, previous_state: WrapperObsType, state: WrapperObsType) -> float:
        """
        calculate the intrinsic reward for some action.
        The episodic reward is scaled based on the life-long reward. as read in chapter 2 of the paper.
        """

        processed_state = self._preproc_obs(state)
        processed_prev_state = self._preproc_obs(previous_state)
        intrinsic_reward = self._episodic_reward(processed_prev_state, processed_state) * np.min( [np.max( [self._life_long_reward(), 1] ), self.L] )

        return intrinsic_reward
    
    def _episodic_reward(self, last_state, state):
        """
        calculate the episodic reward using the embedding network and k-nearest neighbours.

        TODO: implement
        Here we need to map to our embedding network.
        Then with K-Nearest Neighbours we find the reward.
        """

        last_state = np.expand_dims(last_state, axis=0)     # needed to expand dimension by extra batch dim, maybe this can be the direction vector?
        state = np.expand_dims(state, axis=0)

        embedded = self.embedding_network.predict([last_state, state])
        logging.debug(f"INTRINSIC: embedded: {embedded}, with shape: {embedded.shape}")

        self.KNN_memory = np.vstack([self.KNN_memory, embedded])
        logging.debug(f"INTRINSIC: KNN memory: {self.KNN_memory}\n")
        self.KNN.fit(self.KNN_memory)

        if len(self.KNN_memory) < self.k:
            # not enough neighbors, reward = 0
            return 0
        
        distances, _ = self.KNN.kneighbors(embedded)

        # reward is the inv of the mean distances (?) check paper. avoiding division by 0 by adding small term.
        reward = 1 / ((np.mean(distances)) + 1e-8) 
        return reward


    def _life_long_reward(self):
        """
        calculate the life-long (life = one episode) reward. 

        TODO: implement
        """
        reward = 0
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

        logging.debug(f"DOWHAM: state: {state}")
        if state == next_state:
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