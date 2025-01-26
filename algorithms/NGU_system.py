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
tf.random.set_seed(1234)

from keras import models, layers

from gymnasium.core import WrapperActType, WrapperObsType, ObsType, ActType, Any


class NGU_env_wrapper(gym.Wrapper):
    """
    Wrapper class for a(ny) gymnasium environment to add the NGU reward system.
    Initially built on the "Simple" and "dynamic-obstacles" environments.
    """
    def __init__(self, env: gym.Env[ObsType, ActType], beta:float =0.3, L:float = 5.0, k:int = 10, eta:float = 30, useNGU:bool = True, useDoWhaM: bool= True):
        """
        initialize the wrapper.
        
        The arguments are:
        env: The Gymnasium environment.
        beta: the meta-controller to balance extrinsic and intrinsic rewards.                               value of 0.3 is taken directly from the paper
        eta: the decay rate for the DoWhaM reward.                                                          value of 40 is taken directly from the paper
        L: reward scaling factor: for scaling the life-long novelty reward and the episodic reward.         value of 5 is taken directly from the paper
        k: amount of k-nearest neighbours for the embedding network.                                        value of 10 is taken directly from the paper 
        """
        super().__init__(env)
        self.beta = beta

        self.useNGU = useNGU
        self.useDoWhaM = useDoWhaM

        # keep track of previous state for DoWhaM
        self.previous_state = None
        self.extrinsic_reward = None

        logging.info(f"NGU hyperparameters: beta={beta}, eta={eta}, L={L}, k={k}. init with NGU={useNGU} and DoWhaM={useDoWhaM}")
        # initialize the additional reward agents with the hyperparameters.
        # to the intrinsic agent, we pass alpha, L and the observation- and action spaces.
        if useNGU:
            self.intrinsic_agent = intrinsic_agent(L, k, env.observation_space, env.action_space)
        
        if useDoWhaM:
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
        
        self.extrinsic_reward = extrinsic_reward
        # get the intrinsic and DoWhaM rewards
        if self.useNGU:
            intrinsic_reward = self.intrinsic_agent.get_reward(self.previous_state, next_state)
        
        if self.useDoWhaM:
            DoWhaM_reward = self.DoWhaM_agent.get_reward(self.previous_state, action, next_state)

        # calculate the total reward
        if self.useNGU and self.useDoWhaM:
            # DQN + NGU + DoWhaM
            total_reward = extrinsic_reward + self.beta * intrinsic_reward + DoWhaM_reward
        elif self.useNGU and not self.useDoWhaM:
            # DQN + NGU
            total_reward = extrinsic_reward + self.beta *intrinsic_reward
        else:
            # DQN + DoWhaM
            total_reward = extrinsic_reward + DoWhaM_reward

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

        # TODO: when we start training
        # The game is started, so clear the KNN-memory and DoWhaM memory (and reset networks?)

        return observation, info
    
    def get_extrinsic_reward(self):
        return self.extrinsic_reward


class intrinsic_agent:
    """
    agent for calculating the intrinsic reward

    #TODO
    HOW TO TRAIN THE NETWORKS? 
    -> We need to backpropagate the action to the embedding network.
    -> For the life-long predictor network, we need to backpropagate the error.
    """
    
    def __init__(self, L: float, k: int, obs_space, action_space):
        """
        L is the scaling factor for the life-long reward.
        k is the amount of nearest neighbours

        """
        self.L = L
        self.k = k

        # TODO what will the input- and output shape be based on the newly formed, preprocessed obs_space?
        input_shape = obs_space.shape

        # life-long module, random and predictor networks
        self.LL_random_network = self._create_LL_network(input_shape, 64)
        self.LL_predictor_network = self._create_LL_network(input_shape, 64)
        self.LL_step_counter = 0
        self.LL_running_mean = 0
        self.LL_running_std = 1
        self.LL_beta = 0.99  # Decay factor for running mean and std

        # episodic module, embedding network
        self.embedding_network = self._create_embedding_network(input_shape, action_space.n)

        # episodic module, k-nearest neighbors memory storage
        self.KNN = NearestNeighbors(n_neighbors= self.k, algorithm= 'auto')
        self.KNN_memory = np.empty((0,action_space.n),dtype=float)

    def _create_embedding_network(self, input_shape, output_shape):
        """
        We need to create a siamese embedding network, following the structure that can be found in
        Appendix H.1 of the paper. 
        However, because the observation space of our environment is much smaller, we needed to decrease kernel & stride sizes
        """
        logging.info("Creating the embedding network...")

        def create_partial_network(input_shape):
            # create part of the siamese network (twice)
            
            inputs = layers.Input(shape= (input_shape[0],1))
            logging.debug(f"{inputs.shape} siamese network: input dimensions")

            x = layers.Conv1D(32, kernel_size= 3, strides= 2, activation='relu',padding='same')(inputs)
            x = layers.Conv1D(64, kernel_size= 2, strides= 1, activation='relu')(x)
            x = layers.Conv1D(64, kernel_size= 2, strides= 1, activation='relu')(x)

            # now flatten the 3D vector into 1D for the fully connected layer
            x = layers.Flatten()(x)
            x = layers.Dense(32, activation='relu')(x)
            logging.debug(f"{x.shape} siamese network: final dimensions after connecting siamese networks through the flattened, dense layer")
            # return the entire model
            return models.Model(inputs, x)

        # the input for the siamese components are 2 similar inputs for t and t +1
        input_A = layers.Input(shape=(input_shape[0],1))
        input_B = layers.Input(shape=(input_shape[0],1))

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
        logging.debug(f"{x.shape} embedding network: Dimensions after connecting siamese networks")

        # return the entire model
        return models.Model(inputs=[input_A, input_B], outputs= x)

    def _create_LL_network(self, input_shape, output_shape):
        """
        creating the random and predictor networks for the Life-Long module. More info in the life-long-reward method.
        The design is based on the one in the paper. It is (obviously) identical for the random and predictor network.
        
        The only change we made was in the kernel and stride sizes, as our observation space differs from that in the paper, so we needed to decrease those.
        """
        logging.info("Creating the Life-Long random network...")

        input = layers.Input(shape=(input_shape[0],1))
        x = layers.Conv1D(32, kernel_size= 3, strides= 2, activation='linear')(input)
        x = layers.Conv1D(64, kernel_size= 2, strides= 1, activation='relu')(x)
        x = layers.Conv1D(64, kernel_size= 2, strides= 1, activation='relu')(x)
        x = layers.Flatten()(x)
        x = layers.Dense(128, activation='relu')(x)

        # output layer
        x = layers.Dense(output_shape, activation='linear')(x)

        return models.Model(inputs=input, outputs= x)

    def get_reward(self, previous_state: WrapperObsType, state: WrapperObsType) -> float:
        """
        calculate the intrinsic reward for some action.
        The episodic reward is scaled based on the life-long reward. as read in chapter 2 of the paper.
        
        The life-long reward is min/max scaled between 1 and hyperparameter L
        """

        intrinsic_reward = self._episodic_reward(previous_state, state) * np.min( [np.max( [self._life_long_reward(state), 1] ), self.L] )

        return intrinsic_reward
    
    def _episodic_reward(self, last_state, state):
        """
        calculate the episodic reward using the embedding network and k-nearest neighbours.
        """
        last_state = np.expand_dims(last_state,axis=-1)
        last_state = np.expand_dims(last_state,axis=0)

        state = np.expand_dims(state, axis=-1)
        state = np.expand_dims(state, axis=0)

        embedded = self.embedding_network.predict([last_state, state],verbose=0)
        logging.debug(f" output of embedded network for reward of state: {embedded}, with shape: {embedded.shape}")

        self.KNN_memory = np.vstack([self.KNN_memory, embedded])
        logging.debug(f"KNN memory: {self.KNN_memory}\n")
        self.KNN.fit(self.KNN_memory)

        if len(self.KNN_memory) < self.k:
            # not enough neighbors, reward = 0
            return 0
        
        distances, _ = self.KNN.kneighbors(embedded)

        # reward is the inv of the mean distances (?) check paper. avoiding division by 0 by adding small term.
        reward = 1 / ((np.mean(distances)) + 1e-8) 
        return reward

    def _life_long_reward(self, state):
        """
        calculate the life-long (life = one episode) reward. 

        We pass the state through our random network and predictor network. 
        The random network returns a random collection of values, which the predictor network will try to match by minimizing the MSE.
        Thus, for states that are visited often, the predictor network will be better at this prediction, resulting in a lower error. 

        This error is scaled and returned as the life-long reward.

        TODO: Update predictor network how?
        """
        state = np.expand_dims(state, axis=-1)
        state = np.expand_dims(state, axis=0)

        random_output = self.LL_random_network(state)
        predictor_output = self.LL_predictor_network(state)

        error = np.mean(np.square(random_output - predictor_output)) # ||g_hat(x_t) - g(x_t)||^2

        logging.debug(f"Life-Long: calculated random and predictor output: \n random: {random_output} \n predictor: {predictor_output}")

        # calculate & update the running mean and standard dev
        self.LL_step_counter += 1
        old_mean = self.LL_running_mean
        self.LL_running_mean = (error - old_mean) / self.LL_step_counter
        self.LL_running_std = self.LL_running_std + ((error - old_mean) * (error - self.LL_running_mean))

        # calculate & return the reward
        reward = 1 + ((error - self.LL_running_mean) / self.LL_running_std)
        return reward


class DoWhaM_agent:
    """
    DoWhaM additional reward modification.
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