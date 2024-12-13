"""
The DQN network
"""
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from typing import List, Tuple

class DQN:
    def __init__(self, state_shape: Tuple, num_actions: int, learning_rate: float):
        """
        Initialize the DQN Algorithm.

        We create the model and the target model for stabilization.
        """
        self.state_shape = state_shape
        self.num_actions = num_actions
        self.learning_rate = learning_rate

        self.DQN_model = self.build_network()
        self.target_model = self.build_network()
        
        # compile the model with the Adam optimizer
        self.optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate)
        self.DQN_model.compile(optimizer=self.optimizer, loss='mse')

    def build_network(self) -> keras.models.Model:
        """
        building a model with the keras package.
        """
        
        # For now this was taken from GPT as a standard initialization of the DQN for image inputs, just to make it work. 
        # We need to further research how we want to size and structure the layers for this specific implementation.
        
        inputs = layers.Input(shape=self.state_shape)
        x = layers.Conv2D(32, (8, 8), strides=4, activation='relu')(inputs)     # feature mapping
        x = layers.Conv2D(64, (4, 4), strides=2, activation='relu')(x)          # "
        x = layers.Conv2D(64, (3, 3), strides=1, activation='relu')(x)          # "
        x = layers.Flatten()(x)                                                 # Flatten the features
        x = layers.Dense(512, activation='relu')(x)                             # Dense layer of large amount of neurons to learn more complex connections

        # Output layer for Q-values of each action
        outputs = layers.Dense(self.num_actions)(x)                             # 

        model = models.Model(inputs=inputs, outputs=outputs)
        return model

    def predict(self, state):
        """
        Predict Q-values for a state from our Neural Network model
        """

        return self.DQN_model.predict(state)

    
