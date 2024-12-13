"""
The NGU reward system.

This could also be directly integrated into the DQN network, but we wanted to seperate the two as ideally, the NGU reward system can be 
added as an extension to any given Algorithm.
"""


class NGU:
    """
    NGU reward system to define the total reward that is then processed by a chosen algorithm for a certain state-action.
    It combines the standard extrinsic reward from some environment with an intrinsic reward aimed at exploration. 

    In this case, it is further extended with the DoWhaM reward, which tunes the reward even more to favor actions that actually result in effective change (reaching a new state), 
    not just change for the sake of it.
    """

    def __init__(self, beta_controller: float) -> None:
        """
        Initialize the NGU reward system.
        Specify the value of the beta controller
        TODO: this beta controller will later be modified by the Agent57 Meta controller(?) @Natalie

        Keep track of a memory list for the states we have visited before, 
        This will be needed for calculating the intrinsic reward and DoWhaM reward
        """
        self.beta_controller = beta_controller

        self.memory = []

    def compute_total_reward(self,previous_state, state, action,  extrisic_reward: float) -> float:
        """
        Return the total reward for a certain state. It should get the state and extrinsic reward from the environment as arguments.
        It also takes in the previous state, which will be used by DoWhaM to evaluate efficiency.

        Formula:
        total_reward = extrinsic_reward + beta * intrinsic reward + DoWhaM reward
        """

        intrinsic_reward = self._intrinsic_reward(state)
        DoWhaM_reward = self._DoWhaM_reward(state, previous_state)

        return extrisic_reward + self.beta_controller * intrinsic_reward + DoWhaM_reward

    def _intrinsic_reward(self, state) -> float:
        """
        Calcualate the intrinsic reward based on memory. 
        """
        pass

    def _DoWhaM_reward(self, state, previous_state, action) -> float:
        """
        Calculate the DoWhaM reward (Don't Do What Does Not Matter)
        Based on whether or not a state change occurs after some action.
        """
    

        pass