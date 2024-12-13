"""
Main file
"""

import gymnasium as gym

from algorithms.NGU_system import NGU
from algorithms.DQN_agent import DQN

def main():

    env = gym.make("MiniGrid-Empty-16x16-v0")
    # Dict('direction': Discrete(4), 'image': Box(0, 255, (7, 7, 3), uint8), 'mission': MissionSpace(<function EmptyEnv._gen_mission at 0x161d813a0>, None))
    state_shape = env.observation_space['image'].shape
    num_actions = env.action_space
    print(num_actions)
    # (7, 7, 3)

    # basic starting value
    learning_rate = 0.001

    # create DQN agent
    # my_dqn_agent = DQN(state_shape,num_actions, learning_rate)



if __name__ == "__main__":
    main()
    pass