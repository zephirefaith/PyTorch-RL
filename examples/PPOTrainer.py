#!/usr/env python

import argparse
import gym
import os
import sys
import pickle
import time
import math
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils import *
from models.mlp_policy import Policy
from models.mlp_critic import Value
from models.mlp_policy_disc import DiscretePolicy
from torch.autograd import Variable
from core.ppo import ppo_step
from core.common import estimate_advantages
from core.agent import Agent

class PPOTrainer(object):
    def __init__(self):

        super(PPOTrainer, self).__init__()
        # self.policy_args = {'state_dim': , 'action_dim': , 'hidden_size': (x, x), 'learning_rate': ,}
        # self.value_args = {'state_dim':, 'hidden_size': (x, x), 'learning_rate': ,}
        #

    def init_policy_net(self):
        """
        Initialize neural network policy with specified architecture
        """
        # self.policy_net = Policy(state_dim, action_dim, hidden_size=(x, x))
        # self.policy_optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=self.policy_lr)

    def init_value_net(self):
        """
        Initialize neural network value function with specified architecture
        """
        # self.value_net = Value(state_dim, hidden_size=(x, x))
        # self.value_optimizer = torch.optim.Adam(self.value_net.parameters(), lr=self.value_lr)

    def sample_action(self):
        """
        Return an action from a probabilistic policy in continuous
        action space

        @return: action - value of action to take
        """
        # action = PolicyNet(state = (state, goal))

    def update_policy(self, batch):
        """
        Update the parameters of policy neural network

        @param batch: minibatch of experiences from episode rollout
        """
        # separate into s, a, r, s' from batch
        # get state values from ValueNet
        # estimate advantages from rewards and values
        # shuffle the experiences and perform ppo_step() to update
