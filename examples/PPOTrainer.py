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
