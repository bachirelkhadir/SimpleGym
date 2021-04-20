import gym
from gym import spaces
from gym.utils import seeding
import numpy as np


class OneA_ZeroO_OneT_OneR(gym.Env):
    """ One action, zero observation, one timestep long, +1 reward every timestep: This isolates the value network. If my agent can't learn that the value of the only observation it ever sees it 1, there's a problem with the value loss calculation or the optimizer."""
    def __init__(self):
        super().__init__()
        self.observation_space = spaces.Discrete(1)
        self.action_space = spaces.Discrete(1)
        self.state = None
        self.reward = 1

    def reset(self):
        self.state = self.observation_space.sample()
        return self.state

    def step(self, a):
        o, r, d, info = self.state, self.reward, True, {}
        return o, r, d, info


class OneA_RandomO_OneT_PmR(gym.Env):
    """*One action, random +1/-1 observation, one timestep long, obs-dependent +1/-1 reward every time*: If my agent can learn the value in (1.) but not this one - meaning it can learn a constant reward but not a predictable one! - it must be that backpropagation through my network is broken."""

    def __init__(self):
        super().__init__()
        self.observation_space = spaces.Discrete(2)
        self.action_space = spaces.Discrete(1)
        self.state = None
        self.reward = 1

    def reset(self):
        self.state = self.observation_space.sample()
        return self.state

    def step(self, a):
        self.state = self.observation_space.sample()
        r = 1 if self.state == 0 else -1
        return self.state, r, True, {}

class OneA_ZeroThenOneO_TwoT_OneEndR(gym.Env):
    """One action, zero-then-one observation, two timesteps long, +1 reward at the end*: If my agent can learn the value in (2.) but not this one, it must be that my reward discounting is broken."""

    def __init__(self):
        super().__init__()
        self.observation_space = spaces.Discrete(3)
        self.action_space = spaces.Discrete(1)
        self.state = None
        self.reward = 1

    def reset(self):
        self.state = 0
        return self.state

    def step(self, a):
        self.state = self.state + 1
        r = 1 if self.state == 2 else 0
        d = self.state == 2
        return self.state, r, d, {}


class TwoA_ZeroO_OneT_ActionDependentR(gym.Env):

    """Two actions, zero observation, one timestep long, action-dependent +1/-1 reward*: The first env to exercise the policy! If my agent can't learn to pick the better action, there's something wrong with either my advantage calculations, my policy loss or my policy update. That's three things, but it's easy to work out by hand the expected values for each one and check that the values produced by your actual code line up with them."""

    def __init__(self):
        super().__init__()
        self.observation_space = spaces.Discrete(1)
        self.action_space = spaces.Discrete(2)
        self.state = None

    def reset(self):
        self.state = 0
        return self.state

    def step(self, a):
        r = 1 if a == 0 else -1
        d = True
        return self.state, r, d, {}

class TwoA_RandomO_OneT_ActionDependentR(gym.Env):

    """
    Two actions, random +1/-1 observation, one timestep long, action-and-obs dependent +1/-1 reward:
    Now we've got a dependence on both obs and action. The policy and value networks interact here, so there's a couple of things to verify*: that the policy network learns to pick the right action in each of the two states, and that the value network learns that the value of each state is +1. If everything's worked up until now, then if - for example - the value network fails to learn here, it likely means your batching process is feeding the value network stale experience.
    """

    def __init__(self):
        super().__init__()
        self.observation_space = spaces.Discrete(2)
        self.action_space = spaces.Discrete(2)
        self.state = None

    def reset(self):
        self.state = self.observation_space.sample()
        return self.state

    def step(self, a):
        r = 1 if a == self.state else -1
        d = True
        return self.state, r, d, {}
