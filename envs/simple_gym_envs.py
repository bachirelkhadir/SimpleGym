import gym
from gym import spaces
from gym.utils import seeding
import numpy as np

class SimpleEnv:

    def correct_value_fn(self, s):
        return 0

    def error_value_fn(self, v):
        """
        How far is v from the true baseline?
        """
        return np.mean(np.abs([v(s) - self.correct_value_fn(s) for s in
                        [self.observation_space.sample() for _ in range(1000)]]))


    def error_policy(self, pi):
        """
        How far is v from the true baseline?
        """
        return 0


class OneA_ZeroO_OneT_OneR(gym.Env, SimpleEnv):
    """One action, zero observation, one timestep long, +1 reward every timestep:
    This isolates the value network. If my agent can't learn that the value of the
    only observation it ever sees it 1, there's a problem with the value loss
    calculation or the optimizer.
    """

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

    def correct_value_fn(self, s):
        return 1.



class OneA_RandomO_OneT_PmR(gym.Env, SimpleEnv):
    """*One action, random +1/-1 observation, one timestep long, obs-dependent
+1/-1 reward every time*: If my agent can learn the value in (1.) but not this
one - meaning it can learn a constant reward but not a predictable one! - it
must be that backpropagation through my network is broken."""

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
        r = 1 if self.state == 0 else -1
        self.state = self.observation_space.sample()
        return self.state, r, True, {}

    def correct_value_fn(self, s):
        return [1., -1][s]


class OneA_ZeroThenOneO_TwoT_OneEndR(gym.Env, SimpleEnv):
    """One action, zero-then-one observation, two timesteps long, +1 reward at
    the end*: If my agent can learn the value in (2.) but not this one, it must
    be that my reward discounting is broken.

    S: 0 -> 1 -> 2 R: 0 -> 1 -> 0 """

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
        r = 1 if self.state == 1 else 0
        self.state = self.state + 1
        d = self.state == 2
        return self.state, r, d, {}

    def correct_value_fn(self, s):
        return 1. if s == 1 else 0.


class TwoA_ZeroO_OneT_ActionDependentR(gym.Env, SimpleEnv):

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

    def correct_value_fn(self, s):
        return 1.

    def error_policy(self, pi):
        return np.mean(np.abs([pi(s) - 0. for s in
                        [self.observation_space.sample() for _ in range(1000)]]))


class TwoA_RandomO_OneT_ActionDependentR(gym.Env, SimpleEnv):

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

    def correct_value_fn(self, s):
        return 1.


    def error_policy(self, pi):
        # optimal action = state
        return np.mean(np.abs([pi(s) - s for s in
                        [self.observation_space.sample() for _ in range(1000)]]))


# Continuous Envs
class C_OneA_ZeroO_OneT_ActionDependentR(gym.Env, SimpleEnv):
    """One action a in (-1, 1), zero observation, one timestep long,
    action-dependent reward r = f(a) f = 2*exp(-(x-0.5)**2/.1)+
    exp(-(x+0.5)**2/.1)


    Optimal solution
    pi(s) = 1/2
    V(s) = 2

    """

    def __init__(self):
        super().__init__()
        self.observation_space = spaces.Box(-1, 1, shape=(1,))
        self.action_space = spaces.Box(-1, 1, shape=(1,))
        self.reward_fn = lambda x: 2*np.exp(-(x-0.5)**2/.1)+ np.exp(-(x+0.5)**2/.1)
        self.best_action = 0.5 # maximizer

    def reset(self):
        self.state = self.observation_space.sample()
        return self.state

    def step(self, a):
        r = self.reward_fn(a)
        d = True
        return self.state, r, d, {}

    def correct_value_fn(self, s):
        return self.reward_fn(self.best_action)

    def error_policy(self, pi):

        return np.mean(np.abs([pi(s) - 0.5 for s in
                        [self.observation_space.sample() for _ in range(1000)]]))

# Local Variables:
# fill-column: 80
# End:
