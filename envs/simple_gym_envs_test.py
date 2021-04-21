#!/usr/bin/env python3

import gym
from simple_gym_envs import *

ENVS = [OneA_ZeroO_OneT_OneR,
        OneA_RandomO_OneT_PmR,
        OneA_ZeroThenOneO_TwoT_OneEndR,
        TwoA_ZeroO_OneT_ActionDependentR,
        TwoA_RandomO_OneT_ActionDependentR,
        C_OneA_ZeroO_OneT_ActionDependentR]


for env_name in ENVS:
    print("=== Test: {} ===".format(env_name))


    env = env_name() #gym.make(env_name)
    dummy_policy = lambda _: env.observation_space.sample()
    dummy_baseline = lambda _: np.random.randn()
    env.random_start = False
    print("Obs:", env.observation_space)
    print("Action:", env.action_space)
    print("Error/policy:", env.error_policy(dummy_policy))
    print("Error/baseline:", env.error_policy(dummy_baseline))

    env.reset()

    for i in range(5):
        a = env.action_space.sample()
        o, r, d, info = env.step(a)
        print("Obs: {}, Action: {}, Reward: {}, Done flag: {}, Info: {}".format(o, a, r, d, info))
        if d:
            print("Reset")
            env.reset()

    env.close()
    del env
