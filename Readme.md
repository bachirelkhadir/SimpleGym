# Simple Gym environements

- *One action, zero observation, one timestep long, +1 reward every timestep**: This isolates the value network. If my agent can't learn that the value of the only observation it ever sees it 1, there's a problem with the value loss calculation or the optimizer.

- *One action, random +1/-1 observation, one timestep long, obs-dependent +1/-1 reward every time*: If my agent can learn the value in (1.) but not this one - meaning it can learn a constant reward but not a predictable one! - it must be that backpropagation through my network is broken.

- *One action, zero-then-one observation, two timesteps long, +1 reward at the end*: If my agent can learn the value in (2.) but not this one, it must be that my reward discounting is broken.

- *Two actions, zero observation, one timestep long, action-dependent +1/-1 reward*: The first env to exercise the policy! If my agent can't learn to pick the better action, there's something wrong with either my advantage calculations, my policy loss or my policy update. That's three things, but it's easy to work out by hand the expected values for each one and check that the values produced by your actual code line up with them.

- *Two actions, random +1/-1 observation, one timestep long, action-and-obs dependent +1/-1 reward: Now we've got a dependence on both obs and action. The policy and value networks interact here, so there's a couple of things to verify*: that the policy network learns to pick the right action in each of the two states, and that the value network learns that the value of each state is +1. If everything's worked up until now, then if - for example - the value network fails to learn here, it likely means your batching process is feeding the value network stale experience.
