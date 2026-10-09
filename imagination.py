import warnings

import numpy as np
import torch
from gym.spaces import Box


class Imagination:
    def __init__(self, model, n_actors, horizon, measure):
        """
        Imaginary MDP simulated with the learnt dynamics model, in which SAC learns the exploration / task policies.

        Args:
            model: dynamics model (see models.py)
            n_actors: number of parallel episodes
            horizon: length of the episode
            measure: the reward function, either an exploration utility or a task measure
        """
        self.model = model
        self.n_actors = n_actors
        self.horizon = horizon
        self.measure = measure

        self.action_space = Box(low=-1.0, high=1.0, shape=(n_actors, self.model.d_action), dtype=np.float32)
        self.action_space.seed(np.random.randint(np.iinfo(np.uint32).max))

        self.init_state = None
        self.states = None
        self.steps = None

    def step(self, actions):
        actions = actions.to(self.model.device)

        with torch.no_grad():
            next_state_means, next_state_vars = self.model.forward_all(self.states, actions)

        if self.model.predictive == 'moments':
            # sample from the moment-matched Gaussian (epistemic + aleatoric variance)
            next_states = self.model.sample(next_state_means, next_state_vars[0] + next_state_vars[1])
        else:
            # pick a random posterior sample (ensemble member / forward pass / weight sample) for each actor
            n_act, n_samples = next_state_means.shape[:2]
            i = torch.arange(n_act).to(self.model.device)
            j = torch.randint(n_samples, size=(n_act,)).to(self.model.device)
            next_states = self.model.sample(next_state_means[i, j], next_state_vars[i, j])

        if torch.any(torch.isnan(next_states)).item():
            warnings.warn("NaN in sampled next states!")
        if torch.any(torch.isinf(next_states)).item():
            warnings.warn("Inf in sampled next states!")

        measures = self.measure(self.states, actions, next_states, next_state_means, next_state_vars, self.model)

        self.states = next_states
        self.steps += 1
        done = self.steps >= self.horizon

        return next_states, measures, done, {}

    def reset(self):
        states = torch.from_numpy(self.init_state).float()
        states = states.unsqueeze(0).repeat(self.n_actors, 1)
        self.states = states.to(self.model.device)                  # shape: (n_actors, d_state)
        self.steps = 0
        return self.states

    def update_init_state(self, state):
        self.init_state = state
