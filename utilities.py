"""
Exploration utilities u(s, a): the reward that the exploration policy maximises inside the learnt model.

    * JensenRenyiDivergenceUtilityMeasure -- Jensen-Renyi divergence of the sample-based predictive mixture (MAX)
    * GaussianEntropyUtilityMeasure       -- our entropy metric of the epistemic predictive distribution
    * PredictionErrorUtilityMeasure       -- prediction error of the model (PERX, reactive baseline)
"""
import warnings

import numpy as np
import torch


class Measure:
    def __call__(self, states, actions, next_states, next_state_means, next_state_vars, model):
        """
        Args:
            states: (n_actors, d_state)
            actions: (n_actors, d_action)
            next_states: (n_actors, d_state), sampled or observed next states
            next_state_means, next_state_vars: output of `model.forward_all(states, actions)`
            model: the dynamics model

        Returns:
            measure: (n_actors)
        """
        raise NotImplementedError


class UtilityMeasure(Measure):
    def __init__(self, action_norm_penalty=0):
        self.action_norm_penalty = action_norm_penalty

    def compute_utility(self, states, actions, next_states, next_state_means, next_state_vars, model):
        raise NotImplementedError

    def __call__(self, states, actions, next_states, next_state_means, next_state_vars, model):
        utility = self.compute_utility(states, actions, next_states, next_state_means, next_state_vars, model)

        if not np.allclose(self.action_norm_penalty, 0):
            action_norms = (actions ** 2).sum(dim=1)                               # shape: (n_actors)
            utility = utility - self.action_norm_penalty * action_norms

        if torch.any(torch.isnan(utility)).item():
            warnings.warn("NaN in utilities!")
        if torch.any(torch.isinf(utility)).item():
            warnings.warn("Inf in utilities!")
        return utility


class JensenRenyiDivergenceUtilityMeasure(UtilityMeasure):
    def __init__(self, decay, action_norm_penalty=0):
        """
        Jensen-Renyi divergence (alpha = 2) of a mixture of Gaussians, which has a closed form (Shyam et al. 2019).
        Requires a model with predictive == 'samples'.

        Args:
            decay: shrinkage of the aleatoric variances towards their upper bound
        """
        super().__init__(action_norm_penalty=action_norm_penalty)
        self.decay = decay

    def rescale_var(self, var, max_log_var):
        max_var = np.exp(max_log_var)
        return max_var - self.decay * (max_var - var)

    def compute_utility(self, states, actions, next_states, next_state_means, next_state_vars, model):
        state_delta_means = next_state_means - states.to(next_state_means.device).unsqueeze(1)
        mu = model.normalizer.renormalize_state_delta_means(state_delta_means)  # shape: (n_actors, N, d_state)
        var = model.normalizer.renormalize_state_delta_vars(next_state_vars)    # shape: (n_actors, N, d_state)
        n_act, es, d_s = mu.size()

        var = self.rescale_var(var, model.max_log_var)

        # entropy of the mean
        mu_diff = mu.unsqueeze(1) - mu.unsqueeze(2)                           # shape: (n_actors, N, N, d_state)
        var_sum = var.unsqueeze(1) + var.unsqueeze(2)                         # shape: (n_actors, N, N, d_state)

        err = torch.sum(mu_diff * 1 / var_sum * mu_diff, dim=-1)              # shape: (n_actors, N, N)
        det = torch.sum(torch.log(var_sum), dim=-1)                           # shape: (n_actors, N, N)

        log_z = -0.5 * (err + det)
        log_z = log_z.reshape(n_act, es * es)                                 # shape: (n_actors, N * N)
        mx, _ = log_z.max(dim=1, keepdim=True)                                # log-sum-exp trick
        log_z = log_z - mx
        exp = torch.exp(log_z).mean(dim=1, keepdim=True)
        entropy_mean = (-mx - torch.log(exp))[:, 0]                           # shape: (n_actors)

        # mean of entropies
        total_entropy = torch.sum(torch.log(var), dim=-1)                     # shape: (n_actors, N)
        mean_entropy = total_entropy.mean(dim=1) / 2 + d_s * np.log(2) / 2    # shape: (n_actors)

        return entropy_mean - mean_entropy


class GaussianEntropyUtilityMeasure(UtilityMeasure):
    def __init__(self, scale=100., u_min=0., u_max=3.5, action_norm_penalty=0):
        """
        Our entropy metric (Sec. IV-B/C of the paper): the differential entropy of the epistemic part of the
        moment-matched predictive Gaussian, N(mu, diag(var_epistemic)), computed in normalised state-delta space.
        Assuming homoscedastic aleatoric noise, maximising it is equivalent to maximising the information gain.

        The entropy is squashed to [u_min, u_max] with a sigmoid of temperature `scale` to keep the exploration
        reward bounded for SAC. Requires a model with predictive == 'moments'.
        """
        super().__init__(action_norm_penalty=action_norm_penalty)
        self.scale = scale
        self.u_min = u_min
        self.u_max = u_max

    def compute_utility(self, states, actions, next_states, next_state_means, next_state_vars, model):
        var_epistemic = model.normalizer.renormalize_state_delta_vars(next_state_vars[0])   # (n_actors, d_state)
        d_s = var_epistemic.size(-1)

        # entropy of a Gaussian with diagonal covariance: 0.5 * (d (1 + log 2 pi) + log |Sigma|)
        entropy = 0.5 * (d_s * (1 + np.log(2 * np.pi)) + torch.log(var_epistemic).sum(dim=-1))

        return self.u_min + torch.sigmoid(entropy / self.scale) * (self.u_max - self.u_min)


class PredictionErrorUtilityMeasure(UtilityMeasure):
    def compute_utility(self, states, actions, next_states, next_state_means, next_state_vars, model):
        """ squared error between the observed and the (sample-mean) predicted normalised next state """
        predicted_next_states = next_state_means.to(model.device).mean(dim=1)
        next_states = next_states.to(model.device)

        next_states = model.normalizer.normalize_states(next_states)
        predicted_next_states = model.normalizer.normalize_states(predicted_next_states)

        return ((predicted_next_states - next_states) ** 2).sum(dim=1)            # shape: (n_actors)
