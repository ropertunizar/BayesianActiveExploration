"""
Probabilistic dynamics models p(s' | s, a) used for exploration and planning.

All models predict the *state delta* s' - s as a diagonal Gaussian (heteroscedastic aleatoric noise) and capture
epistemic uncertainty with one of three approximate Bayesian inference techniques:

    * EnsembleModel   -- deep ensemble of N independently-initialised networks (MAX baseline, Shyam et al. 2019)
    * MCDropoutModel  -- a single network with dropout kept active at inference, N stochastic forward passes
    * LaplaceModel    -- a single MAP network + (subnetwork) Laplace approximation of the weight posterior,
                         N Monte Carlo samples of the weights (laplace-torch / "Laplace Redux")

Each model exposes the same interface. `forward_all(states, actions)` returns either

    predictive == 'samples': one Gaussian per posterior sample
        next_state_means (batch, N, d_state), next_state_vars (batch, N, d_state)
        -> consumed by the Jensen-Renyi divergence and the prediction-error utilities

    predictive == 'moments': the mixture moment-matched by a single Gaussian
        next_state_mean (batch, d_state), [epistemic_var (batch, d_state), aleatoric_var (batch, d_state)]
        -> consumed by our Gaussian entropy utility

Inputs and targets are normalised internally with a TransitionNormalizer; all outputs are in raw state space.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data_utils
from torch.distributions import Normal

from normalizer import TransitionNormalizer


def swish(x):
    return x * torch.sigmoid(x)


def _init_weight(weight, non_linearity):
    if non_linearity == 'swish':
        nn.init.xavier_uniform_(weight)
    elif non_linearity in ('leaky_relu', 'relu', 'tanh'):
        nn.init.kaiming_normal_(weight)
    elif non_linearity == 'linear':
        nn.init.xavier_normal_(weight)
    else:
        raise ValueError(f'unknown non-linearity: {non_linearity}')


"""
Building blocks
"""


class EnsembleDenseLayer(nn.Module):
    def __init__(self, n_in, n_out, ensemble_size, non_linearity='leaky_relu'):
        """
        Linear + activation layer for `ensemble_size` independent networks, evaluated in a single batched
        matrix multiplication. Inputs have shape (ensemble_size, batch, n_in).
        """
        super().__init__()

        weights = torch.zeros(ensemble_size, n_out, n_in).float()
        biases = torch.zeros(ensemble_size, 1, n_out).float()
        for weight in weights:
            _init_weight(weight, non_linearity)

        self.weights = nn.Parameter(weights)
        self.biases = nn.Parameter(biases)

        self.non_linearity = {'swish': swish,
                              'leaky_relu': F.leaky_relu,
                              'relu': F.relu,
                              'tanh': torch.tanh,
                              'linear': lambda x: x}[non_linearity]

    def forward(self, inp):
        op = torch.baddbmm(self.biases, inp, torch.transpose(self.weights, 1, 2))
        return self.non_linearity(op)


class DenseLayer(nn.Sequential):
    def __init__(self, n_in, n_out, non_linearity='leaky_relu'):
        """
        Linear + activation layer built from standard `nn.Linear` modules so that laplace-torch can handle it.
        """
        linear = nn.Linear(in_features=n_in, out_features=n_out, bias=True)
        _init_weight(linear.weight, non_linearity)
        linear.bias = nn.Parameter(torch.zeros_like(linear.bias))

        if non_linearity == 'linear':
            super().__init__(linear)
        else:
            activation = {'swish': nn.SiLU(),
                          'leaky_relu': nn.LeakyReLU(negative_slope=0.01),
                          'relu': nn.ReLU(),
                          'tanh': nn.Tanh()}[non_linearity]
            super().__init__(linear, activation)


def _output_head(n_in, n_out):
    head = nn.Linear(in_features=n_in, out_features=n_out, bias=True)
    nn.init.xavier_normal_(head.weight)
    head.bias = nn.Parameter(torch.zeros_like(head.bias))
    return head


class GaussianMLP(nn.Module):
    def __init__(self, d_in, d_out, n_hidden, n_layers, non_linearity, p_dropout=None, dropout_layer=None):
        """
        MLP with a shared trunk and two linear heads: the mean and the (raw) log-variance of the state delta.

        Args:
            n_layers: number of hidden layers (>= 2)
            p_dropout, dropout_layer: if given, a dropout layer is inserted after hidden layer `dropout_layer`
        """
        super().__init__()
        trunk = []
        for lyr_idx in range(n_layers):
            trunk.append(DenseLayer(d_in if lyr_idx == 0 else n_hidden, n_hidden, non_linearity=non_linearity))
            if p_dropout is not None and lyr_idx == dropout_layer:
                trunk.append(nn.Dropout(p=p_dropout))

        # NOTE: attribute order matters -- laplace-torch is applied to Sequential(trunk, mean_head)
        self.trunk = nn.Sequential(*trunk)
        self.mean_head = _output_head(n_hidden, d_out)
        self.log_var_head = _output_head(n_hidden, d_out)

    def forward(self, x):
        h = self.trunk(x)
        return self.mean_head(h), self.log_var_head(h)


"""
Models
"""


class DynamicsModel(nn.Module):
    # soft bounds on the (normalised) aleatoric log-variance
    min_log_var = -5
    max_log_var = -1

    # number of independently shuffled copies of each training batch the model expects (one per ensemble member)
    n_bootstrap = 1

    # whether training-time input noise is also applied to the actions
    perturb_actions = True

    def __init__(self, d_state, d_action, n_samples, predictive, device):
        """
        Args:
            d_state, d_action: dimensionality of state and action
            n_samples: number of posterior samples (ensemble members / forward passes / weight samples)
            predictive: 'samples' or 'moments', see module docstring
            device: torch device
        """
        assert predictive in ('samples', 'moments')
        super().__init__()
        self.d_state = d_state
        self.d_action = d_action
        self.n_samples = n_samples
        self.predictive = predictive
        self.device = device
        self.normalizer = None

    """ normalisation """

    def setup_normalizer(self, normalizer):
        self.normalizer = TransitionNormalizer()
        self.normalizer.set_state(normalizer.get_state())

    def _pre_process_model_inputs(self, states, actions):
        states, actions = states.to(self.device), actions.to(self.device)
        if self.normalizer is None:
            return states, actions
        return self.normalizer.normalize_states(states), self.normalizer.normalize_actions(actions)

    def _pre_process_model_targets(self, state_deltas):
        state_deltas = state_deltas.to(self.device)
        if self.normalizer is None:
            return state_deltas
        return self.normalizer.normalize_state_deltas(state_deltas)

    def _denormalize_means(self, delta_mean):
        if self.normalizer is None:
            return delta_mean
        return self.normalizer.denormalize_state_delta_means(delta_mean)

    def _denormalize_vars(self, var):
        if self.normalizer is None:
            return var
        return self.normalizer.denormalize_state_delta_vars(var)

    def _bounded_var(self, log_var):
        log_var = torch.sigmoid(log_var)                                            # in [0, 1]
        log_var = self.min_log_var + (self.max_log_var - self.min_log_var) * log_var
        return torch.exp(log_var)

    """ to be implemented by each model """

    def _propagate_network(self, states, actions):
        """ normalised (states, actions) -> normalised (delta_mean, aleatoric var), used for training """
        raise NotImplementedError

    def _predictive_samples(self, states, actions):
        """
        normalised states (batch, d_state), actions (batch, d_action) ->
        normalised delta means and aleatoric vars, both (n_samples, batch, d_state)
        """
        raise NotImplementedError

    """ prediction """

    def forward_all(self, states, actions):
        """
        Predict the next-state distribution for raw states (batch, d_state) and actions (batch, d_action).
        See the module docstring for the output format.
        """
        normalized_states, normalized_actions = self._pre_process_model_inputs(states, actions)
        delta_mean_samples, var_samples = self._predictive_samples(normalized_states, normalized_actions)
        states = states.to(self.device)

        if self.predictive == 'samples':
            next_state_means = self._denormalize_means(delta_mean_samples) + states.unsqueeze(0)
            next_state_vars = self._denormalize_vars(var_samples)
            return next_state_means.transpose(0, 1), next_state_vars.transpose(0, 1)

        # moment matching of the mixture: the spread of the means is the epistemic variance
        delta_mean = self._denormalize_means(delta_mean_samples.mean(dim=0))
        var_epistemic = self._denormalize_vars(delta_mean_samples.var(dim=0))
        var_aleatoric = self._denormalize_vars(var_samples.mean(dim=0))
        return delta_mean + states, [var_epistemic, var_aleatoric]

    def forward(self, states, actions):
        return self.forward_all(states, actions)

    def sample(self, mean, var):
        return Normal(mean, torch.sqrt(var)).sample()

    """ training """

    def _prior_loss(self):
        return None

    def loss(self, states, actions, state_deltas, training_noise_stdev=0):
        """
        Gaussian negative log-likelihood of the normalised state deltas.

        Args:
            states, actions, state_deltas: raw tensors of shape (n_bootstrap, batch, dim)
            training_noise_stdev: noise added to the normalised inputs and targets
        """
        states, actions = self._pre_process_model_inputs(states, actions)
        targets = self._pre_process_model_targets(state_deltas)

        if self.n_bootstrap == 1:
            states, actions, targets = states.squeeze(dim=0), actions.squeeze(dim=0), targets.squeeze(dim=0)

        if not np.allclose(training_noise_stdev, 0):
            states += torch.randn_like(states) * training_noise_stdev
            if self.perturb_actions:
                actions += torch.randn_like(actions) * training_noise_stdev
            targets += torch.randn_like(targets) * training_noise_stdev

        mu, var = self._propagate_network(states, actions)
        nll = (mu - targets) ** 2 / var + torch.log(var)

        prior_loss = self._prior_loss()
        if prior_loss is None:
            return torch.mean(nll)
        return torch.sum(nll) + prior_loss                                           # MAP objective


class EnsembleModel(DynamicsModel):
    def __init__(self, d_state, d_action, n_hidden, n_layers, ensemble_size, predictive,
                 non_linearity='swish', device=torch.device('cpu')):
        """
        Deep ensemble: `ensemble_size` networks trained on independently shuffled data (MAX, Shyam et al. 2019).
        """
        assert n_layers >= 2, "minimum depth of model is 2"
        super().__init__(d_state, d_action, n_samples=ensemble_size, predictive=predictive, device=device)
        self.n_bootstrap = ensemble_size

        layers = [EnsembleDenseLayer(d_state + d_action, n_hidden, ensemble_size, non_linearity=non_linearity)]
        layers += [EnsembleDenseLayer(n_hidden, n_hidden, ensemble_size, non_linearity=non_linearity)
                   for _ in range(n_layers - 1)]
        layers += [EnsembleDenseLayer(n_hidden, 2 * d_state, ensemble_size, non_linearity='linear')]
        self.layers = nn.Sequential(*layers)
        self.to(device)

    def _propagate_network(self, states, actions):
        op = self.layers(torch.cat((states, actions), dim=-1))
        delta_mean, log_var = torch.split(op, op.size(-1) // 2, dim=-1)
        return delta_mean, self._bounded_var(log_var)

    def _predictive_samples(self, states, actions):
        states = states.unsqueeze(0).repeat(self.n_samples, 1, 1)
        actions = actions.unsqueeze(0).repeat(self.n_samples, 1, 1)
        return self._propagate_network(states, actions)


class MCDropoutModel(DynamicsModel):
    def __init__(self, d_state, d_action, n_hidden, n_layers, n_forward_passes, p_dropout, dropout_layer, predictive,
                 non_linearity='swish', device=torch.device('cpu')):
        """
        Monte Carlo dropout (Gal & Ghahramani, 2016): dropout stays active at inference and the predictive
        distribution is approximated with `n_forward_passes` stochastic forward passes.
        """
        assert n_layers >= 2, "minimum depth of model is 2"
        super().__init__(d_state, d_action, n_samples=n_forward_passes, predictive=predictive, device=device)
        self.net = GaussianMLP(d_state + d_action, d_state, n_hidden, n_layers, non_linearity,
                               p_dropout=p_dropout, dropout_layer=dropout_layer)
        self.to(device)

    def _propagate_network(self, states, actions):
        delta_mean, log_var = self.net(torch.cat((states, actions), dim=-1))
        return delta_mean, self._bounded_var(log_var)

    def _predictive_samples(self, states, actions):
        self.train()                                                                 # keep dropout active
        delta_means, variances = zip(*[self._propagate_network(states, actions) for _ in range(self.n_samples)])
        return torch.stack(delta_means), torch.stack(variances)


class LaplaceModel(DynamicsModel):
    def __init__(self, d_state, d_action, n_hidden, n_layers, n_samples, predictive,
                 batch_size, sigma_noise, prior_prec, subset_of_weights='subnetwork', n_subnet_params=1000,
                 hyper_fitted=True, n_hyper_epochs=50, sigma_noise_factor=5, map_training=False,
                 perturb_actions=False, non_linearity='swish', device=torch.device('cpu')):
        """
        Laplace approximation (MacKay, 1992; Daxberger et al., 2021) of the posterior over the weights of the
        mean network, p(theta | D) ~= N(theta_MAP, H^-1). The predictive distribution is approximated by Monte
        Carlo: `n_samples` weight samples, one forward pass each.

        Args:
            batch_size: batch size used to accumulate the Hessian
            sigma_noise, prior_prec: initial observation noise and prior precision
            subset_of_weights: 'subnetwork' (full Hessian over the `n_subnet_params` largest-magnitude weights)
                               or 'last_layer' (KFAC Hessian of the last layer)
            hyper_fitted: optimise sigma_noise and prior_prec by maximising the marginal likelihood
            n_hyper_epochs: optimisation steps for the marginal likelihood
            sigma_noise_factor: the fitted observation noise is inflated by this factor
            map_training: train with the MAP objective (sum NLL + Gaussian prior) instead of the mean NLL
            perturb_actions: whether training-time input noise is also applied to the actions
        """
        assert n_layers >= 2, "minimum depth of model is 2"
        super().__init__(d_state, d_action, n_samples=n_samples, predictive=predictive, device=device)
        self.net = GaussianMLP(d_state + d_action, d_state, n_hidden, n_layers, non_linearity)
        self.to(device)

        self.batch_size = batch_size
        self.sigma_noise = sigma_noise
        self.prior_prec = prior_prec
        self.subset_of_weights = subset_of_weights
        self.n_subnet_params = n_subnet_params
        self.hyper_fitted = hyper_fitted
        self.n_hyper_epochs = n_hyper_epochs
        self.sigma_noise_factor = sigma_noise_factor
        self.map_training = map_training
        self.perturb_actions = perturb_actions
        self.la = None

    def set_hyperparameters(self, sigma_noise, prior_prec):
        self.sigma_noise = sigma_noise
        self.prior_prec = prior_prec

    def _propagate_network(self, states, actions):
        delta_mean, log_var = self.net(torch.cat((states, actions), dim=-1))
        log_var = np.log(self.sigma_noise ** 2) + log_var
        return delta_mean, self._bounded_var(log_var)

    def _prior_loss(self):
        if not self.map_training:
            return None
        prior = Normal(0, np.sqrt(1 / self.prior_prec))
        weights = torch.cat([p.flatten() for p in self.net.parameters()])
        return - torch.sum(prior.log_prob(weights))

    def _predictive_samples(self, states, actions):
        assert self.la is not None, 'call laplace_fit() before predicting'
        inp = torch.cat((states, actions), dim=-1)
        delta_mean_samples = self.la._nn_predictive_samples(inp, n_samples=self.n_samples)
        _, var = self._propagate_network(states, actions)                            # aleatoric var of the MAP net
        return delta_mean_samples, var.unsqueeze(0).expand_as(delta_mean_samples)

    def laplace_fit(self, buffer):
        """
        Fit the Laplace approximation around the current (MAP) weights using all transitions in `buffer`.

        Returns:
            the (possibly re-estimated) observation noise and prior precision
        """
        from laplace import Laplace
        from laplace.utils import LargestMagnitudeSubnetMask

        n = len(buffer)
        states, actions = self._pre_process_model_inputs(buffer.states[:n], buffer.actions[:n])
        targets = self._pre_process_model_targets(buffer.state_deltas[:n])
        train_loader = data_utils.DataLoader(data_utils.TensorDataset(torch.cat((states, actions), dim=1), targets),
                                             batch_size=self.batch_size, shuffle=False)

        mean_network = nn.Sequential(self.net.trunk, self.net.mean_head)
        if self.subset_of_weights == 'subnetwork':
            subnetwork_mask = LargestMagnitudeSubnetMask(mean_network, n_params_subnet=self.n_subnet_params)
            subnetwork_indices = subnetwork_mask.select().to('cpu')
            self.la = Laplace(mean_network, 'regression', subset_of_weights='subnetwork',
                              hessian_structure='full', subnetwork_indices=subnetwork_indices)
        elif self.subset_of_weights == 'last_layer':
            self.la = Laplace(mean_network, 'regression', subset_of_weights='last_layer', hessian_structure='kron')
        else:
            raise ValueError(f'unknown subset_of_weights: {self.subset_of_weights}')

        self.la.fit(train_loader)

        if self.hyper_fitted:
            # prior precision and observation noise by marginal-likelihood (evidence) maximisation
            log_prior, log_sigma = torch.ones(1, requires_grad=True), torch.ones(1, requires_grad=True)
            hyper_optimizer = torch.optim.Adam([log_prior, log_sigma], lr=1e-1)
            for _ in range(self.n_hyper_epochs):
                hyper_optimizer.zero_grad()
                neg_marglik = - self.la.log_marginal_likelihood(log_prior.exp(), log_sigma.exp())
                neg_marglik.backward()
                hyper_optimizer.step()

            self.la.sigma_noise = self.sigma_noise_factor * self.la.sigma_noise
            self.sigma_noise = self.la.sigma_noise.item()
            self.prior_prec = self.la.prior_precision.item()
        else:
            self.la.sigma_noise = self.sigma_noise * torch.ones(1, requires_grad=True)
            self.la.prior_precision = self.prior_prec * torch.ones(1, requires_grad=True)

        return self.sigma_noise, self.prior_prec
