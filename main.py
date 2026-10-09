#!/usr/bin/env python
"""
Active exploration in Bayesian model-based RL.

Two stages, both run through this script (configuration is handled by sacred):

1. Exploration (task-agnostic): collect a buffer of real transitions by maximising an information-based utility
   inside a Bayesian dynamics model. Buffers are check-pointed every `checkpoint_frequency` steps.

       python main.py with method=LapMCEnt env_name=CoppeliaRLBench-v2

2. Evaluation: for every check-pointed buffer, fit a dynamics model, learn a policy for each task inside the model
   and execute it in the real environment.

       python main.py with mode=evaluate method=LapMCEnt env_name=CoppeliaRLBench-v2 run_dir=logs/<...>

Run `python main.py print_config` to list every option.
"""
import atexit
import csv
import gzip
import os
import pickle
import time
from copy import deepcopy
from datetime import datetime

import gym
import numpy as np
import pandas as pd
import torch
from sacred import Experiment

import envs
from buffer import Buffer
from imagination import Imagination
from logger import get_logger
from models import EnsembleModel, LaplaceModel, MCDropoutModel
from normalizer import TransitionNormalizer
from sac import SAC
from utilities import GaussianEntropyUtilityMeasure, JensenRenyiDivergenceUtilityMeasure, \
    PredictionErrorUtilityMeasure
from wrappers import BoundedActionsEnv, NoisyEnv, RecordedEnv

ex = Experiment('bayesian_active_exploration')
ex.logger = get_logger('bae')

# Methods compared in the paper: (Bayesian model, exploration utility, exploration mode)
METHODS = {
    'LapMCEnt':    dict(model_type='laplace',    utility_measure='gaussian_entropy', exploration_mode='active'),
    'LapMCRenyi':  dict(model_type='laplace',    utility_measure='renyi_div',        exploration_mode='active'),
    'MCDropEnt':   dict(model_type='mc_dropout', utility_measure='gaussian_entropy', exploration_mode='active'),
    'MCDropRenyi': dict(model_type='mc_dropout', utility_measure='renyi_div',        exploration_mode='active'),
    'MAX':         dict(model_type='ensemble',   utility_measure='renyi_div',        exploration_mode='active'),
    'EnsEnt':      dict(model_type='ensemble',   utility_measure='gaussian_entropy', exploration_mode='active'),
    'PERX':        dict(model_type='ensemble',   utility_measure='pred_err',         exploration_mode='reactive'),
    'Random':      dict(model_type='ensemble',   utility_measure='pred_err',         exploration_mode='random'),
}

# filled in `main` with the dimensions of the environment
SPACES = {}


"""
Configuration
"""


# noinspection PyUnusedLocal
@ex.config
def config():
    mode = 'explore'                                # 'explore' or 'evaluate'
    method = 'LapMCEnt'                             # one of METHODS
    eval_method = 'MAX' if method == 'Random' else method   # model used to evaluate the exploration buffers
    env_name = 'MagellanHalfCheetah-v2'             # 'MagellanHalfCheetah-v2' or 'CoppeliaRLBench-v2'
    is_coppelia = env_name in envs.COPPELIA_ENVS

    # exploration
    n_exploration_steps = 20000                     # total number of exploration steps (including warm up)
    n_warm_up_steps = 256                           # initial steps with random actions
    model_train_freq = 25                           # re-fit the model and the exploration policy every n steps
    checkpoint_frequency = 1000                     # dump the buffer (with normalizer) every n steps
    eval_during_exploration = False                 # evaluate the tasks every `eval_freq` steps while exploring
    eval_freq = 2000

    # evaluation (mode=evaluate)
    run_dir = ''                                    # folder of an exploration run, containing the <step>.buffer files
    n_train = 18000                                 # calibration: transitions used for training, the rest for test
    eval_steps = list(range(eval_freq, n_exploration_steps + 1, eval_freq))   # buffers to evaluate
    n_eval_episodes = 3                             # number of real episodes evaluated for each task

    # environment
    if is_coppelia:
        env_noise_stdev = 0. if mode == 'evaluate' else 2e-6
    else:
        env_noise_stdev = 0.02                      # standard deviation of the noise added to the states

    # dynamics model
    n_samples = 32                                  # ensemble members / MC-dropout forward passes / Laplace samples
    n_hidden = 512                                  # width of the hidden layers
    n_layers = 4                                    # number of hidden layers
    non_linearity = 'swish'
    p_dropout = 0.25                                # MC dropout: dropout probability ...
    dropout_layer = 2                               # ... applied after this hidden layer
    laplace_subset_of_weights = 'subnetwork'        # 'subnetwork' or 'last_layer'
    laplace_subnet_size = 1000                      # number of largest-magnitude weights treated as Bayesian
    laplace_sigma_noise = 1.                        # initial observation noise
    laplace_prior_prec = 5.                         # initial prior precision
    laplace_hyper_fitted = True                     # fit noise and prior precision by marginal likelihood
    laplace_hyper_epochs = 50
    laplace_sigma_noise_factor = 5                  # inflation of the fitted observation noise

    # model training
    exploring_model_epochs = 50                     # training epochs of each model fit during exploration
    evaluation_model_epochs = 200                   # training epochs of the model used to solve the tasks
    batch_size = 256
    learning_rate = 1e-3
    weight_decay = 0
    grad_clip = 5
    normalize_data = True                           # normalise states, actions and state deltas
    training_noise_stdev = 0.002 if (is_coppelia and mode == 'explore') else 0.

    # exploration utility
    renyi_decay = 0.1                               # decay of the variances in the Jensen-Renyi divergence
    utility_action_norm_penalty = 0
    action_noise_stdev = 0                          # noise added to the exploration actions

    # SAC policies, learnt inside the model (imagination MDP)
    policy_actors = 128                             # parallel actors in the imagination MDP
    policy_warm_up_episodes = 3                     # episodes with random actions before using the policy
    policy_replay_size = int(1e7)
    policy_batch_size = 4096
    policy_reactive_updates = 100                   # off-policy updates on the real buffer
    policy_active_updates = 1                       # on-policy updates per imagined step
    policy_n_hidden = 256
    policy_lr = 1e-3
    policy_gamma = 0.99
    policy_tau = 0.005
    buffer_reuse = True                             # transfer the real buffer to SAC as off-policy samples
    use_best_policy = False

    policy_explore_horizon = 50
    policy_explore_episodes = 50
    policy_explore_alpha = 0.02                     # SAC entropy coefficient for exploration

    policy_exploit_horizon = 250 if is_coppelia else 100
    policy_exploit_episodes = 100 if is_coppelia else 250
    policy_exploit_alpha = 0.4                      # SAC entropy coefficient for the tasks

    # infrastructure
    verbosity = 1                                   # 0: quiet, 1: progress, 2: training losses, 3: per-step info
    record = is_coppelia and mode == 'evaluate'     # record videos of the evaluation episodes
    save_eval_agents = False                        # save the SAC agents used for evaluation
    disable_cuda = False
    omp_num_threads = 1
    log_dir = 'logs'
    dump_dir = os.path.join(log_dir, env_name, method, f'{datetime.now().strftime("%Y%m%d%H%M%S")}_{os.getpid()}')


"""
Initialisation helpers
"""


@ex.capture
def get_device(disable_cuda):
    return torch.device('cuda' if (not disable_cuda and torch.cuda.is_available()) else 'cpu')


@ex.capture
def get_env(env_name, env_noise_stdev, record, is_coppelia):
    if is_coppelia and record:
        env = gym.make(env_name, render_mode='rgb_array')
    else:
        env = gym.make(env_name)
    env = BoundedActionsEnv(env)

    if env_noise_stdev:
        env = NoisyEnv(env, stdev=env_noise_stdev)
    if record:
        env = RecordedEnv(env)

    env.seed(np.random.randint(np.iinfo(np.uint32).max))
    env.action_space.seed(np.random.randint(np.iinfo(np.uint32).max))
    env.observation_space.seed(np.random.randint(np.iinfo(np.uint32).max))
    atexit.register(lambda: env.close())
    return env


@ex.capture
def get_model(method, n_samples, n_hidden, n_layers, non_linearity, p_dropout, dropout_layer, batch_size,
              laplace_subset_of_weights, laplace_subnet_size, laplace_sigma_noise, laplace_prior_prec,
              laplace_hyper_fitted, laplace_hyper_epochs, laplace_sigma_noise_factor):
    d_state, d_action, device = SPACES['d_state'], SPACES['d_action'], get_device()
    model_type = METHODS[method]['model_type']
    predictive = 'moments' if METHODS[method]['utility_measure'] == 'gaussian_entropy' else 'samples'

    if model_type == 'ensemble':
        return EnsembleModel(d_state=d_state, d_action=d_action, n_hidden=n_hidden, n_layers=n_layers,
                             ensemble_size=n_samples, predictive=predictive, non_linearity=non_linearity,
                             device=device)

    if model_type == 'mc_dropout':
        return MCDropoutModel(d_state=d_state, d_action=d_action, n_hidden=n_hidden, n_layers=n_layers,
                              n_forward_passes=n_samples, p_dropout=p_dropout, dropout_layer=dropout_layer,
                              predictive=predictive, non_linearity=non_linearity, device=device)

    if model_type == 'laplace':
        # Settings used in the paper: the entropy variant is trained by maximum likelihood, while the Renyi
        # variant is trained with the MAP objective on a ReLU network.
        renyi = predictive == 'samples'
        return LaplaceModel(d_state=d_state, d_action=d_action, n_hidden=n_hidden, n_layers=n_layers,
                            n_samples=n_samples, predictive=predictive, batch_size=batch_size,
                            sigma_noise=laplace_sigma_noise, prior_prec=laplace_prior_prec,
                            subset_of_weights=laplace_subset_of_weights, n_subnet_params=laplace_subnet_size,
                            hyper_fitted=laplace_hyper_fitted, n_hyper_epochs=laplace_hyper_epochs,
                            sigma_noise_factor=laplace_sigma_noise_factor, map_training=renyi,
                            perturb_actions=renyi, non_linearity='relu' if renyi else non_linearity,
                            device=device)

    raise ValueError(f'unknown model type: {model_type}')


@ex.capture
def get_buffer(n_exploration_steps, normalize_data):
    buffer = Buffer(d_state=SPACES['d_state'], d_action=SPACES['d_action'], ensemble_size=1,
                    buffer_size=n_exploration_steps + 1)
    if normalize_data:
        buffer.setup_normalizer(TransitionNormalizer())
    return buffer


@ex.capture
def get_optimizer_factory(learning_rate, weight_decay):
    return lambda params: torch.optim.Adam(params, lr=learning_rate, weight_decay=weight_decay)


@ex.capture
def get_utility_measure(utility_measure, utility_action_norm_penalty, renyi_decay):
    if utility_measure == 'renyi_div':
        return JensenRenyiDivergenceUtilityMeasure(decay=renyi_decay, action_norm_penalty=utility_action_norm_penalty)
    if utility_measure == 'gaussian_entropy':
        return GaussianEntropyUtilityMeasure(action_norm_penalty=utility_action_norm_penalty)
    if utility_measure == 'pred_err':
        return PredictionErrorUtilityMeasure(action_norm_penalty=utility_action_norm_penalty)
    raise ValueError(f'invalid utility measure: {utility_measure}')


def _append_csv(filename, row):
    with open(filename, 'a') as f:
        csv.writer(f).writerow(row)


"""
Model training
"""


@ex.capture
def train_epoch(model, buffer, optimizer, batch_size, training_noise_stdev, grad_clip):
    buffer.ensemble_size = model.n_bootstrap

    losses = []
    for tr_states, tr_actions, tr_state_deltas in buffer.train_batches(batch_size=batch_size):
        optimizer.zero_grad()
        loss = model.loss(tr_states, tr_actions, tr_state_deltas, training_noise_stdev=training_noise_stdev)
        losses.append(loss.item())
        loss.backward()
        torch.nn.utils.clip_grad_value_(model.parameters(), grad_clip)
        optimizer.step()

    return np.mean(losses)


@ex.capture
def fit_model(buffer, n_epochs, step_num, method, dump_dir, verbosity, _log, _run):
    """ train a fresh model on `buffer` (and fit its Laplace approximation) """
    model = get_model(method=method)

    # Laplace hyper-parameters fitted on the previous call are kept in the buffer
    if isinstance(model, LaplaceModel) and buffer.sigma_noise is not None:
        model.set_hyperparameters(buffer.sigma_noise, buffer.prior_prec)

    model.setup_normalizer(buffer.normalizer)
    optimizer = get_optimizer_factory()(model.parameters())

    if verbosity:
        _log.info(f"step: {step_num}\t training {method} model")

    start = time.time()
    for epoch_i in range(1, n_epochs + 1):
        tr_loss = train_epoch(model=model, buffer=buffer, optimizer=optimizer)
        if verbosity >= 2:
            _log.info(f'epoch: {epoch_i:3d} training_loss: {tr_loss:.2f}')
    _append_csv(f'{dump_dir}/timing_fitmodel.csv', [step_num, time.time() - start])

    if verbosity:
        _log.info(f"step: {step_num}\t training done for {n_epochs} epochs, final loss: {np.round(tr_loss, 3)}")
    _run.log_scalar("model_loss", tr_loss, step_num)

    if isinstance(model, LaplaceModel):
        start = time.time()
        buffer.sigma_noise, buffer.prior_prec = model.laplace_fit(buffer=buffer)
        _append_csv(f'{dump_dir}/timing_fitlaplace.csv', [step_num, time.time() - start])

    return model


"""
Planning
"""


@ex.capture
def get_policy(buffer, model, measure, mode, policy_replay_size, policy_batch_size, policy_active_updates,
               policy_n_hidden, policy_lr, policy_gamma, policy_tau, policy_explore_alpha, policy_exploit_alpha,
               buffer_reuse, verbosity, _log):
    """ fresh SAC agent, optionally pre-filled with the real transitions rewarded by `measure` """
    policy_alpha = policy_explore_alpha if mode == 'explore' else policy_exploit_alpha

    agent = SAC(d_state=SPACES['d_state'], d_action=SPACES['d_action'], replay_size=policy_replay_size,
                batch_size=policy_batch_size, n_updates=policy_active_updates, n_hidden=policy_n_hidden,
                gamma=policy_gamma, alpha=policy_alpha, lr=policy_lr, tau=policy_tau)
    agent = agent.to(model.device)
    agent.setup_normalizer(model.normalizer)

    if not buffer_reuse:
        return agent

    if verbosity >= 2:
        _log.info("... transferring exploration buffer")

    size = len(buffer)
    for i in range(0, size, 1024):
        j = min(i + 1024, size)
        s, a = buffer.states[i:j], buffer.actions[i:j]
        ns = buffer.states[i:j] + buffer.state_deltas[i:j]
        s, a, ns = s.to(model.device), a.to(model.device), ns.to(model.device)
        with torch.no_grad():
            mu, var = model.forward_all(s, a)
        r = measure(s, a, ns, mu, var, model)
        agent.replay.add(s, a, r, ns)

    return agent


def get_action(mdp, agent):
    current_state = mdp.reset()
    action = agent(current_state, eval=True)[0].detach().data.cpu().numpy()
    policy_value = torch.mean(agent.get_state_value(current_state)).item()
    return action, mdp, agent, policy_value


@ex.capture
def act(state, agent, mdp, buffer, model, measure, mode, method,
        policy_actors, policy_warm_up_episodes, use_best_policy, policy_reactive_updates,
        policy_explore_horizon, policy_exploit_horizon, policy_explore_episodes, policy_exploit_episodes,
        verbosity, _run, _log):
    """
    Return the action for `state`. If `agent` is None, a new SAC agent is first trained inside the model
    (in `mode` 'explore' with the exploration utility, in 'exploit' with the task measure).
    """
    if mode == 'explore':
        policy_horizon, policy_episodes = policy_explore_horizon, policy_explore_episodes
    elif mode == 'exploit':
        policy_horizon, policy_episodes = policy_exploit_horizon, policy_exploit_episodes
    else:
        raise ValueError("invalid acting mode")

    fresh_agent = agent is None

    if mdp is None:
        mdp = Imagination(horizon=policy_horizon, n_actors=policy_actors, model=model, measure=measure)
    if fresh_agent:
        agent = get_policy(buffer=buffer, model=model, measure=measure, mode=mode)

    mdp.update_init_state(state)

    if not fresh_agent:
        return get_action(mdp, agent)

    # reactive updates: off-policy learning from the real transitions
    for _ in range(policy_reactive_updates):
        agent.update()

    # active updates: on-policy learning inside the model
    perform_active_exploration = (mode == 'explore' and METHODS[method]['exploration_mode'] == 'active')
    if perform_active_exploration or mode == 'exploit':
        if perform_active_exploration:
            # only the effect of on-policy (imagined) training remains for active exploration
            agent.reset_replay()

        ep_returns = []
        best_return, best_params = -np.inf, deepcopy(agent.state_dict())
        for ep_i in range(policy_episodes):
            warm_up = ep_i < policy_warm_up_episodes
            ep_return = agent.episode(env=mdp, warm_up=warm_up, verbosity=verbosity, _log=_log)
            ep_returns.append(ep_return)

            if use_best_policy and ep_return > best_return:
                best_return, best_params = ep_return, deepcopy(agent.state_dict())
            if verbosity >= 3:
                _log.info(f"\tep: {ep_i}\taverage step return: {np.round(ep_return / policy_horizon, 3)}")

        if use_best_policy:
            agent.load_state_dict(best_params)

        if mode == 'explore':
            _run.log_scalar("policy_improvement_first_return", ep_returns[0] / policy_horizon)
            _run.log_scalar("policy_improvement_last_return", ep_returns[-1] / policy_horizon)

    return get_action(mdp, agent)


"""
Evaluation and check-pointing
"""


@ex.capture
def transition_novelty(state, action, next_state, model, renyi_decay):
    """ utility of a single real transition, as seen by `model` """
    state = torch.from_numpy(state).float().unsqueeze(0).to(model.device)
    action = torch.from_numpy(action).float().unsqueeze(0).to(model.device)
    next_state = torch.from_numpy(next_state).float().unsqueeze(0).to(model.device)

    with torch.no_grad():
        mu, var = model.forward_all(state, action)

    if model.predictive == 'samples':
        measure = JensenRenyiDivergenceUtilityMeasure(decay=renyi_decay)
    else:
        measure = GaussianEntropyUtilityMeasure()
    return measure(state, action, next_state, mu, var, model).item()


@ex.capture
def evaluate_task(env, model, buffer, task, filename, record, save_eval_agents, is_coppelia, verbosity, _run, _log):
    """ run one real episode of `task`, re-planning at every step """
    state = env.reset(filename=f'{filename}.mp4') if record else env.reset()
    max_steps = env.unwrapped.spec.max_episode_steps

    ep_return = 0
    agent, mdp = None, None
    done = False
    novelty = []
    i = 1
    while not done:
        action, mdp, agent, _ = act(state=state, agent=agent, mdp=mdp, buffer=buffer, model=model,
                                    measure=task.measure, mode='exploit')
        next_state, _, done, info = env.step(action)

        n = transition_novelty(state, action, next_state, model=model)
        novelty.append(n)
        reward = task.reward_function(state, action, next_state)

        if is_coppelia:
            # the return of a manipulation episode is its best reward; success (100) ends the episode
            if done and i < max_steps - 2:
                reward = 100                        # RLBench reports the task success through `done`
            elif reward == 100:
                done = True
            ep_return = max(ep_return, reward)
        else:
            ep_return += reward

        if verbosity >= 3:
            _log.info(f'[{i}] reward: {reward:5.2f} trans_novelty: {n:5.2f} action: {action}')

        if is_coppelia and done and record:
            for _ in range(10):                     # let the scene settle for the video
                env.step(np.zeros_like(action))

        state = next_state
        i += 1
    env.close()

    if record:
        _run.add_artifact(f'{filename}.mp4')
    if save_eval_agents:
        torch.save(agent.state_dict(), f'{filename}_agent.pt')

    return ep_return, np.mean(novelty)


@ex.capture
def evaluate_tasks(buffer, step_num, out_dir, eval_method, n_eval_episodes, evaluation_model_epochs, is_coppelia,
                   _log, _run):
    """
    Fit a model on `buffer` and solve every task of the environment.

    Returns:
        average return over all episodes and the return of each task (for manipulation tasks, the sum over the
        evaluated episodes, as reported in the paper; otherwise the mean)
    """
    model = fit_model(buffer=buffer, n_epochs=evaluation_model_epochs, step_num=step_num, method=eval_method,
                      dump_dir=out_dir)
    env = get_env()
    tasks = env.unwrapped.tasks
    env.close()
    del env

    all_returns, task_returns = [], {}
    for task_name, task in tasks.items():
        returns, novelties = [], []
        for ep_idx in range(1, n_eval_episodes + 1):
            env = get_env()
            ep_return, ep_novelty = evaluate_task(env=env, model=model, buffer=buffer, task=task,
                                                  filename=f"{out_dir}/evaluation_{step_num}_{task_name}_{ep_idx}")
            _log.info(f"task: {task_name}\tepisode: {ep_idx}\treward: {np.round(ep_return, 4)}")
            returns.append(ep_return)
            novelties.append(ep_novelty)

        task_returns[task_name] = np.sum(returns) if is_coppelia else np.mean(returns)
        all_returns.append(returns)
        _log.info(f"task: {task_name}\taverage return: {np.round(np.mean(returns), 4)}")
        _run.log_scalar(f"task_{task_name}_return", np.mean(returns), step_num)
        _run.log_scalar(f"task_{task_name}_episode_novelty", np.mean(novelties), step_num)

    average_return = np.mean(all_returns)
    _run.log_scalar("average_return", average_return, step_num)
    _run.result = average_return
    return average_return, task_returns


def append_performance(performance_data, step_num, average_return, task_returns, filename):
    performance_data.setdefault('step', []).append(step_num)
    performance_data.setdefault('average_performance', []).append(average_return)
    for task_name, task_return in task_returns.items():
        performance_data.setdefault(f'{task_name}_performance', []).append(task_return)
    pd.DataFrame(data=performance_data).to_csv(filename)


@ex.capture
def checkpoint(buffer, step_num, dump_dir, _run):
    buffer_file = f'{dump_dir}/{step_num}.buffer'
    with gzip.open(buffer_file, 'wb') as f:
        pickle.dump(buffer, f)
    _run.add_artifact(buffer_file)


"""
Main functions
"""


@ex.capture
def do_exploration(method, action_noise_stdev, n_exploration_steps, n_warm_up_steps, model_train_freq,
                   exploring_model_epochs, eval_during_exploration, eval_freq, checkpoint_frequency, dump_dir,
                   _log, _run):
    random_exploration = METHODS[method]['exploration_mode'] == 'random'
    env = get_env(record=False)
    buffer = get_buffer()
    exploration_measure = get_utility_measure(utility_measure=METHODS[method]['utility_measure'])

    model, mdp, agent = None, None, None
    performance_data = {}
    state = env.reset()

    for step_num in range(1, n_exploration_steps + 1):
        if step_num > n_warm_up_steps and not random_exploration:
            action, mdp, agent, policy_value = act(state=state, agent=agent, mdp=mdp, buffer=buffer, model=model,
                                                   measure=exploration_measure, mode='explore')
            _run.log_scalar("action_norm", np.sum(np.square(action)), step_num)
            _run.log_scalar("exploration_policy_value", policy_value, step_num)
            if action_noise_stdev:
                action = action + np.random.normal(scale=action_noise_stdev, size=action.shape)
        else:
            action = env.action_space.sample()

        next_state, reward, done, info = env.step(action)
        buffer.add(state, action, next_state)

        if model is not None:
            _run.log_scalar("experience_novelty", transition_novelty(state, action, next_state, model=model), step_num)

        if done:
            _log.info(f"step: {step_num}\tepisode complete")
            agent, mdp = None, None
            next_state = env.reset()
        state = next_state

        if step_num % checkpoint_frequency == 0:
            checkpoint(buffer=buffer, step_num=step_num)

        if random_exploration or step_num < n_warm_up_steps:
            continue

        if step_num % model_train_freq == 0 or step_num == n_warm_up_steps:
            model = fit_model(buffer=buffer, n_epochs=exploring_model_epochs, step_num=step_num)
            mdp, agent = None, None                 # discard the old policy, as the model changed

        if eval_during_exploration and step_num % eval_freq == 0:
            env.close()
            average_return, task_returns = evaluate_tasks(buffer=buffer, step_num=step_num, out_dir=dump_dir)
            append_performance(performance_data, step_num, average_return, task_returns,
                               f"{dump_dir}/performance_data.csv")
            env = get_env(record=False)
            state = env.reset()
            agent, mdp = None, None


@ex.capture
def do_evaluation(run_dir, eval_steps, eval_method, seed, _log):
    assert run_dir, 'set run_dir to the folder of an exploration run'
    out_dir = os.path.join(run_dir, f'evaluation_{eval_method}')
    os.makedirs(out_dir, exist_ok=True)
    ex.commands["save_config"](config_filename=f'{out_dir}/config_evaluation.json')

    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    performance_data = {}
    for step_num in eval_steps:
        with gzip.open(f'{run_dir}/{step_num}.buffer', 'rb') as f:
            buffer = pickle.load(f)
        _log.info(f'evaluating {run_dir}/{step_num}.buffer ({len(buffer)} transitions)')

        average_return, task_returns = evaluate_tasks(buffer=buffer, step_num=step_num, out_dir=out_dir)
        append_performance(performance_data, step_num, average_return, task_returns,
                           f"{out_dir}/performance_data.csv")


@ex.command
def calibration(run_dir, method, n_train, evaluation_model_epochs, seed, _log):
    """
    Calibration and computational cost of a Bayesian model (Table I of the paper): fit the model of `method` on
    the first `n_train` transitions of an exploration run, then compare the utility of the remaining transitions
    with the model's prediction error (AUSE).

        python main.py calibration with method=LapMCEnt run_dir=logs/MagellanHalfCheetah-v2/LapMCEnt/<run>
    """
    from calibration import compute_ause, plot_ause

    assert run_dir, 'set run_dir to the folder of an exploration run'
    setup()
    np.random.seed(seed)
    torch.manual_seed(seed)

    out_dir = os.path.join(run_dir, f'calibration_{method}')
    os.makedirs(out_dir, exist_ok=True)

    buffer_files = sorted((f for f in os.listdir(run_dir) if f.endswith('.buffer')), key=lambda f: int(f[:-7]))
    with gzip.open(f'{run_dir}/{n_train}.buffer', 'rb') as f:
        train_buffer = pickle.load(f)
    with gzip.open(f'{run_dir}/{buffer_files[-1]}', 'rb') as f:
        test_buffer = pickle.load(f)

    start = time.time()
    model = fit_model(buffer=train_buffer, n_epochs=evaluation_model_epochs, step_num=n_train, method=method,
                      dump_dir=out_dir)
    training_time = time.time() - start

    states = test_buffer.states[n_train:len(test_buffer)]
    actions = test_buffer.actions[n_train:len(test_buffer)]
    next_states = states + test_buffer.state_deltas[n_train:len(test_buffer)]

    with torch.no_grad():
        start = time.time()
        next_state_means, next_state_vars = model.forward_all(states, actions)
        inference_time = time.time() - start
        utility = get_utility_measure(utility_measure=METHODS[method]['utility_measure'])
        uncertainty = utility(states, actions, next_states, next_state_means, next_state_vars, model)

    predictions = next_state_means.mean(dim=1) if model.predictive == 'samples' else next_state_means
    error = (predictions.cpu() - next_states).abs().mean(dim=1).numpy()

    scurve_oracle, scurve_model, ause = compute_ause(error, uncertainty.cpu().numpy())
    plot_ause(scurve_oracle, scurve_model, ause, method, f'{out_dir}/ause.png')

    n_params = sum(p.numel() for p in model.parameters())
    if isinstance(model, LaplaceModel):
        n_params += model.n_subnet_params ** 2 if model.subset_of_weights == 'subnetwork' else 0
    results = dict(method=method, ause=ause, training_time=training_time, inference_time=inference_time,
                   n_stored_parameters=n_params)
    pd.DataFrame([results]).to_csv(f'{out_dir}/calibration.csv', index=False)
    _log.info(results)
    return ause


def setup():
    """ store the environment dimensions """
    env = get_env(record=False)
    SPACES['d_state'] = env.observation_space.shape[0]
    SPACES['d_action'] = env.action_space.shape[0]
    env.close()


@ex.automain
def main(mode, method, eval_method, omp_num_threads, dump_dir):
    assert method in METHODS, f'unknown method {method}, choose one of {list(METHODS)}'
    assert eval_method in METHODS, f'unknown eval_method {eval_method}, choose one of {list(METHODS)}'

    torch.set_num_threads(omp_num_threads)
    os.environ['OMP_NUM_THREADS'] = str(omp_num_threads)
    os.environ['MKL_NUM_THREADS'] = str(omp_num_threads)
    setup()

    if mode == 'explore':
        os.makedirs(dump_dir, exist_ok=True)
        ex.commands["save_config"](config_filename=f'{dump_dir}/config_exploration.json')
        do_exploration()
    elif mode == 'evaluate':
        do_evaluation()
    else:
        raise ValueError(f"mode must be 'explore' or 'evaluate', got {mode}")
