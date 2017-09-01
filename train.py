#! /usr/bin/env python3
"""
PPO: Proximal Policy Optimization

Written by Patrick Coady (pat-coady.github.io)

PPO uses a loss function and gradient descent to approximate
Trust Region Policy Optimization (TRPO). See these papers for
details:

TRPO / PPO:
https://arxiv.org/pdf/1502.05477.pdf (Schulman et al., 2016)

Distributed PPO:
https://arxiv.org/abs/1707.02286 (Heess et al., 2017)

Generalized Advantage Estimation:
https://arxiv.org/pdf/1506.02438.pdf

And, also, this GitHub repo which was helpful to me during
implementation:
https://github.com/joschu/modular_rl

This implementation learns policies for continuous environments
in the OpenAI Gym (https://gym.openai.com/). Testing was focused on
the MuJoCo control tasks.
"""
import gym
import numpy as np
from gym import wrappers
from policy import Policy
from value_function import NNValueFunction
import scipy.signal
from utils import Logger, Scaler
from datetime import datetime
import os
import argparse
import signal

from mpi4py import MPI
import os, subprocess, sys
import numpy as np
import mpi_util
import time

class GracefulKiller:
    """ Gracefully exit program on CTRL-C """
    def __init__(self):
        self.kill_now = False
        signal.signal(signal.SIGINT, self.exit_gracefully)
        signal.signal(signal.SIGTERM, self.exit_gracefully)

    def exit_gracefully(self, signum, frame):
        self.kill_now = True


def init_gym(env_name):
    """
    Initialize gym environment, return dimension of observation
    and action spaces.

    Args:
        env_name: str environment name (e.g. "Humanoid-v1")

    Returns: 3-tuple
        gym environment (object)
        number of observation dimensions (int)
        number of action dimensions (int)
    """
    env = gym.make(env_name)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    return env, obs_dim, act_dim


def run_episode(env, policy, scaler, animate=False):
    """ Run single episode with option to animate

    Args:
        env: ai gym environment
        policy: policy object with sample() method
        scaler: scaler object, used to scale/offset each observation dimension
            to a similar range
        animate: boolean, True uses env.render() method to animate episode

    Returns: 4-tuple of NumPy arrays
        observes: shape = (episode len, obs_dim)
        actions: shape = (episode len, act_dim)
        rewards: shape = (episode len,)
        unscaled_obs: useful for training scaler, shape = (episode len, obs_dim)
    """
    obs = env.reset()
    observes, actions, rewards, unscaled_obs = [], [], [], []
    done = False
    step = 0.0
    scale, offset = scaler.get()
    scale[-1] = 1.0  # don't scale time step feature
    offset[-1] = 0.0  # don't offset time step feature
    while not done:
        if animate and mpi_util.rank == 0:
            env.render()
        obs = obs.astype(np.float32).reshape((1, -1))
        obs = np.append(obs, [[step]], axis=1)  # add time step feature
        unscaled_obs.append(obs)
        obs = (obs - offset) * scale  # center and scale observations
        observes.append(obs)
        action = policy.sample(obs).reshape((1, -1)).astype(np.float32)
        actions.append(action)
        obs, reward, done, _ = env.step(np.squeeze(action, axis=0))
        if not isinstance(reward, float):
            reward = np.asscalar(reward)
        rewards.append(reward)
        step += 1e-3  # increment time step feature

    return (np.concatenate(observes), np.concatenate(actions),
            np.array(rewards, dtype=np.float64), np.concatenate(unscaled_obs))


def run_policy(env, policy, scaler, logger, episodes):
    """ Run policy and collect data for a minimum of min_steps and min_episodes

    Args:
        env: ai gym environment
        policy: policy object with sample() method
        scaler: scaler object, used to scale/offset each observation dimension
            to a similar range
        logger: logger object, used to save stats from episodes
        episodes: total episodes to run

    Returns: list of trajectory dictionaries, list length = number of episodes
        'observes' : NumPy array of states from episode
        'actions' : NumPy array of actions from episode
        'rewards' : NumPy array of (un-discounted) rewards from episode
        'unscaled_obs' : NumPy array of (un-discounted) rewards from episode
    """
    total_steps = 0
    trajectories = []
    for e in range(episodes):
        observes, actions, rewards, unscaled_obs = run_episode(env, policy, scaler)
        total_steps += observes.shape[0]
        trajectory = {'observes': observes,
                      'actions': actions,
                      'rewards': rewards,
                      'unscaled_obs': unscaled_obs}
        trajectories.append(trajectory)
    unscaled = np.concatenate([t['unscaled_obs'] for t in trajectories])
    scaler.update(unscaled)  # update running statistics for scaling observations
    # logger.log({'_MeanReward': np.mean([t['rewards'].sum() for t in trajectories]), 'Steps': total_steps})
    logger.log({'_MeanReward':   mpi_util.all_mean(np.array([np.mean([t['rewards'].sum() for t in trajectories])]))[0]  , 'Steps': total_steps})
    return trajectories


def discount(x, gamma):
    """ Calculate discounted forward sum of a sequence at each point """
    return scipy.signal.lfilter([1.0], [1.0, -gamma], x[::-1])[::-1]


def add_disc_sum_rew(trajectories, gamma):
    """ Adds discounted sum of rewards to all time steps of all trajectories

    Args:
        trajectories: as returned by run_policy()
        gamma: discount

    Returns:
        None (mutates trajectories dictionary to add 'disc_sum_rew')
    """
    for trajectory in trajectories:
        if gamma < 0.999:  # don't scale for gamma ~= 1
            rewards = trajectory['rewards'] * (1 - gamma)
        else:
            rewards = trajectory['rewards']
        disc_sum_rew = discount(rewards, gamma)
        trajectory['disc_sum_rew'] = disc_sum_rew


def add_value(trajectories, val_func):
    """ Adds estimated value to all time steps of all trajectories

    Args:
        trajectories: as returned by run_policy()
        val_func: object with predict() method, takes observations
            and returns predicted state value

    Returns:
        None (mutates trajectories dictionary to add 'values')
    """
    for trajectory in trajectories:
        observes = trajectory['observes']
        values = val_func.predict(observes)
        trajectory['values'] = values


def add_gae(trajectories, gamma, lam):
    """ Add generalized advantage estimator.
    https://arxiv.org/pdf/1506.02438.pdf

    Args:
        trajectories: as returned by run_policy(), must include 'values'
            key from add_value().
        gamma: reward discount
        lam: lambda (see paper).
            lam=0 : use TD residuals
            lam=1 : A =  Sum Discounted Rewards - V_hat(s)

    Returns:
        None (mutates trajectories dictionary to add 'advantages')
    """
    for trajectory in trajectories:
        if gamma < 0.999:  # don't scale for gamma ~= 1
            rewards = trajectory['rewards'] * (1 - gamma)
        else:
            rewards = trajectory['rewards']
        values = trajectory['values']
        # temporal differences
        tds = rewards - values + np.append(values[1:] * gamma, 0)
        advantages = discount(tds, gamma * lam)
        trajectory['advantages'] = advantages


def build_train_set(trajectories):
    """

    Args:
        trajectories: trajectories after processing by add_disc_sum_rew(),
            add_value(), and add_gae()

    Returns: 4-tuple of NumPy arrays
        observes: shape = (N, obs_dim)
        actions: shape = (N, act_dim)
        advantages: shape = (N,)
        disc_sum_rew: shape = (N,)
    """
    observes = np.concatenate([t['observes'] for t in trajectories])
    actions = np.concatenate([t['actions'] for t in trajectories])
    disc_sum_rew = np.concatenate([t['disc_sum_rew'] for t in trajectories])
    advantages = np.concatenate([t['advantages'] for t in trajectories])
    # do we even need to concat these from the different processes?  trpo does not. it does specific allmean()s
    if mpi_util.batches_or_update:
        d = mpi_util.rank0_accum_batches({'advantages': advantages, 'actions': actions, 'observes': observes, 'disc_sum_rew': disc_sum_rew})
        observes, actions, disc_sum_rew, advantages = d['observes'], d['actions'], d['disc_sum_rew'], d['advantages']
    # normalize advantages
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-6)
    # observes = np.concatenate(MPI.COMM_WORLD.allgather(observes))
    # actions = np.concatenate(MPI.COMM_WORLD.allgather(actions))
    # advantages = np.concatenate(MPI.COMM_WORLD.allgather(advantages))
    # disc_sum_rew = np.concatenate(MPI.COMM_WORLD.allgather(disc_sum_rew))
    print(str(mpi_util.rank)+'len_of_accum_batches',len(observes))
    return observes, actions, advantages, disc_sum_rew

tstart = time.time()
def log_batch_stats(observes, actions, advantages, disc_sum_rew, logger, episode):
    global tstart
    """ Log various batch statistics """
    logger.log({'_mean_obs': np.mean(observes),
                '_min_obs': np.min(observes),
                '_max_obs': np.max(observes),
                '_std_obs': np.mean(np.var(observes, axis=0)),
                '_mean_act': np.mean(actions),
                '_min_act': np.min(actions),
                '_max_act': np.max(actions),
                '_std_act': np.mean(np.var(actions, axis=0)),
                '_mean_adv': np.mean(advantages),
                '_min_adv': np.min(advantages),
                '_max_adv': np.max(advantages),
                '_std_adv': np.var(advantages),
                '_mean_discrew': np.mean(disc_sum_rew),
                '_min_discrew': np.min(disc_sum_rew),
                '_max_discrew': np.max(disc_sum_rew),
                '_std_discrew': np.var(disc_sum_rew),
                '_Episode': episode,
                })
    print(str(mpi_util.rank)+'_steps_sec', len(observes)/(time.time()-tstart))
    tstart = time.time()


def main(env_name, num_episodes, gamma, lam, kl_targ, batch_size, nprocs, policy_hid_list, valfunc_hid_list):
    """ Main training loop

    Args:
        env_name: OpenAI Gym environment name, e.g. 'Hopper-v1'
        num_episodes: maximum number of episodes to run
        gamma: reward discount factor (float)
        lam: lambda from Generalized Advantage Estimate
        kl_targ: D_KL target for policy update [D_KL(pi_old || pi_new)
        batch_size: number of episodes per policy training batch
    """
    # killer = GracefulKiller()
    if mpi_util.nworkers > 1: batch_size = batch_size//mpi_util.nworkers if batch_size%mpi_util.nworkers==0 else batch_size//mpi_util.nworkers +1
    print('num_episodes per process', batch_size)
    env, obs_dim, act_dim = init_gym(env_name)
    obs_dim += 1  # add 1 to obs dimension for time step feature (see run_episode())
    now = datetime.utcnow().strftime("%b-%d_%H:%M:%S")  # create unique directories
    # rank = MPI.COMM_WORLD.Get_rank()
    print('my rank is:',mpi_util.rank)
    logger = Logger(logname=env_name+str(mpi_util.rank), now=now)
    aigym_path = os.path.join('/tmp', env_name, now)
    if mpi_util.rank == 0:
        env = wrappers.Monitor(env, aigym_path, force=True)
    seed = 100+nprocs
    mpi_util.set_global_seeds(seed+mpi_util.rank)
    env.seed(seed+mpi_util.rank)
    scaler = Scaler(obs_dim)
    val_func = NNValueFunction(obs_dim, valfunc_hid_list)
    policy = Policy(obs_dim, act_dim, kl_targ, policy_hid_list)
    # run a few episodes of untrained policy to initialize scaler:
    run_policy(env, policy, scaler, logger, episodes=5)
    episode = 0
    # mpi_util.all_avg_wts(val_func.sess, val_func.g, 'val')  # initially synchronize the weights
    # mpi_util.all_avg_wts(policy.sess, policy.g, 'policy')
    while episode < num_episodes:
        trajectories = run_policy(env, policy, scaler, logger, episodes=batch_size)
        episode += mpi_util.all_sum(np.array([len(trajectories)]))[0]
        add_value(trajectories, val_func)  # add estimated values to episodes
        add_disc_sum_rew(trajectories, gamma)  # calculated discounted sum of Rs
        add_gae(trajectories, gamma, lam)  # calculate advantage
        # concatenate all episodes into single NumPy arrays
        observes, actions, advantages, disc_sum_rew = build_train_set(trajectories)
        policy.print_sigmas(observes[0:1])
        # add various stats to training log:
        log_batch_stats(observes, actions, advantages, disc_sum_rew, logger, episode)
        if mpi_util.batches_or_update:   # rank0 does all calcs then broadcasts wts
            if mpi_util.rank==0:
                policy.update(observes, actions, advantages, logger)  # update policy
                val_func.fit(observes, disc_sum_rew, logger)  # update value function
            mpi_util.rank0_bcast_wts(val_func.sess, val_func.g, 'val')
            mpi_util.rank0_bcast_wts(policy.sess, policy.g, 'policy')
        else:
            policy.update(observes, actions, advantages, logger)  # update policy
            val_func.fit(observes, disc_sum_rew, logger)  # update value function
        # mpi_util.all_avg_wts(val_func.sess, val_func.g, 'val')  # initially synchronize the weights
        # mpi_util.all_avg_wts(policy.sess, policy.g, 'policy')
        # print('rank',mpi_util.rank,'--------------------------------------------- val --------------------------------------------------')
        if not mpi_util.batches_or_update:   # if all are calculating the policy, the value func wts still need to be averaged here
            mpi_util.all_avg_wts(val_func.sess, val_func.g, 'val')
        logger.write(display=True)  # write logger results to file and stdout
        # if killer.kill_now:
        #     if input('Terminate training (y/[n])? ') == 'y':
        #         break
        #     killer.kill_now = False
    logger.close()
    policy.close_sess()
    val_func.close_sess()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=('Train policy on OpenAI Gym environment '
                                                  'using Proximal Policy Optimizer'))
    parser.add_argument('env_name', type=str, help='OpenAI Gym environment name')
    parser.add_argument('-n', '--num_episodes', type=int, help='Number of episodes to run',
                        default=1000000000000)
    parser.add_argument('-g', '--gamma', type=float, help='Discount factor', default=0.995)
    parser.add_argument('-l', '--lam', type=float, help='Lambda for Generalized Advantage Estimation',
                        default=0.98)
    parser.add_argument('-k', '--kl_targ', type=float, help='D_KL target value',
                        default=0.003)
    parser.add_argument('-b', '--batch_size', type=int,
                        help='Number of episodes per training batch',
                        default=20)
    parser.add_argument('--nprocs', type=int, default=1)
    parser.add_argument('--policy_hid_list', type=str, help='comma separated 3 layer list.  [30,40,25]', default='[]')
    parser.add_argument('--valfunc_hid_list', type=str, help='comma separated 3 layer list.  [30,40,25]', default='[]')

    args = parser.parse_args()
    args.policy_hid_list = eval(args.policy_hid_list)
    args.valfunc_hid_list = eval(args.valfunc_hid_list)
    if "parent" == mpi_util.mpi_fork(args.nprocs): os.exit()
    main(**vars(args))
