# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: Python 3
#     name: python3
# ---

# + colab={"base_uri": "https://localhost:8080/"} id="fhcm7fRqAm-v" executionInfo={"status": "ok", "timestamp": 1677860381600, "user_tz": 0, "elapsed": 1731, "user": {"displayName": "Oskar Hollinsworth", "userId": "00307706571197304608"}} outputId="aa0a6e84-f35e-40fb-c68b-8e6afeee8caa"
from google.colab import drive
drive.mount('/content/drive')
# %cd /content/drive/MyDrive/Colab Notebooks/cartpole-demo

# + colab={"base_uri": "https://localhost:8080/"} id="GgSNZRJh4EjV" executionInfo={"status": "ok", "timestamp": 1677860415136, "user_tz": 0, "elapsed": 33538, "user": {"displayName": "Oskar Hollinsworth", "userId": "00307706571197304608"}} outputId="69250b38-a9d9-4ebb-f165-fa3f344f76f4"
# !pip install einops
# !pip install wandb
# !pip install jupytext

# + colab={"base_uri": "https://localhost:8080/"} id="1g58HZUb8Ltl" executionInfo={"status": "ok", "timestamp": 1677860416428, "user_tz": 0, "elapsed": 1296, "user": {"displayName": "Oskar Hollinsworth", "userId": "00307706571197304608"}} outputId="853b0a34-3f23-4a5f-8f78-2aa950cfd114"
# !jupytext --to py cartpole.ipynb
# !git fetch
# !git status

# + id="vEczQ48wC40O" executionInfo={"status": "ok", "timestamp": 1677860420697, "user_tz": 0, "elapsed": 4271, "user": {"displayName": "Oskar Hollinsworth", "userId": "00307706571197304608"}}
import argparse
import os
import random
import time
import sys
from distutils.util import strtobool
from dataclasses import dataclass
from typing import Optional
import numpy as np
import torch
import torch as t
import gym
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter
from gym.spaces import Discrete
from typing import Any, List, Optional, Union, Tuple, Iterable
from einops import rearrange
import importlib
import wandb


# + id="Q5E93-BGRjuy"
def make_env(env_id: str, seed: int, idx: int, capture_video: bool, run_name: str):
    """Return a function that returns an environment after setting up boilerplate."""
    
    def thunk():
        env = gym.make(env_id, new_step_api=True)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if capture_video:
            if idx == 0:
                env = gym.wrappers.RecordVideo(
                    env, 
                    f"videos/{run_name}", 
                    episode_trigger=lambda x : x % 50 == 0 # Video every 50 runs for env #1
                )
        obs = env.reset(seed=seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env
    
    return thunk


# + id="Kf152ROwHjM_" executionInfo={"status": "ok", "timestamp": 1677860420697, "user_tz": 0, "elapsed": 6, "user": {"displayName": "Oskar Hollinsworth", "userId": "00307706571197304608"}}
def test_minibatch_indexes(minibatch_indexes):
    for n in range(5):
        frac, minibatch_size = np.random.randint(1, 8, size=(2,))
        batch_size = frac * minibatch_size
        indices = minibatch_indexes(batch_size, minibatch_size)
        assert any([isinstance(indices, list), isinstance(indices, np.ndarray)])
        assert isinstance(indices[0], np.ndarray)
        assert len(indices) == frac
        np.testing.assert_equal(np.sort(np.stack(indices).flatten()), np.arange(batch_size))


# + id="mhvduVeOHkln" executionInfo={"status": "ok", "timestamp": 1677860420698, "user_tz": 0, "elapsed": 7, "user": {"displayName": "Oskar Hollinsworth", "userId": "00307706571197304608"}}
def test_calc_entropy_bonus(calc_entropy_bonus):
    probs = Categorical(logits=t.randn((3, 4)))
    ent_coef = 0.5
    expected = ent_coef * probs.entropy().mean()
    actual = calc_entropy_bonus(probs, ent_coef)
    t.testing.assert_close(expected, actual)


# + id="SGjJl_bp35AG" executionInfo={"status": "ok", "timestamp": 1677860420698, "user_tz": 0, "elapsed": 6, "user": {"displayName": "Oskar Hollinsworth", "userId": "00307706571197304608"}}
MAIN = __name__ == "__main__"
os.environ['WANDB_NOTEBOOK_NAME'] = 'cartpole.py'


# + id="Aya60GeCGA5X" executionInfo={"status": "ok", "timestamp": 1677860420698, "user_tz": 0, "elapsed": 6, "user": {"displayName": "Oskar Hollinsworth", "userId": "00307706571197304608"}}
def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    t.nn.init.orthogonal_(layer.weight, std)
    t.nn.init.constant_(layer.bias, bias_const)
    return layer

class Agent(nn.Module):
    critic: nn.Sequential
    actor: nn.Sequential

    def __init__(self, envs: gym.vector.SyncVectorEnv):
        super().__init__()
        obs_shape = np.array(
            (envs.num_envs, ) + envs.single_action_space.shape
        ).prod().astype(int)
        self.actor = nn.Sequential(
            layer_init(nn.Linear(obs_shape, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, envs.single_action_space.n), std=.01),
        )
        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_shape, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1),
        )



# + id="6PwPZHlLGDYu" executionInfo={"status": "ok", "timestamp": 1677860420698, "user_tz": 0, "elapsed": 6, "user": {"displayName": "Oskar Hollinsworth", "userId": "00307706571197304608"}}
# %%
@t.inference_mode()
def compute_advantages(
    next_value: t.Tensor,
    next_done: t.Tensor,
    rewards: t.Tensor,
    values: t.Tensor,
    dones: t.Tensor,
    device: t.device,
    gamma: float,
    gae_lambda: float,
) -> t.Tensor:
    '''Compute advantages using Generalized Advantage Estimation.

    next_value: shape (1, env) - 
        represents V(s_{t+1}) which is needed for the last advantage term
    next_done: shape (env,)
    rewards: shape (t, env)
    values: shape (t, env)
    dones: shape (t, env)

    Return: shape (t, env)
    '''
    assert isinstance(next_value, t.Tensor)
    assert isinstance(next_done, t.Tensor)
    assert isinstance(rewards, t.Tensor)
    assert isinstance(values, t.Tensor)
    assert isinstance(dones, t.Tensor)
    t_max, n_env = values.shape
    next_values = t.concat((values[1:, ], next_value))
    next_dones = t.concat((dones[1:, ], next_done.unsqueeze(0)))
    deltas = rewards + gamma * next_values * (1.0 - next_dones) - values  
    adv = deltas.clone().to(device)
    for to_go in range(1, t_max):
        t_idx = t_max - to_go - 1
        t.testing.assert_close(adv[t_idx], deltas[t_idx])
        adv[t_idx] += (
            gamma * gae_lambda * adv[t_idx + 1] * (1.0 - next_dones[t_idx]) 
        )
    return adv



# + id="uYSSMnF-GPvm" executionInfo={"status": "ok", "timestamp": 1677860420699, "user_tz": 0, "elapsed": 6, "user": {"displayName": "Oskar Hollinsworth", "userId": "00307706571197304608"}}
# %%
@dataclass
class Minibatch:
    obs: t.Tensor
    logprobs: t.Tensor
    actions: t.Tensor
    advantages: t.Tensor
    returns: t.Tensor
    values: t.Tensor

def minibatch_indexes(
    batch_size: int, minibatch_size: int
) -> List[np.ndarray]:
    '''
    Return a list of length (batch_size // minibatch_size) where 
    each element is an array of indexes into the batch.

    Each index should appear exactly once.
    '''
    assert batch_size % minibatch_size == 0
    n = batch_size // minibatch_size
    indices = np.arange(batch_size)
    np.random.shuffle(indices)
    return [indices[i::n] for i in range(n)]

if MAIN:
    test_minibatch_indexes(minibatch_indexes)

def make_minibatches(
    obs: t.Tensor,
    logprobs: t.Tensor,
    actions: t.Tensor,
    advantages: t.Tensor,
    values: t.Tensor,
    obs_shape: tuple,
    action_shape: tuple,
    batch_size: int,
    minibatch_size: int,
) -> List[Minibatch]:
    '''
    Flatten the environment and steps dimension into one batch dimension, 
    then shuffle and split into minibatches.
    '''
    n_steps, n_env = values.shape
    n_dim = n_steps * n_env
    indexes = minibatch_indexes(batch_size=batch_size, minibatch_size=minibatch_size)
    obs_flat = obs.reshape((batch_size,) + obs_shape)
    act_flat = actions.reshape((batch_size,) + action_shape)
    probs_flat = logprobs.reshape((batch_size,) + action_shape)
    adv_flat = advantages.reshape(n_dim)
    val_flat = values.reshape(n_dim)
    return [
        Minibatch(
            obs_flat[idx], probs_flat[idx], act_flat[idx], adv_flat[idx], 
            adv_flat[idx] + val_flat[idx], val_flat[idx]
        )
        for idx in indexes
    ]



# + id="K7wXDJ9MGOWu" executionInfo={"status": "ok", "timestamp": 1677860420699, "user_tz": 0, "elapsed": 6, "user": {"displayName": "Oskar Hollinsworth", "userId": "00307706571197304608"}}
# %%
def calc_policy_loss(
    probs: Categorical, mb_action: t.Tensor, mb_advantages: t.Tensor, 
    mb_logprobs: t.Tensor, clip_coef: float
) -> t.Tensor:
    '''
    Return the policy loss, suitable for maximisation with gradient ascent.

    probs: 
        a distribution containing the actor's unnormalized logits of 
        shape (minibatch, num_actions)

    clip_coef: amount of clipping, denoted by epsilon in Eq 7.

    normalize: if true, normalize mb_advantages to have mean 0, variance 1
    '''
    adv_norm = (mb_advantages - mb_advantages.mean()) / mb_advantages.std()
    ratio = t.exp(probs.log_prob(mb_action)) / t.exp(mb_logprobs)
    min_left = ratio * adv_norm
    min_right = t.clip(ratio, 1 - clip_coef, 1 + clip_coef) * adv_norm
    return t.minimum(min_left, min_right).mean()



# + id="CmyxU6JWGMsG" executionInfo={"status": "ok", "timestamp": 1677860420699, "user_tz": 0, "elapsed": 6, "user": {"displayName": "Oskar Hollinsworth", "userId": "00307706571197304608"}}
# %%
def calc_value_function_loss(
    critic: nn.Sequential, mb_obs: t.Tensor, mb_returns: t.Tensor, v_coef: float
) -> t.Tensor:
    '''Compute the value function portion of the loss function.
    Need to minimise this

    v_coef: 
        the coefficient for the value loss, which weights its contribution to 
        the overall loss. Denoted by c_1 in the paper.
    '''
    output = critic(mb_obs)
    return v_coef * (output - mb_returns).pow(2).mean() / 2



# + id="npyWs6xjGLkP" executionInfo={"status": "ok", "timestamp": 1677860420699, "user_tz": 0, "elapsed": 5, "user": {"displayName": "Oskar Hollinsworth", "userId": "00307706571197304608"}}
# %%
def calc_entropy_loss(probs: Categorical, ent_coef: float):
    '''Return the entropy loss term.
    Need to maximise this

    ent_coef: 
        The coefficient for the entropy loss, which weights its contribution to the overall loss. 
        Denoted by c_2 in the paper.
    '''
    return probs.entropy().mean() * ent_coef

if MAIN:
    test_calc_entropy_bonus(calc_entropy_loss)


# + id="nqJeg1kZGKSG" executionInfo={"status": "ok", "timestamp": 1677860420700, "user_tz": 0, "elapsed": 6, "user": {"displayName": "Oskar Hollinsworth", "userId": "00307706571197304608"}}
# %%
class PPOScheduler:
    def __init__(self, optimizer: optim.Adam, initial_lr: float, end_lr: float, num_updates: int):
        self.optimizer = optimizer
        self.initial_lr = initial_lr
        self.end_lr = end_lr
        self.num_updates = num_updates
        self.n_step_calls = 0

    def step(self):
        '''
        Implement linear learning rate decay so that after num_updates calls to step, 
        the learning rate is end_lr.
        '''
        lr = (
            self.initial_lr + 
            (self.end_lr - self.initial_lr) * self.n_step_calls / self.num_updates
        )
        for param in self.optimizer.param_groups:
            param['lr'] = lr
        self.n_step_calls += 1

def make_optimizer(
    agent: Agent, num_updates: int, initial_lr: float, end_lr: float
) -> Tuple[optim.Adam, PPOScheduler]:
    '''Return an appropriately configured Adam with its attached scheduler.'''
    optimizer = optim.Adam(agent.parameters(), lr=initial_lr, maximize=True)
    scheduler = PPOScheduler(
        optimizer=optimizer, initial_lr=initial_lr, end_lr=end_lr, num_updates=num_updates
    )
    return optimizer, scheduler



# + id="mgZ7-wsRCxJW" executionInfo={"status": "ok", "timestamp": 1677860512190, "user_tz": 0, "elapsed": 198, "user": {"displayName": "Oskar Hollinsworth", "userId": "00307706571197304608"}}
@dataclass
class PPOArgs:
    exp_name: str = 'cartpole.py'    
    seed: int = 1
    torch_deterministic: bool = True
    cuda: bool = True
    track: bool = True
    wandb_project_name: str = "PPOCart"
    wandb_entity: str = None
    capture_video: bool = False
    env_id: str = "CartPole-v1"
    total_timesteps: int = 40_000
    learning_rate: float = 0.00025
    num_envs: int = 4
    num_steps: int = 128
    gamma: float = 0.99
    gae_lambda: float = 0.95
    num_minibatches: int = 4
    update_epochs: int = 4
    clip_coef: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    batch_size: int = 512
    minibatch_size: int = 128


# + id="FHmn5kSUGFFu" executionInfo={"status": "ok", "timestamp": 1677860828050, "user_tz": 0, "elapsed": 674, "user": {"displayName": "Oskar Hollinsworth", "userId": "00307706571197304608"}}
# %%
def train_ppo(args: PPOArgs):
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % "\n".join([f"|{key}|{value}|" 
        for (key, value) in vars(args).items()]),
    )
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    envs = gym.vector.SyncVectorEnv([
        make_env(args.env_id, args.seed + i, i, args.capture_video, run_name) 
        for i in range(args.num_envs)
    ])
    action_shape = envs.single_action_space.shape
    assert action_shape is not None
    assert isinstance(envs.single_action_space, Discrete), "only discrete action space is supported"
    agent = Agent(envs).to(device)
    num_updates = args.total_timesteps // args.batch_size
    (optimizer, scheduler) = make_optimizer(agent, num_updates, args.learning_rate, 0.0)
    obs = torch.zeros(
        (args.num_steps, args.num_envs) + envs.single_observation_space.shape
    ).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + action_shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)
    global_step = 0
    old_approx_kl = 0.0
    approx_kl = 0.0
    value_loss = t.tensor(0.0)
    policy_loss = t.tensor(0.0)
    entropy_loss = t.tensor(0.0)
    clipfracs = []
    info = []
    start_time = time.time()
    next_obs = torch.Tensor(envs.reset()).to(device)
    next_done = torch.zeros(args.num_envs).to(device)
    for _ in range(num_updates):
        for i in range(0, args.num_steps):
            # Rollout phase
            global_step += 1
            curr_obs = next_obs
            done = next_done
            with t.inference_mode():
                logits = agent.actor(curr_obs).detach()
                q_values = agent.critic(curr_obs).detach().squeeze(-1)
            prob = Categorical(logits=logits)
            action = prob.sample()
            logprob = prob.log_prob(action)
            next_obs, reward, next_done, info = envs.step(action.numpy())
            next_obs = t.tensor(next_obs, device=device)
            next_done = t.tensor(next_done, device=device)
            obs[i] = curr_obs
            actions[i] = action
            logprobs[i] = logprob
            rewards[i] = t.tensor(reward, device=device)
            dones[i] = done.detach().clone() # t.tensor(done, device=device)
            values[i] = q_values

            if "episode" in info.keys():
                for item in info['episode']:
                    if item is None or 'r' not in item.keys():
                        continue
                    if global_step % 10 == 0:
                        print(f"global_step={global_step}, episodic_return={item['r']}")
                    writer.add_scalar("charts/episodic_return", item["r"], global_step)
                    writer.add_scalar("charts/episodic_length", item["l"], global_step)
        with t.inference_mode():
            next_value = rearrange(agent.critic(next_obs), "env 1 -> 1 env")
        advantages = compute_advantages(
            next_value, next_done, rewards, values, dones, device, args.gamma, args.gae_lambda
        )
        clipfracs.clear()
        for _ in range(args.update_epochs):
            minibatches = make_minibatches(
                obs,
                logprobs,
                actions,
                advantages,
                values,
                envs.single_observation_space.shape,
                action_shape,
                args.batch_size,
                args.minibatch_size,
            )
            for mb in minibatches:
                probs = Categorical(logits=agent.actor(mb.obs))
                value_loss = calc_value_function_loss(agent.critic, mb.obs, mb.returns, args.vf_coef)
                policy_loss = calc_policy_loss(probs, mb.actions, mb.advantages, mb.logprobs, args.clip_coef)
                entropy_loss = calc_entropy_loss(probs, args.ent_coef)
                loss = policy_loss + entropy_loss - value_loss
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()

        scheduler.step()
        (y_pred, y_true) = (mb.values.cpu().numpy(), mb.returns.cpu().numpy())
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
        with torch.no_grad():
            newlogprob: t.Tensor = probs.log_prob(mb.actions)
            logratio = newlogprob - mb.logprobs
            ratio = logratio.exp()
            old_approx_kl = (-logratio).mean().item()
            approx_kl = (ratio - 1 - logratio).mean().item()
            clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", value_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", policy_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl, global_step)
        writer.add_scalar("losses/approx_kl", approx_kl, global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
        if global_step % 100 == 0:
            print("steps per second (SPS):", int(global_step / (time.time() - start_time)))
            print("losses/value_loss", value_loss.item())
            print("losses/policy_loss", policy_loss.item())
            print("losses/entropy", entropy_loss.item())
    print(f'... training complete after {global_step} steps')
    envs.close()
    writer.close()



# + colab={"base_uri": "https://localhost:8080/", "height": 1000, "referenced_widgets": ["d5dd7fcac74d422a88caedc9ea1fc0f9", "432e821806af4ea98a7113e113e5b41d", "4b4bd01ef1af4dd2a4be4f193cad37cd", "3580bbdb38b8404d8e80db468b6768ac", "66b9fdf9fa3a4c9191371242d08e61d8", "d27a0274421c47b7a2041122f750e4a1", "a7ab2f4449f64758b9b64be7012a3957", "dfd643fd84ea48b3bdf8b6c01c45766d"]} id="JZvMdcw8M5X5" executionInfo={"status": "ok", "timestamp": 1677860862245, "user_tz": 0, "elapsed": 34200, "user": {"displayName": "Oskar Hollinsworth", "userId": "00307706571197304608"}} outputId="a6e5adbf-3ae9-4abc-c9c4-12f87206b054"
if MAIN:
    args = PPOArgs()
    train_ppo(args)

# + id="cXbn7q4EMEQA" executionInfo={"status": "aborted", "timestamp": 1677860421562, "user_tz": 0, "elapsed": 7, "user": {"displayName": "Oskar Hollinsworth", "userId": "00307706571197304608"}}

