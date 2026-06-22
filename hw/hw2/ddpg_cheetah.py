#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import random
from collections import deque

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from tqdm import trange

# Sync W&B by default for final training. Use --wandb-mode offline/disabled for local tests.
os.environ.setdefault("WANDB_MODE", "online")
try:
    import wandb
except ImportError:
    class _NoOpWandb:
        @staticmethod
        def init(*args, **kwargs):
            return None

        @staticmethod
        def log(*args, **kwargs):
            return None

        @staticmethod
        def finish(*args, **kwargs):
            return None

    wandb = _NoOpWandb()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(name):
    if name == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(name)


def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.mul_(1.0 - tau).add_(param.data, alpha=tau)


def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


class ReplayBuffer:
    def __init__(self, obs_dim, act_dim, capacity, device):
        self.capacity = capacity
        self.device = device
        self.ptr = 0
        self.size = 0
        self.states = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, act_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.next_states = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.not_dones = np.zeros((capacity, 1), dtype=np.float32)

    def push(self, state, action, reward, next_state, terminated):
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_state
        self.not_dones[self.ptr] = 1.0 - float(terminated)
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        idx = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.as_tensor(self.states[idx], device=self.device),
            torch.as_tensor(self.actions[idx], device=self.device),
            torch.as_tensor(self.rewards[idx], device=self.device),
            torch.as_tensor(self.next_states[idx], device=self.device),
            torch.as_tensor(self.not_dones[idx], device=self.device),
        )


class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim, max_action, hidden_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, act_dim),
            nn.Tanh(),
        )
        self.register_buffer("max_action", torch.as_tensor(max_action, dtype=torch.float32))
        self.net[-2].weight.data.uniform_(-3e-3, 3e-3)
        self.net[-2].bias.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        return self.net(state) * self.max_action


class Critic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )
        self.net[-1].weight.data.uniform_(-3e-3, 3e-3)
        self.net[-1].bias.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, action):
        return self.net(torch.cat([state, action], dim=-1))


class DDPGAgent:
    def __init__(self, obs_dim, act_dim, max_action, args, device):
        self.device = device
        self.gamma = args.gamma
        self.tau = args.tau
        self.max_action_np = np.asarray(max_action, dtype=np.float32)

        self.actor = Actor(obs_dim, act_dim, max_action, args.hidden_size).to(device)
        self.actor_target = Actor(obs_dim, act_dim, max_action, args.hidden_size).to(device)
        self.actor_optim = Adam(self.actor.parameters(), lr=args.actor_lr)

        self.critic = Critic(obs_dim, act_dim, args.hidden_size).to(device)
        self.critic_target = Critic(obs_dim, act_dim, args.hidden_size).to(device)
        self.critic_optim = Adam(self.critic.parameters(), lr=args.critic_lr)

        hard_update(self.actor_target, self.actor)
        hard_update(self.critic_target, self.critic)

    def select_action(self, state, noise_std=0.0):
        state = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        self.actor.eval()
        with torch.no_grad():
            action = self.actor(state).cpu().numpy()[0]
        self.actor.train()
        if noise_std > 0.0:
            action = action + np.random.normal(0.0, noise_std, size=action.shape) * self.max_action_np
        return np.clip(action, -self.max_action_np, self.max_action_np).astype(np.float32)

    def update(self, replay_buffer, batch_size):
        state, action, reward, next_state, not_done = replay_buffer.sample(batch_size)
        with torch.no_grad():
            next_action = self.actor_target(next_state)
            target_q = self.critic_target(next_state, next_action)
            target_q = reward + self.gamma * not_done * target_q

        current_q = self.critic(state, action)
        critic_loss = F.mse_loss(current_q, target_q)
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        actor_loss = -self.critic(state, self.actor(state)).mean()
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        soft_update(self.actor_target, self.actor, self.tau)
        soft_update(self.critic_target, self.critic, self.tau)
        return critic_loss.item(), actor_loss.item()

    def save(self, out_dir, env_name, suffix):
        os.makedirs(out_dir, exist_ok=True)
        actor_path = os.path.join(out_dir, f"ddpg_actor_{env_name}_{suffix}.pth")
        critic_path = os.path.join(out_dir, f"ddpg_critic_{env_name}_{suffix}.pth")
        torch.save(self.actor.state_dict(), actor_path)
        torch.save(self.critic.state_dict(), critic_path)
        print(f"Saved models to {actor_path} and {critic_path}")
        return actor_path, critic_path


def evaluate(agent, env_name, seed, episodes):
    env = gym.make(env_name)
    returns = []
    for episode in range(episodes):
        state, _ = env.reset(seed=seed + 10000 + episode)
        done = False
        total_reward = 0.0
        while not done:
            action = agent.select_action(state, noise_std=0.0)
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
        returns.append(total_reward)
    env.close()
    return float(np.mean(returns)), float(np.std(returns))


def train(args):
    os.environ["WANDB_MODE"] = args.wandb_mode
    set_seed(args.seed)
    device = get_device(args.device)
    env = gym.make(args.env)
    env.action_space.seed(args.seed)
    state, _ = env.reset(seed=args.seed)

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    max_action = env.action_space.high
    replay_buffer = ReplayBuffer(obs_dim, act_dim, args.replay_size, device)
    agent = DDPGAgent(obs_dim, act_dim, max_action, args, device)
    recent_returns = deque(maxlen=10)
    episode_return = 0.0
    episode_length = 0
    episode_num = 0
    best_eval = -np.inf

    wandb.init(project=args.project, name=args.run_name, config=vars(args))
    progress = trange(1, args.max_steps + 1, desc="DDPG HalfCheetah")
    for step in progress:
        if step <= args.start_steps:
            action = env.action_space.sample()
        else:
            action = agent.select_action(state, noise_std=args.expl_noise)

        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        replay_buffer.push(state, action, reward, next_state, terminated)
        state = next_state
        episode_return += reward
        episode_length += 1

        if replay_buffer.size >= args.batch_size and step > args.start_steps:
            critic_loss, actor_loss = agent.update(replay_buffer, args.batch_size)
            wandb.log(
                {
                    "train/critic_loss": critic_loss,
                    "train/actor_loss": actor_loss,
                    "train/replay_size": replay_buffer.size,
                },
                step=step,
            )

        if done:
            recent_returns.append(episode_return)
            wandb.log(
                {
                    "train/episode_return": episode_return,
                    "train/episode_length": episode_length,
                    "train/episode": episode_num,
                    "train/recent_return": np.mean(recent_returns),
                },
                step=step,
            )
            progress.set_postfix(return_=f"{episode_return:.1f}", episode=episode_num)
            episode_num += 1
            episode_return = 0.0
            episode_length = 0
            state, _ = env.reset()

        if step % args.eval_freq == 0 or step == args.max_steps:
            eval_mean, eval_std = evaluate(agent, args.env, args.seed, args.eval_episodes)
            wandb.log({"eval/mean_return": eval_mean, "eval/std_return": eval_std}, step=step)
            print(f"Step {step}: eval mean {eval_mean:.2f} +/- {eval_std:.2f}")
            if eval_mean > best_eval:
                best_eval = eval_mean
                agent.save(args.out_dir, args.env, f"best_step{step}_score{eval_mean:.1f}")

        if step % args.save_freq == 0:
            agent.save(args.out_dir, args.env, f"step{step}")

    agent.save(args.out_dir, args.env, "final")
    env.close()
    wandb.finish()


def parse_args():
    parser = argparse.ArgumentParser(description="Vanilla DDPG for HalfCheetah-v5")
    parser.add_argument("--env", type=str, default="HalfCheetah-v5")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int, default=500_000)
    parser.add_argument("--start-steps", type=int, default=10_000)
    parser.add_argument("--eval-freq", type=int, default=10_000)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--save-freq", type=int, default=50_000)
    parser.add_argument("--replay-size", type=int, default=1_000_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--actor-lr", type=float, default=3e-4)
    parser.add_argument("--critic-lr", type=float, default=3e-4)
    parser.add_argument("--expl-noise", type=float, default=0.1)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument("--out-dir", type=str, default="preTrained")
    parser.add_argument("--project", type=str, default="ddpg-halfcheetah")
    parser.add_argument("--run-name", type=str, default="ddpg_cheetah_seed42")
    parser.add_argument("--wandb-mode", type=str, default="online", choices=["offline", "online", "disabled"])
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
