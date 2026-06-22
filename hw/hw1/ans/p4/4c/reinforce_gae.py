# Spring 2026, 535510 Reinforcement Learning
# HW1: REINFORCE with Generalized Advantage Estimation (GAE) on LunarLander-v3

import os
import gymnasium as gym
from itertools import count
from collections import namedtuple
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import torch.optim.lr_scheduler as Scheduler
from torch.utils.tensorboard import SummaryWriter

# Define a useful tuple (optional)
SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])

# writer is re-instantiated per lambda in __main__
writer = None


class Policy(nn.Module):
    """
        Implement both policy network and the value network in one model
        - Actor and value networks share the first layer
        - Architecture: Linear(obs_dim -> 128, ReLU) -> action_head + value_head
    """
    def __init__(self):
        super(Policy, self).__init__()

        # Extract the dimensionality of state and action spaces
        self.discrete = isinstance(env.action_space, gym.spaces.Discrete)
        self.observation_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n if self.discrete else env.action_space.shape[0]
        self.hidden_size = 128
        self.double()

        # Shared layer (actor and value networks share this)
        self.fc1 = nn.Linear(self.observation_dim, self.hidden_size)

        # Action head: outputs logits for each action
        self.action_head = nn.Linear(self.hidden_size, self.action_dim)

        # Value head: outputs a scalar state value V(s)
        self.value_head = nn.Linear(self.hidden_size, 1)

        # Random weight initialization (Xavier uniform)
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.action_head.weight)
        nn.init.xavier_uniform_(self.value_head.weight)

        # action & reward memory
        self.saved_actions = []
        self.rewards = []

    def forward(self, state):
        """
            Forward pass of both policy and value networks
            Returns action probability distribution and state value V(s)
        """
        x = F.relu(self.fc1(state))
        action_prob = F.softmax(self.action_head(x), dim=-1)
        state_value = self.value_head(x)

        return action_prob, state_value

    def select_action(self, state):
        """
            Select action from stochastic policy given current state
            Saves (log_prob, value) to buffer for later loss computation
        """
        state = torch.from_numpy(state).unsqueeze(0)
        action_prob, state_value = self.forward(state)
        m = Categorical(action_prob)
        action = m.sample()

        # save to action buffer
        self.saved_actions.append(SavedAction(m.log_prob(action), state_value))

        return action.item()

    def calculate_loss(self, advantages, gamma=0.99):
        """
            Calculate the loss using pre-computed GAE advantages.

            Arguments:
                advantages: list of GAE advantage estimates A_t^{GAE}
                gamma: discount factor (used for rewards-to-go for value loss)

            Policy loss: -log π(a|s) * A_t^{GAE}  (normalized)
            Value loss:  MSE(V(s_t), G_t)  where G_t is the discounted return
        """
        saved_actions = self.saved_actions
        policy_losses = []
        value_losses = []

        # Normalize GAE advantages; dtype=float32 to match network weights
        advantages = torch.tensor(advantages, dtype=torch.float32)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-9)

        # Compute rewards-to-go (G_t) for value network supervision
        R = 0
        returns = []
        for r in self.rewards[::-1]:
            R = r + gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns, dtype=torch.float32)

        # Policy loss uses GAE advantages; value loss uses discounted returns
        for (log_prob, value), adv, ret in zip(saved_actions, advantages, returns):
            # Policy loss: -log π(a|s) * A^GAE
            policy_losses.append(-log_prob * adv)
            # Value loss: MSE between predicted value and actual discounted return
            value_losses.append(F.mse_loss(value.squeeze(), ret))

        # Total loss = policy loss + value loss
        loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()

        return loss

    def clear_memory(self):
        # reset rewards and action buffer
        del self.rewards[:]
        del self.saved_actions[:]


class GAE:
    def __init__(self, gamma, lambda_, num_steps):
        self.gamma = gamma
        self.lambda_ = lambda_
        self.num_steps = num_steps   # set num_steps = None to adapt full batch

    def __call__(self, rewards, values, done):
        """
            Generalized Advantage Estimation (Schulman et al. 2016)

            Formula (computed backwards for efficiency):
                δ_t = r_t + γ * V(s_{t+1}) * (1 - done_t) - V(s_t)
                A_t^GAE = Σ_{l=0}^∞ (γλ)^l * δ_{t+l}

            Arguments:
                rewards: list of length T         [r_0, ..., r_{T-1}]
                values:  list of length T+1       [V(s_0), ..., V(s_{T-1}), V_bootstrap]
                         V_bootstrap = 0 if terminated, V(s_T) if truncated
                done:    list of length T (bool)  [done_0, ..., done_{T-1}]

            Returns:
                advantages: list of length T with GAE estimates
        """
        T = len(rewards)
        advantages = []
        gae = 0.0

        # Iterate backwards through time steps
        for t in reversed(range(T)):
            # TD error: r_t + γ*V(s_{t+1})*(1-done) - V(s_t)
            delta = (rewards[t]
                     + self.gamma * values[t + 1] * (1.0 - float(done[t]))
                     - values[t])
            # Recursive GAE: A_t = δ_t + γλ*(1-done)*A_{t+1}
            gae = delta + self.gamma * self.lambda_ * (1.0 - float(done[t])) * gae
            advantages.insert(0, gae)

        return advantages


def train(lr=0.002, lambda_=0.95):
    """
        Train the model using GAE for advantage estimation.
        - Collects full episodes, computes GAE advantages, then updates networks
        - lambda_ controls bias-variance tradeoff: 0 = high bias/low var, 1 = low bias/high var
    """
    global writer

    gamma = 0.99

    # Instantiate the policy model and the optimizer
    model = Policy()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    gae_fn = GAE(gamma=gamma, lambda_=lambda_, num_steps=None)

    # Learning rate scheduler (optional)
    # scheduler = Scheduler.StepLR(optimizer, step_size=200, gamma=0.9)

    # EWMA reward for tracking the learning progress
    ewma_reward = 0

    # run infinitely many episodes
    for i_episode in count(1):
        # reset environment and episode reward
        state, _ = env.reset()
        ep_reward = 0
        t = 0

        # Track done flags and raw state values for GAE computation
        done_flags = []
        raw_values = []   # V(s_t) as plain floats (detached from graph)

        # Uncomment the following line to use learning rate scheduler
        # scheduler.step()

        # For each episode, only run 9999 steps to avoid infinite loops
        for t in range(1, 10000):
            action = model.select_action(state)
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            model.rewards.append(reward)
            ep_reward += reward
            done_flags.append(done)

            # Extract V(s_t) computed during select_action (detached via .item())
            raw_values.append(model.saved_actions[-1].value.item())

            if done:
                break

        # Bootstrap value for GAE:
        # - If episode terminated (crash/land), future value is 0
        # - If episode truncated (timeout), bootstrap with V(s_T) from network
        if truncated and not terminated:
            with torch.no_grad():
                state_tensor = torch.from_numpy(state).unsqueeze(0)
                _, bootstrap_tensor = model.forward(state_tensor)
                bootstrap_value = bootstrap_tensor.item()
        else:
            bootstrap_value = 0.0

        # Compute GAE advantages: values list must be length T+1
        advantages = gae_fn(model.rewards, raw_values + [bootstrap_value], done_flags)

        # Compute loss with GAE advantages and update networks
        loss = model.calculate_loss(advantages, gamma=gamma)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        model.clear_memory()

        # update EWMA reward and log the results
        ewma_reward = 0.05 * ep_reward + (1 - 0.05) * ewma_reward
        print('Episode {}\tlength: {}\treward: {:.1f}\t ewma reward: {:.2f}\t lambda: {}'.format(
            i_episode, t, ep_reward, ewma_reward, lambda_))

        # Record to Tensorboard
        writer.add_scalar('Loss/train', loss.item(), i_episode)
        writer.add_scalar('Reward/episode', ep_reward, i_episode)
        writer.add_scalar('Reward/ewma', ewma_reward, i_episode)
        writer.add_scalar('Episode/length', t, i_episode)
        writer.add_scalar('LR', optimizer.param_groups[0]['lr'], i_episode)
        writer.add_scalar('Lambda', lambda_, i_episode)

        # LunarLander-v3 is considered solved when EWMA reward > 200
        if ewma_reward > env.spec.reward_threshold:
            if not os.path.isdir("./preTrained"):
                os.mkdir("./preTrained")
            save_name = './preTrained/LunarLander_gae_{}_{}.pth'.format(lr, lambda_)
            torch.save(model.state_dict(), save_name)
            print("Solved! Running reward is now {} and "
                  "the last episode runs to {} time steps!".format(ewma_reward, t))
            break


def test(name, env_name, n_episodes=10):
    """
        Test the learned model (no change needed)
    """
    model = Policy()

    model.load_state_dict(torch.load('./preTrained/{}'.format(name)))
    env = gym.make(env_name)
    max_episode_len = 10000

    for i_episode in range(1, n_episodes+1):
        state, _ = env.reset()
        running_reward = 0
        for t in range(max_episode_len+1):
            action = model.select_action(state)
            state, reward, terminations, truncations, _ = env.step(action)
            done = np.logical_or(terminations, truncations)
            running_reward += reward
            if done:
                break
        print('Episode {}\tReward: {}'.format(i_episode, running_reward))
    env.close()


if __name__ == '__main__':
    # For reproducibility, fix the random seed
    random_seed = 10
    lr = 0.005
    env_name = 'LunarLander-v3'

    # Experiment with three different lambda values as required by the assignment
    # lambda controls the bias-variance tradeoff in GAE:
    #   0.90 -> higher bias, lower variance (faster but less accurate)
    #   0.95 -> balanced tradeoff (empirically strong default)
    #   0.99 -> lower bias, higher variance (closer to Monte Carlo)
    lambdas = [0.90, 0.95, 0.99]

    for lambda_ in lambdas:
        print("\n" + "="*60)
        print(f"Training with lambda = {lambda_}")
        print("="*60)

        # Re-create env and reset seeds for each run to ensure reproducibility
        env = gym.make(env_name)
        env.reset(seed=random_seed)
        torch.manual_seed(random_seed)

        # Separate TensorBoard log directory per lambda for easy comparison
        writer = SummaryWriter(f"./tb_record_lambda_{lambda_}")

        train(lr=lr, lambda_=lambda_)
        test(f'LunarLander_gae_{lr}_{lambda_}.pth', env_name)

        writer.close()
