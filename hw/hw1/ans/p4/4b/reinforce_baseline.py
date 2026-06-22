# Spring 2026, 535510 Reinforcement Learning
# HW1: REINFORCE with baseline (value function) on LunarLander-v3

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

# Define a tensorboard writer
writer = SummaryWriter("./tb_record_1")

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

    def calculate_loss(self, gamma=0.99):
        """
            Calculate the loss (= policy loss + value loss) to perform backprop later

            Key difference from vanilla REINFORCE:
            - Policy loss uses advantage A_t = G_t - V(s_t) instead of raw return G_t
            - Subtracting the baseline V(s_t) reduces variance without introducing bias
            - value.item() detaches V(s_t) from the computation graph for the policy term,
              ensuring gradients only flow through value_losses for the value network
        """

        # Initialize the lists and variables
        R = 0
        saved_actions = self.saved_actions
        policy_losses = []
        value_losses = []
        returns = []

        # Step 1: Calculate rewards-to-go (discounted cumulative reward from each time step)
        for r in self.rewards[::-1]:
            R = r + gamma * R
            returns.insert(0, R)

        # Normalize returns; dtype=float32 to match network weights (self.double() is
        # called before layer definitions so layers stay float32)
        returns = torch.tensor(returns, dtype=torch.float32)
        returns = (returns - returns.mean()) / (returns.std() + 1e-9)

        # Step 2 & 3: Compute policy loss (with baseline) and value loss
        for (log_prob, value), R in zip(saved_actions, returns):
            # Advantage: G_t - V(s_t); use .item() to detach V(s_t) from policy gradient
            advantage = R - value.item()
            # Policy loss: -log π(a|s) * A_t
            policy_losses.append(-log_prob * advantage)
            # Value loss: MSE between predicted value and actual return
            value_losses.append(F.mse_loss(value.squeeze(), R))

        # Total loss = policy loss + value loss
        loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()

        return loss

    def clear_memory(self):
        # reset rewards and action buffer
        del self.rewards[:]
        del self.saved_actions[:]


def train(lr=0.002):
    """
        Train the model using Adam optimizer via backpropagation
        - Runs full episodes, updates policy at the end of each episode
        - Uses value function V(s) as baseline to reduce variance
    """

    # Instantiate the policy model and the optimizer
    model = Policy()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Learning rate scheduler (optional — uncomment if training stalls)
    # scheduler = Scheduler.StepLR(optimizer, step_size=200, gamma=0.9)

    # EWMA reward for tracking the learning progress
    ewma_reward = 0

    # run infinitely many episodes
    for i_episode in count(1):
        # reset environment and episode reward
        state, _ = env.reset()
        ep_reward = 0
        t = 0

        # Uncomment the following line to use learning rate scheduler
        # scheduler.step()

        # For each episode, only run 9999 steps to avoid infinite loops
        for t in range(1, 10000):
            action = model.select_action(state)
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            model.rewards.append(reward)
            ep_reward += reward
            if done:
                break

        # Compute loss and update networks
        loss = model.calculate_loss()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        model.clear_memory()

        # update EWMA reward and log the results
        ewma_reward = 0.05 * ep_reward + (1 - 0.05) * ewma_reward
        print('Episode {}\tlength: {}\treward: {}\t ewma reward: {}'.format(
            i_episode, t, ep_reward, ewma_reward))

        # Record to Tensorboard
        writer.add_scalar('Loss/train', loss.item(), i_episode)
        writer.add_scalar('Reward/episode', ep_reward, i_episode)
        writer.add_scalar('Reward/ewma', ewma_reward, i_episode)
        writer.add_scalar('Episode/length', t, i_episode)
        writer.add_scalar('LR', optimizer.param_groups[0]['lr'], i_episode)

        # LunarLander-v3 is considered solved when EWMA reward > 200 (env.spec.reward_threshold)
        if ewma_reward > env.spec.reward_threshold:
            if not os.path.isdir("./preTrained"):
                os.mkdir("./preTrained")
            torch.save(model.state_dict(), './preTrained/LunarLander_baseline_{}.pth'.format(lr))
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
    lr = 0.002
    env_name = 'LunarLander-v3'
    env = gym.make(env_name)
    obs, _ = env.reset(seed=random_seed)
    torch.manual_seed(random_seed)
    train(lr)
    test(f'LunarLander_baseline_{lr}.pth', env_name)
