import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import sys
import gym
import matplotlib.pyplot as plt
from itertools import count
import numpy as np


# Define the actor network
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = self.softmax(self.fc2(x))
        return x

# Define the critic network
class Critic(nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = self.fc2(x)
        return x

# Define the Actor-Critic algorithm
class ActorCritic():
    def __init__(self, state_dim, action_dim, hidden_dim, lr):
        self.actor = Actor(state_dim, action_dim, hidden_dim)
        self.critic = Critic(state_dim, hidden_dim)
        self.optimizer_actor = optim.Adam(self.actor.parameters(), lr=lr)
        self.optimizer_critic = optim.Adam(self.critic.parameters(), lr=lr)

    def select_action(self, state):
        state = torch.tensor(state, dtype=torch.float32)
        action_probs = self.actor(state)
        dist = torch.distributions.Categorical(action_probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob

    def update(self, state, action, reward, next_state, done, gamma):
        state = torch.tensor(state, dtype=torch.float32)
        next_state = torch.tensor(next_state, dtype=torch.float32)
        action = torch.tensor(action, dtype=torch.int64)
        reward = torch.tensor(reward, dtype=torch.float32)

        # Compute the TD error and the target value
        value = self.critic(state)
        next_value = self.critic(next_state)
        td_error = reward + gamma * next_value * (1 - done) - value
        target_value = reward + gamma * next_value * (1 - done)

        # Update the critic network
        critic_loss = td_error.pow(2).mean()
        self.optimizer_critic.zero_grad()
        critic_loss.backward()
        self.optimizer_critic.step()

        # Update the actor network
        action_probs = self.actor(state)
        dist = torch.distributions.Categorical(action_probs)
        log_prob = dist.log_prob(action)
        actor_loss = -log_prob * td_error.detach()
        self.optimizer_actor.zero_grad()
        actor_loss.backward()
        self.optimizer_actor.step()

def train():
    GAMMA = 0.9

    env = gym.make('CartPole-v0')
    model = ActorCritic(4,2,128,3e-4)

    max_episode_num = 1000
    max_steps = 10000
    numsteps = []
    avg_numsteps = []
    all_rewards = []

    for episode in range(max_episode_num):
        state, info = env.reset()
        
        log_probs = []
        rewards = []
        value_episode = 0 
        for steps in range(max_steps):
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            action, log_prob =model.select_action(state)
       
            new_state, reward, terminated, truncated, _  = env.step(action)
            done = terminated or truncated
        
            log_probs.append(log_prob)
            rewards.append(reward)
            
            model.update(state, action, reward, new_state, done, GAMMA)
            
            if done:
                numsteps.append(steps)
                avg_numsteps.append(np.mean(numsteps[-10:]))
                all_rewards.append(np.sum(rewards))
                if episode % 1 == 0:
                    sys.stdout.write("episode: {}, total reward: {}, average_reward: {}, length: {}\n".format(episode, np.round(np.sum(rewards), decimals = 3),  np.round(np.mean(all_rewards[-10:]), decimals = 3), steps))
                break

            state = new_state

    plt.plot(numsteps)
    plt.plot(avg_numsteps)
    plt.xlabel('Episode')
    plt.show()

if __name__ == '__main__':
    train()