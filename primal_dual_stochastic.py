import gym
import sys
import matplotlib
import numpy as np  
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display
plt.ion()
# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PolicyNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size, learning_rate=3e-4):
        super(PolicyNetwork, self).__init__()

        self.num_actions = num_actions
        self.linear1 = nn.Linear(num_inputs, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, num_actions)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        x = F.softmax(x, dim=1)
        return x 
    
    def get_action(self, state):
        probs = self.forward(Variable(state))
        highest_prob_action = np.random.choice(self.num_actions, p=np.squeeze(probs.cpu().detach().numpy()))
        log_prob = torch.log(probs.squeeze(0)[highest_prob_action])
        return highest_prob_action, log_prob

class ValueNetwork(nn.Module):
    def __init__(self, num_inputs, hidden_size, learning_rate=1e-3):
        super(ValueNetwork, self).__init__()

        self.linear1 = nn.Linear(num_inputs, hidden_size)
        self.linear2 = nn.Linear(hidden_size,hidden_size)
        self.linear3 = nn.Linear(hidden_size, 1)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x 

def update_policy(policy_network, log_prob, delta):
        
    policy_gradient = -log_prob * delta
    
    policy_network.optimizer.zero_grad()
    policy_gradient.backward()
    policy_network.optimizer.step()

def update_value(value_network, policy_network, delta, s):
    with torch.no_grad():
        a = policy_network(s)

    constraints = torch.sum(a * delta + delta.pow(2))
    loss = (1- GAMMA) * value_network(s) +  constraints

    value_network.optimizer.zero_grad()
    loss.backward()
    value_network.optimizer.step()

def main():
    # Constants
    GAMMA = 0.99

    env = gym.make('CartPole-v0')
    policy_net = PolicyNetwork(env.observation_space.shape[0], env.action_space.n, 512).to(device)
    value_net = ValueNetwork(env.observation_space.shape[0], 512).to(device)

    max_episode_num = 500
    max_steps = 10000
    numsteps = []
    avg_numsteps = []
    all_rewards = []

    for episode in range(max_episode_num):
        state, info = env.reset()
        
        rewards = []
        for steps in range(max_steps):
            env.render()
            state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            action, log_prob = policy_net.get_action(state)

            new_state, reward, terminated, truncated, _  = env.step(action)
            done = terminated or truncated
            
            rewards.append(reward)
            
            delta = torch.tensor(reward).to(device) + GAMMA * value_net(torch.tensor(new_state, dtype=torch.float32, device=device).unsqueeze(0)) - value_net(state.to(device))
            update_value(value_net, policy_net, delta, state)
            
            delta = torch.tensor(reward).to(device) + GAMMA * value_net(torch.tensor(new_state, dtype=torch.float32, device=device).unsqueeze(0)) - value_net(state.to(device))
            update_policy(policy_net, log_prob, delta)
            
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
    main()