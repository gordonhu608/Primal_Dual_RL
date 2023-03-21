import random
import statistics
import sys
from decimal import Decimal
from collections import defaultdict
import gym


MAX_EPISODES = 1000000 #10000000
DISCOUNT = 0.9
LEARNING_RATE = 0.2
INITIAL_EPSILON = 0.5
DROP_RATE = 0.5
EPISODES_PER_DROP = 500000


class DiscreteCartpole:
    def __init__(self):
        self.env = gym.make("CartPole-v0")
        self.action_space = self.env.action_space

    def _quantize(self, state):
        return tuple([Decimal(el.item()).quantize(Decimal(".01")) for el in state])

    def reset(self):
        return self._quantize(self.env.reset()[0])

    def step(self, action):
        state, reward, done,_, info = self.env.step(action)
        return self._quantize(state), reward, done, info


class QTable:
    def __init__(self):
        self.storage = defaultdict(int)
        self.reset_stats()

    @property
    def hit_rate(self):
        if self.access_count == 0:
            return 0
        return self.hit_count / self.access_count

    @property
    def nonzero_hit_rate(self):
        if self.access_count == 0:
            return 0
        return self.nonzero_hit_count / self.access_count

    @property
    def size_in_mb(self):
        return sys.getsizeof(self.storage) / 2 ** 20

    def reset_stats(self):
        self.access_count = 0
        self.hit_count = 0
        self.nonzero_hit_count = 0

    def get(self, state, action):
        self.access_count += 1
        if (state, action) in self.storage:
            self.hit_count += 1
        q_value = self.storage[(state, action)]
        if q_value != 0:
            self.nonzero_hit_count += 1
        return q_value

    def set(self, state, action, value):
        self.storage[(state, action)] = value

    def update(self, state, action, next_state, reward):
        curr_q = self.get(state, action)
        max_next_q = max([self.get(next_state, a) for a in [0, 1]])
        discounted_next_q = DISCOUNT * max_next_q
        q_error = LEARNING_RATE * (reward + discounted_next_q - curr_q)
        updated_curr_q = curr_q + q_error
        self.set(state, action, updated_curr_q)


def train():
    env = DiscreteCartpole()
    q_values = QTable()
    is_training = True
    for episode in range(MAX_EPISODES):
        run_episode(episode, env, q_values, is_training)
        evaluate_policy(episode, env, q_values)


def run_episode(episode, env, q_values, is_training):
    state= env.reset()
    done = False
    total_reward = 0
    while not done and total_reward < 200:
        action = get_next_action(episode, q_values, state, is_training)
        next_state, reward, done, info = env.step(action)
        if is_training:
            q_values.update(state, action, next_state, reward)
        total_reward += reward
        state = next_state
    return total_reward


def get_next_action(episode, q_values, state, is_training):
    if random.random() < get_effective_epsilon(episode) and is_training:
        return random.choice([0, 1])
    else:
        return 0 if q_values.get(state, 0) > q_values.get(state, 1) else 1


def evaluate_policy(episode, env, q_values):
    if episode % 1000 != 0:
        return 0,0
    is_training = False
    q_values.reset_stats()
    rewards = [run_episode(i, env, q_values, is_training) for i in range(10)]
    max_reward = max(rewards)
    mean_reward = statistics.mean(rewards)
    median_reward = statistics.median(rewards)
    min_reward = min(rewards)
    print()
    print(f"Iteration {episode}")
    print(f"\tMax steps per run: {max_reward}")
    print(f"\tMedian steps per run: {median_reward}")
    print(f"\tMean steps per run: {mean_reward}")
    print(f"\tMin steps per run: {min_reward}")
    print(f"\tQ Hit Rate: {q_values.hit_rate:.4f}")
    print(f"\tNonzero Q Hit Rate: {q_values.nonzero_hit_rate:.4f}")
    print(f"\tQ table size: {q_values.size_in_mb:.2f} MB")
    print()
    
    if mean_reward is None: 
        mean_reward = 0
    if median_reward is None:
        median_reward = 0
    return (mean_reward, median_reward)

# https://towardsdatascience.com/learning-rate-schedules-and-adaptive-learning-rate-methods-for-deep-learning-2c8f433990d1
def get_effective_epsilon(episode):
    return INITIAL_EPSILON * (DROP_RATE ** (episode / EPISODES_PER_DROP))

if __name__ == "__main__":
    train()