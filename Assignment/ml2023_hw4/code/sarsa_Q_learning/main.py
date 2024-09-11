import gym
import matplotlib.pyplot as plt
from algorithms import QLearning, Sarsa
from utils import render_single_Q, evaluate_Q


# Feel free to run your own debug code in main!
def main():
    num_episodes = 5000
    lr = 0.1
    env = gym.make('Taxi-v3')
    Q_ql, episode_reward_ql = QLearning(env, num_episodes=num_episodes, lr=lr)
    Q_sarsa, episode_reward_sarsa = Sarsa(env, num_episodes=num_episodes, lr=lr)
    plt.plot(episode_reward_sarsa, label='Sarsa')
    plt.plot(episode_reward_ql, label='Q-learning')
    plt.xlabel('Episode')
    plt.ylabel('Episode Reward')
    plt.legend()
    plt.show()
    evaluate_Q(env, Q_ql)
    evaluate_Q(env, Q_sarsa)

if __name__ == '__main__':
    main()
