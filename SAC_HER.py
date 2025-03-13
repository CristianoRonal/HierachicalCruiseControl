import math
import random

import gym
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal

from Environment import CruiseControlEnvironment

from IPython.display import clear_output
import matplotlib.pyplot as plt
from matplotlib import animation
from IPython.display import display
# from reacher import Reacher

import argparse
import time

# 设备配置
GPU = True
device_idx = 0
if GPU and torch.cuda.is_available():
    device = torch.device("cuda:" + str(device_idx))
else:
    device = torch.device("cpu")
print("Using device:", device)

# 参数解析
parser = argparse.ArgumentParser(description='Train or test SAC-HER controller.')
parser.add_argument('--train', dest='train', action='store_true', default=False)
parser.add_argument('--test', dest='test', action='store_true', default=False)
args = parser.parse_args()

# Hindsight Replay Buffer
class HindsightReplayBuffer:
    def __init__(self, capacity, state_dim, action_dim, goal_dim, reward_func, env):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.goal_dim = goal_dim
        self.reward_func = reward_func
        self.env = env

    def push(self, state, action, reward, next_state, done, goal, episode_id):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done, goal, episode_id)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size, strategy="final"):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones, goals = [], [], [], [], [], []
        her_indices = random.sample(range(batch_size), batch_size // 2)

        for i, (s, a, r, s_next, d, g, episode_id) in enumerate(batch):
            states.append(s)
            actions.append(a)
            rewards.append(r)
            next_states.append(s_next)
            dones.append(d)
            goals.append(g)

            if i in her_indices and strategy == "final":
                sT = self.get_episode_final_state(episode_id)  # 获取 episode 的最终状态
                if sT is not None:  # 确保找到有效状态
                    hindsight_goal = np.array([sT[2], sT[0]])  # [x_ego, v_ego] from sT
                    hindsight_reward = self.reward_func(s_next, hindsight_goal, self.env)
                    states.append(s)
                    actions.append(a)
                    rewards.append(hindsight_reward)
                    next_states.append(s_next)
                    dones.append(d)
                    goals.append(hindsight_goal)

        return (np.array(states), np.array(actions), np.array(rewards),
                np.array(next_states), np.array(dones), np.array(goals))

    def get_episode_final_state(self, episode_id):
        """
        找到给定 episode_id 的最终状态 sT, 即 done=True 的状态。
        """
        for experience in reversed(self.buffer):
            s, a, r, s_next, d, g, exp_id = experience
            if exp_id == episode_id and d:  # 匹配 episode_id 且 done=True
                return s_next
        # 备选：返回最后一个匹配的状态
        for experience in reversed(self.buffer):
            s, a, r, s_next, d, g, exp_id = experience
            if exp_id == episode_id:
                return s_next
        return None

    def __len__(self):
        return len(self.buffer)

# 奖励函数
def reward_func(state, goal, env):
    v_ego, a_ego, x_ego, delta_v, delta_d = state
    x_des, v_des = goal
    # 定义一个位置窗口，基于 x_des 的动态区间
    window_size = 50  # 假设目标位置附近 50m 为有效区间
    x_min, x_max = x_des - window_size, x_des + window_size
    # headway = delta_d/v_ego
    if x_ego < 1000:
        v_lim = 120/3.6
    else:
        v_lim = v_des

    R_speed = v_lim - max(0, v_lim - v_ego)

    if x_min <= x_ego <= x_max:
        delta_v_goal = v_des - v_ego
        if delta_v_goal < 0:  # 超速惩罚
            R_des = -200 * delta_v_goal**2 
        else:  # 未达速但鼓励接近 v_des 和 x_des
            R_des = -10 * delta_v_goal**2 
    else:  # 未到达目标区间
        R_des = -abs(x_des - x_ego) * 0.1 + 10 if abs(x_des - x_ego) < 100 else -abs(x_des - x_ego) * 0.1

    R_headway, R_comfort = env.gain_Reward()

    return 0.1 * R_speed + R_headway + R_comfort + 0.5 * R_des

# SAC 网络
class SoftQNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size, init_w=3e-3):
        super(SoftQNetwork, self).__init__()
        self.linear1 = nn.Linear(num_inputs + num_actions, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, hidden_size)
        self.linear4 = nn.Linear(hidden_size, 1)
        self.linear4.weight.data.uniform_(-init_w, init_w)
        self.linear4.bias.data.uniform_(-init_w, init_w)

    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = self.linear4(x)
        return x

class PolicyNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size, action_range=1., init_w=3e-3, log_std_min=-20, log_std_max=2):
        super(PolicyNetwork, self).__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.linear1 = nn.Linear(num_inputs, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, hidden_size)
        self.linear4 = nn.Linear(hidden_size, hidden_size)
        self.mean_linear = nn.Linear(hidden_size, num_actions)
        self.log_std_linear = nn.Linear(hidden_size, num_actions)
        self.mean_linear.weight.data.uniform_(-init_w, init_w)
        self.mean_linear.bias.data.uniform_(-init_w, init_w)
        self.log_std_linear.weight.data.uniform_(-init_w, init_w)
        self.log_std_linear.bias.data.uniform_(-init_w, init_w)
        self.action_range = action_range
        self.num_actions = num_actions

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = F.relu(self.linear4(x))
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mean, log_std

    def evaluate(self, state, epsilon=1e-6):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(0, 1)
        z = normal.sample(mean.shape).to(device)
        action_0 = torch.tanh(mean + std * z)
        action = self.action_range * action_0
        log_prob = Normal(mean, std).log_prob(mean + std * z) - torch.log(1. - action_0.pow(2) + epsilon) - np.log(self.action_range)
        log_prob = log_prob.sum(dim=1, keepdim=True)
        return action, log_prob, z, mean, log_std

    def get_action(self, state, deterministic=False):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        mean, log_std = self.forward(state)
        if deterministic:
            action = self.action_range * torch.tanh(mean)
            return action.detach().cpu().numpy()[0]
        std = log_std.exp()
        normal = Normal(0, 1)
        z = normal.sample(mean.shape).to(device)
        action = self.action_range * torch.tanh(mean + std * z)
        return action.detach().cpu().numpy()[0]

    def sample_action(self):
        a = torch.FloatTensor(self.num_actions).uniform_(-1, 1)
        return self.action_range * a.numpy()

# SAC 训练器
class SAC_Trainer:
    def __init__(self, replay_buffer, hidden_dim, action_range):
        self.replay_buffer = replay_buffer
        self.soft_q_net1 = SoftQNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.soft_q_net2 = SoftQNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.target_soft_q_net1 = SoftQNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.target_soft_q_net2 = SoftQNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.policy_net = PolicyNetwork(state_dim, action_dim, hidden_dim, action_range).to(device)
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)

        print('Soft Q Network (1,2): ', self.soft_q_net1)
        print('Policy Network: ', self.policy_net)

        for target_param, param in zip(self.target_soft_q_net1.parameters(), self.soft_q_net1.parameters()):
            target_param.data.copy_(param.data)
        for target_param, param in zip(self.target_soft_q_net2.parameters(), self.soft_q_net2.parameters()):
            target_param.data.copy_(param.data)

        self.soft_q_criterion1 = nn.MSELoss()
        self.soft_q_criterion2 = nn.MSELoss()
        soft_q_lr = 3e-4
        policy_lr = 3e-4
        alpha_lr = 3e-4
        self.soft_q_optimizer1 = optim.Adam(self.soft_q_net1.parameters(), lr=soft_q_lr)
        self.soft_q_optimizer2 = optim.Adam(self.soft_q_net2.parameters(), lr=soft_q_lr)
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=policy_lr)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=alpha_lr)

    def update(self, batch_size, reward_scale=10., auto_entropy=True, target_entropy=-2, gamma=0.99, soft_tau=1e-2):
        states, actions, rewards, next_states, dones, goals = self.replay_buffer.sample(batch_size)
        states = torch.FloatTensor(states).to(device)
        next_states = torch.FloatTensor(next_states).to(device)
        actions = torch.FloatTensor(actions).to(device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(device)

        predicted_q_value1 = self.soft_q_net1(states, actions)
        predicted_q_value2 = self.soft_q_net2(states, actions)
        new_action, log_prob, _, _, _ = self.policy_net.evaluate(states)
        new_next_action, next_log_prob, _, _, _ = self.policy_net.evaluate(next_states)
        rewards = reward_scale * (rewards - rewards.mean()) / (rewards.std() + 1e-6)

        if auto_entropy:
            alpha_loss = -(self.log_alpha * (log_prob + target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha.exp()
        else:
            self.alpha = 1.

        target_q_min = torch.min(self.target_soft_q_net1(next_states, new_next_action),
                                 self.target_soft_q_net2(next_states, new_next_action)) - self.alpha * next_log_prob
        target_q_value = rewards + (1 - dones) * gamma * target_q_min

        q_value_loss1 = self.soft_q_criterion1(predicted_q_value1, target_q_value.detach())
        q_value_loss2 = self.soft_q_criterion2(predicted_q_value2, target_q_value.detach())
        self.soft_q_optimizer1.zero_grad()
        q_value_loss1.backward()
        self.soft_q_optimizer1.step()
        self.soft_q_optimizer2.zero_grad()
        q_value_loss2.backward()
        self.soft_q_optimizer2.step()

        predicted_new_q_value = torch.min(self.soft_q_net1(states, new_action), self.soft_q_net2(states, new_action))
        policy_loss = (self.alpha * log_prob - predicted_new_q_value).mean()
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        for target_param, param in zip(self.target_soft_q_net1.parameters(), self.soft_q_net1.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - soft_tau) + param.data * soft_tau)
        for target_param, param in zip(self.target_soft_q_net2.parameters(), self.soft_q_net2.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - soft_tau) + param.data * soft_tau)

        return predicted_new_q_value.mean()

    def save_model(self, path):
        torch.save(self.soft_q_net1.state_dict(), path + '_q1')
        torch.save(self.soft_q_net2.state_dict(), path + '_q2')
        torch.save(self.policy_net.state_dict(), path + '_policy')

    def load_model(self, path):
        self.soft_q_net1.load_state_dict(torch.load(path + '_q1'))
        self.soft_q_net2.load_state_dict(torch.load(path + '_q2'))
        self.policy_net.load_state_dict(torch.load(path + '_policy'))
        self.soft_q_net1.eval()
        self.soft_q_net2.eval()
        self.policy_net.eval()

def plot(rewards):
    clear_output(True)
    plt.figure(figsize=(20, 5))
    plt.plot(rewards)
    plt.savefig('sac_her.png')
    # plt.show()

def plot_test(ego_v, ego_dp, ego_x, i_episode):
    ego_v = np.array(ego_v) * 3.6    # km/h
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    ax1.plot(ego_x,ego_v)
    ax1.set_xlim(0,1100)
    ax1.set_ylim(0,130)
    ax1.set_title('velocity_x')
    ax1.set_xlabel('x')
    ax1.set_ylabel('velocity')

    ax2.plot(ego_x, ego_dp)
    ax2.set_xlim(0,1100)
    ax2.set_title('deltadistance_x')
    ax2.set_xlabel('x')
    ax2.set_ylabel('delta distance')

    plt.tight_layout()
    plt.savefig(f'./figure/sac_her_eps{i_episode}')
    plt.close(fig)


# 主程序
if __name__ == "__main__":
    env = CruiseControlEnvironment()
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    goal_dim = 2
    action_range = 1.

    replay_buffer_size = int(1e6)
    replay_buffer = HindsightReplayBuffer(
        capacity=replay_buffer_size,
        state_dim=state_dim,
        action_dim=action_dim,
        goal_dim=goal_dim,
        reward_func=reward_func,
        env = env
    )

    max_episodes = 500
    max_steps = env.max_steps
    batch_size = 300
    explore_steps = 60000
    update_itr = 1
    AUTO_ENTROPY = True
    DETERMINISTIC = False
    hidden_dim = 512
    model_path = './model/sac_her'
    frame_idx = 0

    sac_trainer = SAC_Trainer(replay_buffer, hidden_dim=hidden_dim, action_range=action_range)

    if args.train:
        rewards = []
        for eps in range(max_episodes):
            state = env.reset()
            episode_reward = 0
            v_des = env.gain_vdes()
            goal = np.array([1100, v_des])
            episode_id = eps

            for step in range(max_steps):
                if frame_idx > explore_steps:
                    action = sac_trainer.policy_net.get_action(state, deterministic=DETERMINISTIC)
                else:
                    action = sac_trainer.policy_net.sample_action()

                next_state, reward, done, _ = env.step(action)
                replay_buffer.push(state, action, reward, next_state, done, goal, episode_id)

                state = next_state
                episode_reward += reward
                frame_idx += 1

                if len(replay_buffer) > batch_size:
                    for i in range(update_itr):
                        _ = sac_trainer.update(batch_size, reward_scale=10., auto_entropy=AUTO_ENTROPY,
                                               target_entropy=-1 * action_dim)
                if done or step == max_steps - 1:
                    env.stop()
                    break
            if eps % 20 == 0 and eps > 0:  # plot and model saving interval
                plot(rewards)
                np.save('rewards', rewards)
                sac_trainer.save_model(model_path)

            print('Episode: ', eps, '| Episode Reward: ', episode_reward, '| Total Steps: ', frame_idx)
            rewards.append(episode_reward)

        sac_trainer.save_model(model_path)

    if args.test:
        sac_trainer.load_model(model_path)
        for eps in range(10):
            ego_v=[]
            ego_dp=[]
            ego_x=[]
            state = env.reset()
            episode_reward = 0
            v_des = env.gain_vdes()


            for step in range(max_steps):
                action = sac_trainer.policy_net.get_action(state, deterministic=True)
                next_state, reward, done, _ = env.step(action)
                state = next_state
                episode_reward += reward
                ego_v.append(state[0])
                ego_dp.append(state[4])
                ego_x.append(state[2])

                if step == (max_steps - 1) or done:
                    break

            plot_test(ego_v, ego_dp, ego_x, eps)
            print('Episode: ', eps, '| Episode Reward: ', episode_reward, '| Desired Velocity: ', v_des*3.6)
