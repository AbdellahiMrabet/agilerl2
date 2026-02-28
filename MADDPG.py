from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gymnasium as gym
from collections import deque
import random

class Agent():
    def __init__(self, state_dim, action_dim, num_agents, hidden_dim=64):
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.num_agents = num_agents
        self.actor = Actor(state_dim, action_dim, hidden_dim)
        self.critic = Critic(state_dim, action_dim, num_agents, hidden_dim)
        self.target_actor = Actor(state_dim, action_dim, hidden_dim)
        self.target_critic = Critic(state_dim, action_dim, num_agents, hidden_dim)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=0.001)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=0.001)
        self.update_target_networks(tau=1.0)
        self.ou_noise = OUNoise(self.action_dim).noise()
        self.replay = ReplayBuffer(capacity=100000)
    
    def update_target_networks(self, tau=0.01):
        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
            
    def train(self, batch_size=64, train_starting=10000, gamma=0.95):
        if self.replay.__len__() < train_starting:
            return
        states, actions, rewards, next_states, dones = self.replay.sample(batch_size)
        states = list(states)
        actions = list(actions)
        rewards = list(rewards)
        next_states = list(next_states)
        dones = list(dones)
        for i in range(batch_size):
            states[i] = np.array([states[i]['agent_'+str(j)] for j in range(self.num_agents)])
            actions[i] = np.array([actions[i]['agent_'+str(j)] for j in range(self.num_agents)])
            rewards[i] = [rewards[i]['agent_'+str(j)] for j in range(self.num_agents)]
            next_states[i] = np.array([next_states[i]['agent_'+str(j)] for j in range(self.num_agents)])
            dones[i] = [dones[i]['agent_'+str(j)] for j in range(self.num_agents)]
            
            # Update Critic
                
            target_actions = self.target_actor(torch.FloatTensor(next_states[i]).unsqueeze(0))
            target_q_value = self.target_critic(
                    torch.FloatTensor(next_states[i]).flatten(start_dim=0).unsqueeze(0),
                                         torch.FloatTensor(target_actions).flatten(start_dim=0).unsqueeze(0))
                
            target = torch.FloatTensor(np.array([sum(rewards[i])])).unsqueeze(0)\
                    + (1 - torch.FloatTensor(np.array([all(dones[i])])).unsqueeze(0))\
                        * gamma * target_q_value
            
            current_q_value = self.critic(
                torch.FloatTensor(states[i]).flatten(start_dim=0).unsqueeze(0),
                                         torch.FloatTensor(actions[i]).flatten(start_dim=0).unsqueeze(0))
            #print('target:', target, 'current_q_value:', current_q_value)
            critic_loss = nn.MSELoss()(current_q_value, target.detach())
            
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()
            
            # Update Actor
            actor_loss = -self.critic(
                torch.FloatTensor(states[i]).flatten(start_dim=0).unsqueeze(0),
                                     torch.FloatTensor(actions[i]).flatten(start_dim=0).unsqueeze(0)).mean()
            
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
        
        # Update target networks
        self.update_target_networks()   
              
    def get_action(self, states, noise=0.0):
        discret = {}
        continuos = {}
        for i in range(self.num_agents):
            state = torch.FloatTensor(states['agent_'+str(i)]).unsqueeze(0)
            action = self.actor(state).detach().numpy()[0]
            action += noise
            discret['agent_'+str(i)] = np.array([np.array(action.argmax(), dtype=int)])
            continuos['agent_'+str(i)] = action
        
        return discret, continuos
    def save_checkpoint(self, filepath, episode=None, additional_info=None):
        """
        Save a checkpoint for the MADDPG model.
        """
        checkpoint = {
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'episode': episode,
            'additional_info': additional_info
        }
        torch.save(checkpoint, filepath)
        print(f"Checkpoint saved at {filepath}")
        
    def load(self, filepath):
        """
        Load a checkpoint for the MADDPG model.
        """
        checkpoint = torch.load(filepath)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        episode = checkpoint['episode']
        additional_info = checkpoint.get('additional_info', None)
        print(f"Checkpoint loaded from {filepath}, resuming from episode {episode}")
        #return checkpoint
    
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        action = torch.tanh(self.fc3(x))
        return action


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, num_agents, hidden_dim):
        super(Critic, self).__init__()
        input_dim = state_dim * num_agents + action_dim * num_agents
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, states, actions):
        x = torch.cat([states.flatten(start_dim=1), actions.flatten(start_dim=1)], dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        q_value = self.fc3(x)
        return q_value
    
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)
    
class OUNoise:
    def __init__(self, action_dim, mu=0.0, theta=0.15, sigma=0.2):
        self.action_dim = action_dim
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.action_dim) * self.mu

    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu

    def noise(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state

