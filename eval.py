import csv
import os
from typing import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import new_env_fog3 as simple_speaker_listener_v4

from agilerl.algorithms import MATD3

class eval:
    def __init__(self, n_fogs = 3, n_agents = 4, algo = None):
        self.n_agents = n_agents
        self.n_fogs = n_fogs
        self.algo = algo
        extension = ".pt"
        self.path = "./models/"+self.algo+"/"+self.algo+"_trained_agent_N_"+str(self.n_agents)
        if self.n_fogs == 5:
            import new_env_fog5 as simple_speaker_listener_v4
            self.path += "_fog5"
        if self.n_agents == 4:
            self.env = simple_speaker_listener_v4.parallel_env_4(render_mode="rgb_array")
        elif self.n_agents == 9:
            self.env = simple_speaker_listener_v4.parallel_env(render_mode="rgb_array")  
        elif self.n_agents == 14:
            self.env = simple_speaker_listener_v4.parallel_env_14(render_mode="rgb_array")
        self.env.reset()
        if self.algo == 'MATD3':
            self.algo_value = MATD3
            self.path += extension
        elif self.algo == 'MADDPG':
            from MADDPG import Agent
            STATE_DIM = self.env.observation_space('agent_0').shape[0]
            ACTION_DIM = self.env.action_space('agent_0').n
            maddpg = Agent(STATE_DIM, ACTION_DIM, 1 + self.n_agents)
            self.algo_value = maddpg
            self.path = "./models/"+self.algo+"/"+self.algo+"_trained_agent_N_"+str(self.n_agents)+".pth"
        if self.algo is not None:
            if self.algo == 'MADDPG':
                self.loaded_algo = self.algo_value
                self.loaded_algo.load(self.path)
            else:
                self.loaded_algo = self.algo_value.load(self.path,
                                torch.device("cuda" if torch.cuda.is_available() else "cpu"))
            
        self.csv_path = "./csv/"
        
        self.agent_ids = self.env.agents
        self.episodes = 50  # Number of episodes to test agent on
        self.max_steps = 25  # Max number of steps to take in the environment in each episode   
        self.rewards = []  # List to collect total episodic reward
        self.exection_time = []
        self.pay = []
        self.energy = []
        self.indi_agent_rewards = {
            agent_id: [] for agent_id in self.agent_ids
        }  # Dictionary to collect
    
    #save results as csv files
    def save(self):
        os.makedirs(self.csv_path, exist_ok=True)
        rewards, exection_time, energy, pay, _, obs = self.evaluate()
        with open(os.path.join(self.csv_path, self.scrore_filename), 'w', newline='') as file:
                writer = csv.writer(file)
                for i in range(len(rewards)):
                    writer.writerow([i, rewards[i], exection_time[i], energy[i], pay[i]])
                    
        file.close()
        with open(os.path.join(self.csv_path, self.fog_filename), 'w', newline='') as file:
                writer = csv.writer(file)
                print('obs[agent_3]', obs['agent_3'])
                for i in range(len(obs['agent_3'])):
                    writer.writerow([i, obs['agent_3'][i]])
        file.close()
        
class evaluate_agent(eval):
    def __init__(self, n_fogs = 3, n_agents = 4, algo = 'MATD3'):
        super().__init__(n_fogs, n_agents, algo)
        self.scrore_filename = 'eval'+algo+'_score_N'+str(n_agents)+'.csv'
        self.fog_filename = 'eval'+algo+'_fog_N_'+str(n_agents)+'.csv'
        
    def evaluate(self):
        for ep in range(self.episodes):
            obs, info = self.env.reset()
            agent_reward = {agent_id: 0 for agent_id in self.agent_ids}
            extm = {agent_id: 0 for agent_id in self.agent_ids}
            enrg = {agent_id: 0 for agent_id in self.agent_ids}
            pymnt = {agent_id: 0 for agent_id in self.agent_ids}
            score = 0
            score1 = 0
            score2 = 0
            score3 = 0
            for _ in range(self.max_steps):
                # Get next action from agent
                #for ag in obs:
                if self.algo == 'MADDPG':
                    action, _ = self.loaded_algo.get_action(obs, noise=self.loaded_algo.ou_noise)
                else:
                    action, _ = self.loaded_algo.get_action(obs, infos=info)               
                
                print('obs', obs, 'action', action)
                for agent_id in self.agent_ids:
                    extm[agent_id] += self.env.exectime(action, agent_id)
                        
                    enrg[agent_id] += self.env.energy(action, agent_id)
                        
                    pymnt[agent_id] += self.env.pay(action, agent_id)

                # Take action in environment
                obs, reward, termination, truncation, info = self.env.step(
                    {agent: a.squeeze() for agent, a in action.items()}
                )
                # Save agent's reward for this step in this episode
                for agent_id, r in reward.items():
                    agent_reward[agent_id] += r
                    
                # Determine total score for the episode and then append to rewards list
                score = sum(agent_reward.values())
                score1 = sum(extm.values())
                score2 = sum(enrg.values())
                score3 = sum(pymnt.values())

                # Stop episode if any agents have terminated
                # Stop episode if any agents have terminated
                if any(truncation.values()) or any(termination.values()):
                    break
            self.rewards.append(score)
            self.exection_time.append(score1)
            self.energy.append(score2)
            self.pay.append(score3)
            
            # Record agent specific episodic reward
            for agent_id in self.agent_ids:
                self.indi_agent_rewards[agent_id].append(agent_reward[agent_id])

            print("-" * 15, f"Episode: {ep}", "-" * 15)
            print("Episodic Reward: ", self.rewards[-1])
            for agent_id, reward_list in self.indi_agent_rewards.items():
                print(f"{agent_id} reward: {reward_list[-1]}")
        self.env.close()
        return self.rewards, self.exection_time, self.energy, self.pay, _, obs
        
class evaluate_dto(eval):
    def __init__(self, n_agents = 4):
        super().__init__(n_agents)
        self.scrore_filename = 'evaldto_score_N'+str(n_agents)+'.csv'
        self.fog_filename = 'evaldto_fog_N_'+str(n_agents)+'.csv'
        
    def evaluate(self):
        for ep in range(self.episodes):
            obs, info = self.env.reset()
            agent_reward = {agent_id: 0 for agent_id in self.agent_ids}
            extm = {agent_id: 0 for agent_id in self.agent_ids}
            enrg = {agent_id: 0 for agent_id in self.agent_ids}
            pymnt = {agent_id: 0 for agent_id in self.agent_ids}
            score = 0
            score1 = 0
            score2 = 0
            score3 = 0
            action = OrderedDict()
            for agent_id in self.agent_ids:
                action[agent_id] = np.array([np.random.randint(3)])
            for _ in range(self.max_steps):
                # Get next action from agent
                #for ag in obs:
                print('obs', obs, 'action', action)
                for agent_id in self.agent_ids:
                    extm[agent_id] += self.env.exectime(action, agent_id)
                        
                    enrg[agent_id] += self.env.energy(action, agent_id)
                        
                    pymnt[agent_id] += self.env.pay(action, agent_id)

                action2 = OrderedDict()
                for agent_id in self.agent_ids:
                    action2[agent_id] = np.array([np.random.randint(3)])
                #calculate the cost of action and action2
                cost = {agent_id: self.env.utility(action, agent_id) for agent_id in self.agent_ids}
                
                cost2 = {agent_id: self.env.utility(action2, agent_id) for agent_id in self.agent_ids}
                
                for agent_id in self.agent_ids:
                    if cost2[agent_id] < cost[agent_id]:
                        action[agent_id] = action2[agent_id]
                # Take action in environment
                obs, reward, termination, truncation, info = self.env.step(
                        {agent: a.squeeze() for agent, a in action.items()}
                    )
                # Save agent's reward for this step in this episode
                for agent_id, r in reward.items():
                    agent_reward[agent_id] += r
                    
                # Determine total score for the episode and then append to rewards list
                score = sum(agent_reward.values())
                score1 = sum(extm.values())
                score2 = sum(enrg.values())
                score3 = sum(pymnt.values())

                # Stop episode if any agents have terminated
                # Stop episode if any agents have terminated
                if any(truncation.values()) or any(termination.values()):
                    break
            self.rewards.append(score)
            self.exection_time.append(score1)
            self.energy.append(score2)
            self.pay.append(score3)
            
            # Record agent specific episodic reward
            for agent_id in self.agent_ids:
                self.indi_agent_rewards[agent_id].append(agent_reward[agent_id])

            print("-" * 15, f"Episode: {ep}", "-" * 15)
            print("Episodic Reward: ", self.rewards[-1])
            for agent_id, reward_list in self.indi_agent_rewards.items():
                print(f"{agent_id} reward: {reward_list[-1]}")
        self.env.close()
        return self.rewards, self.exection_time, self.energy, self.pay, _, obs