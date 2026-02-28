import csv
import math
import os
from typing import OrderedDict

import imageio
import numpy as np
import torch
import new_env as simple_speaker_listener_v4

from agilerl.algorithms import MATD3
import matplotlib.pyplot as plt


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Configure the environment
    env = simple_speaker_listener_v4.parallel_env(render_mode="rgb_array")
    env.reset()

    # Append number of agents and agent IDs to the initial hyperparameter dictionary
    n_agents = env.num_agents
    agent_ids = env.agents

    # Load the saved agent
    path = "./models/MATD3"

    # Define test loop parameters
    episodes = 50  # Number of episodes to test agent on
    max_steps = 25  # Max number of steps to take in the environment in each episode

    rewards = []  # List to collect total episodic reward
    exection_time = []
    pay = []
    energy = []
    indi_agent_rewards = {
        agent_id: [] for agent_id in agent_ids
    }  # Dictionary to collect inidivdual agent rewards
    for ep in range(episodes):
        obs, info = env.reset()
        agent_reward = {agent_id: 0 for agent_id in agent_ids}
        extm = {agent_id: 0 for agent_id in agent_ids}
        enrg = {agent_id: 0 for agent_id in agent_ids}
        pymnt = {agent_id: 0 for agent_id in agent_ids}
        score = 0
        score1 = 0
        score2 = 0
        score3 = 0
        action = OrderedDict()
        for agent_id in agent_ids:
            action[agent_id] = np.array([np.random.randint(3)])
        #action, _ = matd3.get_action(obs, infos=info)
        for _ in range(max_steps):
            # Get next action from agent
            #for ag in obs:
            #caclutate the cost of action and action2
            
            print('obs', obs, 'action', action)
            for agent_id in agent_ids:
                extm[agent_id] += env.exectime(action, agent_id)
                    
                enrg[agent_id] += env.energy(action, agent_id)
                    
                pymnt[agent_id] += env.pay(action, agent_id)
            
            action2 = OrderedDict()
            for agent_id in agent_ids:
                action2[agent_id] = np.array([np.random.randint(3)])
            #calculate the cost of action and action2
            cost = {agent_id: env.utility(action, agent_id) for agent_id in agent_ids}
            
            cost2 = {agent_id: env.utility(action2, agent_id) for agent_id in agent_ids}
            
            for agent_id in agent_ids:
                if cost2[agent_id] < cost[agent_id]:
                    action[agent_id] = action2[agent_id]
            # Take action in environment
            obs, reward, termination, truncation, info = env.step(
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
        rewards.append(score)
        exection_time.append(score1)
        energy.append(score2)
        pay.append(score3)
        # Record agent specific episodic reward
        for agent_id in agent_ids:
            indi_agent_rewards[agent_id].append(agent_reward[agent_id])

        print("-" * 15, f"Episode: {ep}", "-" * 15)
        print("Episodic Reward: ", rewards[-1])
        for agent_id, reward_list in indi_agent_rewards.items():
            print(f"{agent_id} reward: {reward_list[-1]}")
    env.close()

    # Save the gif to specified path
    gif_path = "./videos/"
    os.makedirs(gif_path, exist_ok=True)
    print('obs', obs['agent_3'])
    with open(os.path.join(path, 'evaldto_score_N_9.csv'), 'w', newline='') as file:
                writer = csv.writer(file)
                for i in range(len(rewards)):
                    writer.writerow([i, rewards[i], exection_time[i], energy[i], pay[i]])
    with open(os.path.join(path, 'evaldto_fog_N_9.csv'), 'w', newline='') as file:
                writer = csv.writer(file)
                for i in range(len(obs['agent_3'])):
                    writer.writerow([i, obs['agent_3'][i]])
    # plt.figure()
    # plt.plot(range(1, len(DI)+1), DI, marker='o')
    # plt.xlabel('Episode')
    # plt.ylabel('DI')
    # #plt.grid()
    # plt.xticks(range(1, len(DI)+1))
    # plt.savefig(os.path.join(gif_path, 'degree_of_imbalance.png'))
    # plt.show()  
    #plot individual agent rewards over episodes
    plt.figure()
    for agent_id, reward_list in indi_agent_rewards.items():
        plt.plot(range(1, episodes+1), reward_list, marker='o', label=agent_id)
    plt.title('Individual Agent Rewards over Episodes')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    #plt.grid()
    #plt.xticks(range(1, episodes+1))
    plt.legend()
    plt.savefig(os.path.join(gif_path, 'individual_agent_rewards.png'))
    plt.show()
    #plot rewards over episodes
    plt.figure()
    plt.plot(range(1, episodes+1), rewards, marker='o')
    plt.title('Total Episodic Rewards over Episodes')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    #plt.grid()
    plt.xticks(range(1, episodes+1))
    plt.savefig(os.path.join(gif_path, 'totaldto_episodic_rewards_n9.png'))
    plt.show()
    plt.figure()
    plt.plot(range(1, episodes+1), exection_time, marker='o')
    #plt.title('Total Episodic exection_time over Episodes')
    plt.xlabel('Episode')
    plt.ylabel('Total Exection Time')
    #plt.grid()
    #plt.xticks(range(1, episodes+1))
    plt.savefig(os.path.join(gif_path, 'totaldto_exection_time_n9.png'))
    plt.show()
    plt.figure()
    plt.plot(range(1, episodes+1), energy, marker='o')
    #plt.title('Total Episodic energy over Episodes')
    plt.xlabel('Episode')
    plt.ylabel('Total Energy Consumption')
    #plt.grid()
    #plt.xticks(range(1, episodes+1))
    plt.savefig(os.path.join(gif_path, 'totaldto_energy_n9.png'))
    plt.show()
    plt.figure()
    plt.plot(range(1, episodes+1), pay, marker='o')
    #plt.title('Total Episodic payment cost over Episodes')
    plt.xlabel('Episode')
    plt.ylabel('Total Payment Cost')
    #plt.grid()
    #plt.xticks(range(1, episodes+1))
    plt.savefig(os.path.join(gif_path, 'totaldto_cost_n9.png'))
    plt.show() 