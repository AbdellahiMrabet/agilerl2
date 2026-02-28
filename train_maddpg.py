import os
from matplotlib import pyplot as plt
from pyparsing import deque
import new_env_fog5 as simple_speaker_listener_v4
from MADDPG import Agent
import numpy as np
import torch
import torch.nn as nn
from agilerl.utils.utils import default_progress_bar

# Hyperparameters
REPLAY_BUFFER_SIZE = 100000
REPLAY_START_SIZE = 10000
BATCH_SIZE = 64
GAMMA = 0.95
if __name__ == "__main__":
    # Initialize the environment
    env = simple_speaker_listener_v4.parallel_env()
    env.reset()
    NUM_AGENTS = len(env.agents)
    agent_ids = env.agents
    STATE_DIM = env.observation_space('agent_0').shape[0]
    ACTION_DIM = env.action_space('agent_0').n
    path = "./models/MADDPG/"

    #initialize agent
    maddpg = Agent(STATE_DIM, ACTION_DIM, NUM_AGENTS)
    maddpg.replay.buffer = deque(maxlen=REPLAY_BUFFER_SIZE)
    # Training loop
    global_rewards = []
    episodes = []
    EPISODES = 2_900_000
    UPDATE_RATE = 50
    pbar = default_progress_bar(EPISODES)
    for episode in range(EPISODES):
        states, _ = env.reset()
        episode_rewards = [0 for _ in range(maddpg.num_agents)]
        for step in range(200):
            actions, c_actions = maddpg.get_action(states, maddpg.ou_noise)
           
            next_states, rewards, dones, _, _ = env.step(
                {agent: a.squeeze() for agent, a in actions.items()}
            )
            
            maddpg.replay.add(states, c_actions, rewards, next_states, dones)
            
            states = next_states
            for i in range(maddpg.num_agents):
                episode_rewards[i] += rewards['agent_'+str(i)]

            if all(dones):
                break
        
        #update pbar
        if episode % UPDATE_RATE == 0 and episode > 0:
            pbar.update(UPDATE_RATE)
            maddpg.train(BATCH_SIZE, REPLAY_START_SIZE, GAMMA)
            global_rewards.append(sum(episode_rewards))
            episodes.append(episode)
        #     print(f'Episode {episode}: Rewards = {episode_rewards}')
            
            #maddpg.save_checkpoint(f'{path}checkpoint_maddpg_{episode}.pth', episode)
            
    #saving checkpoint
    filename = "MADDPG_trained_agent_N_{}.pth".format(NUM_AGENTS)
    os.makedirs(path, exist_ok=True)
    save_path = os.path.join(path, filename)
    maddpg.save_checkpoint(save_path)
    pbar.close()
    env.close()  
    plt.figure()
    plt.plot(episodes, global_rewards, marker='o')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('MADDPG Training Rewards')
    plt.grid()
    plt.savefig(f'{path}training.png')
    plt.show()