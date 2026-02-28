# Author: Michael Pratt
import os

import imageio
from agilerl.components.data import Transition, ReplayDataset, to_tensordict
import gymnasium as gym
import numpy as np
import torch
from tqdm import trange

from agilerl.algorithms.td3 import TD3
from agilerl.components.replay_buffer import ReplayBuffer
from agilerl.algorithms.core.registry import HyperparameterConfig, RLParameter
from agilerl.hpo.mutation import Mutations
from agilerl.hpo.tournament import TournamentSelection
from agilerl.training.train_off_policy import train_off_policy
from agilerl.utils.utils import (
    create_population,
    make_vect_envs,
    observation_space_channels_to_first
)

if __name__ == "__main__":

    # Initial hyperparameters
    INIT_HP = {
        "ALGO": "TD3",
        "POP_SIZE": 4,  # Population size
        "BATCH_SIZE": 128,  # Batch size
        "LR_ACTOR": 0.0001,  # Actor learning rate
        "LR_CRITIC": 0.001,  # Critic learning rate
        "O_U_NOISE": True,  # Ornstein-Uhlenbeck action noise
        "EXPL_NOISE": 0.1,  # Action noise scale
        "MEAN_NOISE": 0.0,  # Mean action noise
        "THETA": 0.15,  # Rate of mean reversion in OU noise
        "DT": 0.01,  # Timestep for OU noise
        "GAMMA": 0.99,  # Discount factor
        "MEMORY_SIZE": 100_000,  # Max memory buffer size
        "POLICY_FREQ": 2,  # Policy network update frequency
        "LEARN_STEP": 1,  # Learning frequency
        "TAU": 0.005,  # For soft update of target parameters
        "EPISODES": 1000,  # Number of episodes to train for
        "EVO_EPOCHS": 20,  # Evolution frequency, i.e. evolve after every 20 episodes
        "TARGET_SCORE": 200.0,  # Target score that will beat the environment
        "EVO_LOOP": 3,  # Number of evaluation episodes
        "MAX_STEPS": 500,  # Maximum number of steps an agent takes in an environment
        "LEARNING_DELAY": 1000,  # Steps before starting learning
        "EVO_STEPS": 10000,  # Evolution frequency
        "EVAL_STEPS": None,  # Number of evaluation steps per episode
        "EVAL_LOOP": 1,  # Number of evaluation episodes
        "TOURN_SIZE": 2,  # Tournament size
        "ELITISM": True,  # Elitism in tournament selection
    }

    # Mutation parameters
    MUT_P = {
        # Mutation probabilities
        "NO_MUT": 0.4,  # No mutation
        "ARCH_MUT": 0.2,  # Architecture mutation
        "NEW_LAYER": 0.2,  # New layer mutation
        "PARAMS_MUT": 0.2,  # Network parameters mutation
        "ACT_MUT": 0.2,  # Activation layer mutation
        "RL_HP_MUT": 0.2,  # Learning HP mutation
        "MUT_SD": 0.1,  # Mutation strength
        "RAND_SEED": 42,  # Random seed
        # Define max and min limits for mutating RL hyperparams
        "MIN_LR": 0.0001,
        "MAX_LR": 0.01,
        "MIN_BATCH_SIZE": 8,
        "MAX_BATCH_SIZE": 1024,
        "MIN_LEARN_STEP": 1,
        "MAX_LEARN_STEP": 16,
    }

    # Create vectorized environment
    num_envs = 8
    env = make_vect_envs("LunarLanderContinuous-v3", num_envs=num_envs)  # Create environment

    observation_space = env.single_observation_space
    action_space = env.single_action_space

    # Set-up the device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Define the network configuration of a simple mlp with two hidden layers, each with 64 nodes
    net_config = {
        "encoder_config": {"hidden_size": [64, 64]},  # Encoder hidden size
        "head_config": {"hidden_size": [64, 64]},  # Head hidden size
    }

    # Mutation config for RL hyperparameters
    hp_config = HyperparameterConfig(
        lr_actor = RLParameter(min=1e-4, max=1e-2),
        lr_critic = RLParameter(min=1e-4, max=1e-2),
        learn_step = RLParameter(min=1, max=16),
        batch_size = RLParameter(min=8, max=512),
    )

    # Define a population
    pop = create_population(
        algo="TD3", # Algorithm
        observation_space=observation_space,  # State dimension
        action_space=action_space,  # Action dimension
        net_config=net_config,  # Network configuration
        INIT_HP=INIT_HP,  # Initial hyperparameters
        hp_config=hp_config,  # RL hyperparameter configuration
        population_size=INIT_HP["POP_SIZE"],  # Population size
        num_envs=num_envs,
        device=device,
    )

    memory = ReplayBuffer(
        max_size=10000,  # Max replay buffer size
        device=device,
    )


    tournament = TournamentSelection(
        INIT_HP["TOURN_SIZE"],
        INIT_HP["ELITISM"],
        INIT_HP["POP_SIZE"],
        INIT_HP["EVAL_LOOP"],
    )

    mutations = Mutations(
        no_mutation=MUT_P["NO_MUT"],
        architecture=MUT_P["ARCH_MUT"],
        new_layer_prob=MUT_P["NEW_LAYER"],
        parameters=MUT_P["PARAMS_MUT"],
        activation=MUT_P["ACT_MUT"],
        rl_hp=MUT_P["RL_HP_MUT"],
        mutation_sd=MUT_P["MUT_SD"],
        rand_seed=MUT_P["RAND_SEED"],
        device=device,
    )

    # Define training loop parameters
    max_steps = 500  # Max steps (default: 2000000)
    learning_delay = 0  # Steps before starting learning
    evo_steps = 10_000  # Evolution frequency
    eval_steps = None  # Evaluation steps per episode - go until done
    eval_loop = 1  # Number of evaluation episodes
    elite = pop[0]  # Assign a placeholder "elite" agent
    total_steps = 0

    total_steps = 0

    # TRAINING LOOP
    print("Training...")
    pbar = trange(max_steps, unit="step")
    while np.less([agent.steps[-1] for agent in pop], max_steps).all():
        pop_episode_scores = []
        for agent in pop:  # Loop through population
            obs, info = env.reset()  # Reset environment at start of episode
            scores = np.zeros(num_envs)
            completed_episode_scores = []
            steps = 0

            for idx_step in range(INIT_HP["EVO_STEPS"] // num_envs):
                action = agent.get_action(obs)  # Get next action from agent

                # Act in environment
                next_obs, reward, terminated, truncated, info = env.step(action)
                scores += np.array(reward)
                steps += num_envs
                total_steps += num_envs

                # Collect scores for completed episodes
                reset_noise_indices = []
                for idx, (d, t) in enumerate(zip(terminated, truncated)):
                    if d or t:
                        completed_episode_scores.append(scores[idx])
                        agent.scores.append(scores[idx])
                        scores[idx] = 0
                        reset_noise_indices.append(idx)

                # Reset action noise
                agent.reset_action_noise(reset_noise_indices)

                # Save experience to replay buffer
                done = terminated or truncated
                transition = Transition(
                    obs=obs,
                    action=action,
                    reward=reward,
                    next_obs=next_obs,
                    done=done,
                    batch_size=[num_envs]
                )
                transition = transition.to_tensordict()
                memory.add(transition)

                # Learn according to learning frequency
                if memory.size > INIT_HP["LEARNING_DELAY"] and len(memory) >= agent.batch_size:
                    for _ in range(num_envs // agent.learn_step):
                        # Sample replay buffer
                        experiences = memory.sample(agent.batch_size)
                        # Learn according to agent's RL algorithm
                        agent.learn(experiences)

                obs = next_obs

            pbar.update(INIT_HP["EVO_STEPS"] // len(pop))
            agent.steps[-1] += steps
            pop_episode_scores.append(completed_episode_scores)

        # Evaluate population
        fitnesses = [
            agent.test(
                env,
                max_steps=INIT_HP["EVAL_STEPS"],
                loop=INIT_HP["EVAL_LOOP"],
            )
            for agent in pop
        ]
        mean_scores = [
            (
                np.mean(episode_scores)
                if len(episode_scores) > 0
                else "0 completed episodes"
            )
            for episode_scores in pop_episode_scores
        ]

        print(f"--- Global steps {total_steps} ---")
        print(f"Steps {[agent.steps[-1] for agent in pop]}")
        print(f"Scores: {mean_scores}")
        print(f'Fitnesses: {["%.2f"%fitness for fitness in fitnesses]}')
        print(
            f'5 fitness avgs: {["%.2f"%np.mean(agent.fitness[-5:]) for agent in pop]}'
        )

        # Tournament selection and population mutation
        elite, pop = tournament.select(pop)
        pop = mutations.mutation(pop)

        # Update step counter
        for agent in pop:
            agent.steps.append(agent.steps[-1])

    # Save the trained algorithm
    path = "./models/TD3"
    filename = "TD3_trained_agent.pt"
    os.makedirs(path, exist_ok=True)
    save_path = os.path.join(path, filename)
    elite.save_checkpoint(save_path)
    pbar.close()
    env.close()



