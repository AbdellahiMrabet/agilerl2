# noqa: D212, D415
"""
# Simple Adversary

```{figure} mpe_simple_adversary.gif
:width: 140px
:name: simple_adversary
```

```{eval-rst}
.. warning::

    The environment `pettingzoo.mpe.simple_adversary_v3` has been moved to the new `MPE2 package <https://mpe2.farama.org>`_, and will be removed from PettingZoo in a future release.
    Please update your import to `mpe2.simple_adversary_v3`.

```

This environment is part of the <a href='..'>MPE environments</a>. Please read that page first for general information.

| Import             | `from pettingzoo.mpe import simple_adversary_v3` |
|--------------------|--------------------------------------------------|
| Actions            | Discrete/Continuous                              |
| Parallel API       | Yes                                              |
| Manual Control     | No                                               |
| Agents             | `agents= [adversary_0, agent_0,agent_1]`         |
| Agents             | 3                                                |
| Action Shape       | (5)                                              |
| Action Values      | Discrete(5)/Box(0.0, 1.0, (5))                   |
| Observation Shape  | (8),(10)                                         |
| Observation Values | (-inf,inf)                                       |
| State Shape        | (28,)                                            |
| State Values       | (-inf,inf)                                       |


In this environment, there is 1 adversary (red), N good agents (green), N landmarks (default N=2). All agents observe the position of landmarks and other agents. One landmark is the 'target landmark' (colored green). Good agents are rewarded based on how close the closest one of them is to the
target landmark, but negatively rewarded based on how close the adversary is to the target landmark. The adversary is rewarded based on distance to the target, but it doesn't know which landmark is the target landmark. All rewards are unscaled Euclidean distance (see main MPE documentation for
average distance). This means good agents have to learn to 'split up' and cover all landmarks to deceive the adversary.

Agent observation space: `[goal_rel_position, landmark_rel_position, other_agent_rel_positions]`

Adversary observation space: `[landmark_rel_position, other_agents_rel_positions]`

Agent action space: `[no_action, move_left, move_right, move_down, move_up]`

Adversary action space: `[no_action, move_left, move_right, move_down, move_up]`

### Arguments

``` python
simple_adversary_v3.env(N=2, max_cycles=25, continuous_actions=False, dynamic_rescaling=False)
```



`N`:  number of good agents and landmarks

`max_cycles`:  number of frames (a step for each agent) until game terminates

`continuous_actions`: Whether agent action spaces are discrete(default) or continuous

`dynamic_rescaling`: Whether to rescale the size of agents and landmarks based on the screen size

"""

import math
import numpy as np
from gymnasium.utils import EzPickle

from core import Agent, Landmark, World, Fog
from pettingzoo.mpe._mpe_utils.scenario import BaseScenario
from simple_env import SimpleEnv, make_env
from pettingzoo.utils.conversions import parallel_wrapper_fn


class raw_env(SimpleEnv, EzPickle):
    def __init__(
        self,
        N=4,
        max_cycles=25,
        continuous_actions=False,
        render_mode=None,
        dynamic_rescaling=False,
    ):
        EzPickle.__init__(
            self,
            N=N,
            max_cycles=max_cycles,
            continuous_actions=continuous_actions,
            render_mode=render_mode,
        )
        scenario = Scenario()
        world = scenario.make_world(N=N)
        SimpleEnv.__init__(
            self,
            scenario=scenario,
            world=world,
            render_mode=render_mode,
            max_cycles=max_cycles,
            continuous_actions=continuous_actions,
            dynamic_rescaling=dynamic_rescaling,
        )
        self.metadata["name"] = "simple_adversary_v3"

class raw_env_9(SimpleEnv, EzPickle):
    def __init__(
        self,
        N=9,
        max_cycles=25,
        continuous_actions=False,
        render_mode=None,
        dynamic_rescaling=False,
    ):
        EzPickle.__init__(
            self,
            N=N,
            max_cycles=max_cycles,
            continuous_actions=continuous_actions,
            render_mode=render_mode,
        )
        scenario = Scenario()
        world = scenario.make_world(N=N)
        SimpleEnv.__init__(
            self,
            scenario=scenario,
            world=world,
            render_mode=render_mode,
            max_cycles=max_cycles,
            continuous_actions=continuous_actions,
            dynamic_rescaling=dynamic_rescaling,
        )

class raw_env_14(SimpleEnv, EzPickle):
    def __init__(
        self,
        N=14,
        max_cycles=25,
        continuous_actions=False,
        render_mode=None,
        dynamic_rescaling=False,
    ):
        EzPickle.__init__(
            self,
            N=N,
            max_cycles=max_cycles,
            continuous_actions=continuous_actions,
            render_mode=render_mode,
        )
        scenario = Scenario()
        world = scenario.make_world(N=N)
        SimpleEnv.__init__(
            self,
            scenario=scenario,
            world=world,
            render_mode=render_mode,
            max_cycles=max_cycles,
            continuous_actions=continuous_actions,
            dynamic_rescaling=dynamic_rescaling,
        )

class raw_env_19(SimpleEnv, EzPickle):
    def __init__(
        self,
        N=19,
        max_cycles=25,
        continuous_actions=False,
        render_mode=None,
        dynamic_rescaling=False,
    ):
        EzPickle.__init__(
            self,
            N=N,
            max_cycles=max_cycles,
            continuous_actions=continuous_actions,
            render_mode=render_mode,
        )
        scenario = Scenario()
        world = scenario.make_world(N=N)
        SimpleEnv.__init__(
            self,
            scenario=scenario,
            world=world,
            render_mode=render_mode,
            max_cycles=max_cycles,
            continuous_actions=continuous_actions,
            dynamic_rescaling=dynamic_rescaling,
        )
        
env = make_env(raw_env_9)
env_4 = make_env(raw_env)
env_14 = make_env(raw_env_14)
env_19 = make_env(raw_env_19)

parallel_env = parallel_wrapper_fn(env)
parallel_env_4 = parallel_wrapper_fn(env_4)
parallel_env_14 = parallel_wrapper_fn(env_14)
parallel_env_19 = parallel_wrapper_fn(env_19)

class Scenario(BaseScenario):
    def make_world(self, N=19):
        world = World()
        # set any world properties first
        world.dim_c = 2
        num_agents = N + 1
        world.num_agents = num_agents
        num_adversaries = 0
        num_landmarks = num_agents - 1
        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        world.fog_nb = 3 # Number of rays for fog of war
        world.fogs = [Fog() for i in range(world.fog_nb)]
        for i, agent in enumerate(world.agents):
            agent.adversary = True if i < num_adversaries else False
            base_name = "adversary" if agent.adversary else "agent"
            base_index = i if i < num_adversaries else i - num_adversaries
            agent.name = f"{base_name}_{base_index}"
            agent.collide = False
            agent.silent = True
            agent.size = 0.15
        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = "landmark %d" % i
            landmark.collide = False
            landmark.movable = False
            landmark.size = 0.08
        return world

    def reset_world(self, world, np_random):
        # random properties for agents
        world.agents[0].color = np.array([0.85, 0.35, 0.35])
        for i in range(1, world.num_agents):
            world.agents[i].color = np.array([0.35, 0.35, 0.85])
        # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.15, 0.15, 0.15])
        # set goal landmark
        goal = np_random.choice(world.landmarks)
        goal.color = np.array([0.15, 0.65, 0.15])
        for agent in world.agents:
            agent.goal_a = goal
            agent.task.den = np.random.randint(1000, 2000)
            agent.task.ln = np.random.randint(1000, 2000)
        # set random initial states
        for fog in world.fogs:
            fog.utilization = np.random.uniform(20000, 30000)
            fog.den = np.random.uniform(100000, 200000)
            fog.R = 0
            fog.cost = np.random.randint(3) * 10 ** (-9)
        for agent in world.agents:
            agent.state.p_pos = np_random.uniform(-1, +1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
        for i, landmark in enumerate(world.landmarks):
            landmark.state.p_pos = np_random.uniform(-1, +1, world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)

    def benchmark_data(self, agent, world):
        # returns data for benchmarking purposes
        if agent.adversary:
            return np.sum(np.square(agent.state.p_pos - agent.goal_a.state.p_pos))
        else:
            dists = []
            for lm in world.landmarks:
                dists.append(np.sum(np.square(agent.state.p_pos - lm.state.p_pos)))
            dists.append(
                np.sum(np.square(agent.state.p_pos - agent.goal_a.state.p_pos))
            )
            return tuple(dists)

    # return all agents that are not adversaries
    def good_agents(self, world):
        return [agent for agent in world.agents if not agent.adversary]
    
    def setLoad(self, fog, load):
        fog.utilization += load

    # return all adversarial agents
    def adversaries(self, world):
        return [agent for agent in world.agents if agent.adversary]
  
    def reward(self, agent, world):
        # Agents are rewarded based on minimum agent distance to each landmark
        action = agent.action.a
        fog = np.where(action == 1)[0]
        fog = fog[0]
        rate = 0
        load = 0
        for ag in world.agents:
            if ag.action.a.argmax() == fog:
                rate += 1
                load += ag.task.den
        world.fogs[fog].setLoad(load)
        cost = world.fogs[fog].getLoad()
        rate *= 16
        rate += 0.1
        rate = rate**-1
        rate *= 16
        rate += 1
        rate = math.log2(rate)
        rate *= (5 * (10 ** 6))
        cost += agent.task.ln / rate
        cost *= 0.55
        cost += agent.task.ln * world.fogs[fog].getCost()
        return -cost

    def agent_reward(self, agent, world):
        # Rewarded based on how close any good agent is to the goal landmark, and how far the adversary is from it
        shaped_reward = True
        shaped_adv_reward = True

        # Calculate negative reward for adversary
        adversary_agents = self.adversaries(world)
        if shaped_adv_reward:  # distance-based adversary reward
            adv_rew = sum(
                np.sqrt(np.sum(np.square(a.state.p_pos - a.goal_a.state.p_pos)))
                for a in adversary_agents
            )
        else:  # proximity-based adversary reward (binary)
            adv_rew = 0
            for a in adversary_agents:
                if (
                    np.sqrt(np.sum(np.square(a.state.p_pos - a.goal_a.state.p_pos)))
                    < 2 * a.goal_a.size
                ):
                    adv_rew -= 5

        # Calculate positive reward for agents
        good_agents = self.good_agents(world)
        if shaped_reward:  # distance-based agent reward
            pos_rew = -min(
                np.sqrt(np.sum(np.square(a.state.p_pos - a.goal_a.state.p_pos)))
                for a in good_agents
            )
        else:  # proximity-based agent reward (binary)
            pos_rew = 0
            if (
                min(
                    np.sqrt(np.sum(np.square(a.state.p_pos - a.goal_a.state.p_pos)))
                    for a in good_agents
                )
                < 2 * agent.goal_a.size
            ):
                pos_rew += 5
            pos_rew -= min(
                np.sqrt(np.sum(np.square(a.state.p_pos - a.goal_a.state.p_pos)))
                for a in good_agents
            )
        return pos_rew + adv_rew

    def adversary_reward(self, agent, world):
        # Rewarded based on proximity to the goal landmark
        shaped_reward = True
        if shaped_reward:  # distance-based reward
            return -np.sqrt(
                np.sum(np.square(agent.state.p_pos - agent.goal_a.state.p_pos))
            )
        else:  # proximity-based reward (binary)
            adv_rew = 0
            if (
                np.sqrt(np.sum(np.square(agent.state.p_pos - agent.goal_a.state.p_pos)))
                < 2 * agent.goal_a.size
            ):
                adv_rew += 5
            return adv_rew

    def observation(self, agent, world):
        # get positions of all entities in this agent's reference frame
        
        other_action = []
        fogs = []
        for other in world.agents:
            if other.action.a is None:
                other.action.a = np.zeros(world.fog_nb)
            if other is agent:
                continue
            other_action.append(other.action.a)
        if agent.action.a is None:
            agent.action.a = np.zeros(world.fog_nb)
        for fog in world.fogs:
            fogs.append(fog.getLoad())
        fogs = np.array(fogs)
        #print('other_action', other_action, 'fogs', fogs)
        return np.concatenate(
                fogs + other_action 
            )