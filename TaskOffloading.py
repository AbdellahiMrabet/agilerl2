import gym
from gym import spaces
import random
import numpy as np
import math

class TOEnv(gym.Env):
    metadata = {"render_modes": ["human"], "name": "to_v10"}

    def __init__(self, render_mode=None, iot_nb=20, fog_nb=5):
        self.fog_nb = fog_nb  # The number of fog nodes
        self.fogs = ["fog"+str(i) for i in range(self.fog_nb)]
        self.fogs_set = {"fog"+str(i) for i in range(self.fog_nb)}

        self.iot_nb = iot_nb  # The number of iot devices
        self.iots = ["iot"+str(i) for i in range(self.iot_nb)]

        # Observations are dictionaries with the fog's and the local's utilisation.
        
        self.observation_space = spaces.Box(0, 1, shape=(1+self.fog_nb,), dtype=np.float64()) 

        # chooce to execute task locally or offload it to a fog node
        arr = np.full((self.iot_nb), -1)

        arr2 = np.full((self.iot_nb), 1+self.fog_nb)
        self.action_space = spaces.Box(np.float64(arr), np.float64(arr2), dtype=np.float64)

        self._local_den = {iot: random.uniform(10000, 20000) for iot in self.iots}
        self._fog_den = {fog: random.uniform(100000, 200000) for fog in self.fogs}
        self._trans_rate = {fog: random.uniform(10000, 20000) for fog in self.fogs}
        self._tasks_on = {dv: [] for dv in self.iots+self.fogs}
        self._queue = []
        
        self._local_utilization = {iot: random.uniform(1000, 1533) for iot in self.iots}

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """

    def get_den(self, server, type):

        if type == "iot":
            return self._local_den[server]
        else:
            return self._fog_den[server]
  

    def _get_obs(self, iot):
        obs = [round(self._local_utilization[iot]/self._local_den[iot], 5)]
        
        for j in range(self.fog_nb):
            obs.append(round(self._fog_utilization[self.fogs[j]]/self._fog_den[self.fogs[j]], 5))
        return np.array(obs)

    def _get_info(self):
        return {"selected_server": ""} #
    

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # Choose the server's utilisation uniformly at random
        #first we initialize the server's 
      
        self._local_utilization = {iot: random.uniform(1000, 1533) for iot in self.iots}
        self._fog_utilization = {fog: random.uniform(20000, 30000) for fog in self.fogs}
        
        self._local_den = {iot: random.uniform(10000, 20000) for iot in self.iots}
        
        self._fog_den = {fog: random.uniform(100000, 200000) for fog in self.fogs}
        
        self._tasks_on = {dv: [] for dv in self.iots+self.fogs}
        self._queue = []
        
        iot = 'iot'+str(random.randrange(self.iot_nb))
        observation =  self._get_obs(iot)
        info = self._get_info()
        info['current'] = iot
        
        return observation, info
    
    def createTask(self, strt, dn, lt, device, orig=None):
        obs = []
        task = (strt, dn, lt)
        if device == 'queue':
            self._queue.append(task)

        elif device in self.iots:
            obs = self._get_obs(device)
            self._local_utilization[device] += dn
            tasks = self._tasks_on[device]
            tasks.append((len(tasks), *task, device))
            
            obs = self._get_obs(device)

        elif device in self.fogs:
            obs = self._get_obs(orig)
            self._fog_utilization[device] += dn
            tasks = self._tasks_on[device]
            tasks.append((len(tasks), *task, device, orig))
            obs = self._get_obs(orig)
        else:
            print('you must choose either queue, iot or fog.')
            pass

        return obs
    
    def endTask(self, tsk, device):
        iot = random.choice(self.iots)
        tasks = self._tasks_on[device]
        if tsk in tasks:
            if device in self.fogs:
                try:
                    self._fog_utilization[device] -= tsk[2]
                except:
                    pass
            else:
                iot = device
                try:
                    self._local_utilization[device] -= tsk[2]
                except:
                    pass
        tasks.remove(tsk)
        for i in range(len(tasks)):
            tasks[i] = (i, *tasks[i][1:])

        obs = self._get_obs(iot)
        
        return obs

    def iot_initialize(self, iot, util):
        self._local_utilization[iot] = util
        self._local_den[iot] = 10 * util

    def fog_initialize(self, fog, util):
        self._fog_utilization[fog] = util
        self._fog_den[fog] = 10 * util

    def step(self, action):
        
        new_obs, rewards, terminated, truncated, new_info = {iot: self._get_obs(iot) for iot in self.iots}\
        , {iot: 0 for iot in self.iots}, {iot: False for iot in self.iots}, {iot: False for iot in self.iots},\
        {iot: {} for iot in self.iots}
        initial_rew = (2+self.fog_nb)//2
        cost = np.full([self.iot_nb, 2 + self.fog_nb], np.zeros(int(2 + self.fog_nb), dtype=float))

        #actions
        self._A = np.full([1 + self.fog_nb, self.iot_nb], np.zeros(int(self.iot_nb), dtype=int))

        #reload actions
        self._A4 = np.full([self.fog_nb, self.iot_nb], np.zeros(int(self.iot_nb), dtype=int))

        #transmission rates
        self._R = np.zeros(int(self.fog_nb), dtype=int)
        for k in range(self.iot_nb):
            if action[k] != -1:
                    
                location = 0
                cost[k][0] = 0.55 * self._get_obs(self.iots[k])[0]
                cost[k][1+self.fog_nb] = 0.55 * self._get_obs(self.iots[k])[0]
                for a in range(1, 1+self.fog_nb):
                    cost[k][a] = 0.55 * self._get_obs(self.iots[k])[a] + 0.01

                if action[k] == 0:
                    if self._queue == []:
                        print('no task in the queue')
                    else:
                        tsk = self._queue[-1]
                        #move task from queue to IoT device
                        self.createTask(*tsk, self.iots[k])
                        self._queue.remove(tsk)
                elif action[k] in range(1, 1 + self.fog_nb):
                    if self._tasks_on[self.iots[k]] == []:
                        print('no task on {}'.format(self.iots[k]))
                    else:
                        tsk = self._tasks_on[self.iots[k]][-1]
                        self._R[action[k]-1] += 1
                        
                        #cost function
                        for a in range(1, 1+self.fog_nb):
                            
                            rate = self._R[a-1]
                            if rate == 0:
                                rate = 1
                            rate *= 16
                            rate += 0.1
                            rate = rate**-1
                            rate *= 16
                            rate += 1
                            rate = math.log(rate)
                            rate /= math.log(2)
                            rate *= (5 * (10 ** 6))
                            cost[k][a] += tsk[2] / rate

                        #move task from IoT device to fog node
                        self.createTask(*tsk[1:-1], 'fog'+str(action[k] - 1), self.iots[k])
                        self.endTask(tsk, self.iots[k])

                elif action[k] == 1 + self.fog_nb:
                        
                        for a in range(1, 1+self.fog_nb):
                            tsks2 = self._tasks_on['fog'+str(a-1)]
                            tsksiot2 = [ts for ts in tsks2 if ts[-1] in [self.iots[k]]]
                            if tsksiot2 != []:
                                tsk = tsksiot2[-1]
                                self._A4[a-1] = 1        
                                self._R[a-1] += 1
                                    
                                rate = self._R[a-1]
                                if rate == 0:
                                    rate = 1
                                rate *= 16
                                rate += 0.1
                                rate = rate**-1
                                rate *= 16
                                rate += 1
                                rate = math.log(rate)
                                rate /= math.log(2)
                                rate *= (5 * (10 ** 6))
                                #cost[k][a] += tsk[2] / rate
                                #cost[k][1+self.fog_nb] += tsk[2] / rate
                                cost[k][0] = cost[k][a]
                                cost[k][action[k]] = cost[k][1+self.fog_nb]
                                #update the state
                                self.createTask(*tsk[1:-2], self.iots[k])
                                self.endTask(tsk, 'fog'+str(a-1))
                                break
                else:
                    print('action should be 0, 1 or 2')

                obs5 = sorted(cost[k])
                location = obs5.index(cost[k][action[k]])

                REWARD_MAP = {(0): (initial_rew)}
                
                for i in range(1, self.fog_nb+2):
                    REWARD_MAP[i] = initial_rew - i
                #update the values to be rended
                new_obs[self.iots[k]] = self._get_obs(self.iots[k])
                rewards[self.iots[k]] = REWARD_MAP[location]
                if rewards[self.iots[k]] == initial_rew:
                    terminated[self.iots[k]] = True
                new_info[self.iots[k]] = cost[k][action[k]]

        return new_obs, rewards, terminated, False, new_info

    def render(self):
        """
        Renders the environment. In human mode, it can print to terminal, open
        up a graphical window, or open up some other display that a human can see and understand.
        """
        if self.render_mode is None:
            gym.logger.warn(
                "You are calling render method without specifying any render mode."
            )
            return

        
        obs = self._get_obs('iot0')
        string = ''
        in_format = []
        for iot in self.iots:
            obs = self._get_obs(iot)
            string += "local utilization of {} : {:.2f} %\n"
            in_format.append(iot)
            in_format.append((obs[0]*100))
        for i in range(self.fog_nb):
            string += "fog"+str(i)+" utilization : {:.2f} %\n"
            in_format.append((obs[i+1]*100))

        string = string.format(*tuple(in_format))
        
        
        print(string)

    def iot_state(self, iot):
        tasks = self._tasks_on[iot]
        last = None
        if tasks != []:
            last = tasks[-1]
        util = self._get_obs(iot)

        return tasks, last, util

    def fog_state(self, fog):
        tasks = self._tasks_on[fog]
        
        return tasks
    
    def close(self):
        pass