import gym
import numpy as np
from .classes.GM import GridMap
from .config.GW import GridWorldConfig

import os
import matplotlib as mpl
if os.environ.get('DISPLAY','') == '':
    print('no display found. Using non-interactive Agg backend')
    mpl.use('Agg')
# mpl.use('Agg')


def create_environment(env_params):
    env_name = env_params['environment']
    if env_name == 'ContinuousBanditsNormalized':
        return ContinuousBanditsNormalized(env_params)
    elif 'ContinuousMaze' in env_name or 'ContinuousWorld' in env_name:
        return ContinuousMazeEnvironment(env_params)
    elif 'GridWorld' in env_name:
        return GridWorldEnvironment(env_params)
    else:
        return ContinuousEnvironment(env_params)


class ContinuousBanditsNormalized(object):
    def __init__(self, env_params):

        self.name = env_params['environment']
        self.eval_interval = env_params['EvalIntervalMilSteps'] * 1000000
        self.eval_episodes = env_params['EvalEpisodes']

        # total number of steps allowed in a run
        self.TOTAL_STEPS_LIMIT = env_params['TotalMilSteps'] * 1000000

        # maximum number of steps allowed for each episode
        # if -1 takes default setting from gym
        if env_params['EpisodeSteps'] != -1:
            self.EPISODE_STEPS_LIMIT = env_params['EpisodeSteps']

        else:
            self.EPISODE_STEPS_LIMIT = 1  # only one state env

        # state info
        self.state_dim = 1
        self.state_range = np.array([0.])
        self.state_min = np.array([0.])
        self.state_max = np.array([0.])
        self.state_bounded = True

        # action info
        self.action_dim = 1
        self.action_range = np.array([2.])
        self.action_min = np.array([-1.])
        self.action_max = np.array([1.])

    def set_random_seed(self, random_seed):
        pass

    # Reset the environment for a new episode. return the initial state
    def reset(self):

        # starts at 0.
        self.state = np.array([0.])
        return self.state

    def step(self, action):
        self.state = self.state + action  # terminal state
        reward = self.reward_func(action)
        done = True
        info = {}

        return self.state, reward, done, info

    @staticmethod
    def reward_func(action):

        maxima1 = -1.0
        maxima2 = 1.0

        stddev1 = 0.2
        stddev2 = 0.2

        # Reward function.
        # Two gaussian functions.
        modal1 = 1. * np.exp(-0.5 * ((2 * action - maxima1) / stddev1) ** 2)
        modal2 = 1.5 * np.exp(-0.5 * ((2 * action - maxima2) / stddev2) ** 2)

        return modal1 + modal2

    @staticmethod
    def get_max():
        return 0.5

    # Close the environment and clear memory
    def close(self):
        pass


# This file provide environments to interact with, consider actions as continuous, need to rewrite otherwise
class ContinuousEnvironment(object):
    def __init__(self, env_params):

        self.name = env_params['environment']
        self.eval_interval = env_params['EvalIntervalMilSteps'] * 1000000
        self.eval_episodes = env_params['EvalEpisodes']
        self.render = env_params['render']

        self.instance = self.init_instance(env_params)

        # total number of steps allowed in a run
        self.TOTAL_STEPS_LIMIT = env_params['TotalMilSteps'] * 1000000
        # self.TOTAL_EPISODES_LIMIT = env_params['TotalEpisodes']

        # maximum number of steps allowed for each episode
        # if -1 takes default setting from gym
        if env_params['EpisodeSteps'] != -1:
            self.EPISODE_STEPS_LIMIT = env_params['EpisodeSteps']
            self.instance._max_episode_steps = env_params['EpisodeSteps']

        else:
            self.EPISODE_STEPS_LIMIT = self.instance._max_episode_steps

        # state info
        self.state_dim = self.get_state_dim()
        self.state_range = self.get_state_range()
        self.state_min = self.get_state_min()
        self.state_max = self.get_state_max()
        self.state_bounded = False if np.any(np.isinf(self.instance.observation_space.high)) or np.any(np.isinf(self.instance.observation_space.low)) else True
        
        # action info
        self.action_dim = self.get_action_dim()
        self.action_range = self.get_action_range()
        self.action_min = self.get_action_min()
        self.action_max = self.get_action_max()

    def init_instance(self, env_params):
        return gym.make(env_params['environment'])

    def set_random_seed(self, random_seed):
        self.instance.seed(random_seed)

    # Reset the environment for a new episode. return the initial state
    def reset(self):
        state = self.instance.reset()
        '''
        if self.state_bounded:
            # normalize to [-1,1]
            scaled_state = 2.*(state - self.state_min)/self.state_range - 1.
            return scaled_state
        '''
        return state

    def step(self, action):
        state, reward, done, info = self.instance.step(action)

        '''
        if self.state_bounded:
            scaled_state = 2.*(state - self.state_min)/self.state_range - 1.
            return (scaled_state, reward, done, info)
        '''
        return (state, reward, done, info)

    def get_state_dim(self):
        return self.instance.observation_space.shape[0]
  
    # this will be the output units in NN
    def get_action_dim(self):
        if hasattr(self.instance.action_space, 'n'):
            return int(self.instance.action_space.n-1)
        return int(self.instance.action_space.sample().shape[0])

    # Return action ranges, NOT IN USE
    def get_action_range(self):
        if hasattr(self.instance.action_space, 'high'):
            return self.instance.action_space.high - self.instance.action_space.low
        return self.instance.action_space.n - 1    

    # Return action ranges
    def get_action_max(self):
        if hasattr(self.instance.action_space, 'high'):
            return self.instance.action_space.high
        return self.instance.action_space.n - 1    

    # Return action min
    def get_action_min(self):
        if hasattr(self.instance.action_space, 'low'):
            return self.instance.action_space.low
        return 0

    # Return state range
    def get_state_range(self):
        return self.instance.observation_space.high - self.instance.observation_space.low
    
    # Return state min
    def get_state_min(self):
        return self.instance.observation_space.low

    def get_state_max(self):
        return self.instance.observation_space.high

    # Close the environment and clear memory
    def close(self):
        self.instance.close()

class ContinuousMazeEnvironment(object):
    def __init__(self, env_params):

        self.name = env_params['environment']
        self.eval_interval = env_params['EvalIntervalMilSteps'] * 1000000
        self.eval_episodes = env_params['EvalEpisodes']

        ## Gridmap is not the same as GridWorld!!!
        working_dir = env_params['working_dir']
        self.instance = GridMap.GridMapEnv( workingDir=working_dir)
        self.instance.load(working_dir, "initial_maze_env.json")
        self.instance.reset()

        self.render = env_params['render']
        self.render_time = -1 if not self.render else 0.001

        self.inner_step_count = 0
        self.timeout_steps = env_params['TimeoutSteps']

        # total number of steps allowed in a run
        self.TOTAL_STEPS_LIMIT = env_params['TotalMilSteps'] * 1000000
        # self.TOTAL_EPISODES_LIMIT = env_params['TotalEpisodes']

        # maximum number of steps allowed for each episode
        # if -1 takes default setting from gym
        self.EPISODE_STEPS_LIMIT = env_params['EpisodeSteps']
        
        # state info
        self.state_dim = self.get_state_dim()
        self.state_range = self.get_state_range()
        self.state_min = self.get_state_min()
        self.state_max = self.get_state_max()
        self.state_bounded = True
        
        # action info
        self.action_dim = self.get_action_dim()
        self.action_range = self.get_action_range()
        self.action_min = self.get_action_min()
        self.action_max = self.get_action_max()

    def set_random_seed(self, random_seed):
        pass

    # Reset the environment for a new episode. return the initial state
    def reset(self):
        return self.instance.reset()

    def step(self, action):
        if self.inner_step_count == self.timeout_steps:
            self.inner_step_count = 0
            state = self.reset()
            return (state, 0, False, None)
        if action[0] == 0 and action[1] == 0:
            action[0] = 1e-10
        act = GridMap.BlockCoorDelta( action[0], action[1] )
        state, reward, done, _ = self.instance.step(act)
        self.inner_step_count += 1
        if self.render:
            self.instance.render(self.render_time, flagSave=False)
        return (state, reward, done, None)

    def get_state_dim(self):
        return self.instance.get_state_size()
  
    # this will be the output units in NN
    def get_action_dim(self):
        return self.instance.get_action_size()

    # Return action ranges
    def get_action_range(self):
        return self.instance.get_action_range()

    # Return action ranges
    def get_action_max(self):
        return self.instance.get_action_max()

    # Return action min
    def get_action_min(self):
        return self.instance.get_action_min()

    # Return state range
    def get_state_range(self):
        return self.instance.get_state_range()
    
    # Return state min
    def get_state_min(self):
        return self.instance.get_state_min()

    def get_state_max(self):
        return self.instance.get_state_max()

    # Close the environment and clear memory
    def close(self):
        self.instance.finalize()

class GridWorldEnvironment(ContinuousEnvironment):
    def __init__(self, env_params):
        super().__init__(env_params)
        self.inner_step_count = 0
        self.timeout_steps = env_params['TimeoutSteps']

    def init_instance(self, env_params):
        from .classes.GW.continous_grids import GridWorld
        return GridWorld(silent_mode=not env_params['render'], max_episode_len=env_params['EpisodeSteps'], **GridWorldConfig.config())

    def step(self, action):
        if self.inner_step_count == self.timeout_steps:
            self.inner_step_count = 0
            state = self.reset()
            return (state, 0, False, None)
        if action[0] == 0 and action[1] == 0:
            action[0] = 1e-10
        state, reward, done, _ = self.instance.step(action)
        self.inner_step_count += 1
        return (state, reward, done, None)