import os
import sys
sys.path.append(os.getcwd())

import gym
import numpy as np
import matplotlib.pyplot as plt
import minatar
import re
import itertools
import numpy as np

import src.utils.math_utils as utils
import src.representations.rbf as rbf

class MinAtar():
	def __init__(self, name):
		self.env = minatar.Environment(name)
		self.obs_dim = 10 * 10 * self.env.n_channels
		self.in_channels = self.env.state_shape()[2]
		self.action_dim = self.env.num_actions()
		self.num_actions = self.env.num_actions()
		self.name = self.env.game_name()

	def step(self, action):
		reward, done = self.env.act(action)
		return self.env.state(), reward, done, None

	def reset(self):
		self.env.reset()
		return self.env.state()

	def render(self):
		pass

class Env():
	def __init__(self):
		self.env = None

	def step(self, action):
		return self.env.step(action)

	def reset(self):
		return self.env.reset()

	def render(self):
		return self.env.render()

class InvertedPendulum(Env):
	def __init__(self, use_rbf=False):
		self.env = gym.make("InvertedPendulumPyBulletEnv-v0")
		self.name = "InvertedPendulum"
		self.env._max_episode_steps = int(1e10)
		# self.tilecode = tilecode 
		self.use_rbf = use_rbf
		self.action_dim = 1
		if self.use_rbf is True:
			self.obs_dim = 256
			self.rbf = rbf.Fourier(5, self.obs_dim, dtype = "numpy")
			# self.tc = tilecoding.TileCoder([4, 4, 4, 4], [(-2.4, 2.4), (-10., 10), (-41.8, 41.8), (-10, 10)], 4)
			# self.obs_dim = self.tc.n_tiles
			# print(self.obs_dim)
		else:
			self.obs_dim = 5
		self.action_type = "continuous"

	def step(self, action):
		sp, r, d, _ = self.env.step(action)
		if self.use_rbf is True:
			sp = self.rbf(sp)
		return sp, r, d, _

	def reset(self):
		s = self.env.reset()
		if self.use_rbf is True:
			s = self.rbf(s)
			# print("reset tc")
		# print(self.tc[s])
		return s

class InvertedDoublePendulum(Env):
	def __init__(self, use_rbf=False):
		self.env = gym.make("InvertedDoublePendulumMuJoCoEnv-v0")
		self.name = "InvertedDoublePendulum"
		self.env._max_episode_steps = int(1e10)
		# self.tilecode = tilecode 
		self.use_rbf = use_rbf
		self.action_dim = 1
		if self.use_rbf is True:
			self.obs_dim = 512
			self.rbf = rbf.Fourier(11, self.obs_dim, dtype = "numpy")
			# self.tc = tilecoding.TileCoder([4, 4, 4, 4], [(-2.4, 2.4), (-10., 10), (-41.8, 41.8), (-10, 10)], 4)
			# self.obs_dim = self.tc.n_tiles
			# print(self.obs_dim)
		else:
			self.obs_dim = 11
		self.action_type = "continuous"

	def step(self, action):
		sp, r, d, _ = self.env.step(action)
		if self.use_rbf is True:
			sp = self.rbf(sp)
		return sp, r, d, _

	def reset(self):
		s = self.env.reset()
		if self.use_rbf is True:
			s = self.rbf(s)
			# print("reset tc")
		# print(self.tc[s])
		return s

class DiscretePendulum(Env):
	def __init__(self, action_dim = 10):
		self.env = gym.make("Pendulum-v0")
		self.obs_dim = 3
		self.action_type = "discrete"
		self.action_dim = action_dim
		self.name = "DiscretePendulum{}".format(action_dim)

	def step(self, action):
		# real action space is -2, 2
		a = -2. + 4 * action / (self.action_dim - 1)
		# print(a)
		return self.env.step([a])

class Pendulum(Env):
	def __init__(self, use_rbf = False):
		self.env = gym.make("Pendulum-v0")
		self.action_dim = 1
		self.action_type = "continuous"
		self.name = "Pendulum"
		self.env._max_episode_steps = int(1e10)
		self.use_rbf = use_rbf
		if self.use_rbf is True:
			self.obs_dim = 128
			self.rbf = rbf.Fourier(3, self.obs_dim, dtype = "numpy")
		else:
			self.obs_dim = 3
		self.action_type = "continuous"

	def step(self, action):
		# print(2 * action)
		sp, r, d, _ = self.env.step(2 * action) # scale since actions taken are [-2, 2] and agents give [-1, 1]
		if self.use_rbf is True:
			sp = self.rbf(sp)
		return sp, r, d, _
	
	def reset(self):
		s = self.env.reset()
		if self.use_rbf is True:
			s = self.rbf(s)
		return s



class LunarLander(Env):
	def __init__(self, use_rbf = False, n_features=256):
		self.env = gym.make("LunarLander-v2")
		self.env._max_episode_steps = int(1e10)
		self.action_dim = 4
		self.name = "LunarLander"

		self.use_rbf = use_rbf
		if self.use_rbf is True:
			self.obs_dim = n_features
			self.rbf = rbf.Fourier(8, self.obs_dim, dtype = "numpy")
		else:
			self.obs_dim = 8
		self.action_type = "discrete"

	def step(self, action):
		sp, r, d, _ = self.env.step(action)
		if self.use_rbf is True:
			sp = self.rbf(sp)
		return sp, r, d, _

	
	def reset(self):
		s = self.env.reset()
		if self.use_rbf is True:
			s = self.rbf(s)
		return s


class LunarLanderContinuous(Env):
	def __init__(self):
		self.env = gym.make("LunarLanderContinuous-v2")
		self.env._max_episode_steps = int(1e10)
		self.obs_dim = 8
		self.action_type = "continuous"
		self.action_dim = 2
		self.name = "LunarLanderContinuous"


class Acrobot(Env):
	def __init__(self, use_rbf = False, n_features = None):
		self.env = gym.make("Acrobot-v1")
		self.name = "Acrobot"
		self.env._max_episode_steps = int(1e10)
		self.action_dim = 3
		self.action_type = "discrete"
		self.use_rbf = use_rbf
		if self.use_rbf is True:
			self.obs_dim = n_features
			self.rbf = rbf.Fourier(6, self.obs_dim, dtype = "numpy")
		else:
			self.obs_dim = 6

	def step(self, action):
		sp, r, d, _ = self.env.step(action)
		if self.use_rbf is True:
			sp = self.rbf(sp)
		return sp, r, d, _

	def reset(self):
		s = self.env.reset()
		if self.use_rbf is True:
			s = self.rbf(s)
		return s

class CartPole(Env):
	def __init__(self, use_rbf=False, n_features = None):
		self.env = gym.make("CartPole-v0")
		self.name = "CartPole"
		self.env._max_episode_steps = int(1e10)
		self.use_rbf = use_rbf
		self.action_dim = 2
		if self.use_rbf is True:
			self.obs_dim = n_features
			self.rbf = rbf.Fourier(4, self.obs_dim, dtype = "numpy")
		else:
			self.obs_dim = 4
		self.action_type = "discrete"

	def step(self, action):
		sp, r, d, _ = self.env.step(action)
		if self.use_rbf is True:
			sp = self.rbf(sp)
		return sp, r, d, _

	def reset(self):
		s = self.env.reset()
		if self.use_rbf is True:
			s = self.rbf(s)
		return s


class SparseCartPole(Env):
	def __init__(self, use_rbf=False, n_features = None):
		self.env = gym.make("CartPole-v0")
		self.name = "SparseCartPole"
		self.env._max_episode_steps = int(1e10)
		self.use_rbf = use_rbf
		self.action_dim = 2
		if self.use_rbf is True:
			self.obs_dim = n_features
			self.rbf = rbf.Fourier(4, self.obs_dim, dtype = "numpy")
		else:
			self.obs_dim = 4
		self.action_type = "discrete"

	def step(self, action):
		sp, r, d, _ = self.env.step(action)
		if self.use_rbf is True:
			sp = self.rbf(sp)
		if d is True:
			r = -1
		else:
			r = 0
		return sp, r, d, _

	def reset(self):
		s = self.env.reset()
		if self.use_rbf is True:
			s = self.rbf(s)
		return s


class MountainCar(Env):
	def __init__(self):
		self.env = gym.make("MountainCar-v0")
		self.env._max_episode_steps = int(1e10)
		self.name = "MountainCar"
		self.obs_dim = 2
		self.action_dim = 3
		self.action_type = "discrete"
