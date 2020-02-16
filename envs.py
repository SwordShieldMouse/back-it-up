import gym
import numpy as np
import matplotlib.pyplot as plt
import os
import re
import itertools
from emdp import examples
from emdp import actions
from emdp import analytic
from emdp.chainworld import build_chain_MDP
from emdp.gridworld import build_simple_grid
import utils
import numpy as np
import rbf
	

class EMDP():
	""" From https://github.com/zafarali/emdp/tree/master/emdp """
	def __init__(self):
		pass
	
	def true_Q(self, policy, gamma, temp = None):
		""" if a temperature temp is given, return the soft action value """
		# first calculate V pi
		entropy = np.zeros(self.n_states)
		R = np.zeros(self.n_states) # for E[R]
		for j in range(self.n_states):
			R[j] = np.sum(self.mdp.R[j, :] * policy[j])
			# print(policy[j], j)
			entropy[j] = -(policy[j] * np.log(policy[j])).sum()
		P = self.mdp.P # holds p(s, a, s') 
		# calculate E[R]
		if temp is not None:
			R += temp * entropy
		P_ssp = np.zeros((self.n_states, self.n_states))
		for j in range(self.n_states):
			P_ssp[j, :] = (P[j, :, :] * policy[j].reshape(self.n_actions[j], 1)).sum(axis = 0)
		# calculate true V: V = R + gamma Ps,sp V ==> V = (I - gamma Ps,sp)^{-1} R
		V = np.matmul(np.linalg.pinv(np.eye(self.n_states) - gamma * P_ssp), R)
		# calculate true Q with Q = R + gamma Ps,a,sp V
		Q = self.mdp.R + gamma * np.multiply(P, V.reshape(1, 1, self.n_states)).sum(axis = -1)
		return Q

	def step(self, a):
		# print(a, actions.UP, actions.DOWN, actions.LEFT, actions.RIGHT)
		sp, r, done, _ = self.mdp.step(self.actions[a])
		sp = sp.tolist().index(1)
		return sp, r, done, _

	
	def reset(self):
		return self.mdp.reset().tolist().index(1)
	

class FiveWorld(EMDP):
	def __init__(self, init_policy = True):
		self.name = "FiveWorld"
		self.env_type = "tabular"
		self.n_states = 5
		self.mdp = build_chain_MDP(n_states=5, p_success=1., reward_spec=[(2, actions.RIGHT, -1), (2, actions.LEFT, 1), (3, actions.RIGHT, 10), (1, actions.RIGHT, -1)],
                    starting_distribution=np.array([0,0,1,0,0]),
                    terminal_states=[0, 4], gamma=0.99)
		self.actions = [actions.LEFT, actions.RIGHT]
		self.n_actions = [2] * self.n_states
		self.empty_Q = np.zeros([5, 2])
		self.empty_policy =  np.zeros([5, 2])
		if init_policy is True:
			# Want to initialize to make it likely to take the short-sighted action (left)
			# init policy
			# self.empty_policy[2, 0] = np.log(0.9)
			# self.empty_policy[2, 1] = np.log(0.1)
			# self.empty_policy[1, 0] = np.log(0.9)
			# self.empty_policy[1, 1] = np.log(0.1)
			# init Q estimate
			self.empty_Q[2, 0] = 10
			self.empty_Q[1, 0] = 10
			self.name += "WithInit"


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
	def __init__(self, use_rbf = False):
		self.env = gym.make("LunarLander-v2")
		self.env._max_episode_steps = int(1e10)
		self.action_dim = 4
		self.name = "LunarLander"

		self.use_rbf = use_rbf
		if self.use_rbf is True:
			self.obs_dim = 256
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



class Acrobot(Env):
	def __init__(self, use_rbf = False):
		self.env = gym.make("Acrobot-v1")
		self.name = "Acrobot"
		self.env._max_episode_steps = int(1e10)
		self.action_dim = 3
		self.action_type = "discrete"
		self.use_rbf = use_rbf
		if self.use_rbf is True:
			self.obs_dim = 128
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
	def __init__(self, use_rbf=False):
		self.env = gym.make("CartPole-v0")
		self.name = "CartPole"
		self.env._max_episode_steps = int(1e10)
		self.use_rbf = use_rbf
		self.action_dim = 2
		if self.use_rbf is True:
			self.obs_dim = 128
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


class MountainCar(Env):
	def __init__(self):
		self.env = gym.make("MountainCar-v0")
		self.env._max_episode_steps = int(1e10)
		self.name = "MountainCar"
		self.obs_dim = 2
		self.action_dim = 3
		self.action_type = "discrete"
